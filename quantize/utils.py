import os
import torch
from modeling_mixtral import MixtralForCausalLM
from transformers import AutoTokenizer
import transformers
from hqq.core.peft import PeftUtils

# Test Model
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

def get_lora_params(dtype, test=False):
    ### lora_params for the model
    base_lora_params = {'lora_type':'default', 'r':128, 'lora_alpha':128, 'dropout':0.05, 'train_dtype':dtype}

    if test:
        lora_params      = {'block_sparse_moe.experts.w3'   : base_lora_params}
    else:
        lora_params      = {'self_attn.q_proj': base_lora_params,
                    'self_attn.k_proj': base_lora_params,
                    'self_attn.v_proj': base_lora_params,
                    'self_attn.o_proj': base_lora_params,
                    'block_sparse_moe.experts.w1'   : base_lora_params,
                    'block_sparse_moe.experts.w3'   : base_lora_params,
                    'block_sparse_moe.experts.w2'   : base_lora_params}
    return lora_params

class CustomTrainer(transformers.Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        # 如果没有指定output_dir，则使用训练参数中的输出目录
        if output_dir is None:
            output_dir = self.args.output_dir #这里的args不是该脚本的输入，而是TrainerArgs

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 保存完整的模型参数
        # torch.save(self.model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        
        self.model.eval()
        PeftUtils.cast_lora_weights(self.model, dtype=torch.float16)

        #Save LoRA weights
        PeftUtils.save_lora_weights(self.model, output_dir+'_lora_combine.pt')

        PeftUtils.cast_lora_weights(self.model, dtype=torch.bfloat16)
        self.model.train()

        # 保存配置文件和tokenizer
        self.model.config.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

class CompensatedModel(torch.nn.Module):
    def __init__(self, model, path, layerid, expertid, dtype=torch.float16, device='cuda:0'):
        super(CompensatedModel, self).__init__()
        self.model = model
        ### self.A and self.B_prime are initialized as the values loaded from the file
        self.A = torch.load(path + f'A_{layerid}_{expertid}.pt', map_location='cuda:0').to(dtype).to(device)
        self.B_prime = torch.load(path + f'B_prime_{layerid}_{expertid}.pt', map_location='cuda:0').to(dtype).to(device)
        
    def forward(self, input_ids):
        outputs = self.model(input_ids)
        residual = (input_ids @ self.A.T) @ self.B_prime.T
        outputs += residual
        return outputs

def get_model(model_name, device_map, dtype=torch.bfloat16, use_cache=True):
    if 'Llama' in model_name:
        from modeling_llama_down import LlamaForCausalLM
        # load_thresholds()
        llm = LlamaForCausalLM.from_pretrained(
            model_name,
            # device_map=device_map,
            use_cache=False,
            torch_dtype=torch.float16,
        ).cuda(0)
    elif 'Mixtral' in model_name:
        llm = MixtralForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            use_cache=use_cache,
            torch_dtype=dtype,
        ) 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return llm, tokenizer

def evaluate(task_name_list, model, tokenizer, num_fewshot, device):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
    model=hflm,
    tasks=task_name_list,
    num_fewshot=num_fewshot)
    print(results['results'])

def myevaluate(task_name_list, model, tokenizer, num_fewshot, device):
    model.eval()
    evaluate(task_name_list, model, tokenizer, num_fewshot, device)
    avg_list = []
    for layerid in range(32):
        # for expertid in range(8):
        avg_list.append(model.model.layers[layerid].mlp.get_ratio())
        model.model.layers[layerid].mlp.print_ratio()
        # avg_list.append(model.model.layers[layerid].block_sparse_moe.experts[expertid].get_ratio())
    
    print('Average Sparsity: ', f'{sum(avg_list)/len(avg_list):.4f}')
    print('Max Sparsity: {:.4f}'.format(max(avg_list)))
    print('Min Sparsity: {:.4f}'.format(min(avg_list)))
