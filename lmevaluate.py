# Test Model
import os
import json
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, MixtralForCausalLM
from convert_model import convert_llama_model, convert_mixtral_model
# export HF_ENDPOINT="https://hf-mirror.com"
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator


def _load_model(model_name = "Llama3-8b"):
    print(f"Loading model {model_name}")
    ### from path.json read paths of model and dataset
    with open('path.json', 'r') as file:
        paths = json.load(file)
        model_path = paths.get(model_name, '')
    if "Llama" in model_name:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            use_cache=True,
            torch_dtype=torch.float16,
        )
    else:
        model = MixtralForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            use_cache=True,
            torch_dtype=torch.float16,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer

def evaluate(task_name_list, model, tokenizer, num_fewshot, device):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
    model=hflm,
    tasks=task_name_list,
    num_fewshot=num_fewshot)
    print(results['results'])


def main(task_name_list, model_name, sparsity, start_num, end_num, token_sparsity, is_sparsity, device, num_fewshot, beta=0.1, gamma=0.3):
    model, tokenizer = _load_model(model_name)
    
    if is_sparsity == True:
        if "Llama" in model_name:
            model = convert_llama_model(model, sparsity, start_num, end_num, token_sparsity, 
                                        beta=beta, gamma=gamma, use_core=False)
        else:
            convert_mixtral_model(model, start_num=-1, end_num=32, gamma=0.2,)

    evaluate(task_name_list, model, tokenizer, num_fewshot, device)
    for layerid in range(start_num+1,end_num):
        for expertid in range(8):
            model.model.layers[layerid].block_sparse_moe.experts[expertid].print_ratio()

# triviaqa
task_list=['winogrande','sciq','openbookqa','arc_challenge','arc_easy']
# 'boolq',
# task_list=['truthfulqa_gen','triviaqa_gen']
num_fewshot = 0
beta = 0.1
gammma = 0.2
start_num = -1

# task_list=['truthfulqa_gen','boolq']
main(task_name_list=task_list, model_name="mixtral", sparsity=0.1, start_num=start_num, end_num=32, token_sparsity=0.1,
      is_sparsity=True, device='cuda', num_fewshot=num_fewshot, beta=beta, gamma=gammma)
