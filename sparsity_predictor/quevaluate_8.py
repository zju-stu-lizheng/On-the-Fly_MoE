import sys
sys.path.append("../quantize")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"
import torch
import math
from modeling_mixtral_predict import MixtralForCausalLM, load_thresholds
from transformers import AutoTokenizer, BitsAndBytesConfig
from utils import myevaluate
import json 
import argparse
from peft import PeftModelForCausalLM

import transformers
from hqq.core.quantize import *
from hqq.models.hf.mixtral import MixtralPatch
import transformers
from hqq.models.base import BaseHQQModel
from accelerate import init_empty_weights


class BaseHQQHFModel(BaseHQQModel):
    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir):
        model.config.save_pretrained(save_dir)

    # Create empty model from config
    @classmethod
    def create_model(cls, save_dir, kwargs):
        model_kwargs = {}
        for key in ["attn_implementation"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        config = transformers.AutoConfig.from_pretrained(
            cls.get_config_file(save_dir)
        )

        with init_empty_weights():
            model = MixtralForCausalLM._from_config(config, **model_kwargs)

        return model

class MixtralHQQ(MixtralPatch, BaseHQQHFModel):
    pass
 
def get_model(model_name, device_map, dtype=torch.bfloat16, use_cache=True):
	quantization_config = BitsAndBytesConfig(load_in_8bit=True,
										  llm_int8_skip_modules=["embed_tokens", "lm_head", "w3"])

	llm = MixtralForCausalLM.from_pretrained(
		model_name,
		device_map=device_map,
		use_cache=use_cache,
		quantization_config=quantization_config,
		torch_dtype=dtype,
	) 
	# save_dir = '/home/bcds/On-the-Fly_MoE_Inference/offloading/hqqsaved'
	# dtype = torch.float16
	# llm = MixtralHQQ.from_quantized(save_dir, compute_dtype=dtype, device='cuda:0', use_cache=True)
	# HQQLinear.set_backend(HQQBackend.PYTORCH)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id

	return llm, tokenizer

def doeval(dtype, lora_save_path, args):
	threshold_path_name=args.threshold_path
	use_average = args.use_average
	with open('../path.json', 'r') as f:
		path = json.load(f)
		model_name = path['mixtral']
		threshold_path = path[threshold_path_name]

	with open('../quantize/device_map_1.json', 'r') as f:
		device_map = json.load(f)
	
	## 开启稀疏模式
	# set_profile_mode(False)
	filepath = str(args.sparsity_level).replace('.', '_')
	if math.fabs(args.sparsity_level - 0) < 1e-5:
		print('use zero sparsity')
		load_thresholds(f'{threshold_path}/thresholds_{filepath}.pt', use_average=use_average, zero=True)
	else:
		load_thresholds(f'{threshold_path}/thresholds_{filepath}.pt', use_average=use_average,)
	
	llm, tokenizer = get_model(model_name, device_map, dtype=dtype, use_cache=False)
	if lora_save_path != './saved/training/lora_weights.pt':
		print(f'load lora model: {lora_save_path}')
		llm = PeftModelForCausalLM.from_pretrained(llm, lora_save_path, 'default')
		# 合并 LoRA 权重进行推理
		llm = llm.merge_and_unload()
		print('merge done')
	else:
		print('not loading lora model')

	# q3_config    = BaseQuantizeConfig(nbits=2, group_size=64)

	# quant_config = {
	# 	'block_sparse_moe.experts.w3'  :q3_config,
	# }
	# MixtralHQQ.quantize_model(llm, quant_config=quant_config, compute_dtype=dtype, device=device_map)  
	print(llm)
	# task_name_list=['arc_challenge']
	task_name_list = args.task_name_list
	num_fewshot = 0
	myevaluate(task_name_list, llm, tokenizer, num_fewshot, 'cuda')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--lora_path", type=str, default='./saved/training/lora_weights.pt')
	parser.add_argument('--task_name_list', nargs='+')
	parser.add_argument("--threshold_path", type=str, default='training_sparsity_path')
	parser.add_argument("--use_average", action='store_true', help='use average threshold')
	parser.add_argument("--sparsity_level", type=float, default=0.8)
	args = parser.parse_args()
	lora_save_path = args.lora_path
	dtype = torch.float16
	print('lora_save_path: ', lora_save_path, dtype)
	print('task_name_list: ', args.task_name_list)
	doeval(dtype, lora_save_path, args)