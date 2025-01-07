import torch
from modeling_mixtral import set_profile_mode, load_thresholds
from utils import myevaluate, get_model, CompensatedModel
import json 
import argparse
from hqq.core.quantize import *
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.peft import PeftUtils

def doeval(dtype, lora_save_path):
	with open('../path.json', 'r') as f:
		path = json.load(f)
		model_name = path['mixtral']
		threshold_path = path['training_sparsity_path']

	with open('./device_map_1.json', 'r') as f:
		device_map = json.load(f)
	
	## 开启稀疏模式
	set_profile_mode(False)
	load_thresholds(f'{threshold_path}/thresholds_0_8.pt')
	llm, tokenizer = get_model(model_name, device_map, dtype=dtype)

	q4_config    = BaseQuantizeConfig(nbits=8, group_size=64) 
	q3_config    = BaseQuantizeConfig(nbits=2, group_size=64)

	quant_config = {
	'block_sparse_moe.experts.w3'  :q3_config,
	}
	AutoHQQHFModel.quantize_model(llm, quant_config=quant_config, compute_dtype=dtype, device=device_map)  
		
	for i in range(32):
		if i == 31:
			print(f"Layer {i} done...")
		for j in range(8):
			llmdevice = llm.model.layers[i].block_sparse_moe.experts[j].w3.device
			llm.model.layers[i].block_sparse_moe.experts[j].w3 = \
			CompensatedModel(llm.model.layers[i].block_sparse_moe.experts[j].w3, '/home/lz/On-the-Fly_MoE_Inference/quantize/saved/eora/', layerid=i, expertid=j, dtype=dtype, device=llmdevice).to(llmdevice)

	base_lora_params = {'lora_type':'default', 'r':128, 'lora_alpha':128, 'dropout':0.05, 'train_dtype':dtype}

	lora_params      = {'self_attn.q_proj': base_lora_params,
					'self_attn.k_proj': base_lora_params,
					'self_attn.v_proj': base_lora_params,
					'self_attn.o_proj': base_lora_params,
					'block_sparse_moe.experts.w1'   : base_lora_params,
					'block_sparse_moe.experts.w3'   : base_lora_params,
					'block_sparse_moe.experts.w2'   : base_lora_params}
	PeftUtils.add_lora(llm, lora_params)
	PeftUtils.load_lora_weights(llm, lora_save_path)
			
	task_name_list=['winogrande','sciq','openbookqa','arc_challenge','arc_easy']
	num_fewshot = 0
	myevaluate(task_name_list, llm, tokenizer, num_fewshot, 'cuda')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--lora_path", type=str, default='./saved/training/lora_weights.pt')
	args = parser.parse_args()
	lora_save_path = args.lora_path
	dtype = torch.float16
	print('lora_save_path: ', lora_save_path, dtype)
	doeval(dtype, lora_save_path)