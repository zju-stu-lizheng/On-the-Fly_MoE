import torch
from modeling_mixtral import set_profile_mode, load_thresholds
from utils import myevaluate, get_model, CompensatedModel, get_lora_params
import json 
import argparse
from hqq.core.quantize import *
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.peft import PeftUtils

def doeval(dtype, lora_save_path, threshold_path_name):
	with open('../path.json', 'r') as f:
		path = json.load(f)
		model_name = path['mixtral']
		threshold_path = path[threshold_path_name]

	with open('./device_map_1.json', 'r') as f:
		device_map = json.load(f)
	
	## 开启稀疏模式
	set_profile_mode(False)
	load_thresholds(f'{threshold_path}/thresholds_0_8.pt', use_average=False)
	llm, tokenizer = get_model(model_name, device_map, dtype=dtype)

	q4_config    = BaseQuantizeConfig(nbits=8, group_size=64) 
	q3_config    = BaseQuantizeConfig(nbits=2, group_size=64)

	quant_config = {
	'block_sparse_moe.experts.w3'  :q3_config,
	}
	AutoHQQHFModel.quantize_model(llm, quant_config=quant_config, compute_dtype=dtype, device=device_map)  

	lora_params = get_lora_params(dtype, test=False)
	print(lora_params)
	PeftUtils.add_lora(llm, lora_params)
	PeftUtils.load_lora_weights(llm, lora_save_path)
	#### 加载量化后的权重, w3: lora+eora
	for i in range(32):
		if i == 31:
			print(f"Layer {i} done...")
		for j in range(8):
			llmdevice = llm.model.layers[i].block_sparse_moe.experts[j].w3.linear_layer.device
			llm.model.layers[i].block_sparse_moe.experts[j].w3.linear_layer = \
			CompensatedModel(llm.model.layers[i].block_sparse_moe.experts[j].w3.linear_layer, '/home/lz/On-the-Fly_MoE_Inference/quantize/saved/eora/', layerid=i, expertid=j, device=llmdevice)
			
	task_name_list=['winogrande','sciq','openbookqa','arc_challenge','arc_easy']
	num_fewshot = 0
	myevaluate(task_name_list, llm, tokenizer, num_fewshot, 'cuda')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--lora_path", type=str, default='./saved/training/lora_weights.pt')
	parser.add_argument("--threshold_path", type=str, default='training_sparsity_path')
	args = parser.parse_args()
	lora_save_path = args.lora_path
	dtype = torch.float16
	print('lora_save_path: ', lora_save_path, dtype)
	doeval(dtype, lora_save_path, threshold_path_name=args.threshold_path)