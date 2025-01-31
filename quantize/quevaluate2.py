import os
# 获取当前 Python 文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置工作目录为当前 Python 文件所在的目录
os.chdir(current_dir)
import torch
# from modeling_mixtral import set_profile_mode, load_thresholds
from transformers import MixtralForCausalLM, AutoTokenizer
from utils import myevaluate, get_model, CompensatedModel, get_lora_params
import json 
import argparse
from hqq.core.quantize import *
# from hqq.models.hf.base import AutoHQQHFModel
from hqq.models.hf.mixtral import MixtralHQQ
from hqq.core.peft import PeftUtils

def doeval(dtype, lora_save_path, args):
	threshold_path_name=args.threshold_path
	use_average = args.use_average
	with open('../path.json', 'r') as f:
		path = json.load(f)
		model_name = path['mixtral']
		threshold_path = path[threshold_path_name]

	with open('./device_map_1.json', 'r') as f:
		device_map = json.load(f)
	
	## 开启稀疏模式
	# set_profile_mode(False)
	# load_thresholds(f'{threshold_path}/thresholds_0_8.pt', use_average=use_average, zero=True)	
	# llm, tokenizer = get_model(model_name, device_map, dtype=dtype)
	llm = MixtralForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        use_cache=True,
        torch_dtype=dtype,
    ) 
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id

	# q4_config    = BaseQuantizeConfig(nbits=8, group_size=64) 
	q3_config    = BaseQuantizeConfig(nbits=args.quantization_bit, group_size=64)

	quant_config = {
		'self_attn.q_proj':q3_config,
		'self_attn.k_proj':q3_config,
		'self_attn.v_proj':q3_config,
		'self_attn.o_proj':q3_config,
		'block_sparse_moe.experts.w1'  :q3_config,
		'block_sparse_moe.experts.w2'  :q3_config,
		'block_sparse_moe.experts.w3'  :q3_config,
	}
	MixtralHQQ.quantize_model(llm, quant_config=quant_config, compute_dtype=dtype, device=device_map)  

	# lora_params = get_lora_params(dtype, test=False)
	# print(lora_params)
	# PeftUtils.add_lora(llm, lora_params)
	# PeftUtils.load_lora_weights(llm, lora_save_path)
	# PeftUtils.cast_lora_weights(llm, dtype)
	#### 加载量化后的权重, w3: lora+eora
	# for i in range(32):
	# 	if i == 31:
	# 		print(f"Layer {i} done...")
	# 	for j in range(8):
	# 		llmdevice = llm.model.layers[i].block_sparse_moe.experts[j].w3.linear_layer.device
	# 		llm.model.layers[i].block_sparse_moe.experts[j].w3.linear_layer = \
	# 		CompensatedModel(llm.model.layers[i].block_sparse_moe.experts[j].w3.linear_layer, '/home/lz/On-the-Fly_MoE_Inference/quantize/saved/eora/', layerid=i, expertid=j, device=llmdevice)
			
	# task_name_list=['boolq','sciq','openbookqa', 'winogrande','arc_challenge','arc_easy']
	# num_fewshot = 0
	task_name_list = args.task_name_list
	num_fewshot = args.num_fewshot

	myevaluate(task_name_list, llm, tokenizer, num_fewshot, 'cuda')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--lora_path", type=str, default='./saved/training/lora_weights.pt')
	parser.add_argument("--threshold_path", type=str, default='training_sparsity_path')
	parser.add_argument("--use_average", action='store_true', help='use average threshold')
	parser.add_argument("--num_fewshot", type=int, default=0)
	parser.add_argument('--task_name_list', nargs='+')
	parser.add_argument('--quantization_bit', type=int, default=2)

	args = parser.parse_args()
	lora_save_path = args.lora_path
	dtype = torch.float16
	print(args)
	print('lora_save_path: ', lora_save_path, dtype)
	doeval(dtype, lora_save_path, args)