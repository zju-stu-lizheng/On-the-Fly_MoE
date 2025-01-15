import sys
sys.path.append("../quantize")
import torch
import math
from modeling_mixtral import load_thresholds
from utils import myevaluate, get_model
import json 
import argparse

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