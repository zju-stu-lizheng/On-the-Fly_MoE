import torch
from modeling_mixtral import set_profile_mode, load_thresholds
from utils import myevaluate, get_model
import json 
import argparse
from peft import PeftModelForCausalLM

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
	set_profile_mode(False)
	load_thresholds(f'{threshold_path}/thresholds_0_8.pt', use_average=use_average)	
	llm, tokenizer = get_model(model_name, device_map, dtype=dtype)

	if lora_save_path != './saved/training/lora_weights.pt':
		PeftModelForCausalLM.from_pretrained(llm, lora_save_path, 'default')
			
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
	args = parser.parse_args()
	lora_save_path = args.lora_path
	dtype = torch.float16
	print('lora_save_path: ', lora_save_path, dtype)
	print('task_name_list: ', args.task_name_list)
	doeval(dtype, lora_save_path, args)