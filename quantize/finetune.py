import torch
from modeling_mixtral import set_profile_mode, load_thresholds
import json
from utils import get_model
from hqq.core.quantize import *
from datasets import load_dataset, Dataset
import functools
from transformers import AutoTokenizer, BitsAndBytesConfig, AdamW
from transformers import (
	DataCollatorForSeq2Seq,
	Trainer,
	TrainingArguments
)
import argparse
import tensorboard
from peft import LoraConfig, get_peft_model
import math

def get_model_for_training(model_name, dtype, device_map, threshold_path, sparsity_level=0.8, use_average=True):
	set_profile_mode(False)
	filepath = str(sparsity_level).replace('.', '_')
	if math.fabs(args.sparsity_level - 0) < 1e-5:
		print('use zero sparsity')
		load_thresholds(f'{threshold_path}/thresholds_{filepath}.pt', use_average=use_average, zero=True)
	else:
		load_thresholds(f'{threshold_path}/thresholds_{filepath}.pt', use_average=use_average)
	print('using ',dtype)
	model, tokenizer = get_model(model_name, device_map, dtype=dtype)

	target_modules = ["w1", "w2", "w3", "q_proj", "k_proj", "v_proj", "o_proj"]
	peft_config = LoraConfig(
			lora_alpha=32,
			lora_dropout=0.01,
			r=64,
			bias="none",
			target_modules=target_modules,
			task_type="CAUSAL_LM"
		)
	for name, param in model.named_parameters():
		# freeze base model's layers
		param.requires_grad = False

	model = get_peft_model(model, peft_config) 

	return model, tokenizer

def preprocess_data(batch, tokenizer):
	# 使用 tokenizer 将文本数据转换为模型输入
	inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
	inputs["labels"] = inputs.input_ids.clone()
	return inputs

### get openmath and fineweb dataset
def get_combined_dataset(fineweb_path, tokenizer, test_num = 0.1, seed = 42):
	if fineweb_path.endswith("parquet"):
		openmath = load_dataset("parquet",data_files="/home/bcds/venv/dilab/floe/dataset/open-web-math/data/train-00045-of-00114-dae3a4ce38fbb868.parquet")
		#55397
		fineweb = load_dataset("parquet",data_files=fineweb_path) #726000
	else:
		openmath = load_dataset("/home/lz/web-math/",data_files="/home/lz/web-math/openmath1.json")
		fineweb = load_dataset(fineweb_path)
	openmath_text = openmath['train']['text'][:8000] 
	fineweb_text = fineweb['train']['text'][:35000] 

	# combined_text = fineweb_text
	combined_text = openmath_text + fineweb_text
	combined_dataset = Dataset.from_dict({"text": combined_text})
	combined_train = combined_dataset.train_test_split(test_size=test_num, seed=seed)
	train_data = combined_train['train']
	test_data = combined_train['test']

	new_train_data = train_data.map(
		functools.partial(
		preprocess_data,
		tokenizer=tokenizer
	), batched=True)
	new_test_data = test_data.map(
		functools.partial(
		preprocess_data,
		tokenizer=tokenizer
	), batched=True)
	new_train_data.shuffle(seed)
	new_test_data.shuffle(seed)

	return new_train_data, new_test_data


def get_bagel_dataset(bagel_path, tokenizer, test_num = 0.1, seed = 42):
	bagel = load_dataset("json", data_files=bagel_path)
	bagel = bagel['train']['text'][:15000] 
	combined_dataset = Dataset.from_dict({"text": bagel})

	combined_train = combined_dataset.train_test_split(test_size=test_num, seed=seed)
	train_data = combined_train['train']
	test_data = combined_train['test']

	new_train_data = train_data.map(
		functools.partial(
		preprocess_data,
		tokenizer=tokenizer
	), batched=True)
	new_test_data = test_data.map(
		functools.partial(
		preprocess_data,
		tokenizer=tokenizer
	), batched=True)
	new_train_data.shuffle(seed)
	new_test_data.shuffle(seed)

	return new_train_data, new_test_data

def dotrain(dtype, args, save_steps = 300):
	model_save_path = args.model_save_path
	epochs = args.epoch
	training_steps = args.training_steps
	use_average = args.use_average

	# # 加载 C4 数据集的验证集
	with open('../path.json', 'r') as file:
		paths = json.load(file)
		fineweb_path = paths.get('fineweb', '')
		model_name = paths.get('mixtral','')
		threshold_path = paths.get('chess_up_threshold','')
		bagel_path = paths.get("bagel_json","")

	with open('./device_map.json', 'r') as f:
		device_map = json.load(f)

	### get peft model for training
	llm, tokenizer = get_model_for_training(model_name, dtype, device_map, threshold_path, args.sparsity_level, use_average=use_average)
	# new_train_data, new_test_data = get_combined_dataset(fineweb_path, tokenizer, test_num = 0.1, seed = 42)
	new_train_data, new_test_data = get_bagel_dataset(bagel_path, tokenizer)
	# model_save_path='./saved/training/less_new'
	learning_rate = 1e-4
	micro_batch_size=8
	# epochs = 2
	save_total_limit = 6
	sample_num = training_steps
	optimizer=AdamW(filter(lambda p : p.requires_grad, llm.parameters()),lr=learning_rate)
	linear_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(sample_num*epochs) // micro_batch_size)
	training_args = TrainingArguments(
		output_dir=model_save_path,
		num_train_epochs=epochs,
		# max_steps=opt.max_steps,
		# fp16=True,
		bf16=True,
		optim="adamw_torch",# paged_adamw_8bit
		learning_rate=learning_rate,
		lr_scheduler_type="cosine",
		per_device_train_batch_size=micro_batch_size,
		gradient_accumulation_steps=1,
		gradient_checkpointing=False,   ### 先设置成False
		group_by_length=False,
		logging_steps=20,
		eval_steps=20,
		save_strategy="steps",
		save_only_model=True,
		save_steps=save_steps,
		save_total_limit=save_total_limit,
		disable_tqdm=False,
		report_to='tensorboard',
		logging_dir=f'{model_save_path}/logs/'
	)

	trainer = Trainer(
		model=llm,
		train_dataset=new_train_data.select(range(training_steps)),
		eval_dataset=new_test_data.select(range(int(training_steps/10))),
		args=training_args,
		optimizers=(optimizer, linear_scheduler),
		data_collator=DataCollatorForSeq2Seq(
		tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
	)

	# silence the warnings. re-enable for inference!
	llm.config.use_cache = False
	trainer.train()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_save_path", type=str, default='./saved/training/less_new')
	parser.add_argument("--use_average", action='store_true')
	parser.add_argument("--epoch", type=int, default=2)
	parser.add_argument("--sparsity_level", type=float, default=0.8)
	parser.add_argument("--training_steps", type=int, default=3000, help='the number of sentences from datasets(3000-10000)')

	args = parser.parse_args()

	dtype = torch.bfloat16
	print('model_save_path: ', args.model_save_path, dtype)
	dotrain(dtype, args, save_steps = 600)