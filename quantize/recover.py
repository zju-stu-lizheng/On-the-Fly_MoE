import torch
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
import transformers
from modeling_mixtral import set_profile_mode, load_thresholds
import json
from utils import get_model, CompensatedModel, get_lora_params, CustomTrainer
from hqq.core.quantize import *
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.peft import PeftUtils
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

def get_peft_model(model_name, dtype, device_map, threshold_path):
	set_profile_mode(False)
	load_thresholds(f'{threshold_path}/thresholds_0_8.pt', use_average=False)
	print('using ',dtype)
	llm, tokenizer = get_model(model_name, device_map, dtype=dtype)

	q4_config    = BaseQuantizeConfig(nbits=8, group_size=64) 
	q3_config    = BaseQuantizeConfig(nbits=2, group_size=64)

	quant_config      = {'block_sparse_moe.experts.w3'   : q3_config}
	AutoHQQHFModel.quantize_model(llm, quant_config=quant_config, compute_dtype=dtype, device=device_map)

	lora_params = get_lora_params(dtype, test=True)
	print(lora_params)
	PeftUtils.add_lora(llm, lora_params)

	for i in range(32):
		if i == 31:
			print(f"Layer {i} done...")
		for j in range(8):
			llmdevice = llm.model.layers[i].block_sparse_moe.experts[j].w3.linear_layer.device
			llm.model.layers[i].block_sparse_moe.experts[j].w3.linear_layer = \
			CompensatedModel(llm.model.layers[i].block_sparse_moe.experts[j].w3.linear_layer, '/home/lz/On-the-Fly_MoE_Inference/quantize/saved/eora/', layerid=i, expertid=j, device=llmdevice)
	return llm, tokenizer

def preprocess_data(batch, tokenizer):
	# 使用 tokenizer 将文本数据转换为模型输入
	inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
	inputs["labels"] = inputs.input_ids.clone()
	return inputs

### get openmath and fineweb dataset
def get_combined_dataset(fineweb_path, tokenizer, test_num = 0.1, seed = 42):
	openmath = load_dataset("/home/lz/web-math/",data_files="/home/lz/web-math/openmath1.json")
	fineweb = load_dataset(fineweb_path)
	openmath_text = openmath['train']['text'][:4000] 
	fineweb_text = fineweb['train']['text'][:10000]

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


def dotrain(dtype, args):
	model_save_path = args.model_save_path
	epochs = args.epoch
	training_steps = args.training_steps

	# # 加载 C4 数据集的验证集
	with open('../path.json', 'r') as file:
		paths = json.load(file)
		fineweb_path = paths.get('fineweb', '')
		model_name = paths.get('mixtral','')
		threshold_path = paths.get('chess_up_sparsity_threshold','')

	with open('./device_map.json', 'r') as f:
		device_map = json.load(f)
	### get peft model for training
	llm, tokenizer = get_peft_model(model_name, dtype, device_map, threshold_path)
	new_train_data, new_test_data = get_combined_dataset(fineweb_path, tokenizer, test_num = 0.1, seed = 42)

	# model_save_path='./saved/training/less_new'
	learning_rate = 1e-4
	micro_batch_size=8
	# epochs = 2
	save_steps = 300
	save_total_limit = 6
	sample_num = len(new_train_data)
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

	trainer = CustomTrainer(
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
	parser.add_argument("--epoch", type=int, default=2)
	parser.add_argument("--training_steps", type=int, default=3000, help='the number of sentences from datasets(3000-10000)')

	args = parser.parse_args()

	dtype = torch.bfloat16
	print('model_save_path: ', args.model_save_path, dtype)
	dotrain(dtype, args)