import torch
# from modeling_mixtral import set_profile_mode, load_thresholds
from transformers import LlamaForCausalLM as LlamaTeacher
from modeling_llama import LlamaForCausalLM as LlamaStudent
from modeling_llama import load_thresholds
import json
# from hqq.core.quantize import *
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
import torch.nn.functional as F


# Configuration
config = {
    "project_name": "distil-logits",
    "dataset": {
        "name": "/home/bcds/On-the-Fly_MoE_Inference/OpenHermes-2.5/openhermes2_5.json",
        "split": "train",
        "num_samples": 20000, # You can pass a number here to limit the number of samples to use.
        "seed": 42
    },
    "models": {
        "teacher": "/home/bcds/venv/dilab/Meta-Llama-3.1-8B",
        "student": "/home/bcds/venv/dilab/Meta-Llama-3.1-8B"
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "training": {
        "output_dir": "./results/finetune",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "model_config": {
        "use_flash_attention": True
    }
    # "spectrum": {
    #     "layers_to_unfreeze": "/workspace/spectrum/snr_results_Qwen-Qwen2-1.5B_unfrozenparameters_50percent.yaml" # You can pass a spectrum yaml file here to freeze layers identified by spectrum.
    # }
}


def get_model_for_training(dtype, device_map, threshold_path, sparsity_level=0.8, use_average=True):
	# set_th_sparsity(int(sparsity_level*100))
	filepath = str(sparsity_level).replace('.', '_')
	if math.fabs(args.sparsity_level - 0) < 1e-5:
		print('use zero sparsity')
		load_thresholds(f'{threshold_path}/thresholds_{filepath}.pt', use_average=use_average, zero=True)
	else:
		load_thresholds(f'{threshold_path}/thresholds_{filepath}.pt', use_average=use_average)
	print('using ',dtype)
	student = LlamaStudent.from_pretrained(
		config["models"]["student"],
		use_cache=False,
		torch_dtype=dtype,
    ).cuda(0)
	teacher = LlamaTeacher.from_pretrained(
		config["models"]["teacher"],
		use_cache=False,
		torch_dtype=dtype,
    ).cuda(0)
	tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id

	target_modules = ["w1", "w2", "w3", "q_proj", "k_proj", "v_proj", "o_proj"]
	peft_config = LoraConfig(
			lora_alpha=64,
			lora_dropout=0.01,
			r=64,
			bias="none",
			target_modules=target_modules,
			task_type="CAUSAL_LM"
		)
	for name, param in student.named_parameters():
		# freeze base model's layers
		param.requires_grad = False

	student = get_peft_model(student, peft_config) 

	return student, teacher, tokenizer

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
	fineweb_text = fineweb['train']['text'][:30000] 

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

def get_wiki_dataset(tokenizer, sample_num=24000, seed=42):
	# 加载fineweb数据集
	test_num = min(sample_num//10, 1000)

	### use bagel
	wikitext = load_dataset("parquet", data_files="/home/bcds/On-the-Fly_MoE_Inference/wikitext-2/data/train-00000-of-00001.parquet")
	fineweb = load_dataset("json", data_files="/home/bcds/On-the-Fly_MoE_Inference/bagel-v0.5/processed_data.json")
	# fineweb = load_dataset("parquet", data_files="/home/bcds/venv/dilab/floe/dataset/finewebedu/sample/10BT/000_00000.parquet")
	# fineweb = load_dataset("json", data_files="/home/zyx/moe/fineweb-edu/fineweb_edu_sample100000.json")

	dataset_1 = wikitext['train']['text'][:15000] 
	dataset_2 = fineweb['train']['text'][:15000] 
	
	## 随机从fineweb中抽取sample_num条数据
	combined_text = dataset_1 + dataset_2
	combined_dataset = Dataset.from_dict({"text": combined_text})

	fineweb_train = combined_dataset.train_test_split(test_size=test_num, seed=seed)
	train_data = fineweb_train['train'].select(range(sample_num))
	test_data = fineweb_train['test']

	fineweb_train_data = train_data.map(
		functools.partial(
		preprocess_data,
		tokenizer=tokenizer
	), batched=True)
	fineweb_test_data = test_data.map(
		functools.partial(
		preprocess_data,
		tokenizer=tokenizer
	), batched=True)
	fineweb_train_data.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
	fineweb_test_data.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
	fineweb_train_data.shuffle(seed)
	fineweb_test_data.shuffle(seed)

	return fineweb_train_data, fineweb_test_data


def get_bagel_dataset(sft_path, fineweb_path, tokenizer, test_num = 0.1, seed = 42, merge=False):
	bagel = load_dataset("json", data_files=sft_path)
	bagel_text = bagel['train']['text'][:15000] 
	combined_text = bagel_text 

	if merge:
		print("merge datasets with", fineweb_path)
		filetype = fineweb_path.split(".")[-1]
		fineweb = load_dataset(filetype, data_files=fineweb_path)
		fineweb_text = fineweb['train']['text'][:15000] 
		combined_text = bagel_text + fineweb_text 

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

def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits

class LogitsTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in inputs.items()}
        # self.teacher_model = self.teacher_model.to(model.device)
        # print(inputs)
        
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):
        student_logits, teacher_logits = pad_logits(student_logits.to(self.model.device), teacher_logits.to(self.model.device))
        
        student_logits_scaled = student_logits / config["distillation"]["temperature"]
        teacher_logits_scaled = teacher_logits / config["distillation"]["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (config["distillation"]["temperature"] ** 2) / config["tokenizer"]["max_length"]

		### 这里是两个各0.5，所以总的loss比之前要低
        return config["distillation"]["alpha"] * loss_kd + (1 - config["distillation"]["alpha"]) * original_loss

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
		threshold_path = paths.get('llama_threshold','')
		bagel_path = paths.get("bagel_json","")
		openhermes_path = paths.get("openhermes","")

	with open('./device_map.json', 'r') as f:
		device_map = json.load(f)

	### get peft model for training
	student, teacher, tokenizer = get_model_for_training(dtype, device_map, threshold_path, args.sparsity_level, use_average=use_average)

	new_train_data, new_test_data = get_combined_dataset(fineweb_path, tokenizer, test_num = 0.1, seed = 42)
	# new_train_data, new_test_data = get_bagel_dataset(bagel_path, fineweb_path, tokenizer, merge=args.merge)
	# training_steps = 20000
	# new_train_data, new_test_data = get_wiki_dataset(tokenizer, sample_num=training_steps)
	# model_save_path='./saved/training/less_new'
	learning_rate = 1e-4
	micro_batch_size = 1
	# epochs = 2
	save_total_limit = 6
	sample_num = training_steps
	optimizer=AdamW(filter(lambda p : p.requires_grad, student.parameters()),lr=learning_rate)
	linear_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(sample_num*epochs) // micro_batch_size)
	training_args = TrainingArguments(
		output_dir=config["training"]["output_dir"],
		num_train_epochs=epochs,
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

	if args.use_distill:
		### 继承Trainer并重写compute_loss
		trainer = LogitsTrainer(
			model=student,
			train_dataset=new_train_data.select(range(training_steps)),
			eval_dataset=new_test_data.select(range(1000)),
			args=training_args,
			optimizers=(optimizer, linear_scheduler),
			data_collator=DataCollatorForSeq2Seq(
			tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
		)
	else:
		print("not use distill")
		trainer = Trainer(
			model=student,
			train_dataset=new_train_data.select(range(training_steps)),
			eval_dataset=new_test_data.select(range(1000)),
			args=training_args,
			optimizers=(optimizer, linear_scheduler),
			data_collator=DataCollatorForSeq2Seq(
			tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
		)
	if args.use_distill:
		# Add the teacher model to the trainer
		trainer.teacher_model = teacher

	# silence the warnings. re-enable for inference!
	student.config.use_cache = False
	trainer.train()
	# Save the final model
	# trainer.save_model(config["training"]["output_dir"])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_save_path", type=str, default='./saved/training/less_new')
	parser.add_argument("--use_average", action='store_true')
	parser.add_argument("--use_distill", action='store_true')
	parser.add_argument("--merge", action='store_true', help='merge dataset in training')
	parser.add_argument("--epoch", type=int, default=2)
	parser.add_argument("--sparsity_level", type=float, default=0.8)
	parser.add_argument("--training_steps", type=int, default=20000, help='the number of sentences from datasets(3000-10000)')

	args = parser.parse_args()

	dtype = torch.bfloat16
	print(args)
	dotrain(dtype, args, save_steps = 600)