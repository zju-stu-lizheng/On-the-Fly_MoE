### 0808 定义一个newtrain.py文件，用于定义一个新的Trainer类，继承自transformers.Trainer，重写compute_loss方法，用于计算增加蒸馏部分的损失函数。
### 0810 teacher和student放到两个device上，增大batch size。
### mixtral模型一张卡放不下，每个模型在两张卡上
### sd : 0,1, td:6,7
import argparse
import random
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, MixtralConfig
from transformers import MixtralForCausalLM as MixtralTeacher
from modeling_mixtral import MixtralForCausalLM, set_profile_mode, load_thresholds
from peft import LoraConfig, get_peft_model, PeftModel
import functools
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import ast
import json
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--sample_num',type=int, default=2000, help='the number of samples to train')
parser.add_argument('--do_eval', action="store_true", help="Whether to evaluate the model")
parser.add_argument('--has_atten', action="store_true", help="Whether to use lora in attention layer") 
parser.add_argument('--has_lora', action="store_true", help="Whether to use lora")
parser.add_argument('--use_distill', action="store_true", help="Whether to use distill loss")
# parser.add_argument('--lora_path', type=str, default='/home/lz/workspace/llama2-7b/HQQ/notebooks/output/fineweb-90/checkpoint-2500', help='lora path')
parser.add_argument('--lora_path', nargs='+', help='list of lora paths')
parser.add_argument('--batch_size',type=int, default=6)
parser.add_argument('--sparsity',type=int, default=80)
parser.add_argument('--epochs',type=int, default=3)
parser.add_argument('--accum_iter',type=int, default=4, help='the number of iterations to accumulate gradients')
parser.add_argument('--learning_rate',type=float, default=1e-4)
parser.add_argument('--align_list', type=ast.literal_eval, default=[32])
opt = parser.parse_args()
print(opt)

with open("./device_map.json", "r") as f:
	sd = json.load(f)

with open("./device_map_2.json", "r") as f:
	td = json.load(f)

def prepare_model(model_name, is_eval=False, has_atten=False, sparsity=80):
	config = MixtralConfig(output_router_logits=True, use_cache=False, output_hidden_states=True)
	if is_eval:
		# set_teacher_sparsity(50,'c4')
		model = MixtralTeacher.from_pretrained(
			pretrained_model_name_or_path=model_name,
			config=config,
			torch_dtype=torch.bfloat16,
			device_map=td,
			# attn_implementation="flash_attention_2"
			)
		### 加载lora_path：bagel训练的
		lora_path_teacher = '/home/bcds/On-the-Fly_MoE_Inference/quantize/saved/training/bagel0/checkpoint-1200'
		#### 加载lora模型并merge
		
		print(f"load lora model: {lora_path_teacher}")
		model = PeftModel.from_pretrained(model, lora_path_teacher, adapter_name=f"load_teacher")
		model.set_adapter(f"load_teacher")
		model = model.merge_and_unload()
		model.eval()
	else:
		set_profile_mode(mode=False)
		load_thresholds("/home/bcds/On-the-Fly_MoE_Inference/saving/threshold/c4_mixtral/thresholds_0_8.pt", use_average=False)
		model = MixtralForCausalLM.from_pretrained(
			pretrained_model_name_or_path=model_name,
			config=config,
			torch_dtype=torch.bfloat16,
			device_map=sd,
			# attn_implementation="flash_attention_2"
		)
		print(f"set sparsity to {sparsity}")
		### 包装lora模块
		rank = 32
		target_modules = ["w1","w2","w3","gate"]
		if has_atten:
			target_modules += ["q_proj","k_proj","v_proj","o_proj",]
		# target_modules = ["gate"]
		peft_config = LoraConfig(
			lora_alpha=32,
			lora_dropout=0.01,
			r=rank,
			bias="none",
			target_modules=target_modules,
			layers_to_transform=[i for i in range(32) if i != 0],
			# layers_to_transform=[1],
			task_type="CAUSAL_LM"
		)
		#### 加载lora模型并merge
		# if opt.has_lora:
		#     for i in range(len(opt.lora_path)):
		#         print(f"load lora model_{i}: {opt.lora_path[i]}")
		#         model = PeftModel.from_pretrained(model, opt.lora_path[i], adapter_name=f"load_{i}")
		#         model.set_adapter(f"load_{i}")
		#         model = model.merge_and_unload()
			
		model = get_peft_model(model, peft_config) 
		
		# if opt.has_lora:
		#     print("load lora model")
		#     loaded_state_dict = load_file(opt.lora_path)
		#     new_loaded_state_dict = {}
		#     for name, param in loaded_state_dict.items():
		#         name = name.replace('weight','default.weight')
		#         new_loaded_state_dict[name] = param
		#     model.load_state_dict(new_loaded_state_dict, strict=False)
		for name,param in model.named_parameters():
			if not any(nd in name for nd in ["lora_A","lora_B"]):
				param.requires_grad = False
			else:
				param.requires_grad = True
	return model

def compute_norm_loss(model, input_ids, attention_mask, labels):
	"""
	compute loss for student model
	"""
	outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
	norm_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
	return norm_loss

def reverse_kl(logits, teacher_logits, input_ids, attention_mask, labels):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (attention_mask != 0).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def forward_kl(logits, teacher_logits, input_ids, attention_mask, labels):
	teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
	inf_mask = torch.isinf(logits)
	student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
	prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
	
	# print("prod_probs", prod_probs.shape) #torch.Size([6, 512, 32000])
	x = torch.sum(prod_probs, dim=-1).view(-1)
	# print("x", x.shape) # torch.Size([3072])
	mask = (attention_mask != 0).int()  #### padding的部分, attention_mask==0
	# print("mask", mask.shape) # mask torch.Size([6, 512])
	distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
	return distil_loss

def expert_logits_loss(logits, teacher_logits, attention_mask, criterion=nn.KLDivLoss()):
	expert_loss = torch.tensor(0.0, device=logits[0].device, dtype=torch.float32)
	for layerid in range(1, 32):
		tensor1 = logits[layerid].to(torch.float32)
		tensor2 = teacher_logits[layerid].to(torch.float32)

		mask = attention_mask.unsqueeze(-1).expand_as(tensor1)
		masked_tensor1 = tensor1 * mask.to(tensor1.device)
		masked_tensor2 = tensor2 * mask.to(tensor2.device)
		
		tensor1 = F.softmax(masked_tensor1, dim=-1)
		tensor2 = F.softmax(masked_tensor2, dim=-1)
		cur_loss = criterion(tensor1.log(), tensor2.to(tensor1.device))
		# print(f'layer {layerid}: ', cur_loss.item())
		expert_loss += cur_loss

	return expert_loss

def compute_loss(model, teacher_model, input_ids, attention_mask, labels, align_list=[1,32]):
	"""
	compute loss for student model
	"""
	criterion=nn.KLDivLoss()

	outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
	norm_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
	last_logits = outputs["logits"] # torch.Size([6, 512, 32000])
	
	with torch.no_grad():
		teacher_outputs = teacher_model(input_ids.to(teacher_model.device), 
										attention_mask=attention_mask.to(teacher_model.device))
		teacher_logits = teacher_outputs["logits"]

	tensor1 = outputs["router_logits"]
	tensor2 = teacher_outputs["router_logits"]
	# print(attention_mask.size(), attention_mask.sum())
	expert_loss = expert_logits_loss(tensor1, tensor2, attention_mask.view(-1), criterion=criterion).to(last_logits.dtype)
	dl_list = []
	dl = forward_kl(last_logits, teacher_logits.to(model.device), input_ids, attention_mask, labels).to(last_logits.dtype) + \
		reverse_kl(last_logits, teacher_logits.to(model.device), input_ids, attention_mask, labels).to(last_logits.dtype)
	dl_list.append(dl)

	loss = expert_loss + dl + norm_loss
	return loss, dl, norm_loss, dl_list, expert_loss

def train_model(model, teacher_model, train_loader, val_loader, opt):
	epochs = opt.epochs
	learning_rate = opt.learning_rate
	accum_iter  = opt.accum_iter  ## 累计梯度更新

	### 只取出需要训练的参数
	optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs*len(train_loader)/accum_iter,eta_min=1e-6)
	align_list = opt.align_list
	
	writer = SummaryWriter(f'runs/{opt.save_name}_new')
	for epoch in range(epochs):
		if opt.do_eval:
			model.eval()
			with torch.no_grad():
				val_loss = 0
				for data in tqdm(val_loader):
					input_ids, attention_mask, labels = data['input_ids'].to(model.device), data['attention_mask'].to(model.device), data['labels'].to(model.device)
					if opt.use_distill:
						loss, distill_loss, norm_loss, dl_list, expert_loss = compute_loss(model, teacher_model, input_ids, attention_mask, labels, align_list)
					else:
						loss = compute_norm_loss(model, input_ids, attention_mask, labels)
					val_loss += loss.detach().item()
				print(f'Epoch {epoch}, Validation Loss: {val_loss / len(val_loader)}')
		model.train()
		nl, dl, el = 0, 0, 0
		dl_list_all = [0 for _ in range(len(align_list))]
		for batch_idx, data in enumerate(tqdm(train_loader)):
			input_ids, attention_mask, labels = data['input_ids'].to(model.device), data['attention_mask'].to(model.device), data['labels'].to(teacher_model.device)

			#### computing loss
			if opt.use_distill:
				loss, distill_loss, norm_loss, dl_list, expert_loss = compute_loss(model, teacher_model, input_ids, attention_mask, labels, align_list)
			else:
				norm_loss = compute_norm_loss(model, input_ids, attention_mask, labels)
				loss = norm_loss
				distill_loss = 0
			if batch_idx == 0:
				print("nl,dl,el:",norm_loss, distill_loss, expert_loss)

			nl += norm_loss.detach().item()
			if opt.use_distill:
				el += expert_loss.detach().item()
				dl += distill_loss.detach().item()
			#     for i, layer_idx in enumerate(align_list):
			#         dl_list_all[i] += dl_list[i].detach().item()
			#     distill_loss = distill_loss / accum_iter
			#     distill_loss.backward()
			# else:
			loss = loss / accum_iter # 取各个累计batch的平均损失，从而在.backward()时得到平均梯度
			loss.backward()
			
			if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
				### 这里记录的实际上是一个batch的损失，而非accum_iter个batch的平均损失（所以对比不公平）
				# writer.add_scalar('norm_loss/Train', norm_loss.item(), epoch * len(train_loader) + batch_idx)
				# if opt.use_distill:
				#     writer.add_scalar('distill_loss/Train', distill_loss.item(), epoch * len(train_loader) + batch_idx)
				writer.add_scalar('norm_loss/Train', nl / accum_iter, epoch * len(train_loader) + batch_idx)
				if opt.use_distill:
				#     for i, layer_idx in enumerate(align_list):
				#         writer.add_scalar(f'distill_loss/Train_{layer_idx}', dl_list_all[i], epoch * len(train_loader) + batch_idx)
					writer.add_scalar('distill_loss/Train', dl / accum_iter, epoch * len(train_loader) + batch_idx)
					writer.add_scalar('expert_loss/Train', el / accum_iter, epoch * len(train_loader) + batch_idx)
				nl, dl = 0, 0
				dl_list_all = [0 for _ in range(len(align_list))]
				optimizer.step()        # 更新模型
				optimizer.zero_grad()   # 梯度清零
				scheduler.step()
		### 按epoch 保存模型
		model.save_pretrained(f'output/mixtral/{opt.save_name}/{epoch}')
		
def preprocess_data(batch, tokenizer):
	# 使用 tokenizer 将文本数据转换为模型输入
	inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
	inputs["labels"] = inputs.input_ids.clone()
	return inputs

def get_all_dataset(tokenizer, sample_num, seed):
	test_num = min(sample_num//10, 100)
	with open('../path.json','r') as f:
		paths = json.load(f)
		c4_path = paths["c4"]
		fineweb_path = paths["fineweb"]
		bagel_path = paths["bagel_json"]
		openhermes_path = paths["openhermes"]
		wikitext_path = paths["wikitext"]

	c4 = load_dataset("parquet", data_dir=c4_path)
	bagel = load_dataset("json", data_files=bagel_path)
	openhermes = load_dataset("json", data_files=openhermes_path)
	fineweb = load_dataset("parquet", data_files=fineweb_path)
	wikitext = load_dataset("parquet", data_files=wikitext_path)

	all_text = c4["validation"]["text"][:30000] + fineweb["train"]["text"][:25000] + bagel["train"]["text"][:15000] + openhermes["train"]["text"][:20000] + wikitext["train"]["text"][:15000]
	combined_dataset = Dataset.from_dict({"text": all_text})

	fineweb_train = combined_dataset.train_test_split(test_size=test_num, seed=seed)
	train_data = fineweb_train['train'].select(range(sample_num))
	test_data = fineweb_train['test']
	# train_data.shuffle(seed)
	all_train_data = train_data.map(
		functools.partial(
		preprocess_data,
		tokenizer=tokenizer
	), batched=True)
	all_test_data = test_data.map(
		functools.partial(
		preprocess_data,
		tokenizer=tokenizer
	), batched=True)
	all_train_data.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
	all_test_data.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
	all_train_data.shuffle(seed)
	all_test_data.shuffle(seed)
	return all_train_data, all_test_data

def get_c4_dataset(tokenizer, sample_num, seed):
	test_num = min(sample_num//10, 100)
	with open('../path.json','r') as f:
		c4_path = json.load(f)["c4"]

	c4 = load_dataset("parquet", data_dir=c4_path)
	print(f"use c4 datasets, {c4_path}")
	c4dataset = c4['validation']['text']
	combined_dataset = Dataset.from_dict({"text": c4dataset})

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

def get_fineweb_dataset(tokenizer, sample_num=24000, seed=42):
	# 加载fineweb数据集
	test_num = min(sample_num//10, 100)

	### use bagel
	# wikitext = load_dataset("parquet", data_files="/home/bcds/On-the-Fly_MoE_Inference/wikitext-2/data/train-00000-of-00001.parquet")
	wikitext = load_dataset("json", data_files="/home/bcds/On-the-Fly_MoE_Inference/bagel-v0.5/processed_data.json")
	fineweb = load_dataset("parquet", data_files="/home/bcds/venv/dilab/floe/dataset/finewebedu/sample/10BT/000_00000.parquet")
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


def main():
	### 加载两个 model
	model_name = '/home/bcds/venv/dilab/Mixtral-8x7B-v0.1'
	sparsity = opt.sparsity
	if opt.use_distill:
		teacher = prepare_model(model_name, is_eval=True)
	else:
		teacher = None
	student = prepare_model(model_name, is_eval=False, has_atten=opt.has_atten, sparsity=sparsity)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"

	sample_num  = opt.sample_num
	batch_size  = opt.batch_size

	random.seed(42)
	# fineweb_train_data, fineweb_test_data = get_all_dataset(tokenizer, sample_num=sample_num, seed = 42)
	fineweb_train_data, fineweb_test_data = get_fineweb_dataset(tokenizer, sample_num=sample_num)
	train_loader = DataLoader(fineweb_train_data, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(fineweb_test_data, batch_size=batch_size, shuffle=False)

	# opt.save_name = f'90_{sample_num}'
	# opt.save_name = f'{sparsity}_{sample_num}_new_{opt.align_list[0]}'
	opt.save_name = f'{sparsity}_{sample_num}'
	if opt.has_atten:
		opt.save_name += '_gate_atten_2'
	print(f"Start training {opt.save_name}")
	train_model(student, teacher, train_loader, val_loader, opt=opt)

if __name__ == '__main__':
	main()