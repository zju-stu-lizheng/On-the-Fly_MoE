import torch
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
from transformers import AutoTokenizer, BitsAndBytesConfig, MixtralForCausalLM
import json

with open('../path.json', 'r') as f:
    path = json.load(f)
    model_name = path['mixtral']

with open('./device_map.json', 'r') as f:
    device_map = json.load(f)

llm = MixtralForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    use_cache=True,
    torch_dtype=torch.float16,
) 
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
from datasets import load_dataset
def preprocess_data(batch):
    # 使用 tokenizer 将文本数据转换为模型输入
    inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    inputs["labels"] = inputs.input_ids.clone()
    return inputs

# 定义一个函数来选择特征并丢弃不需要的
def select_features(example):
    return {
        'input_ids': example['input_ids'],
        'attention_mask': example['attention_mask'],
        'labels': example['labels']
    }

tokenizer.pad_token = tokenizer.eos_token
# # 加载 C4 数据集的验证集
with open('../path.json', 'r') as file:
    paths = json.load(file)
    c4_path = paths.get('c4', '')
c4 = load_dataset(c4_path)
# 对数据集进行预处理
c4_dataset = c4.map(preprocess_data, batched=True)
# c4_dataset = c4_dataset.map(select_features, batched=True)
c4_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
# c4_dataset
top_four_thousand_data = c4_dataset['validation'].select(range(100))

import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

from torch.utils.data import DataLoader
from tqdm import tqdm
set_seed(42)

# 定义数据加载器
batch_size = 4
# dataloader = DataLoader(c4_dataset['validation'], batch_size=batch_size)
dataloader = DataLoader(top_four_thousand_data, batch_size=batch_size)

# # 计算评估损失
# total_loss = 0.0
# num_batches = 0

# for batch in tqdm(dataloader):
#     input_ids = batch['input_ids'].to(llm.device)
#     attention_mask = batch['attention_mask'].to(llm.device)
#     labels = batch['labels'].to(llm.device)
    
#     # 禁用梯度计算
#     with torch.no_grad():
#         outputs = llm(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         total_loss += loss.item()
#         num_batches += 1
#         if num_batches % 100 == 0:
#             print(f"[{num_batches}], Eval Loss: {total_loss / (num_batches)}")

# # 计算平均损失
# eval_loss = total_loss / num_batches
# print(f"Eval Loss: {eval_loss}")

# %%
import torch
import os

llm_base = MixtralForCausalLM.from_pretrained(
    model_name,
    device_map='cpu',
    use_cache=True,
    torch_dtype=torch.float16,
    # attn_implementation="flash_attention_2"
) 
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
def profle_svdllm(name, model, calib_loader, dev):
    # model.to(dev)
    if "llama" in name or "mixtral" in name or "vicuna" in name:
        layers = model.model.layers
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if "w3" in name:
            # print(name)
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
            
    for batch in tqdm(calib_loader):
        inputs = batch['input_ids'].to(llm.device)
        model(inputs)
    for name, module in model.named_modules():
        if "w3" in name:
            module._forward_hooks.clear()
            # print(module.raw_scaling_diag_matrix)
    torch.cuda.empty_cache()

    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    
    layer_profile = {}
    for name, module in model.named_modules():
        if "w3" in name:
            covariance = module.raw_scaling_diag_matrix.double().to(dev)
            if not torch.allclose(covariance, covariance.t(), atol=1e-6):
                raise ValueError("Covariance matrix is not symmetric.")
                    # Perform eigen decomposition
            Lambda, Q = torch.linalg.eigh(covariance, UPLO='U')
            if torch.isnan(Lambda).any() or torch.isinf(Lambda).any():
                raise ValueError("Lambda contains NaN or Inf values.")

            # 检查 Lambda 是否包含负值
            if (Lambda < 0).any():
                print("Lambda contains negative values. Clamping to zero.")
                eigenvalues = torch.linalg.eigvalsh(covariance)
                covariance += (- eigenvalues[0] + 2e-6) * torch.eye(covariance.shape[0]).cuda()
                Lambda, Q = torch.linalg.eigh(covariance, UPLO='U')
                print(f"Lambda min: {Lambda.min().item()}, Lambda max: {Lambda.max().item()}")
            # 现在进行平方根操作
            Lambda_diag = torch.diag(torch.sqrt(Lambda))
            # Sort eigenvalues and eigenvectors in descending order
            indices = torch.argsort(Lambda, descending=True)
            Lambda = Lambda[indices]
            Q = Q[:, indices]

            # Compute Q_prime = Q * sqrt(Lambda)
            Lambda_diag = torch.diag(torch.sqrt(Lambda))
            Q_prime = torch.matmul(Q, Lambda_diag)
            layer_profile[name] = Q_prime.cpu()
            profiling_mat[name] = layer_profile
    return profiling_mat
profiling_mat=profle_svdllm("mixtral", llm, dataloader, "cuda")


# %%
#Quantize
from hqq.core.quantize import *
q4_config    = BaseQuantizeConfig(nbits=8, group_size=64) 
q3_config    = BaseQuantizeConfig(nbits=2, group_size=64)

quant_config = {
  'block_sparse_moe.experts.w3'  :q3_config,
}
from hqq.models.hf.base import AutoHQQHFModel
AutoHQQHFModel.quantize_model(llm, quant_config=quant_config, compute_dtype=torch.float16, device=device_map)

class CompensatedModel(torch.nn.Module):
    def __init__(self, model, B_prime, A):
        super(CompensatedModel, self).__init__()
        self.model = model
        self.B_prime = torch.nn.Parameter(torch.tensor(B_prime)).to(torch.float16)
        self.A = torch.nn.Parameter(torch.tensor(A)).to(torch.float16)
        # print(self.A.shape,self.B_prime.shape)
    def forward(self, input_ids):
        outputs = self.model(input_ids)
        # 假设在特定层添加残差连接，根据实际模型结构进行修改
        # print(self.B_prime.shape,self.A.shape,input_ids.shape)
        residual = input_ids @ (self.B_prime @ self.A).T
        outputs += residual
    
        return outputs
    
for i in range(32):
    print(f"Layer {i} done...")
    for j in range(8):
        llmdevice = llm.model.layers[i].block_sparse_moe.experts[j].w3.device
        Delta_W = llm_base.model.layers[i].block_sparse_moe.experts[j].w3.weight.to(llmdevice) - llm.model.layers[i].block_sparse_moe.experts[j].w3.dequantize()
        Q_prime = profiling_mat[f"model.layers.{i}.block_sparse_moe.experts.{j}.w3"][f"model.layers.{i}.block_sparse_moe.experts.{j}.w3"].cuda().float()
        Delta_W_prime =  Delta_W.to(torch.float32).to(llmdevice) @ Q_prime.to(torch.float32).to(llmdevice)
        llm_base.model.layers[i].block_sparse_moe.experts[j].w3.cpu()
        # 步骤5: 进行SVD分解并取前r个奇异值
        rank = 1024  # 设置 desired rank
        U_prime, Sigma_prime, V_prime = torch.linalg.svd(Delta_W_prime, full_matrices=False)
        U_prime = U_prime[:, :rank]
        Sigma_prime = Sigma_prime[:rank]
        V_prime = V_prime[:rank, :]

        B_prime = U_prime @ torch.diag(Sigma_prime)
        A_prime = V_prime

        # 步骤6: 投影回原空间
        A = A_prime.to(llmdevice) @ torch.linalg.inv(Q_prime).to(llmdevice)
        llm.model.layers[i].block_sparse_moe.experts[j].w3 = CompensatedModel(llm.model.layers[i].block_sparse_moe.experts[j].w3, B_prime, A).to(llmdevice)
    # compensated_model = CompensatedModel(student.base, B_prime, A).to("cuda")

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
del dataloader
del profiling_mat
del llm_base

# %%
def evaluate(task_name_list, model, tokenizer, num_fewshot, device):
    hflm = HFLM(pretrained=llm, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
    model=hflm,
    tasks=task_name_list,
    num_fewshot=num_fewshot)
    print(results['results'])


# triviaqa
task_list=['winogrande','sciq','openbookqa','arc_challenge','arc_easy']
evaluate(task_list, llm, tokenizer, 0, "cuda")

# {'arc_easy': {'acc,none': 0.8345959595959596, 'acc_stderr,none': 0.007623938582125698, 'acc_norm,none': 0.8198653198653199, 'acc_norm_stderr,none': 0.007885661261794779, 'alias': 'arc_easy'}, 
# 'arc_challenge': {'acc,none': 0.5477815699658704, 'acc_stderr,none': 0.014544519880633822, 'acc_norm,none': 0.5793515358361775, 'acc_norm_stderr,none': 0.01442621125250841, 'alias': 'arc_challenge'}, 
# 'openbookqa': {'acc,none': 0.338, 'acc_stderr,none': 0.02117566569520941, 'acc_norm,none': 0.462, 'acc_norm_stderr,none': 0.022318338119870527, 'alias': 'openbookqa'}, 
# 'sciq': {'acc,none': 0.97, 'acc_stderr,none': 0.005397140829099195, 'acc_norm,none': 0.956, 'acc_norm_stderr,none': 0.00648892179842741, 'alias': 'sciq'}, 
# 'winogrande': {'acc,none': 0.7482241515390686, 'acc_stderr,none': 0.012198489100259778, 'alias': 'winogrande'}}