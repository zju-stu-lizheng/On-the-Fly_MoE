import torch
import json
import os
import csv
from utils import get_model, set_seed
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
### from path.json read paths of model and dataset
# model_name = "mixtral"
model_name = "llama"
dataset_name = "fineweb"
with open('../path.json', 'r') as file:
    paths = json.load(file)
    model_path = paths.get(model_name, '')
    dataset_path = paths.get(dataset_name, '')
    save_path = paths.get('llama_down_threshold','')
    print('model path:', model_path, '\ndataset path:', dataset_path, '\nsave path:', save_path)

set_seed(42)
with open("../quantize/device_map_1.json", "r") as f:
    device_map = json.load(f)
model = get_model(model_path, device_map=device_map)

# %%
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import numpy as np

# 定义文件路径
dataset_file = '/home/bcds/On-the-Fly_MoE_Inference/saving/threshold/c4_llama3_up/datasets-llama.pt'
# 检查文件是否存在
if os.path.exists(dataset_file):
    # 如果文件存在，直接加载
    print(f"'{dataset_file}' 文件已存在，直接加载...")
    datasets = torch.load(dataset_file)
else:
    # 如果文件不存在，重新生成
    print(f"'{dataset_file}' 文件不存在，重新生成...")

    # 加载原始数据集
    raw_datasets = load_dataset("parquet", data_files=dataset_path)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 定义处理函数
    def process(example):
        ids = tokenizer.encode(example['text'])
        out = {'ids': ids, 'len': len(ids)}
        return out

    # 对数据集进行分词处理
    tokenized = raw_datasets.map(process, desc='tokenizing raw datasets', num_proc=64)

    # 将分词后的数据转换为 numpy 数组并保存
    datasets = dict()
    for split, dset in tokenized.items():
        datasets[split] = []
        length = np.sum(dset['len'])
        datasets[split] = np.ndarray((length,), np.uint32)
        idx = 0
        for row in dset:
            datasets[split][idx:idx + row['len']] = row['ids']
            idx += row['len']

    # 保存生成的数据集
    torch.save(datasets, dataset_file)
    print(f"'{dataset_file}' 文件已生成并保存。")

def get_batch(data, batch_size, block_size):
    start_idxs = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in start_idxs])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in start_idxs])
    return x, y

parser = argparse.ArgumentParser()
parser.add_argument("--sparsity_level", type=float, default=0.8)
args = parser.parse_args()
print(args)

sparsity_level = args.sparsity_level
print(sparsity_level)
device = 'cuda:0'
device_2 = 'cuda:2'
avg_loss = 0.0
n_batch = 64
# n_batch = 2
accum_steps = 4 
# accum_steps = 2
batch_size = 1
block_size = 1024
torch.manual_seed(42)
n_layers = len(model.model.layers)

up_proj_states_thresholds = [torch.zeros([1,], device=device_2) for _ in range(len(model.model.layers))]
down_proj_states_thresholds = [torch.zeros([1,], device=device_2) for _ in range(len(model.model.layers))]

gate_proj_states_mean_squares = [torch.zeros(model.config.intermediate_size, device=device_2) for _ in range(len(model.model.layers))]

gate_proj_states = [torch.zeros([accum_steps * batch_size * block_size, model.config.intermediate_size]) for _ in range(len(model.model.layers))]
up_proj_states = [torch.zeros([accum_steps * batch_size * block_size, model.config.intermediate_size]) for _ in range(len(model.model.layers))]
down_proj_states = [torch.zeros([accum_steps * batch_size * block_size, model.config.intermediate_size]) for _ in range(len(model.model.layers))]

with torch.no_grad():
    for step in range(n_batch // accum_steps):
        print(step * accum_steps)
        for batch_idx in range(accum_steps):
            inputs, labels = get_batch(datasets['train'], batch_size, block_size)
            inputs = inputs.to(device)
            outputs = model(inputs, labels=inputs)
            avg_loss = avg_loss + outputs.loss / n_batch

            for layer_idx in range(len(model.model.layers)):
                states = model.model.layers[layer_idx].mlp.gate_proj_states
                gate_proj_states[layer_idx][batch_idx * batch_size * block_size : (batch_idx + 1) * batch_size * block_size, :] = states.reshape(-1, states.size(-1))

                states = model.model.layers[layer_idx].mlp.up_proj_states
                up_proj_states[layer_idx][batch_idx * batch_size * block_size : (batch_idx + 1) * batch_size * block_size, :] = states.reshape(-1, states.size(-1))


                states = model.model.layers[layer_idx].mlp.down_proj_states
                down_proj_states[layer_idx][batch_idx * batch_size * block_size : (batch_idx + 1) * batch_size * block_size, :] = states.reshape(-1, states.size(-1))

        for layer_idx in range(len(model.model.layers)):   
            up_proj_states_thresholds[layer_idx] += up_proj_states[layer_idx].to(device_2).abs().flatten().kthvalue(int(up_proj_states[layer_idx].numel() * sparsity_level)).values.to(device_2)
            down_proj_states_thresholds[layer_idx] += down_proj_states[layer_idx].to(device_2).abs().flatten().kthvalue(int(down_proj_states[layer_idx].numel() * sparsity_level)).values.to(device_2)
            gate_proj_states_mean_squares[layer_idx] += (torch.sum(gate_proj_states[layer_idx].to(device_2) ** 2, dim=0).to(device_2) / gate_proj_states[layer_idx].size(0)).to(device_2)

for layer_idx in range(len(model.model.layers)):
    up_proj_states_thresholds[layer_idx] /= n_batch // accum_steps
    down_proj_states_thresholds[layer_idx] /= n_batch // accum_steps
    gate_proj_states_mean_squares[layer_idx] /= n_batch // accum_steps
# importance_thresholds = [torch.zeros([1,], device=device_2) for _ in range(len(model.model.layers))]
# up_proj_states_thresholds_2 = [torch.zeros(model.config.intermediate_size) for _ in range(len(model.model.layers))]

# with torch.no_grad():
#     for step in range(n_batch // accum_steps):
#         print(step * accum_steps)
#         for batch_idx in range(accum_steps):
#             inputs, labels = get_batch(datasets['train'], batch_size, block_size)
#             inputs = inputs.to(device)
#             outputs = model(inputs, labels=inputs)
#             avg_loss = avg_loss + outputs.loss / n_batch

#             for layer_idx in range(len(model.model.layers)):
#                 states = model.model.layers[layer_idx].mlp.up_proj_states
#                 up_proj_states[layer_idx][batch_idx * batch_size * block_size : (batch_idx + 1) * batch_size * block_size, :] = states.reshape(-1, states.size(-1))
        
#         for layer_idx in range(len(model.model.layers)):   
#             importance_scores = up_proj_states[layer_idx].to(device_2) ** 2 * gate_proj_states_mean_squares[layer_idx]
#             importance_thresholds[layer_idx] += importance_scores.to(device_2).flatten().kthvalue(int(importance_scores.numel() * sparsity_level)).values.to(device_2)

# for layer_idx in range(len(model.model.layers)):
#     importance_thresholds[layer_idx] /= n_batch // accum_steps
#     up_proj_states_thresholds_2[layer_idx] = (importance_thresholds[layer_idx].expand_as(up_proj_states_thresholds_2[layer_idx]) / gate_proj_states_mean_squares[layer_idx]) ** 0.5

thresholds = {'down_proj_states_thresholds': down_proj_states_thresholds}

# torch.save(thresholds, f'{save_path}/thresholds_0_8.pt')
sp = str(sparsity_level).replace('.', '_')
print('save in:', f'{save_path}/thresholds_{sp}.pt')
torch.save(thresholds, f'{save_path}/thresholds_{sp}.pt')
