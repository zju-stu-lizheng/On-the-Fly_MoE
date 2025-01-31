import torch
import json
import os
import csv
from utils import get_model, set_seed
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from modeling_phimoe import load_thresholds, set_profile_mode

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
### from path.json read paths of model and dataset
model_name = "phi"
dataset_name = "c4"
with open('../path.json', 'r') as file:
    paths = json.load(file)
    fineweb = paths['fineweb']
    model_path = paths.get(model_name, '')
    dataset_path = paths.get(dataset_name, '')
    save_path = paths.get('phi_threshold','')
    print('model path:', model_path, '\ndataset path:', dataset_path, '\nsave path:', save_path)

with open("../quantize/device_map_1.json", "r") as f:
    device_map = json.load(f)

set_seed(42)
load_thresholds(save_path+"thresholds_0_5.pt",zero=True)
set_profile_mode(True)
model = get_model(model_path, device_map=device_map)

print(model.config.intermediate_size)
import torch
import numpy as np

# 定义文件路径
dataset_file = 'datasets-phi.pt'

# 检查文件是否存在
if os.path.exists(dataset_file):
    # 如果文件存在，直接加载
    print(f"'{dataset_file}' 文件已存在，直接加载...")
    datasets = torch.load(dataset_file)
else:
    # 如果文件不存在，重新生成
    print(f"'{dataset_file}' 文件不存在，重新生成...")

    # 加载原始数据集
    raw_datasets = load_dataset("parquet", data_files=fineweb)

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

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--sparsity_level", type=float, default=0.8)
args = parser.parse_args()
print(args)

sparsity_level = args.sparsity_level
# device = 'cuda:1'
device_2 = 'cuda:2'
avg_loss = 0.0
n_batch = 64
# n_batch = 2
# accum_steps = 4 
accum_steps = 1
batch_size = 2
block_size = 1024
torch.manual_seed(42)
n_layers = len(model.model.layers)
n_experts = len(model.model.layers[0].block_sparse_moe.experts)
print(n_layers, n_experts)


up_proj_states_thresholds = [torch.zeros([n_experts,]) for _ in range(n_layers)]
gate_proj_states_thresholds = [torch.zeros([n_experts,]) for _ in range(n_layers)]
down_proj_states_thresholds = [torch.zeros([n_experts,]) for _ in range(n_layers)]

up_states = [[torch.zeros([accum_steps * batch_size * block_size , model.config.intermediate_size], device=device_2) for _ in range(n_experts)] for _ in range(n_layers)]
gate_states = [[torch.zeros([accum_steps * batch_size * block_size , model.config.intermediate_size], device=device_2) for _ in range(n_experts)] for _ in range(n_layers)]
down_states = [[torch.zeros([accum_steps * batch_size * block_size , model.config.intermediate_size], device=device_2) for _ in range(n_experts)] for _ in range(n_layers)]


with torch.no_grad():
    for step in range(n_batch // accum_steps):
        print(step * accum_steps)
        all_counts = [0 for _ in range(n_layers * n_experts)]
        for batch_idx in range(accum_steps):
            # print('batch_idx:', batch_idx)
            inputs, labels = get_batch(datasets['train'], batch_size, block_size)
            inputs = inputs.cuda()
            outputs = model(inputs, labels=inputs)
            avg_loss = avg_loss + outputs.loss / n_batch

            for layer_idx in range(n_layers):
                for expert_idx in range(n_experts):
                    counts = all_counts[layer_idx * n_experts + expert_idx]
                    
                    try:
                        states = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx].up_proj_states.reshape(-1, model.config.intermediate_size)
                        cur_counts = states.size(0)
                        # print('counts and cur_counts:',counts, cur_counts)
                        # print(states.size())
                        # print(up_states[layer_idx][expert_idx][counts : counts+cur_counts, :].size())
                        up_states[layer_idx][expert_idx][counts : counts+cur_counts, :] = states

                        states = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx].gate_proj_states.reshape(-1, model.config.intermediate_size)
                        gate_states[layer_idx][expert_idx][counts : counts+cur_counts, :] = states
                        # counts += cur_counts
                        states = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx].down_proj_states.reshape(-1, model.config.intermediate_size)
                        down_states[layer_idx][expert_idx][counts : counts+cur_counts, :] = states

                        all_counts[layer_idx * n_experts + expert_idx] += cur_counts
                    except:
                        print('have error in :', layer_idx, expert_idx)

        for layer_idx in range(n_layers):   
            for expert_idx in range(n_experts):
                try:
                    # print('layer_idx:', layer_idx, 'expert_idx:', expert_idx)
                    useful_num = all_counts[layer_idx * n_experts + expert_idx]
                    topk_num = int(useful_num * model.config.intermediate_size * sparsity_level)
                    up_proj_states_thresholds[layer_idx][expert_idx] += up_states[layer_idx][expert_idx][0:useful_num,:].to(device_2).abs().flatten().kthvalue(topk_num).values.to('cpu')
                    gate_proj_states_thresholds[layer_idx][expert_idx] += gate_states[layer_idx][expert_idx][0:useful_num,:].to(device_2).abs().flatten().kthvalue(topk_num).values.to('cpu')
                    down_proj_states_thresholds[layer_idx][expert_idx] += down_states[layer_idx][expert_idx][0:useful_num,:].to(device_2).abs().flatten().kthvalue(topk_num).values.to('cpu')
                except:
                    print('have error in :', layer_idx, expert_idx)

for layer_idx in range(n_layers):
    for expert_idx in range(n_experts):
        up_proj_states_thresholds[layer_idx][expert_idx] /= n_batch // accum_steps
        gate_proj_states_thresholds[layer_idx][expert_idx] /= n_batch // accum_steps
        down_proj_states_thresholds[layer_idx][expert_idx] /= n_batch // accum_steps


# %%
# importance_thresholds = [torch.zeros([n_experts,]) for _ in range(n_layers)]
# up_proj_states_thresholds_2 = [[torch.zeros(model.config.intermediate_size) for _ in range(n_experts)] for _ in range(n_layers)]

# with torch.no_grad():
#     for step in range(n_batch // accum_steps):
#         print(step * accum_steps)
#         all_counts = [0 for _ in range(n_layers * n_experts)]
#         for batch_idx in range(accum_steps):
#             inputs, labels = get_batch(datasets['validation'], batch_size, block_size)
#             inputs = inputs.cuda()
#             outputs = model(inputs, labels=inputs)
#             avg_loss = avg_loss + outputs.loss / n_batch

#             for layer_idx in range(n_layers):
#                 for expert_idx in range(n_experts):
#                     counts = all_counts[layer_idx * n_experts + expert_idx]
#                     states = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx].up_proj_states.reshape(-1, states.size(-1))
#                     cur_counts = states.size(0)
#                     up_states[layer_idx][expert_idx][counts:cur_counts+counts, :] = states
#                     # counts += cur_counts
#                     all_counts[layer_idx * n_experts + expert_idx] += cur_counts
                
#         for layer_idx in range(n_layers):   
#             for expert_idx in range(n_experts):
#                 useful_num = all_counts[layer_idx * n_experts + expert_idx]
#                 importance_scores = up_states[layer_idx][expert_idx][:useful_num,:] ** 2 * gate_proj_states_mean_squares[layer_idx][expert_idx]
#                 importance_thresholds[layer_idx][expert_idx] += importance_scores.to(device_2).flatten().kthvalue(int(importance_scores.numel() * sparsity_level)).values.to('cpu')

# for layer_idx in range(n_layers):
#     for expert_idx in range(n_experts):
#         importance_thresholds[layer_idx][expert_idx] /= n_batch // accum_steps
#         up_proj_states_thresholds_2[layer_idx][expert_idx] = (importance_thresholds[layer_idx][expert_idx].expand_as(up_proj_states_thresholds_2[layer_idx][expert_idx]) / gate_proj_states_mean_squares[layer_idx][expert_idx]) ** 0.5

thresholds = {'up_proj_states_thresholds': up_proj_states_thresholds, 
            'gate_proj_states_thresholds': gate_proj_states_thresholds,
            'down_proj_states_thresholds': down_proj_states_thresholds}

sp = str(sparsity_level).replace('.', '_')
print('save in:', f'{save_path}/thresholds_{sp}.pt')
torch.save(thresholds, f'{save_path}/thresholds_{sp}.pt')
