import torch
import json
import os
import csv
from utils import get_model, set_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
### from path.json read paths of model and dataset
model_name = "mixtral"
dataset_name = "c4"
with open('../path.json', 'r') as file:
    paths = json.load(file)
    model_path = paths.get(model_name, '')
    dataset_path = paths.get(dataset_name, '')
    save_path = paths.get('chess_up_threshold','')
    print('model path:', model_path, '\ndataset path:', dataset_path, '\nsave path:', save_path)

set_seed(42)
model = get_model(model_path)

# %%
datasets = torch.load('./threshold/chess/datasets.pt')
import torch
import numpy as np
def get_batch(data, batch_size, block_size):
    start_idxs = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in start_idxs])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in start_idxs])
    return x, y

# %%
sparsity_level = 0.9
# device = 'cuda:1'
device_2 = 'cpu'
avg_loss = 0.0
n_batch = 64 * 2
# n_batch = 2
# accum_steps = 4 
accum_steps = 2
batch_size = 1
block_size = 2048
torch.manual_seed(42)
n_layers = len(model.model.layers)
n_experts = len(model.model.layers[0].block_sparse_moe.experts)

up_proj_states_thresholds = [torch.zeros([n_experts,]) for _ in range(n_layers)]
gate_proj_states_mean_squares = [[torch.zeros(model.config.intermediate_size) for _ in range(n_experts)] for _ in range(n_layers)]

up_states = [[torch.zeros([accum_steps * batch_size * block_size , model.config.intermediate_size]) for _ in range(n_experts)] for _ in range(n_layers)]
gate_states = [[torch.zeros([accum_steps * batch_size * block_size , model.config.intermediate_size]) for _ in range(n_experts)] for _ in range(n_layers)]

with torch.no_grad():
    for step in range(n_batch // accum_steps):
        print(step * accum_steps)
        all_counts = [0 for _ in range(n_layers * n_experts)]
        for batch_idx in range(accum_steps):
            # print('batch_idx:', batch_idx)
            inputs, labels = get_batch(datasets['validation'], batch_size, block_size)
            inputs = inputs.cuda()
            outputs = model(inputs, labels=inputs)
            avg_loss = avg_loss + outputs.loss / n_batch

            for layer_idx in range(n_layers):
                for expert_idx in range(n_experts):
                    counts = all_counts[layer_idx * n_experts + expert_idx]

                    states = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx].up_proj_states.reshape(-1, model.config.intermediate_size)
                    cur_counts = states.size(0)
                    # print('counts and cur_counts:',counts, cur_counts)
                    # print(states.size())
                    # print(up_states[layer_idx][expert_idx][counts : counts+cur_counts, :].size())
                    up_states[layer_idx][expert_idx][counts : counts+cur_counts, :] = states

                    states = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx].gate_proj_states.reshape(-1, model.config.intermediate_size)
                    gate_states[layer_idx][expert_idx][counts : counts+cur_counts, :] = states
                    # counts += cur_counts
                    all_counts[layer_idx * n_experts + expert_idx] += cur_counts

        for layer_idx in range(n_layers):   
            for expert_idx in range(n_experts):
                # print('layer_idx:', layer_idx, 'expert_idx:', expert_idx)
                useful_num = all_counts[layer_idx * n_experts + expert_idx]
                topk_num = int(useful_num * model.config.intermediate_size * sparsity_level)
                up_proj_states_thresholds[layer_idx][expert_idx] += up_states[layer_idx][expert_idx][0:useful_num,:].to(device_2).abs().flatten().kthvalue(topk_num).values.to('cpu')
                gate_proj_states_mean_squares[layer_idx][expert_idx] += (torch.sum(gate_states[layer_idx][expert_idx][0:useful_num,:].to(device_2) ** 2, dim=0).to('cpu') / useful_num).to('cpu')

for layer_idx in range(n_layers):
    for expert_idx in range(n_experts):
        gate_proj_states_mean_squares[layer_idx][expert_idx] /= n_batch // accum_steps
        up_proj_states_thresholds[layer_idx][expert_idx] /= n_batch // accum_steps


# %%
importance_thresholds = [torch.zeros([n_experts,]) for _ in range(n_layers)]
up_proj_states_thresholds_2 = [[torch.zeros(model.config.intermediate_size) for _ in range(n_experts)] for _ in range(n_layers)]

with torch.no_grad():
    for step in range(n_batch // accum_steps):
        print(step * accum_steps)
        all_counts = [0 for _ in range(n_layers * n_experts)]
        for batch_idx in range(accum_steps):
            inputs, labels = get_batch(datasets['validation'], batch_size, block_size)
            inputs = inputs.cuda()
            outputs = model(inputs, labels=inputs)
            avg_loss = avg_loss + outputs.loss / n_batch

            for layer_idx in range(n_layers):
                for expert_idx in range(n_experts):
                    counts = all_counts[layer_idx * n_experts + expert_idx]
                    states = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx].up_proj_states.reshape(-1, states.size(-1))
                    cur_counts = states.size(0)
                    up_states[layer_idx][expert_idx][counts:cur_counts+counts, :] = states
                    # counts += cur_counts
                    all_counts[layer_idx * n_experts + expert_idx] += cur_counts
                
        for layer_idx in range(n_layers):   
            for expert_idx in range(n_experts):
                useful_num = all_counts[layer_idx * n_experts + expert_idx]
                importance_scores = up_states[layer_idx][expert_idx][:useful_num,:] ** 2 * gate_proj_states_mean_squares[layer_idx][expert_idx]
                importance_thresholds[layer_idx][expert_idx] += importance_scores.to(device_2).flatten().kthvalue(int(importance_scores.numel() * sparsity_level)).values.to('cpu')

for layer_idx in range(n_layers):
    for expert_idx in range(n_experts):
        importance_thresholds[layer_idx][expert_idx] /= n_batch // accum_steps
        up_proj_states_thresholds_2[layer_idx][expert_idx] = (importance_thresholds[layer_idx][expert_idx].expand_as(up_proj_states_thresholds_2[layer_idx][expert_idx]) / gate_proj_states_mean_squares[layer_idx][expert_idx]) ** 0.5

thresholds = {'up_proj_states_thresholds': up_proj_states_thresholds, 'up_proj_states_thresholds_2': up_proj_states_thresholds_2}

# torch.save(thresholds, f'{save_path}/thresholds_0_8.pt')
sp = str(sparsity_level).replace('.', '_')
print('save in:', f'{save_path}/thresholds_{sp}.pt')
torch.save(thresholds, f'{save_path}/thresholds_{sp}.pt')
