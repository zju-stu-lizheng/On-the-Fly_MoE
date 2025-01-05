import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
from modeling_mixtral import MixtralForCausalLM, set_profile_mode
import json
from utils import get_model

# # 加载 C4 数据集的验证集
with open('../path.json', 'r') as file:
    paths = json.load(file)
    c4_path = paths.get('c4', '')
    model_name = paths.get('mixtral','')

with open('./device_map.json', 'r') as f:
    device_map = json.load(f)

set_profile_mode(False)
llm, tokenizer = get_model(model_name, device_map)
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

import torch

class CompensatedModel(torch.nn.Module):
    def __init__(self, model, path, layerid, expertid):
        super(CompensatedModel, self).__init__()
        self.model = model
        ### self.A and self.B_prime are initialized as the values loaded from the file
        self.A = torch.load(path + f'A_{layerid}_{expertid}.pt').to(torch.float16)
        self.B_prime = torch.load(path + f'B_prime_{layerid}_{expertid}.pt').to(torch.float16)
        

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        residual = (input_ids @ self.A.T) @ self.B_prime.T
        torch.add(outputs, residual, out = outputs)
    
        return outputs

for i in range(32):
    print(f"Layer {i} done...")
    for j in range(8):
        llmdevice = llm.model.layers[i].block_sparse_moe.experts[j].w3.device
        llm.model.layers[i].block_sparse_moe.experts[j].w3 = \
        CompensatedModel(llm.model.layers[i].block_sparse_moe.experts[j].w3, '/home/lz/On-the-Fly_MoE_Inference/quantize/saved/', layerid=i, expertid=j).to(llmdevice)

import torch
import numpy as np
datasets = torch.load('../saving/threshold/chess/datasets.pt')
set_profile_mode(True)
def get_batch(data, batch_size, block_size):
    start_idxs = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in start_idxs])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in start_idxs])
    return x, y

sparsity_level = 0.9
# device = 'cuda:1'
device_2 = 'cpu'
avg_loss = 0.0
n_batch = 64
# accum_steps = 4 
accum_steps = 2
batch_size = 1
block_size = 2048
torch.manual_seed(42)

model = llm

n_layers = len(model.model.layers)
n_experts = len(model.model.layers[0].block_sparse_moe.experts)

up_proj_states_thresholds = [torch.zeros([n_experts,]) for _ in range(n_layers)]
gate_proj_states_mean_squares = [[torch.zeros(model.config.intermediate_size) for _ in range(n_experts)] for _ in range(n_layers)]

up_states = [[torch.zeros([accum_steps * batch_size * block_size //2, model.config.intermediate_size]) for _ in range(n_experts)] for _ in range(n_layers)]
gate_states = [[torch.zeros([accum_steps * batch_size * block_size //2, model.config.intermediate_size]) for _ in range(n_experts)] for _ in range(n_layers)]

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

save_path = './threshold/c4_mixtral_up'

sp = str(sparsity_level).replace('.', '_')
print('save in:', save_path)
torch.save(thresholds, f'{save_path}/thresholds_{sp}.pt')