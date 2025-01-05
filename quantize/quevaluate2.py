import torch
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
from transformers import AutoTokenizer
from modeling_mixtral import MixtralForCausalLM, set_profile_mode
from utils import myevaluate, get_model
import json 

## 开启稀疏模式
set_profile_mode(False)

with open('../path.json', 'r') as f:
    path = json.load(f)
    model_name = path['mixtral']

with open('./device_map.json', 'r') as f:
    device_map = json.load(f)

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
        CompensatedModel(llm.model.layers[i].block_sparse_moe.experts[j].w3, './saved/', layerid=i, expertid=j).to(llmdevice)
        

task_name_list=['winogrande','sciq','openbookqa','arc_challenge','arc_easy']
num_fewshot = 0
myevaluate(task_name_list, llm, tokenizer, num_fewshot, 'cuda')

# {'arc_easy': {'acc,none': 0.8345959595959596, 'acc_stderr,none': 0.007623938582125698, 'acc_norm,none': 0.8198653198653199, 'acc_norm_stderr,none': 0.007885661261794779, 'alias': 'arc_easy'}, 
# 'arc_challenge': {'acc,none': 0.5477815699658704, 'acc_stderr,none': 0.014544519880633822, 'acc_norm,none': 0.5793515358361775, 'acc_norm_stderr,none': 0.01442621125250841, 'alias': 'arc_challenge'}, 
# 'openbookqa': {'acc,none': 0.338, 'acc_stderr,none': 0.02117566569520941, 'acc_norm,none': 0.462, 'acc_norm_stderr,none': 0.022318338119870527, 'alias': 'openbookqa'}, 
# 'sciq': {'acc,none': 0.97, 'acc_stderr,none': 0.005397140829099195, 'acc_norm,none': 0.956, 'acc_norm_stderr,none': 0.00648892179842741, 'alias': 'sciq'}, 
# 'winogrande': {'acc,none': 0.7482241515390686, 'acc_stderr,none': 0.012198489100259778, 'alias': 'winogrande'}}