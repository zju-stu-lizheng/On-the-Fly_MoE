import torch
import os
import sys
sys.path.append('/home/lz/On-the-Fly_MoE_Inference')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from modeling_mixtral import set_profile_mode
import json 
from utils import myevaluate, get_model

## 开启稀疏模式
set_profile_mode(False)

with open('../path.json', 'r') as f:
    path = json.load(f)
    model_name = path['mixtral']

with open('./device_map.json', 'r') as f:
    device_map = json.load(f)

llm, tokenizer = get_model(model_name, device_map)

#Quantize
from hqq.core.quantize import *
q4_config    = BaseQuantizeConfig(nbits=8, group_size=64, offload_meta=True) 
q3_config    = BaseQuantizeConfig(nbits=2, group_size=32)

quant_config = {
  'block_sparse_moe.experts.w3'  :q3_config,
}
from hqq.models.hf.base import AutoHQQHFModel
AutoHQQHFModel.quantize_model(llm, quant_config=quant_config, compute_dtype=torch.float16, device=device_map)

### for test
for layerid in range(1):
    for expertid in range(1):
        llm.model.layers[layerid].block_sparse_moe.experts[expertid].print_ratio()


task_name_list=['winogrande','sciq','openbookqa','arc_challenge','arc_easy']
num_fewshot = 0
myevaluate(task_name_list, llm, tokenizer, num_fewshot, 'cuda')

# up: 2bit
# {'arc_easy': {'acc,none': 0.8236531986531986, 'acc_stderr,none': 0.007820313817947478, 'acc_norm,none': 0.8202861952861953, 'acc_norm_stderr,none': 0.007878465068489398, 'alias': 'arc_easy'}, 
# 'arc_challenge': {'acc,none': 0.5358361774744027, 'acc_stderr,none': 0.014573813664735622, 'acc_norm,none': 0.5802047781569966, 'acc_norm_stderr,none': 0.014422181226303012, 'alias': 'arc_challenge'}, 
# 'openbookqa': {'acc,none': 0.358, 'acc_stderr,none': 0.021461434862859133, 'acc_norm,none': 0.464, 'acc_norm_stderr,none': 0.022324981738385333, 'alias': 'openbookqa'}, 
# 'sciq': {'acc,none': 0.964, 'acc_stderr,none': 0.00589395781616553, 'acc_norm,none': 0.957, 'acc_norm_stderr,none': 0.006418114379799739, 'alias': 'sciq'}, 
# 'winogrande': {'acc,none': 0.7426992896606156, 'acc_stderr,none': 0.012285989618865697, 'alias': 'winogrande'}}

# + sparsity 25% (大概，用的是之前的稀疏阈值)
# {'arc_easy': {'acc,none': 0.8055555555555556, 'acc_stderr,none': 0.008121078550852043, 'acc_norm,none': 0.7946127946127947, 'acc_norm_stderr,none': 0.008289582587432948, 'alias': 'arc_easy'}, 
# 'arc_challenge': {'acc,none': 0.5110921501706485, 'acc_stderr,none': 0.01460779491401306, 'acc_norm,none': 0.5503412969283277, 'acc_norm_stderr,none': 0.014537144444284738, 'alias': 'arc_challenge'}, 
# 'openbookqa': {'acc,none': 0.33, 'acc_stderr,none': 0.021049612166134803, 'acc_norm,none': 0.474, 'acc_norm_stderr,none': 0.022352791650914163, 'alias': 'openbookqa'}, 
# 'sciq': {'acc,none': 0.952, 'acc_stderr,none': 0.006763264133666694, 'acc_norm,none': 0.938, 'acc_norm_stderr,none': 0.007629823996280313, 'alias': 'sciq'}, 
# 'winogrande': {'acc,none': 0.7245461720599842, 'acc_stderr,none': 0.012555690055709525, 'alias': 'winogrande'}}

