import torch
import os
import sys
sys.path.append('/home/lz/On-the-Fly_MoE_Inference')
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
from transformers import AutoTokenizer
from modeling_mixtral import MixtralForCausalLM
import json 

with open('../path.json', 'r') as f:
    path = json.load(f)
    model_name = path['mixtral']

with open('./device_map.json', 'r') as f:
    device_map = json.load(f)

llm = MixtralForCausalLM.from_pretrained(
    model_name,
    # device_map='auto',
    device_map=device_map,
    use_cache=True,
    torch_dtype=torch.float16,
    # attn_implementation="flash_attention_2"
) 
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# convert_mixtral_model(llm, start_num=-1, end_num=32, gamma=0.2,)
# for name, param in llm.named_parameters():
#     print(name)

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

# Test Model
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

def evaluate(task_name_list, model, tokenizer, num_fewshot, device):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
    model=hflm,
    tasks=task_name_list,
    num_fewshot=num_fewshot)
    print(results['results'])

task_name_list=['winogrande','sciq','openbookqa','arc_challenge','arc_easy']
num_fewshot = 0

evaluate(task_name_list, llm, tokenizer, num_fewshot, 'cuda')
for layerid in range(32):
    for expertid in range(8):
        llm.model.layers[layerid].block_sparse_moe.experts[expertid].print_ratio()

# up: 2bit
# {'arc_easy': {'acc,none': 0.8236531986531986, 'acc_stderr,none': 0.007820313817947478, 'acc_norm,none': 0.8202861952861953, 'acc_norm_stderr,none': 0.007878465068489398, 'alias': 'arc_easy'}, 
# 'arc_challenge': {'acc,none': 0.5358361774744027, 'acc_stderr,none': 0.014573813664735622, 'acc_norm,none': 0.5802047781569966, 'acc_norm_stderr,none': 0.014422181226303012, 'alias': 'arc_challenge'}, 
# 'openbookqa': {'acc,none': 0.358, 'acc_stderr,none': 0.021461434862859133, 'acc_norm,none': 0.464, 'acc_norm_stderr,none': 0.022324981738385333, 'alias': 'openbookqa'}, 
# 'sciq': {'acc,none': 0.964, 'acc_stderr,none': 0.00589395781616553, 'acc_norm,none': 0.957, 'acc_norm_stderr,none': 0.006418114379799739, 'alias': 'sciq'}, 
# 'winogrande': {'acc,none': 0.7426992896606156, 'acc_stderr,none': 0.012285989618865697, 'alias': 'winogrande'}}

# + sparsity
