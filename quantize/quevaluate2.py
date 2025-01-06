import torch
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
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

dtype = torch.float16
lora_save_path='./saved/training/lora_weights.pt'

llm, tokenizer = get_model(model_name, device_map, dtype=dtype)

#Quantize
from hqq.core.quantize import *
q4_config    = BaseQuantizeConfig(nbits=8, group_size=64) 
q3_config    = BaseQuantizeConfig(nbits=2, group_size=64)

quant_config = {
  'block_sparse_moe.experts.w3'  :q3_config,
}
from hqq.models.hf.base import AutoHQQHFModel
AutoHQQHFModel.quantize_model(llm, quant_config=quant_config, compute_dtype=dtype, device=device_map)     

#First, quantize/load a quantized HQQ model the
from hqq.core.peft import PeftUtils

PeftUtils.load_lora_weights(llm, lora_save_path)
        
task_name_list=['winogrande','sciq','openbookqa','arc_challenge','arc_easy']
num_fewshot = 0
myevaluate(task_name_list, llm, tokenizer, num_fewshot, 'cuda')