import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
os.environ["TOKENIZERS_PARALLELISM"] = "False"
from modeling_mixtral import MixtralForCausalLM
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from typing import Optional
import json

with open('../path.json', 'r') as f:
    path = json.load(f)
    model_name = path['mixtral']
    # threshold_path = path[threshold_path_name]

with open("../quantize/device_map.json", "r") as f:
    device_map = json.load(f)

def get_model(model_name, device_map, dtype=torch.bfloat16, use_cache=True):
    llm = MixtralForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        use_cache=use_cache,
        torch_dtype=dtype,
    ) 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return llm, tokenizer

dtype = torch.float16
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
llm, tokenizer = get_model(model_name, 'balanced_low_0', dtype=dtype)


### HQQ量化
from hqq.core.quantize import *
from hqq.models.hf.mixtral import MixtralHQQ

save_dir = './hqqup2'
### 第一次加载
# q4_config    = BaseQuantizeConfig(nbits=8, group_size=64)
q3_config    = BaseQuantizeConfig(nbits=2, group_size=64)
quant_config      = {'block_sparse_moe.experts.w3'   : q3_config}
MixtralHQQ.quantize_model(llm, quant_config=quant_config, compute_dtype=dtype, device=device_map)
### 先放CUDA量化，然后再传回CPU
MixtralHQQ.save_quantized(llm, save_dir)

### 从保存的权重中加载
# llm = MixtralHQQ.from_quantized(save_dir, compute_dtype=dtype, device='cpu')
# HQQLinear.set_backend(HQQBackend.PYTORCH)


# backend       = "gemlite" #'torchao_int4' #"torchao_int4" (4-bit only) or "gemlite" (4-bit + 2-bit)
# # #Optimize
# from hqq.utils.patching import prepare_for_inference
# prepare_for_inference(llm, backend=backend, verbose=True)
# #Load GemLite cache
# if(backend == 'gemlite'):
# 	import gemlite
# 	gemlite.core.GEMLITE_TRITON_RESTRICT_M = True
# 	gemlite.core.GemLiteLinear.load_config('/tmp/gemlite_config.json')