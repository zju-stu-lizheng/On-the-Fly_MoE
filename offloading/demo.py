# OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python demo.py 
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"
os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from modeling_mixtral import MixtralForCausalLM
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from typing import Optional
import json

with open('../path.json', 'r') as f:
    path = json.load(f)
    model_name = path['mixtral']

save_dir = './hqqsaved'
backend       = "bitblas" #'torchao_int4' #"torchao_int4" (4-bit only) or "gemlite" (4-bit + 2-bit)
dtype = torch.bfloat16 if backend=="torchao_int4" else torch.float16
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

### HQQ量化
from hqq.core.quantize import *
from hqq.models.hf.mixtral import MixtralPatch
import transformers
from hqq.models.base import BaseHQQModel
from accelerate import init_empty_weights

class BaseHQQHFModel(BaseHQQModel):
    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir):
        model.config.save_pretrained(save_dir)

    # Create empty model from config
    @classmethod
    def create_model(cls, save_dir, kwargs):
        model_kwargs = {}
        for key in ["attn_implementation"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        config = transformers.AutoConfig.from_pretrained(
            cls.get_config_file(save_dir)
        )

        with init_empty_weights():
            model = MixtralForCausalLM._from_config(config, **model_kwargs)

        return model

class MixtralHQQ(MixtralPatch, BaseHQQHFModel):
    pass
 

llm = MixtralHQQ.from_quantized(save_dir, compute_dtype=dtype, device='cuda:0', use_cache=True)
HQQLinear.set_backend(HQQBackend.PYTORCH)


# #Optimize
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(llm, backend=backend, verbose=True)