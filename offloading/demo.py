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
# from hqq.utils.patching import prepare_for_inference
# prepare_for_inference(llm, backend=backend, verbose=True)


import json
from datasets import load_dataset, Dataset
from transformers import GenerationConfig

input_length = 8
MAX_LENGTH = input_length
output_length = 16
device_id = 0
test_samples = 1

def preprocess_data(data, tokenizer):
	# 使用 tokenizer 将文本数据转换为模型输入
	inputs = tokenizer(data, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
	inputs["labels"] = inputs.input_ids.clone()
	return inputs

generated_all, decode_time, prefill_time = 0, 0, 0
# print("max output length is {}".format(output_length))
text = "The future of AI is here, and "

inputs = preprocess_data(text, tokenizer)
# # 测试时间
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# 开始计时
# torch.cuda.synchronize()
# start_event.record()

# 前向传播
with torch.no_grad():
    output = llm.generate(
        input_ids=inputs["input_ids"].cuda(device_id),
        attention_mask=inputs["attention_mask"].cuda(device_id),
        max_length=input_length + output_length,  # 总长度为输入长度 + 输出长度
        generation_config=GenerationConfig(do_sample=False),
        pad_token_id=tokenizer.pad_token_id, 
        # cache_implementation="static" ## moe not support
    )

# 结束计时
# end_event.record()
# torch.cuda.synchronize()

# 计算时间
# elapsed_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
# decode_time += elapsed_time
# print(f"Generated length: {len(output[0]) - input_length}", f"Time taken: {elapsed_time:.2f} s")
# print(output)
print(tokenizer.batch_decode(output, skip_special_tokens=True))

# generated_all += (len(output[0]) - input_length -1)

# timepertoken = (decode_time) / (generated_all)
# print("decode phase speed:", '{:.4f}'.format(1/timepertoken) , ' token/s')