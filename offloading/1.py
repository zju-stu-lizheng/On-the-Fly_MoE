import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,1,2"
os.environ["TOKENIZERS_PARALLELISM"] = "False"
from modeling_mixtral import MixtralForCausalLM
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from typing import Optional
import json
### HQQ量化
from hqq.core.quantize import *
from hqq.models.hf.mixtral import MixtralPatch
import transformers
from hqq.models.base import BaseHQQModel
from accelerate import init_empty_weights
import random 
import numpy as np

seed = 42
# 设置 PyTorch 的随机数种子
torch.manual_seed(seed)
# 设置 Python 的随机数种子
random.seed(seed)
# 设置 NumPy 的随机数种子
np.random.seed(seed)
# 设置 CUDA 的随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

with open('../path.json', 'r') as f:
    path = json.load(f)
    model_name = '/data/Mixtral-8x7B'
    threshold_path = '/data/On-the-Fly_MoE_Inference/saving/threshold/c4_mixtral_up'
    fineweb_path = '/data/datasets/fineweb_edu_sample100000.json'

save_dir = './hqqsaved'
dtype = torch.float16
backend       = "bitblas" #'torchao_int4' #"torchao_int4" (4-bit only) or "gemlite" (4-bit + 2-bit)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
### 从保存的权重中加载
llm = MixtralHQQ.from_quantized(save_dir, compute_dtype=dtype, device='cpu')
HQQLinear.set_backend(HQQBackend.PYTORCH)
# # #Optimize
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(llm, backend=backend, verbose=True)

# %%
device_map = {layer_idx: 'cuda:1' if layer_idx <= 16 else 'cuda:2' for layer_idx in range(1, 32)}
from convert import convert_mixtral_to_cached_mlp

prefill_layers = 6  ### 固定在device上的MLP层数
device_id = 0
llm, cached_mlps = convert_mixtral_to_cached_mlp(llm, dtype, sparsity=0.8, backends=backend, 
                                                 device='cuda:0', device_map=device_map, threshold_path = threshold_path, prefill_layers=prefill_layers)

# if(backend == 'gemlite'):
# 	gemlite.core.GEMLITE_TRITON_RESTRICT_M = True
# 	gemlite.core.GemLiteLinear.load_config('/tmp/gemlite_config.json')

# %%
from pipelinellm import PipelineLLM
# 创建流水线模型
PLLM = PipelineLLM(llm, cached_mlps, 1, 3, training_epoch=20,
                   device='cuda:0', device_map=device_map, prefill_layers=prefill_layers, print_layer_info=True) ### use ep

# %%
# class CUDAGraphRunner():
#     def __init__(self, model):
#         self.model = model
#         self.cuda_graph = None
#         self.graph_input = torch.zeros((1,4096), dtype=torch.float16, device=f'cuda:{device_id}')
#         self.graph_output = None
    
#     def capture(self, x,):
#         assert self.cuda_graph is None
#         self.graph_input = self.graph_input.copy_(x).to(x.device)
#         self.cuda_graph = torch.cuda.CUDAGraph()
#         # self.cuda_graph.enable_debug_mode()
#         with torch.cuda.graph(self.cuda_graph):
#             self.graph_output = self.model(self.graph_input,)
#         torch.cuda.synchronize()
        
#     def forward(self, x,):
#         self.graph_input.copy_(x)
#         self.cuda_graph.replay()
#         return self.graph_output

#     def __call__(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)
    
# inp = torch.randn(1, 4096).half().cuda(device_id)
# for i in range(prefill_layers):
#     for j in range(len(llm.model.layers[0].block_sparse_moe.experts)):
#         expert=llm.model.layers[i].block_sparse_moe.experts[j]
#         # print(expert(inp))
#         graph_runner = CUDAGraphRunner(expert)
#         graph_runner.capture(inp)
#         # print(graph_runner(inp))

#         llm.model.layers[i].block_sparse_moe.experts[j].graph = graph_runner

# %% [markdown]
# ### 测试时间开销

# %%
import json
# import torch._dynamo.config
from datasets import load_dataset, Dataset
from transformers import GenerationConfig

input_length = 128
MAX_LENGTH = 128
output_length = 10
test_samples = 3

def preprocess_data(data, tokenizer):
	# 使用 tokenizer 将文本数据转换为模型输入
	inputs = tokenizer(data, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
	inputs["labels"] = inputs.input_ids.clone()
	return inputs

filt_type = fineweb_path.split('.')[-1]
fineweb = load_dataset(filt_type, data_files=fineweb_path) #726000
fineweb_text = fineweb['train']['text']

print("warm up ...")
# 预热（避免第一次运行时的额外开销）
for text in fineweb_text[:test_samples]:
    inputs = preprocess_data(text, tokenizer)
    with torch.no_grad():
        output = llm(input_ids=inputs["input_ids"].cuda(device_id), attention_mask=inputs["attention_mask"].cuda(device_id))

# %%
from transformers import GenerationConfig
import torch

# 定义 input_length 和 output_length 的范围
# input_length_range = [128]
# output_length_range = [128]
# input_length_range = [16,]  # 16到32
input_length_range = [32,64,128]
output_length_range = [256,512,1024]  # 128到1024

test_samples = 5
device_id = 0

def preprocess_data(data, tokenizer, max_length):
    # 使用 tokenizer 将文本数据转换为模型输入
    inputs = tokenizer(data, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    inputs["labels"] = inputs.input_ids.clone()
    return inputs

for input_length in input_length_range:
    for output_length in output_length_range:
        MAX_LENGTH = input_length
        generated_all = 0
        prefill_time, decode_time = 0, 0
        reloaded_experts = 0

        # 打开文件以写入结果
        with open(f"{input_length}-{output_length}.out", "w") as f:
            print(f"output length is {output_length}", file=f)
            for text in fineweb_text[2:2+test_samples]:
                inputs = preprocess_data(text, tokenizer, MAX_LENGTH)
                ### 清空统计数据
                PLLM.get_prefill_time()
                PLLM.get_reload_experts()

                # 测试时间
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # 开始计时
                torch.cuda.synchronize()
                start_event.record()

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
                end_event.record()
                torch.cuda.synchronize()

                if (len(output[0]) - input_length) == output_length:
                    # 计算时间
                    elapsed_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
                    decode_time += elapsed_time
                    cur_prefill_time = PLLM.get_prefill_time()
                    prefill_time += cur_prefill_time
                    generated_all += (len(output[0]) - input_length)
                    reloaded_experts += PLLM.get_reload_experts()
                print(f"Generated length: {len(output[0]) - input_length}", f"Time taken: {elapsed_time:.2f} s,", f"prefill time: {cur_prefill_time:.2f} s", file=f)
                # print(output, file=f)
                print(tokenizer.batch_decode(output, skip_special_tokens=True), file=f)

            print("Generate speed:", '{:.4f}'.format((generated_all) / decode_time) , 'token/s', file=f)
            timepertoken = (decode_time - prefill_time) / (generated_all)
            print("decode phase speed(not cover prefill phase):", '{:.4f}'.format(1/timepertoken) , 'token/s', file=f)
            expertpertoken = reloaded_experts / generated_all
            print("the number of reloaded experts per token:", '{:.3f}, ({:.2f}%)'.format(expertpertoken, 100 * expertpertoken / ((32-prefill_layers) * 2)), file=f)

