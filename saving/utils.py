import functools
import torch
from modeling_llama_up import LlamaForCausalLM, set_th_sparsity
from modeling_mixtral_up import MixtralForCausalLM
import modeling_mixtral_up
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MAX_LENGTH = 512

def get_model(model_path):
    sparsity=0
    if 'Llama' in model_path:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            use_cache=False,
            torch_dtype=torch.float16,
        )
        set_th_sparsity(sparsity)
    else:
        model = MixtralForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            use_cache=False,
            torch_dtype='float16',
            # attn_implementation="flash_attention_2"
        )
        modeling_mixtral_up.set_th_sparsity(sparsity)
    
    print(f'with sparsity of {sparsity}')
    return model

def preprocess_data(batch, tokenizer):
    # 使用 tokenizer 将文本数据转换为模型输入
    # inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=100, return_tensors="pt")
    inputs = tokenizer(batch['text'], padding=False, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs["labels"] = inputs.input_ids.clone()
    return inputs

def get_c4_data(model_path, dataset_path, sample_num = 4000):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    # # 加载 C4 数据集的验证集
    c4 = load_dataset(dataset_path)
    # 对数据集进行预处理
    c4_dataset = c4.map(
        functools.partial(
        preprocess_data,
        tokenizer=tokenizer
    ))
    c4_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    top_four_thousand_data = c4_dataset['validation'].select(range(sample_num))
    return top_four_thousand_data

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
