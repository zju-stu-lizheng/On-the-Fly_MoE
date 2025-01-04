import os
import torch
from modeling_mixtral import MixtralForCausalLM
from transformers import AutoTokenizer

# Test Model
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

def get_model(model_name, device_map):
    llm = MixtralForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        use_cache=True,
        torch_dtype=torch.float16,
    ) 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return llm, tokenizer

def evaluate(task_name_list, model, tokenizer, num_fewshot, device):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
    model=hflm,
    tasks=task_name_list,
    num_fewshot=num_fewshot)
    print(results['results'])

def myevaluate(task_name_list, model, tokenizer, num_fewshot, device):
    evaluate(task_name_list, model, tokenizer, num_fewshot, device)
    for layerid in range(32):
        for expertid in range(8):
            model.model.layers[layerid].block_sparse_moe.experts[expertid].print_ratio()

