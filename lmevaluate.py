# Test Model
import os
import json
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from convert_llama import convert_llama_model
# export HF_ENDPOINT="https://hf-mirror.com"
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator


def _load_model(model_name = "Llama3-8b"):
    print(f"Loading model {model_name}")
    ### from path.json read paths of model and dataset
    with open('path.json', 'r') as file:
        paths = json.load(file)
        model_path = paths.get(model_name, '')

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        use_cache=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer

def evaluate(task_name_list, model, tokenizer, num_fewshot, device):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
    model=hflm,
    tasks=task_name_list,
    num_fewshot=num_fewshot)
    print(results['results'])


def main(task_name_list, model_name, sparsity, start_num, end_num, token_sparsity, is_sparsity, device, num_fewshot, gamma=0.3):
    model, tokenizer = _load_model(model_name)
    
    if is_sparsity == True:
        model = convert_llama_model(model, sparsity, start_num, end_num, token_sparsity, gamma=gamma, use_core=False)

    evaluate(task_name_list, model, tokenizer, num_fewshot, device)

# triviaqa
task_list=['boolq','sciq','openbookqa','winogrande','arc_challenge','arc_easy']
num_fewshot = 0
predictor_ratio = 0.5

# task_list=['truthfulqa_gen','boolq']
main(task_name_list=task_list, model_name="Llama3-8b", sparsity=0.1, start_num=21, end_num=32, token_sparsity=0.1, is_sparsity=True, device='cuda', num_fewshot=num_fewshot, gamma=predictor_ratio)