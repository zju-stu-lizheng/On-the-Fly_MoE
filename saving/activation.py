import functools
import torch
import json
from modeling_llama_up import LlamaForCausalLM, set_th_sparsity, dataset_x, dataset_y, dataset_x1, set_skip_layer_idx
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
### from path.json read paths of model and dataset
model_name = "Llama3-8b"
dataset_name = "c4"
MAX_LENGTH = 512
with open('../path.json', 'r') as file:
    paths = json.load(file)
    model_path = paths.get(model_name, '')
    dataset_path = paths.get(dataset_name, '')
    save_path = paths.get('gatemulup_path','')

def get_model(model_path):
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        use_cache=False,
        torch_dtype=torch.float16,
    )
    sparsity=0
    set_th_sparsity(sparsity)
    print(f'with sparsity of {sparsity}')
    return model

def preprocess_data(batch, tokenizer):
    # 使用 tokenizer 将文本数据转换为模型输入
    # inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=100, return_tensors="pt")
    inputs = tokenizer(batch['text'], padding=False, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs["labels"] = inputs.input_ids.clone()
    return inputs

def get_c4_data(sample_num = 4000):
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

def save_datasets(fileid,layerid=1,use_x1=True):
    print(dataset_x[0].shape)
    dx = torch.cat(dataset_x,dim=1)
    dataset_x.clear()
    dy = torch.cat(dataset_y,dim=1)
    dataset_y.clear()
    torch.cuda.empty_cache()
    if use_x1:
        dx1 = torch.cat(dataset_x1,dim=1)
        dataset_x1.clear()
        torch.cuda.empty_cache()
        d = [dx, dx1, dy]
    else:
        d = [dx, dy]
    torch.save(d, f'{save_path}/{fileid}-{layerid}-more.pth')
    del dx
    if use_x1:
        del dx1
    del dy
    torch.cuda.empty_cache()

# 定义数据加载器
# batch_size = 1
# dataloader = DataLoader(c4_dataset['validation'], batch_size=batch_size)
# dataloader = DataLoader(top_four_thousand_data, batch_size=batch_size)

def run_c4(c4data, model, layerid = 15, sample_nums = 400):
    # 计算评估损失
    total_loss = 0.0
    num_batches = 0
    set_skip_layer_idx(layerid)

    for batch_idx, data in enumerate(tqdm(c4data)):
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        labels = data['labels'].cuda()
        
        # 禁用梯度计算
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            if num_batches % sample_nums == 0:
                print(f"[{num_batches}], Eval Loss: {total_loss / (num_batches)}")
                save_datasets(num_batches // sample_nums, layerid, use_x1=False)

    # 计算平均损失
    eval_loss = total_loss / num_batches
    print(f"Eval Loss: {eval_loss}")

set_seed(42)
c4data = get_c4_data(sample_num = 4000)
model = get_model(model_path)
for layerid in range(22, 32):
    run_c4(c4data, model, layerid=layerid)