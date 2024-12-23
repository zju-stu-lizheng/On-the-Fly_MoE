import torch
import json
from modeling_mixtral_up import dataset_x, dataset_y, set_skip_layer_idx
import os
from utils import get_c4_data, get_model, set_seed
from tqdm import tqdm
# import argparser

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
### from path.json read paths of model and dataset
model_name = "mixtral"
dataset_name = "c4"
MAX_LENGTH = 512
with open('../path.json', 'r') as file:
    paths = json.load(file)
    model_path = paths.get(model_name, '')
    dataset_path = paths.get(dataset_name, '')
    save_path = paths.get('gatemulup_path','')

def save_datasets(fileid,layerid=1,use_x1=False):
    for i in range(8):
        # print(i)
        # print(dataset_x[i])
        dx = torch.cat(dataset_x[i])
        dy = torch.cat(dataset_y[i])
        dataset_x[i].clear()
        dataset_y[i].clear()
        torch.cuda.empty_cache()
        d = [dx, dy]
        # if not (i != 2 and i != 5):
        torch.save(d, f'{save_path}/{fileid}-{layerid}-mixtral-{i}.pth')
        del dx
        del dy

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
c4data = get_c4_data(model_path, dataset_path, sample_num = 8000)
model = get_model(model_path)
for layerid in range(15,32):
    run_c4(c4data, model, layerid=layerid, sample_nums=1000)
