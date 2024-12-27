import torch
import json
from modeling_llama_up import dataset_x, dataset_y, dataset_x1, set_skip_layer_idx, set_profile_mode
import os
from utils import get_c4_data, get_model, set_seed
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
### from path.json read paths of model and dataset
model_name = "Llama3-8b"
dataset_name = "c4"
with open('../path.json', 'r') as file:
    paths = json.load(file)
    model_path = paths.get(model_name, '')
    dataset_path = paths.get(dataset_name, '')
    save_path = paths.get('channel_up_path','')
    ### channel_gate_path： more对应的是up, gate才是silu*gate
    ### gate_path: (i-1)-th layer x and i-th layer silu(gate)

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
    torch.save(d, f'{save_path}/{fileid}-{layerid}-gate.pth')
    del dx
    if use_x1:
        del dx1
    del dy
    torch.cuda.empty_cache()

# 定义数据加载器
# batch_size = 1
# dataloader = DataLoader(c4_dataset['validation'], batch_size=batch_size)
# dataloader = DataLoader(top_four_thousand_data, batch_size=batch_size)

def run_c4(c4data, model, layerid = 15, samples_per_file = 400):
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
            if num_batches % samples_per_file == 0:
                print(f"[{num_batches}], Eval Loss: {total_loss / (num_batches)}")
                save_datasets(num_batches // samples_per_file, layerid, use_x1=False)

    # 计算平均损失
    eval_loss = total_loss / num_batches
    print(f"Eval Loss: {eval_loss}")

set_seed(42)
sample_num = 4000
set_profile_mode(False) ### not to profile threshold
c4data = get_c4_data(model_path, dataset_path, sample_num = sample_num)
model = get_model(model_path)
for layerid in range(0, 32):
    run_c4(c4data, model, layerid=layerid, samples_per_file=sample_num//5)