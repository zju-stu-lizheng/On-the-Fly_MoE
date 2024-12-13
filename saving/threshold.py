import torch
import json
from modeling_llama_up import step, x_small, x_all
import os
import csv
from utils import get_c4_data, get_model, set_seed
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
### from path.json read paths of model and dataset
model_name = "Llama3-8b"
dataset_name = "c4"
MAX_LENGTH = 512
with open('../path.json', 'r') as file:
    paths = json.load(file)
    model_path = paths.get(model_name, '')
    dataset_path = paths.get(dataset_name, '')
    save_path = paths.get('threshold_path','')

def run_c4(c4data, model, sample_nums = 400):
    # 计算评估损失
    total_loss = 0.0

    for batch_idx, data in enumerate(tqdm(c4data)):
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        labels = data['labels'].cuda()
        
        # 禁用梯度计算
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    # 计算平均损失
    eval_loss = total_loss / len(c4data)
    print(f"Eval Loss: {eval_loss}")

set_seed(42)
c4data = get_c4_data(model_path, dataset_path, sample_num = 400)
model = get_model(model_path)
run_c4(c4data, model)

layer_num = 32
expert_num = 1
output_file = f'{save_path}/output-c4-llama3-{step}.csv'

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    col=[]
    for i in range(step):
        col.append((1.0/step)*(i+1))
    writer.writerow(col)

    for layer in range(0, layer_num):
        for expert in range(expert_num):
            row = []
            if x_all[layer][expert] != 0:
                row.append("%.4f" % (x_small[layer][expert][0] * 1.0 / x_all[layer][expert]))
                for i in range(step - 1):
                    row.append("%.4f" % ((x_small[layer][expert][i + 1] - x_small[layer][expert][i]) * 1.0 / x_all[layer][expert]))
            else:
                for i in range(step):
                    row.append("%.4f" % 0.0)
            writer.writerow(row)

print(f"数据已成功写入 {output_file} 文件")

data = []
with open(output_file, mode='r', newline='') as file:
    reader = csv.reader(file)
    
    # 读取每一行并添加到数据列表中
    for row in reader:
        # 将读取的字符串转换为浮点数
        float_row = [float(item) for item in row]
        data.append(float_row)

def get_threshold(th = 0.7):
    t=[[0 for _ in range(expert_num)] for _ in range(layer_num)]
    for mlp, row in enumerate(data):
        if mlp == 0: continue
        p=0
        j=0
        for index, item in enumerate(row):
            # print(index, item)
            p += item   ## p 是累加的概率
            if expert_num == 1:
                t[(mlp-1)] = [(index+0.5)*(1.0/step)]
            else:
                t[(mlp - 1) // expert_num][(mlp - 1) % expert_num] = (index+0.5)*(1.0/step)
            if p >= th:
                break

    print(t)
    threshold = int(th*100)
    with open(f'{save_path}/th_{threshold}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(t)

for th in range(5, 10):
    get_threshold(th * 0.1)

get_threshold(0.95)
get_threshold(0)