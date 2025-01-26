import argparse
from torch import nn
import torch.nn.init as init
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast  # 用于混合精度训练
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch
import random
import numpy as np

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int, default=2)
parser.add_argument('--sparsity',type=int, default=90)
parser.add_argument('--layerid',type=int, default=31)
parser.add_argument('--predictions', type=float, default=0.2)
opt = parser.parse_args()
print(opt)
set_seed(2024)
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)


def load_datasets(layerid = 1, expertid = 0, dataset_name="arc"):
    print("Loading dataset", dataset_name)
    datasets_x = []
    # datasets_x1 = []
    datasets_y = []
    if dataset_name == "arc":
        for fileid in range(1, 9):
            # print(fileid)
            d = torch.load(f'/mnt/newdata/lz/sparsity/arc_llama3/{fileid}-{layerid}.pth', map_location=lambda storage, loc: storage.cuda(0))
            datasets_x.append(d[0])
            datasets_y.append(d[1])
    else:
        for fileid in range(1, 4):
            # print(fileid)
            d = torch.load(f'/mnt/newdata/lz/sparsity/c4_llama/adapter/{fileid}-{layerid}.pth', map_location='cpu')
            datasets_x.append(d[0])
            # datasets_x1.append(d[1])
            datasets_y.append(d[1])
    x,y = torch.cat(datasets_x), torch.cat(datasets_y)
    datasets_x.clear()
    # datasets_x1.clear()
    datasets_y.clear()
    x = x.reshape(-1, 4096)
    # x1 = x1.reshape(-1, 14336)
    y = y.reshape(-1, 14336)
    # print(x[0].shape)
    return x,y

class CustomDataset(Dataset):
    def __init__(self, layerid = 1, expertid = 0):
        # 加载数据
        self.data_x, self.data_y = load_datasets(layerid, dataset_name='fineweb')

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        return self.data_x[idx],self.data_y[idx]


class SimpleLinearModel(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=32):
        super(SimpleLinearModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim,bias=False)
        # self.activation = nn.SiLU() # 添加激活函数
        self.linear2 = nn.Linear(hidden_dim,output_dim,bias=False)  

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
from tqdm import tqdm
import csv
# 读取CSV文件

def read_csv_to_2d_list(filename):
    data_2d_list = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # 将每一行的字符串转换为浮点数
            float_row = [float(value) for value in row]
            data_2d_list.append(float_row)
    return data_2d_list

def set_th_sparsity():
    """
    根据稀疏程度读取对应的阈值
    """
    global th
    filename="/mnt/storage/zyx/Meta-Llama-3-8B/conbine8b1000_th_90.csv"
    th = read_csv_to_2d_list(filename)

    
def sparse_row(row, keep_ratio=0.1, use_abs = False):
    # 根据绝对值最大搜索
    if use_abs:
        row = torch.abs(row)
    # 设置阈值
    # print(row.type)
    threshold = torch.quantile(row.float(), 1-keep_ratio)  # 选择前20%的值
    sparse_row = (row > threshold).float()
    
    return sparse_row

def generate_label(y, keep_ratio, use_abs=False):
    # 对每一行进行稀疏化
    sparse_tensor = torch.stack([sparse_row(row, keep_ratio, use_abs) for row in y])
    return sparse_tensor

def test_model(model, val_loader, sparsity=0.1):
    model.eval()
    # 初始化总的统计变量
    total_correct_preds = 0
    total_preds = 0
    total_labels = 0
    total_masks = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            with autocast():
                outputs = model(inputs)
            # print(outputs.shape)

            # preds, truth = generate_test_label(outputs, targets, threshold)
            preds = generate_label(outputs, keep_ratio=sparsity)
            truth = generate_label(targets, keep_ratio=0.1, use_abs = True)
            # truth = targets
            
            # 计算当前batch的精度
            dif = truth - preds
            miss = dif > 0.0 # classifier didn't activated target neuron

            total_correct_preds += (truth.sum(dim=1).float() - miss.sum(dim=1).float()).mean().item()
            total_preds += (preds.sum(dim=1).float()).mean().item()
            total_labels += (truth.sum(dim=1).float()).mean().item()
            total_masks += (preds.numel() / len(preds))
    
    print('大于阈值的维度:', (total_labels / total_masks))
    print('预测与标签选取的数量比:',(total_preds / total_labels))
    print('覆盖率(Recall):',(total_correct_preds / total_labels))

def train_model(model, train_loader, val_loader, criterion, optimizer, writer, epochs=25, layerid=1, sparsity=0.2):
    scaler = GradScaler()  # 创建 GradScaler 对象
    for epoch in range(epochs):
        if epoch % 5 == 0:
            print(f'---------after training {epoch} epochs---------')
            test_model(model, val_loader, sparsity=sparsity)
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()

            targets = generate_label(targets, sparsity, use_abs =True)

            # 使用 autocast 来进行自动混合精度处理
            with autocast():
                outputs = model(inputs)
                probs = outputs.sigmoid()
                # cross_entropy
                loss = criterion(probs, targets)

            # 使用 GradScaler 来缩放损失，然后进行反向传播
            # 注意：反向传播不包含在 autocast() 块中
            scaler.scale(loss).backward()
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_loader) + batch_idx)
            # 调用 scaler.step() 来更新模型权重，并调用 scaler.update() 准备下一步
            scaler.step(optimizer)
            scaler.update()
    print(f'---------after training {epochs} epochs---------')
    test_model(model, val_loader, sparsity=sparsity)
    torch.save(model.state_dict(), f'/home/lz/workspace/llama2-7b/moe-offloading/notebooks/output/sparsity/{layerid}.pt')


layerid = opt.layerid
sparsity = opt.predictions
set_th_sparsity()

threshold = th[layerid][0]

hidden_dim = 256
model=SimpleLinearModel(4096,14336,hidden_dim=hidden_dim)
model.to("cuda")  # 假设使用 GPU
# criterion = nn.MSELoss().to("cuda")
criterion = nn.CrossEntropyLoss().to("cuda")
# criterion = nn.KLDivLoss(reduction='batchmean').to("cuda")
optimizer = optim.Adam(model.parameters(), lr=2e-4) #lr=5e-5
writer = SummaryWriter('runs/predictor_sparsity')
dataset = CustomDataset(layerid)
# print(len(dataset), dataset[0][0].shape, dataset[0][1].shape) # torch.Size([512, 4096])
# 划分训练集和验证集
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
print(len(train_dataset),'hidden_dim:',hidden_dim)

test_model(model, val_loader, sparsity=sparsity)

# train_model(model, train_loader, val_loader, criterion, optimizer, writer=writer, epochs=10, layerid=layerid, sparsity=opt.predictions)