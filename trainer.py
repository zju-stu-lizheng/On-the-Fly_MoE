from torch import nn
import torch.nn.init as init
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast  
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch
import json
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

with open('path.json', 'r') as file:
    paths = json.load(file)
    save_path = paths.get('gatemulup_path','')

class CustomDataset(Dataset):
    def __init__(self, layerid = 1, expertid = 0, startid=1, endid=4, use_x1 =False):
        # 加载数据self.data_x1,
        self.use_x1 = use_x1
        if use_x1:
            self.data_x, self.data_x1, self.data_y = self.load_datasets(layerid,startid=startid,endid=endid,use_x1=use_x1)
            print(len(self.data_x1),len(self.data_x),len(self.data_y))
        else:
            self.data_x, self.data_y = self.load_datasets(layerid,startid=startid,endid=endid,use_x1=use_x1)
            print(len(self.data_x),len(self.data_y))

    def load_datasets(self, layerid = 1, expertid = 0, startid=1, endid=4, use_x1 = False):   
        datasets_x = []
        datasets_y = []
        datasets_x1 = []
        for fileid in range(startid, endid):
            # print(fileid)
            # 加一个map_location
            d = torch.load(f'{save_path}/{fileid}-{layerid}-more.pth', map_location=lambda storage, loc: storage.cuda(0))
            datasets_x.append(d[0])
            if use_x1:
                datasets_x1.append(d[1])
            datasets_y.append(d[-1])
        x,y = torch.cat(datasets_x,dim=1), torch.cat(datasets_y,dim=1)
        datasets_x.clear()
        datasets_y.clear()
        x = x.reshape(-1, 4096)
        y = y.reshape(-1, 14336)
        # print(x[0].shape)
        if use_x1:
            x1 = torch.cat(datasets_x1,dim=1)
            datasets_x1.clear()
            x1 = x1.reshape(-1, 14336)
            return x,x1,y
        return x,y

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        if self.use_x1:
            return self.data_x[idx],self.data_x1[idx],self.data_y[idx]
        else:
            return self.data_x[idx],self.data_y[idx]

class SimpleLinearModel(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=32):
        super(SimpleLinearModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim,bias=False)
        # self.activation = nn.SiLU() # 添加激活函数
        self.linear2 = nn.Linear(hidden_dim,output_dim,bias=False)  
        init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.linear2.weight, mode='fan_out', nonlinearity='relu')
        # self.linear1.bias.data.fill_(0)
        # self.linear2.bias.data.fill_(0)

    def forward(self, x):
        # x= self.activation(x)
        return self.linear2(self.linear1(x))
    
cnt = 0

def sparse_row(row, keep_ratio=0.1, use_abs = False):
    # 计算需要保留的参数数量
    num_to_keep = int(keep_ratio * row.numel())
    
    # 找到绝对值最大的 num_to_keep 个参数的索引
    if use_abs:
        row = torch.abs(row)
    topk_indices = torch.topk(row, num_to_keep).indices
    # topk_indices = torch.topk(row, num_to_keep).indices
    
    # 创建一个与 row 相同大小的零张量
    sparse_row = torch.zeros_like(row)
    
    # 将 topk_indices 对应的值置为 1
    sparse_row[topk_indices] = 1
    
    return sparse_row

def generate_label(y, sparsity, use_abs=False):
    # 对每一行进行稀疏化
    sparse_tensor = torch.stack([sparse_row(row, sparsity, use_abs) for row in y])
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
            with autocast():
                outputs = model(inputs)

            preds = generate_label(outputs, sparsity)
            truth = generate_label(targets, 0.1, use_abs = True)
            # truth = targets
            
            # 计算当前batch的精度
            dif = truth - preds
            miss = dif > 0.0 # classifier didn't activated target neuron

            total_correct_preds += (truth.sum(dim=1).float() - miss.sum(dim=1).float()).mean().item()
            total_preds += (preds.sum(dim=1).float()).mean().item()
            total_labels += (truth.sum(dim=1).float()).mean().item()

    # print('预测占比:{:.4f}'.format((total_preds/total_masks).item()))
    # print('标签占比:{:.4f}'.format((total_labels/total_masks).item()))
    print('预测与标签选取的数量比:',(total_preds / total_labels))
    print('覆盖率(Recall):',(total_correct_preds / total_labels))

def train_model(model, train_loader, val_loader, criterion, optimizer, writer, epochs=25, layerid=1):
    scaler = GradScaler()  # 创建 GradScaler 对象
    for epoch in range(epochs):
        if epoch % 2 == 0:
            print(f'---------after training {epoch} epochs---------')
            test_model(model, val_loader, sparsity=0.2)
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()

            targets = generate_label(targets, 0.2, use_abs =True)

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
    test_model(model, val_loader, sparsity=0.2)
    global cnt
    torch.save(model.state_dict(), f'./output/sparsity/{layerid}-{cnt}.pt')
    cnt += 1


for layerid in range(22,32):
    print(layerid)
    model=SimpleLinearModel(4096,14336,hidden_dim=1024)
    model.to("cuda")  # 假设使用 GPU
    # criterion = nn.MSELoss().to("cuda")
    criterion = nn.CrossEntropyLoss().to("cuda")
    # criterion = nn.KLDivLoss(reduction='batchmean').to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=5e-4) #lr=5e-5
    writer = SummaryWriter('runs/predictor_sparsity')
    
    cnt = 0
    for startid, endid in [(1,4),(4,7),(7,10)]:
        dataset = CustomDataset(layerid, startid=startid, endid=endid)
        print(len(dataset)) # torch.Size([512, 4096])
        # 划分训练集和验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

        train_model(model, train_loader, val_loader, criterion, optimizer, writer=writer, epochs=4, layerid=layerid)
