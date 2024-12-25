import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.data import Dataset
import torch.optim as optim
import torch
import json

with open('../path.json', 'r') as file:
    paths = json.load(file)
    save_path = paths.get('channel_gate_path','')

def load_datasets(layerid = 1, expertid = 0, startid=1, endid=4, use_x1 = False):   
    datasets_x = []
    datasets_y = []
    datasets_x1 = []
    for fileid in range(startid, endid):
        # print(fileid)
        # 加一个map_location
        d = torch.load(f'{save_path}/{fileid}-{layerid}-gate.pth', map_location=lambda storage, loc: storage.cuda(0))
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


class CustomDataset(Dataset):
    def __init__(self, layerid = 1, expertid = 0, startid=1, endid=4, use_x1 =False):
        # 加载数据self.data_x1,
        self.use_x1 = use_x1
        if use_x1:
            self.data_x, self.data_x1, self.data_y = load_datasets(layerid,startid=startid,endid=endid,use_x1=use_x1)
            print(len(self.data_x1),len(self.data_x),len(self.data_y))
        else:
            self.data_x, self.data_y = load_datasets(layerid,startid=startid,endid=endid,use_x1=use_x1)
            print(len(self.data_x),len(self.data_y))

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        if self.use_x1:
            return self.data_x[idx],self.data_x1[idx],self.data_y[idx]
        else:
            return self.data_x[idx],self.data_y[idx]
        

for layerid in range(32):
    dataset = CustomDataset(layerid, startid=1, endid=5)
    print(layerid, len(dataset), dataset[0][0].shape, dataset[0][1].shape) # torch.Size([512, 4096])
    
    ### 统计的个数
    counts = len(dataset)
    
    updata_sum = torch.zeros_like(dataset[0][1])

    for i in range(counts):
        updata_sum += torch.abs(dataset[i][1])

    updata_sum /= counts
    torch.save(updata_sum, f'{save_path}/{layerid}-average.pth')
