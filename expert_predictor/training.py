### 专家预测器
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast # 用于混合精度训练
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def top_k_position_accuracy_unordered(output, target, k=1):
	"""Compute the accuracy based on the intersection of top-k values between output and target,
	   regardless of their order."""
	with torch.no_grad():
		# 获取 output 和 target 中 top-k 最大值的索引
		_, topk_pred_indices = output.topk(k, 1, True)
		_, topk_target_indices = target.topk(k, 1, True)
		# 初始化批次的正确计数
		batch_size = output.size(0)
		correct_counts = 0
		
		# 检查每个样本的预测top-k是否包含在真实的top-k中
		for i in range(batch_size):
			# 将预测和目标的top-k索引转换为集合
			set_pred = set(topk_pred_indices[i].tolist())
			set_target = set(topk_target_indices[i].tolist())
			
			# 计算交集
			intersection = set_pred.intersection(set_target)
			
			# 计算正确的预测个数
			correct_counts = correct_counts+len(intersection)
		
		# 计算平均正确率
		return correct_counts,batch_size*k

def eval_model(model, val_loader,):
	# Example validation loop
	model.eval()
	total_topk_accuracy_1 = 0
	total_topk_accuracy_2 = 0
	cont=0
	len1=0
	len2=0
	with torch.no_grad():
		for inputs, targets in val_loader:
			inputs, targets = inputs.to("cuda"), targets.to("cuda")
			with autocast():
				outputs = model(inputs)
			# 计算 top-K 准确率（不考虑顺序）
			topk_accuracy_1 = top_k_position_accuracy_unordered(outputs, targets, k=1)
			topk_accuracy_2 = top_k_position_accuracy_unordered(outputs, targets, k=2)
			total_topk_accuracy_1 += topk_accuracy_1[0]
			total_topk_accuracy_2 += topk_accuracy_2[0]
			len1+= topk_accuracy_1[1]
			len2+= topk_accuracy_2[1]   
		avg_topk_accuracy_1 = total_topk_accuracy_1 / len1
		avg_topk_accuracy_2 = total_topk_accuracy_2 / len2
		# print(len2)
		print(f'Top-{1} Accuracy: {avg_topk_accuracy_1:.4f}', f'Top-{2} Accuracy (unordered): {avg_topk_accuracy_2:.4f}')

class CustomDataset(Dataset):
	def __init__(self, file_paths):
		# 加载数据
		self.data = []
		
		# 遍历文件路径列表，加载每个文件
		for file_path in file_paths:
			# 加载当前文件的数据
			file_data = torch.load(file_path)
			# 将当前文件的数据追加到总数据列表中
			self.data.extend(file_data)
		
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		x, y = self.data[idx]
		return x.detach().clone(), y.detach().clone()

def sparse_row(row, topk=2, use_abs=False):
	"""
	对每一行保留 topk 个最大值的索引为 1，其余为 0。
	
	参数:
		row (torch.Tensor): 输入的一行数据。
		topk (int): 保留的最大值的数量。
		use_abs (bool): 是否使用绝对值进行排序。
	
	返回:
		sparse_row (torch.Tensor): 稀疏化后的行。
	"""
	if use_abs:
		row = torch.abs(row)  # 如果需要使用绝对值，先取绝对值
	
	# 找到 topk 个最大值的索引
	topk_indices = torch.topk(row, topk).indices
	
	# 创建一个与 row 相同大小的零张量
	sparse_row = torch.zeros_like(row)
	
	# 将 topk_indices 对应的值置为 1
	sparse_row[topk_indices] = 1
	
	return sparse_row

def generate_label(y, topk=2, use_abs=False):
	"""
	对输入的张量 y 的每一行进行稀疏化，保留 topk 个最大值的索引为 1，其余为 0。
	
	参数:
		y (torch.Tensor): 输入的张量。
		topk (int): 保留的最大值的数量。
		use_abs (bool): 是否使用绝对值进行排序。
	
	返回:
		sparse_tensor (torch.Tensor): 稀疏化后的张量。
	"""
	# 对每一行进行稀疏化
	sparse_tensor = torch.stack([sparse_row(row, topk, use_abs) for row in y])
	return sparse_tensor

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25, writer=None, layeridx = 0):
	scaler = GradScaler()  # 创建 GradScaler 对象
	for epoch in range(epochs):
		model.train()
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			inputs, targets = inputs.cuda(), targets.cuda()

			optimizer.zero_grad()

			# 使用 autocast 来进行自动混合精度处理
			with torch.cuda.amp.autocast():
				outputs = model(inputs)
				### targets 按照大小编码成 0,1 
				loss = criterion(outputs, generate_label(targets))

			# 使用 GradScaler 来缩放损失，然后进行反向传播
			# 注意：反向传播不包含在 autocast() 块中
			scaler.scale(loss).backward()
			writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_loader) + batch_idx)
			# 调用 scaler.step() 来更新模型权重，并调用 scaler.update() 准备下一步
			scaler.step(optimizer)
			scaler.update()
		if epoch % 5 == 0:
			model.eval()
			eval_model(model, val_loader,)
			torch.save(model.state_dict(), f"./training/{layeridx}-{epoch}.pth")

# #### 重新训练router

# %%
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast  # 用于混合精度训练
import torch.nn.functional as F
import torch.nn.init as init

class SimpleLinearModel(nn.Module):
	def __init__(self,input_dim,output_dim,hidden_dim=32):
		super(SimpleLinearModel, self).__init__()
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.activation = nn.SiLU() # 添加 ReLU 激活函数
		self.linear2 = nn.Linear(hidden_dim,output_dim)  # 添加一个 8x8 线性层
		init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
		init.kaiming_normal_(self.linear2.weight, mode='fan_out', nonlinearity='relu')
		self.linear1.bias.data.fill_(0)
		self.linear2.bias.data.fill_(0)

	def forward(self, x):
		x= self.linear1(x)
		x= self.activation(x)
		return self.linear2(x)


def train_ep(args):
	for i in [6,7,8,9,31]:
	# for i in range(1, 4):
		print("layer ", i)
		file_names = [f'merge/a2ef_{i}_{j}.pth' for j in range(10)]
		dataset = CustomDataset(file_paths=file_names)
		# 划分训练集和验证集
		train_size = int(0.8 * len(dataset))
		val_size = len(dataset) - train_size
		train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
		train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
		print(len(train_dataset))
		val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

		model=SimpleLinearModel(4096,8, hidden_dim=512)
		model.to("cuda")  # 假设使用 GPU
		eval_model(model, val_loader)
		# criterion = nn.MSELoss().to("cuda")
		# criterion = nn.CrossEntropyLoss().to("cuda")
		criterion = nn.SmoothL1Loss()
		# criterion = nn.KLDivLoss(reduction='batchmean').to("cuda")
		optimizer = optim.Adam(model.parameters(), lr=5e-4) #lr=5e-5
		writer = SummaryWriter('runs/predictor_multilayer')
		train_model(model, train_loader, val_loader, criterion, optimizer, epochs=args.epochs, writer=writer, layeridx=i)
		torch.save(model.state_dict(), f"./training/{i}-{args.epochs}.pth")
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--threshold_path", type=str, default='training_sparsity_path')
	parser.add_argument("--use_average", action='store_true', help='use average threshold')
	parser.add_argument("--sparsity_level", type=float, default=0.8)
	parser.add_argument("--epochs", type=int, default=4)
	args = parser.parse_args()
	dtype = torch.float16
	train_ep(args)
