import torch
import torch.nn as nn
import time
import torch.nn.functional as F

class CachedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dtype, sparsity: float = 0.2,
    device = 'cuda:0', device_map=None):
        super(CachedMLP, self).__init__()
        self.sparsity = sparsity
        self.activenum = int((1 - sparsity) * hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device
        self.device_map = device_map
        self.compute_compensate = 0.000229
        print("active neural num ",self.activenum)

        self.stream = torch.cuda.Stream(device=device)

        self.activation = nn.SiLU()

        self.w3_expert0, self.w1_expert0, self.w2_expert0 = None, None, None
        self.w3_expert1, self.w1_expert1, self.w2_expert1 = None, None, None

        self.indices0 = torch.empty((self.activenum), dtype=torch.int, device=self.device)
        self.indices1 = torch.empty((self.activenum), dtype=torch.int, device=self.device)

        # 将GPU缓存张量改为列表存储
        self.w_gpu = [
            torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device=self.device) for _ in range(4)
        ]  # [w1_gpu, w2_gpu, w1_gpu_expert1, w2_gpu_expert1]

        self.use_pin = False
        # 将Pinned Memory缓冲区改为列表存储
        if self.device_map[1] == "cpu":
            self.use_pin = True
            
            self.sparse_w_cpu = [
                torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device=self.device_map[1]).pin_memory() for _ in range(4)
            ]  # [sparse_w1_cpu, sparse_w2_cpu, sparse_w1_cpu_expert1, sparse_w2_cpu_expert1]

        ### 增加两个专家序号
        self.expert_ids = torch.tensor([0,1], device=self.device)

    def get_predict_experts(self):
        return self.expert_ids
    
    def load_conflict_cpu(self, cpu_mlp_new, stream: torch.cuda.Stream, hidden_states, replace_idx,
            layer_idx = 0):
        """
        同样地，这里也只复制指针
        """
        ### 也需要sleep一下，这里模拟需要补充代价
        if replace_idx == 0:
            self.w1_expert0 = cpu_mlp_new['w1']
            self.w2_expert0 = cpu_mlp_new['w2']
            self.w3_expert0 = cpu_mlp_new['w3']
        else:
            self.w1_expert1 = cpu_mlp_new['w1']
            self.w2_expert1 = cpu_mlp_new['w2']
            self.w3_expert1 = cpu_mlp_new['w3']

    def load_from_cpu(self, cpu_mlp, cpu_mlp_expert1, stream: torch.cuda.Stream, hidden_states,
            layer_idx=0):
        """
        只复制指针，不进行实际的数据搬运
        
        参数:
            cpu_mlp: 包含CPU上参数的字典（第一个专家）
            cpu_mlp_expert1: 包含CPU上参数的字典（第二个专家）。
            stream: 用于数据传输的CUDA流。
        """
        for expert, cpu_mlp_data in zip(['expert0', 'expert1'], [cpu_mlp, cpu_mlp_expert1]):
            setattr(self, f'w1_{expert}', cpu_mlp_data['w1'])
            setattr(self, f'w2_{expert}', cpu_mlp_data['w2'])
            setattr(self, f'w3_{expert}', cpu_mlp_data['w3'])

        return 

    def load_expert_weights(self, expert_ids):
        # print('loading next expert: ', expert_ids)   ## tensor([7, 6], device='cuda:0')
        self.expert_ids.copy_(expert_ids)

    def parallel_computation(self, hidden_states, layer_idx):
        """
        并行计算函数，支持跨设备计算和异步传输，并对两个专家分别计算。

        参数:
            self: 包含模型参数和设备的对象。
            hidden_states (torch.Tensor): 输入张量，位于 device1 上。
            layer_idx (int): 当前层的索引，用于确定 device2。

        返回:
            tuple: 包含两个专家的计算结果，均位于 device1 上。
        """
        # 获取设备
        device1 = self.device
        device2 = self.device_map[layer_idx]

        # 第1步：将 x 从 device1 传输到 device2（异步）
        with torch.cuda.stream(self.stream):
            x_device2 = hidden_states.to(device2, non_blocking=True)  # 异步传输

        # 第2步：在 device1 上计算 expert0 和 expert1 的 up(x)
        with torch.cuda.stream(self.stream):
            up_x_expert0 = self.w3_expert0(hidden_states)  # expert0 的计算
            up_x_expert1 = self.w3_expert1(hidden_states)  # expert1 的计算

        # 第3步：在 device2 上计算 expert0 和 expert1 的 gate(x_device2)
        gate_x_expert0 = self.activation(torch.matmul(x_device2, self.w1_expert0.T))
        gate_x_expert1 = self.activation(torch.matmul(x_device2, self.w1_expert1.T))

        # 第4步：将 up_x_expert0 和 up_x_expert1 从 device1 传输到 device2（异步）
        with torch.cuda.stream(self.stream):
            up_x_device2_expert0 = up_x_expert0.to(device2, non_blocking=True)  # expert0 的传输
            up_x_device2_expert1 = up_x_expert1.to(device2, non_blocking=True)  # expert1 的传输

        # 第5步：在 device2 上计算 expert0 和 expert1 的 down(gate_x * up_x_device2)
        down_x_device2_expert0 = torch.matmul(gate_x_expert0 * up_x_device2_expert0, self.w2_expert0)
        down_x_device2_expert1 = torch.matmul(gate_x_expert1 * up_x_device2_expert1, self.w2_expert1)

        # 第6步：将 down_x_expert0 和 down_x_expert1 从 device2 传输回 device1（异步）
        with torch.cuda.stream(self.stream):
            down_x_device1_expert0 = down_x_device2_expert0.to(device1, non_blocking=True)  # expert0 的传输
            down_x_device1_expert1 = down_x_device2_expert1.to(device1, non_blocking=True)  # expert1 的传输

        # 同步 Stream，确保所有操作完成
        torch.cuda.synchronize(device1)
        torch.cuda.synchronize(device2)

        # 返回两个专家的计算结果
        return down_x_device1_expert0, down_x_device1_expert1
    
    def forward(self, hidden_states, expert_weights, expert_ids, layer_idx=0):
        """
        根据hidden_states， 分别计算两个专家的输出
        """
        if self.expert_ids[0] != expert_ids[0]:
            # print("----replace expert weights")
            expert_weights[0], expert_weights[1] = expert_weights[1], expert_weights[0]

        down_x_device1_expert0, down_x_device1_expert1 = \
                self.parallel_computation(hidden_states, layer_idx)
        time.sleep(self.compute_compensate)

        final_hidden_states = down_x_device1_expert0 * expert_weights[0] + down_x_device1_expert1 * expert_weights[1]
        return final_hidden_states
          

def load_thresholds(threshold_path, use_average=True):
    """
    load thresholds from path
    """
    # f"{chess_up_threshold}/thresholds_0_8.pt"
    if use_average:
        up_th = torch.load(threshold_path, map_location='cuda')["up_proj_states_thresholds_2"]
    else:
        up_th = torch.load(threshold_path, map_location='cuda')["up_proj_states_thresholds"]
    print(f"Thresholds loaded from {threshold_path}")
    
    return up_th

def convert_mixtral_to_cached_mlp(llm, dtype, sparsity=0.9, backends='bitblas', 
    device='cuda:0', device_map=None, threshold_path='../saving/threshold/c4_mixtral_up', prefill_layers = 1):
    """
    prefill_layers 指固定在device上的层数，便于prefill阶段的使用
    """
    device_number = int(device[-1])
    ### 其他部分存放在device上
    llm.model.embed_tokens.cuda(device_number)
    for i in range(len(llm.model.layers)):
        llm.model.layers[i].self_attn.cuda(device_number)
        llm.model.layers[i].input_layernorm.cuda(device_number)
        llm.model.layers[i].post_attention_layernorm.cuda(device_number)
        ### 原始的gate
        llm.model.layers[i].block_sparse_moe.gate.cuda(device_number)
        for j in range(len(llm.model.layers[0].block_sparse_moe.experts)):
            # if backends == 'gemlite':
            #     llm.model.layers[i].block_sparse_moe.experts[j].w3.cuda(device_number)
            if backends == "bitblas":
                llm.model.layers[i].block_sparse_moe.experts[j].w3.W_q = \
                    llm.model.layers[i].block_sparse_moe.experts[j].w3.W_q.cuda(device_number)
                llm.model.layers[i].block_sparse_moe.experts[j].w3.scale = \
                    llm.model.layers[i].block_sparse_moe.experts[j].w3.scale.cuda(device_number)
                llm.model.layers[i].block_sparse_moe.experts[j].w3.zero = \
                    llm.model.layers[i].block_sparse_moe.experts[j].w3.zero.cuda(device_number)
                llm.model.layers[i].block_sparse_moe.experts[j].w3.device = device
            else:
                llm.model.layers[i].block_sparse_moe.experts[j].w3.cuda(device_number)
                llm.model.layers[i].block_sparse_moe.experts[j].w3.device = device

    #### 先替换第 0-prefill_layers 层的forward函数
    up_th = load_thresholds(f'{threshold_path}/thresholds_0_8.pt', use_average=False)
    for i in range(0, prefill_layers):
        for j in range(len(llm.model.layers[0].block_sparse_moe.experts)):
            llm.model.layers[i].block_sparse_moe.experts[j].threshold = up_th[i][j].cuda(device_number)
            llm.model.layers[i].block_sparse_moe.experts[j].w2t = llm.model.layers[i].block_sparse_moe.experts[j].w2.weight.T.contiguous().cuda(device_number)
            llm.model.layers[i].block_sparse_moe.experts[j].w1.cuda(device_number)
            llm.model.layers[i].block_sparse_moe.experts[j].w3.cuda(device_number)
            # llm.model.layers[i].block_sparse_moe.experts[j].forward = llm.model.layers[i].block_sparse_moe.experts[j].kernel_forward
            llm.model.layers[i].block_sparse_moe.experts[j].forward = llm.model.layers[i].block_sparse_moe.experts[j].old_forward

    for i in range(prefill_layers, 32):
        for j in range(len(llm.model.layers[0].block_sparse_moe.experts)):
            llm.model.layers[i].block_sparse_moe.experts[j].forward = llm.model.layers[i].block_sparse_moe.experts[j].old_forward

    llm.model.norm.cuda(device_number)
    llm.lm_head.cuda(device_number)
    
    # 创建两个共享的CachedMLP实例
    buffer0 = CachedMLP(
        input_dim=llm.config.hidden_size,
        hidden_dim=llm.config.intermediate_size,
        dtype=dtype,
        sparsity=sparsity,
        device=device,
        device_map = device_map
    )
    buffer1 = CachedMLP(
        input_dim=llm.config.hidden_size,
        hidden_dim=llm.config.intermediate_size,
        dtype=dtype,
        sparsity=sparsity,
        device=device,
        device_map = device_map
    )
    cached_mlps = [buffer0, buffer1]
    
    for i, layer in enumerate(llm.model.layers):
        if i < prefill_layers :
            continue
        print(f"... loading layer {i} for pipelineLLM")
        # 将专家的forward方法替换为PipelineLLM管理的方式
        for j, expert in enumerate(layer.block_sparse_moe.experts):
            cur_device = device_map[i]
            expert.cpu_mlp = {
                "w1": expert.w1.to(cur_device).weight,
                "w2": expert.w2.to(cur_device).weight.T.contiguous(),
                "w3": expert.w3,
            }
    return llm, cached_mlps