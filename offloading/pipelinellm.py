from typing import Tuple, Optional
import torch
import torch.nn as nn
import threading
import json
import torch.nn.functional as F

class Expert_Predictor(nn.Module):
    def __init__(self, layer_idx: int = 0, input_dim: int = 4096, hidden_dim: int = 512, output_dim: int = 8, dtype = torch.float16,
                 training_epoch: int = 10):
        super(Expert_Predictor, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, dtype=dtype)
        self.activation = nn.SiLU() 
        self.linear2 = nn.Linear(hidden_dim,output_dim, dtype=dtype) 

        self.load_state_dict(torch.load(f'../expert_predictor/training/{layer_idx}-{training_epoch}.pth'))

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.linear2(hidden_states)

#### 增加专家preload失败后的重新加载
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
        print("active neural num ",self.activenum)

        self.activation = nn.SiLU()

        self.w3_expert0 = None
        self.w3_expert1 = None

        self.indices0 = None
        self.indices1 = None

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
        # 只更新需要加载的专家权重
        _, indices = torch.topk(torch.abs(cpu_mlp_new['w3'](hidden_states)), self.activenum, dim=1)
        indices = indices[0].to(self.device_map[layer_idx])
        if replace_idx == 0:
            # 更新CPU数据
            self.w3_expert0 = cpu_mlp_new['w3']
            self.indices0 = indices
            if self.use_pin:
                self.sparse_w_cpu[0].copy_(cpu_mlp_new['w1'].data[indices, :])
                self.sparse_w_cpu[1].copy_(cpu_mlp_new['w2'].data[indices, :])
                # 异步复制到GPU
                with torch.cuda.stream(stream):
                    self.w_gpu[0].copy_(self.sparse_w_cpu[0], non_blocking=True)
                    self.w_gpu[1].copy_(self.sparse_w_cpu[1], non_blocking=True)
            else:
                # 异步复制到GPU
                with torch.cuda.stream(stream):
                    self.w_gpu[0].copy_(cpu_mlp_new['w1'].data[indices, :], non_blocking=True)
                    self.w_gpu[1].copy_(cpu_mlp_new['w2'].data[indices, :], non_blocking=True)   
        else:
            # 更新CPU数据
            self.w3_expert1 = cpu_mlp_new['w3']
            self.indices1 = indices
            if self.use_pin:
                self.sparse_w_cpu[2].copy_(cpu_mlp_new['w1'].data[indices, :])
                self.sparse_w_cpu[3].copy_(cpu_mlp_new['w2'].data[indices, :])
                
                # 异步复制到GPU
                with torch.cuda.stream(stream):
                    self.w_gpu[2].copy_(self.sparse_w_cpu[2], non_blocking=True)
                    self.w_gpu[3].copy_(self.sparse_w_cpu[3], non_blocking=True)
            else:
                # 异步复制到GPU
                with torch.cuda.stream(stream):
                    self.w_gpu[2].copy_(cpu_mlp_new['w1'].data[indices, :], non_blocking=True)
                    self.w_gpu[3].copy_(cpu_mlp_new['w2'].data[indices, :], non_blocking=True)

    def load_from_cpu(self, cpu_mlp, cpu_mlp_expert1, stream: torch.cuda.Stream, hidden_states,
            layer_idx=0):
        """
        从CPU加载参数，并使用指定的CUDA流进行异步复制到GPU。
        
        参数:
            cpu_mlp: 包含CPU上参数的字典（第一个专家）
            cpu_mlp_expert1: 包含CPU上参数的字典（第二个专家）。
            stream: 用于数据传输的CUDA流。
        """
        ### 根据up计算的结果进行稀疏化
        self.w3_expert0 = cpu_mlp['w3']
        self.w3_expert1 = cpu_mlp_expert1['w3']
        up_result0 = self.w3_expert0(hidden_states)
        up_result1 = self.w3_expert1(hidden_states)
        # 提取 up_result0 的值并计算 top-k 索引
        _, indices0 = torch.topk(torch.abs(up_result0), self.activenum, dim=1)  # 在第二个维度上取 top-k
        _, indices1 = torch.topk(torch.abs(up_result1), self.activenum, dim=1)  # 在第二个维度上取 top-k
    
        self.indices0 = indices0
        self.indices1 = indices1
        cur_device = self.device_map[layer_idx]
        indices0 = indices0[0].to(cur_device)
        indices1 = indices1[0].to(cur_device) 

        if self.use_pin:
            # 使用列表索引更新CPU数据
            self.sparse_w_cpu[0].copy_(cpu_mlp['w1'].data[indices0, :])
            self.sparse_w_cpu[1].copy_(cpu_mlp['w2'].data[indices0, :])

            # 异步复制到GPU
            with torch.cuda.stream(stream):
                self.w_gpu[0].copy_(self.sparse_w_cpu[0], non_blocking=True)
                self.w_gpu[1].copy_(self.sparse_w_cpu[1], non_blocking=True)

            self.sparse_w_cpu[2].copy_(cpu_mlp_expert1['w1'].data[indices1, :])
            self.sparse_w_cpu[3].copy_(cpu_mlp_expert1['w2'].data[indices1, :])
            
            # 异步复制到GPU
            with torch.cuda.stream(stream):
                self.w_gpu[2].copy_(self.sparse_w_cpu[2], non_blocking=True)
                self.w_gpu[3].copy_(self.sparse_w_cpu[3], non_blocking=True)
        else:
            # 异步复制到GPU
            with torch.cuda.stream(stream):
                w01 = cpu_mlp['w1'].data[indices0, :]
                w02 = cpu_mlp['w2'].data[indices0, :]
                w11 = cpu_mlp_expert1['w1'].data[indices1, :]
                w12 = cpu_mlp_expert1['w2'].data[indices1, :]
                self.w_gpu[0].copy_(w01, non_blocking=True)
                self.w_gpu[1].copy_(w02, non_blocking=True)
                self.w_gpu[2].copy_(w11, non_blocking=True)
                self.w_gpu[3].copy_(w12, non_blocking=True)

    def load_expert_weights(self, expert_ids):
        # print('loading next expert: ', expert_ids)   ## tensor([7, 6], device='cuda:0')
        self.expert_ids.copy_(expert_ids)

    def forward(self, hidden_states, expert_weights, expert_ids):
        """
        根据hidden_states， 分别计算两个专家的输出
        """
        # print(expert_weights, expert_ids)
        # print(self.expert_ids)
        ### 如果取出来的专家 第一个对不上，可能是预测的 1，2号专家顺序颠倒了，替换
        if self.expert_ids[0] != expert_ids[0]:
            # print("----replace expert weights")
            expert_weights[0], expert_weights[1] = expert_weights[1], expert_weights[0]

        w3_output = self.w3_expert0(hidden_states)[:, self.indices0.to(self.device)]
        w1_output = self.activation(torch.matmul(hidden_states, self.w_gpu[0].T))
        hidden_states_expert0 = torch.matmul(w1_output * w3_output, self.w_gpu[1])

        # 第二个专家的计算
        w3_output_expert1 = self.w3_expert1(hidden_states)[:, self.indices0.to(self.device)]
        w1_output_expert1 = self.activation(torch.matmul(hidden_states, self.w_gpu[2].T))
        hidden_states_expert1 = torch.matmul(w1_output_expert1 * w3_output_expert1, self.w_gpu[3])

        final_hidden_states = hidden_states_expert0 * expert_weights[0] + hidden_states_expert1 * expert_weights[1]
        return final_hidden_states
                    

class PipelineLLM:
    def __init__(self, llm, cached_mlps, start_ep_idx = 1, end_ep_idx = 3, 
                 device = 'cuda:0', device_map = None,
                 training_epoch: int = 10, print_layer_info = False):
        """
        初始化 PipelineLLM，替换模型每一层的 forward 方法。
        
        参数:
            llm: 原始的大模型
            cached_mlps: 两个 CachedMLP 实例列表
            start_ep_idx, end_ep_idx: [sidx, eidx] 需要使用训练的router范围，闭区间
            training_epoch: router的训练轮次
        """
        self.llm = llm
        self.device = device
        self.device_map = device_map
        self.cached_mlps = cached_mlps  # [buffer0, buffer1]
        self.num_layers = len(llm.model.layers)

        self.use_buffer0 = True  # 标记当前使用哪个缓冲区
        self.print_layer_info = print_layer_info

        # self.stream0 = torch.cuda.Stream(device=device)
        # self.stream1 = torch.cuda.Stream(device=device)
        self.stream0 = {'cuda:1':torch.cuda.Stream(device='cuda:1'),
                        'cuda:2':torch.cuda.Stream(device='cuda:2'),}
        self.stream1 = {'cuda:1':torch.cuda.Stream(device='cuda:1'),
                        'cuda:2':torch.cuda.Stream(device='cuda:2'),}

        self.stream_conflict = torch.cuda.Stream(device=device)

        self.top_k = 2
        self.activation = nn.SiLU()

        # 用于统计时间的变量
        self.total_reload_experts = 0

        self.prefill_time = 0

        #### 增加[1,3]的Expert_Predictor
        self.eps = []
        self.start_layer = start_ep_idx
        self.end_layer = end_ep_idx
        for i in range(self.start_layer, self.end_layer+1):
            ep = Expert_Predictor(layer_idx=i, training_epoch=training_epoch).to(self.device)
            self.eps.append(ep)

        self._replace_forward_methods()
    
    def get_reload_experts(self):
        tmp = self.total_reload_experts
        self.total_reload_experts = 0
        return tmp

    def get_prefill_time(self):
        tmp = self.prefill_time
        self.prefill_time = 0
        return tmp

    def _load_layer(self, layer_idx, buffer, expert_ids, stream,
                    hidden_states):
        """
        加载指定层的参数到指定的缓冲区。
        
        参数:
            layer_idx: 层的索引
            buffer_index: 缓冲区的索引（0 或 1）
        """
        layer = self.llm.model.layers[layer_idx]
        expert0 = layer.block_sparse_moe.experts[expert_ids[0]]
        expert1 = layer.block_sparse_moe.experts[expert_ids[1]]

        cpu_mlp = expert0.cpu_mlp
        cpu_mlp_expert1 = expert1.cpu_mlp

        ### weights应该用正确的来算
        buffer.load_expert_weights(expert_ids)
        # 异步加载参数
        buffer.load_from_cpu(cpu_mlp, cpu_mlp_expert1, stream, hidden_states, layer_idx=layer_idx)

    def _load_conflict_layer(self, layer_idx, buffer, expert_ids, hidden_states,):
        """
        处理专家预测冲突的情况，尽可能复用已加载的专家权重
        
        参数:
            layer_idx: 层索引
            buffer_index: 缓冲区索引
            expert_ids: 实际需要的专家ID
            predict_experts: 预测的专家ID
            hidden_states: 输入hidden_states
        """
        predict_experts = buffer.get_predict_experts()
        # print(f"predict_experts in layer {layer_idx}: {predict_experts}")
        # 找出需要加载的新专家
        required_experts = set(expert_ids.tolist()) - set(predict_experts.tolist())
        
        # 如果只需要加载一个专家
        if len(required_experts) == 1:
            new_expert_id = list(required_experts)[0]
            # 找出需要被替换的专家位置
            if predict_experts[0] in expert_ids:
                replace_idx = 1   
                ### 更新buffer中保存的专家序号
                new_expert_ids = torch.tensor([predict_experts[0], new_expert_id], device=self.device)
            else:
                replace_idx = 0
                new_expert_ids = torch.tensor([new_expert_id, predict_experts[1]], device=self.device)

            # 更新专家ID
            buffer.load_expert_weights(new_expert_ids)

            print(f"reloading 1 experts, {new_expert_id}")
            self.total_reload_experts += 1
            
            # 获取需要加载的专家
            layer = self.llm.model.layers[layer_idx]
            new_expert = layer.block_sparse_moe.experts[new_expert_id]
            cpu_mlp_new = new_expert.cpu_mlp
            
            # 只更新需要加载的专家权重
            buffer.load_conflict_cpu(cpu_mlp_new, self.stream_conflict, hidden_states, replace_idx, layer_idx=layer_idx)
            
        else:
            print(f"reloading 2 experts, {expert_ids[:]}")
            self.total_reload_experts += 2
            # 需要加载两个专家，直接调用原始方法
            self._load_layer(layer_idx, buffer, expert_ids, self.stream_conflict, hidden_states)

    def _replace_forward_methods(self):
        """
        替换模型每一层的 forward 方法，添加参数预加载逻辑和注意力计算。
        """
        for i, layer in enumerate(self.llm.model.layers):
            def new_forward(hidden_states: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None,
                        position_ids: Optional[torch.LongTensor] = None,
                        past_key_value: Optional[Tuple[torch.Tensor]] = None,
                        output_attentions: Optional[bool] = False,
                        output_router_logits: Optional[bool] = False,
                        use_cache: Optional[bool] = False,
                        cache_position: Optional[torch.LongTensor] = None,
                        layer=layer,
                        layer_idx=i):
                batch_size, sequence_length, hidden_dim = hidden_states.shape

                # 处理当前层
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)

                # Self Attention
                hidden_states, self_attn_weights, present_key_value = layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)

                if sequence_length > 1:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    if self.print_layer_info:
                        print("in prefill layer ", layer_idx)
                    # 对于prefill阶段，仅将experts加载到GPU计算
                    experts = layer.block_sparse_moe.experts

                    # 将experts移动到GPU
                    if layer_idx != 0:
                        for expert in experts:
                            expert.to(self.device)
                    # 在GPU上进行MoE计算（gate保持在CPU）
                    final_hidden_states, router_logits = layer.block_sparse_moe(hidden_states)
                    # 计算完成后将experts移回CPU
                    if layer_idx != 0:
                        cur_device = self.device_map[layer_idx]
                        for expert in experts:
                            expert.w1.to(cur_device)
                            expert.w2.to(cur_device)
                    end_event.record()
                    torch.cuda.synchronize()

                    # 计算时间
                    self.prefill_time += start_event.elapsed_time(end_event) / 1000 
                else:
                    #### decode phase ####
                    ### 加载应该在atten算完之后[专家预测的才更准]
                    # 选择当前使用的缓冲区
                    current_buffer = self.cached_mlps[0] if self.use_buffer0 else self.cached_mlps[1]
                    cur_stream = self.stream0 if self.use_buffer0 else self.stream1
                    
                    next_buffer_index = 1 if self.use_buffer0 else 0
                    next_layer_idx = layer_idx + 1
                    if next_layer_idx < self.num_layers:
                        ### 使用下一个缓冲区进行加载
                        next_buffer = self.cached_mlps[next_buffer_index]
                        stream = self.stream1 if self.use_buffer0 else self.stream0

                        # 预加载下一层的参数
                        # batch_size, sequence_length, hidden_dim = hidden_states.shape
                        hidden_states = hidden_states.view(-1, hidden_dim)
                        # router_logits: (batch * sequence_length, n_experts)
                        if self.start_layer <= next_layer_idx <= self.end_layer:
                            ### 使用训练好的专家预测矩阵
                            router = self.eps[next_layer_idx - self.start_layer]
                        else:
                            ### 使用next_layer_idx对应的gate矩阵
                            router = self.llm.model.layers[next_layer_idx].block_sparse_moe.gate
                        router_logits = router(hidden_states)

                        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                        _, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                        # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

                        self._load_layer(
                            next_layer_idx,
                            buffer=next_buffer,
                            expert_ids=selected_experts[0],
                            stream=stream[self.device_map[next_layer_idx]],
                            hidden_states=hidden_states,
                        )
                        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)

                        # 切换缓冲区
                        self.use_buffer0 = not self.use_buffer0
                    # print("in decode layer", layer_idx)
                    hidden_states = hidden_states.view(-1, hidden_dim)
                    ### 根据router计算需要使用的专家 ###
                    router = layer.block_sparse_moe.gate
                    # router_logits: (batch * sequence_length, n_experts)
                    router_logits = router(hidden_states)

                    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                    # we cast back to the input dtype
                    routing_weights = routing_weights.to(hidden_states.dtype)
                    expert_ids = selected_experts[0]
                    expert_weights = routing_weights[0]
                    # print('the right experts is', expert_ids)
                    ## tensor([6, 7], device='cuda:0')
                    if layer_idx > 0:
                        ### 等待完全加载
                        cur_device = self.device_map[layer_idx]
                        cur_stream[cur_device].synchronize()  # 等待 cur_stream 中的所有操作完成
                        ### 判断加载是否正确
                        predict_experts = current_buffer.get_predict_experts()
                        
                        # 判断expert_ids和predict_experts是否包含相同数据（忽略顺序）
                        # if not torch.equal(torch.sort(expert_ids)[0], torch.sort(predict_experts)[0]):

                        if not torch.equal((1 << expert_ids[0]) + (1 << expert_ids[1] ),(1 << predict_experts[0]) + (1 << predict_experts[1])):
                            ### 不吻合，重新加载
                            self._load_conflict_layer(
                                layer_idx,
                                buffer=current_buffer,## 用cur_buffer_index
                                expert_ids=expert_ids,
                                hidden_states=hidden_states,
                            )
                            self.stream_conflict.synchronize()  # 等待当前层的参数传递                          
                        
                        ### 使用当前缓冲区进行 MLP 计算 ###
                        final_hidden_states = current_buffer(hidden_states, expert_weights, expert_ids)
                    else:
                        final_hidden_states_expert0 = layer.block_sparse_moe.experts[expert_ids[0]](
                            hidden_states) * expert_weights[0]

                        final_hidden_states_expert1 = layer.block_sparse_moe.experts[expert_ids[1]](
                            hidden_states) * expert_weights[1]

                        # 将两个专家的结果相加
                        final_hidden_states = final_hidden_states_expert0 + final_hidden_states_expert1

                    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

                hidden_states = residual + final_hidden_states

                outputs = (hidden_states,)

                if output_attentions:
                    outputs += (self_attn_weights,)

                if use_cache:
                    outputs += (present_key_value,)

                return outputs

            # 替换forward方法
            layer.forward = new_forward

def convert_mixtral_to_cached_mlp(llm, dtype, sparsity=0.9, backends='bitblas', 
    device='cuda:0', device_map=None):
    ### 其他部分存放在device上
    llm.model.embed_tokens.to(device)
    for i in range(len(llm.model.layers)):
        llm.model.layers[i].self_attn.to(device)
        llm.model.layers[i].input_layernorm.to(device)
        llm.model.layers[i].post_attention_layernorm.to(device)
        ### 原始的gate
        llm.model.layers[i].block_sparse_moe.gate.to(device)
        for j in range(len(llm.model.layers[0].block_sparse_moe.experts)):
            if backends == 'gemlite':
                llm.model.layers[i].block_sparse_moe.experts[j].w3.to(device)
            elif backends == "bitblas":
                llm.model.layers[i].block_sparse_moe.experts[j].w3.W_q = \
                    llm.model.layers[i].block_sparse_moe.experts[j].w3.W_q.to(device)
                llm.model.layers[i].block_sparse_moe.experts[j].w3.scale = \
                    llm.model.layers[i].block_sparse_moe.experts[j].w3.scale.to(device)
                llm.model.layers[i].block_sparse_moe.experts[j].w3.zero = \
                    llm.model.layers[i].block_sparse_moe.experts[j].w3.zero.to(device)
                llm.model.layers[i].block_sparse_moe.experts[j].w3.device = device
            else:
                llm.model.layers[i].block_sparse_moe.experts[j].w3.W_q.to(device)
                llm.model.layers[i].block_sparse_moe.experts[j].w3.meta['scale'] = \
                    llm.model.layers[i].block_sparse_moe.experts[j].w3.meta['scale'].to(device)
                llm.model.layers[i].block_sparse_moe.experts[j].w3.meta['zero'] = \
                    llm.model.layers[i].block_sparse_moe.experts[j].w3.meta['zero'].to(device)
                llm.model.layers[i].block_sparse_moe.experts[j].w3.device = device
                
    ### 第0层的专家存放在GPU上
    for j in range(len(llm.model.layers[0].block_sparse_moe.experts)):
        llm.model.layers[0].block_sparse_moe.experts[j].w3.device = device
        llm.model.layers[0].block_sparse_moe.experts[j].to(device)

    llm.model.norm.to(device)
    llm.lm_head.to(device)
    
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
        print(f"... loading layer {i}")
        # 将专家的forward方法替换为PipelineLLM管理的方式
        for j, expert in enumerate(layer.block_sparse_moe.experts):
            if i==0:
                #### 要保证w1,w2在device上
                llm.model.layers[i].block_sparse_moe.experts[j].w2.to(device)
                llm.model.layers[i].block_sparse_moe.experts[j].w1.to(device)
            else:
                cur_device = device_map[i]
                curw2 = expert.w2.weight.T.contiguous().to(cur_device)
                expert.cpu_mlp = {
                    "w1": expert.w1.to(cur_device).weight,
                    "w2": curw2,
                    "w3": expert.w3.to(device),
                }

        torch.cuda.empty_cache()
    return llm, cached_mlps