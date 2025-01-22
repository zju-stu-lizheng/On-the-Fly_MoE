from typing import Tuple, Optional
import torch
import torch.nn as nn
import threading
import time
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

          
class PipelineLLM:
    def __init__(self, llm, cached_mlps, start_ep_idx = 1, end_ep_idx = 3, 
                 device = 'cuda:0', device_map = None,
                 training_epoch: int = 10, prefill_layers = 1, print_layer_info = False):
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
        self.offload_startid = prefill_layers

        self.use_buffer0 = True  # 标记当前使用哪个缓冲区
        self.print_layer_info = print_layer_info

        self.transmission_compensate = 0.002    ### 传输延迟
        self.one_layer_compensate = 0.00875451 * 8 ### 完整的一层(8个专家)

        self.stream0 = torch.cuda.Stream(device=device)
        self.stream1 = torch.cuda.Stream(device=device)
        self.stream_conflict = torch.cuda.Stream(device=device)

        self.top_k = 2
        self.activation = nn.SiLU()

        # 用于统计时间的变量 和 重新加载专家的个数
        self.total_reload_experts = 0
        self.prefill_time = 0

        #### 用于模拟: 保证routing gate保存在对应的device上
        for i in range(self.offload_startid, 32):
            self.llm.model.layers[i].block_sparse_moe.gate.cuda(self.device)

        #### 增加[1,3]的Expert_Predictor
        self.eps = []
        self.start_layer = start_ep_idx
        self.end_layer = end_ep_idx
        for i in range(self.start_layer, self.end_layer+1):
            ep = Expert_Predictor(layer_idx=i, training_epoch=training_epoch).cuda(int(self.device[-1]))
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

            # print(f"reloading 1 experts, {new_expert_id}")
            self.total_reload_experts += 1
            
            # 获取需要加载的专家
            layer = self.llm.model.layers[layer_idx]
            new_expert = layer.block_sparse_moe.experts[new_expert_id]
            cpu_mlp_new = new_expert.cpu_mlp
            
            # 只更新需要加载的专家权重
            buffer.load_conflict_cpu(cpu_mlp_new, self.stream_conflict, hidden_states, replace_idx, layer_idx=layer_idx)
            time.sleep(self.transmission_compensate)
        else:
            # print(f"reloading 2 experts, {expert_ids[:]}")
            self.total_reload_experts += 2
            # 需要加载两个专家，直接调用原始方法
            self._load_layer(layer_idx, buffer, expert_ids, self.stream_conflict, hidden_states)
            time.sleep(self.transmission_compensate * 2)

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
        with torch.cuda.stream(self.stream0):
            x_device2 = hidden_states.to(device2, non_blocking=True)  # 异步传输

        # 第2步：在 device1 上计算 expert0 和 expert1 的 up(x)
        with torch.cuda.stream(self.stream0):
            up_x_expert0 = self.w3_expert0(hidden_states)  # expert0 的计算
            up_x_expert1 = self.w3_expert1(hidden_states)  # expert1 的计算

        # 第3步：在 device2 上计算 expert0 和 expert1 的 gate(x_device2)
        gate_x_expert0 = self.activation(torch.matmul(x_device2, self.w1_expert0.T))
        gate_x_expert1 = self.activation(torch.matmul(x_device2, self.w1_expert1.T))

        # 第4步：将 up_x_expert0 和 up_x_expert1 从 device1 传输到 device2（异步）
        with torch.cuda.stream(self.stream0):
            up_x_device2_expert0 = up_x_expert0.to(device2, non_blocking=True)  # expert0 的传输
            up_x_device2_expert1 = up_x_expert1.to(device2, non_blocking=True)  # expert1 的传输

        # 第5步：在 device2 上计算 expert0 和 expert1 的 down(gate_x * up_x_device2)
        down_x_device2_expert0 = torch.matmul(gate_x_expert0 * up_x_device2_expert0, self.w2_expert0)
        down_x_device2_expert1 = torch.matmul(gate_x_expert1 * up_x_device2_expert1, self.w2_expert1)

        # 第6步：将 down_x_expert0 和 down_x_expert1 从 device2 传输回 device1（异步）
        with torch.cuda.stream(self.stream0):
            down_x_device1_expert0 = down_x_device2_expert0.to(device1, non_blocking=True)  # expert0 的传输
            down_x_device1_expert1 = down_x_device2_expert1.to(device1, non_blocking=True)  # expert1 的传输

        # 同步 Stream，确保所有操作完成
        torch.cuda.synchronize(device1)
        torch.cuda.synchronize(device2)

        # 返回两个专家的计算结果
        return down_x_device1_expert0, down_x_device1_expert1

    def load_prefill_layer(self, layer_idx=0, expert_ids=None):
        """
        只复制指针，不进行实际的数据搬运
        
        参数:
            cpu_mlp: 包含CPU上参数的字典（第一个专家）
            cpu_mlp_expert1: 包含CPU上参数的字典（第二个专家）。
        """
        self.w1_expert0 = self.llm.model.layers[layer_idx].block_sparse_moe.experts[expert_ids[0]].w1.weight
        self.w2_expert0 = self.llm.model.layers[layer_idx].block_sparse_moe.experts[expert_ids[0]].w2.weight.T
        self.w3_expert0 = self.llm.model.layers[layer_idx].block_sparse_moe.experts[expert_ids[0]].w3
        
        self.w1_expert1 = self.llm.model.layers[layer_idx].block_sparse_moe.experts[expert_ids[1]].w1.weight
        self.w2_expert1 = self.llm.model.layers[layer_idx].block_sparse_moe.experts[expert_ids[1]].w2.weight.T
        self.w3_expert1 = self.llm.model.layers[layer_idx].block_sparse_moe.experts[expert_ids[1]].w3

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

                    if layer_idx < self.offload_startid:
                        final_hidden_states, router_logits = layer.block_sparse_moe(hidden_states)
                    else:
                        cur_device = self.device_map[layer_idx]

                        ### 模拟，这里把hidden_states传到experts在的地方
                        # hidden_states = hidden_states.to(cur_device)
                        time.sleep(self.one_layer_compensate)  

                        # 在GPU上进行MoE计算
                        hidden_states = hidden_states.view(-1, hidden_dim)
                        # router_logits: (batch * sequence_length, n_experts)
                        router = self.llm.model.layers[layer_idx].block_sparse_moe.gate.to(self.device)
                        router_logits = router(hidden_states)

                        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                        _, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                        # Initialize output tensor
                        final_hidden_states = torch.zeros_like(hidden_states)
                        # 计算完成后将experts移回CPU
                        for i in range(batch_size * sequence_length):
                            # Get current token's hidden state
                            current_state = hidden_states[i]
                            
                            # Get selected experts and their weights for this token
                            current_weights = routing_weights[i]
                            current_experts = selected_experts[i]
                            
                            self.load_prefill_layer(layer_idx, current_experts)
                            down_x_device1_expert0, down_x_device1_expert1 = self.parallel_computation(hidden_states=current_state.unsqueeze(0), layer_idx=layer_idx)

                            expert_output = down_x_device1_expert0.squeeze(0) * current_weights[0] + down_x_device1_expert1.squeeze(0) * current_weights[1]
                            
                            # Store result in final output tensor
                            final_hidden_states[i] = expert_output
                        
                        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

                        ### 把final_hidden_states传回
                        final_hidden_states = final_hidden_states.to(self.device)
                    
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
                    if self.offload_startid <= next_layer_idx < self.num_layers:
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

                        # [self.device_map[next_layer_idx]]
                        self._load_layer(
                            next_layer_idx,
                            buffer=next_buffer,
                            expert_ids=selected_experts[0],
                            stream=stream,
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
                    if layer_idx >= self.offload_startid:
                        ### 等待完全加载
                        cur_stream.synchronize()
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
                        final_hidden_states = current_buffer(hidden_states, expert_weights, 
                                                             expert_ids, layer_idx = layer_idx)
                    else:
                        final_hidden_states_expert0 = layer.block_sparse_moe.experts[expert_ids[0]](
                            hidden_states=hidden_states) * expert_weights[0]

                        final_hidden_states_expert1 = layer.block_sparse_moe.experts[expert_ids[1]](
                            hidden_states=hidden_states) * expert_weights[1]

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
