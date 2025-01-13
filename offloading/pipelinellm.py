from typing import Tuple, Optional
import torch
import torch.nn as nn
import threading
import json
import torch.nn.functional as F
from queue import Queue

class CachedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dtype, sparsity: float = 0.2):
        super(CachedMLP, self).__init__()
        self.sparsity = sparsity
        self.activenum = int((1 - sparsity) * hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        self.activation = nn.SiLU()

        # GPU 缓存张量
        self.w1_gpu = torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device='cuda')
        self.w2_gpu = torch.empty((self.input_dim, self.activenum), dtype=self.dtype, device='cuda')
        self.w3_gpu = torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device='cuda')

        # 第二个专家的 GPU 缓存张量
        self.w1_gpu_expert1 = torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device='cuda')
        self.w2_gpu_expert1 = torch.empty((self.input_dim, self.activenum), dtype=self.dtype, device='cuda')
        self.w3_gpu_expert1 = torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device='cuda')

        # Pinned Memory 缓冲区
        self.register_buffer('sparse_w1_cpu', torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device='cpu'))
        self.register_buffer('sparse_w2_cpu', torch.empty((self.input_dim, self.activenum), dtype=self.dtype, device='cpu'))
        self.register_buffer('sparse_w3_cpu', torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device='cpu'))
        self.sparse_w1_cpu = self.sparse_w1_cpu.pin_memory()
        self.sparse_w2_cpu = self.sparse_w2_cpu.pin_memory()
        self.sparse_w3_cpu = self.sparse_w3_cpu.pin_memory()

        # 第二个专家的 Pinned Memory 缓冲区
        self.register_buffer('sparse_w1_cpu_expert1', torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device='cpu'))
        self.register_buffer('sparse_w2_cpu_expert1', torch.empty((self.input_dim, self.activenum), dtype=self.dtype, device='cpu'))
        self.register_buffer('sparse_w3_cpu_expert1', torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device='cpu'))
        self.sparse_w1_cpu_expert1 = self.sparse_w1_cpu_expert1.pin_memory()
        self.sparse_w2_cpu_expert1 = self.sparse_w2_cpu_expert1.pin_memory()
        self.sparse_w3_cpu_expert1 = self.sparse_w3_cpu_expert1.pin_memory()

        self.expert0_weight = torch.tensor(0)
        self.expert1_weight = torch.tensor(0)

        # 统计信息
        self.load_from_cpu_time = 0.0
        self.load_from_cpu_calls = 0

    def load_expert_weights(self, expert_weights):
        self.expert0_weight = expert_weights[0]
        self.expert1_weight = expert_weights[1]

    def forward(self, hidden_states):
        """
        根据hidden_states， 分别计算两个专家的输出
        """
        # 第一个专家的计算
        w3_output = torch.matmul(hidden_states, self.w3_gpu.T)
        w1_output = self.activation(torch.matmul(hidden_states, self.w1_gpu.T))
        w2 = self.w2_gpu.T
        hidden_states_expert0 = torch.matmul(w1_output * w3_output, w2)

        # 第二个专家的计算
        w3_output_expert1 = torch.matmul(hidden_states, self.w3_gpu_expert1.T)
        w1_output_expert1 = self.activation(torch.matmul(hidden_states, self.w1_gpu_expert1.T))
        w2_expert1 = self.w2_gpu_expert1.T
        hidden_states_expert1 = torch.matmul(w1_output_expert1 * w3_output_expert1, w2_expert1)

        final_hidden_states = hidden_states_expert0* self.expert0_weight + hidden_states_expert1* self.expert1_weight
        
        return final_hidden_states
                        

    def load_from_cpu(self, cpu_mlp, cpu_mlp_expert1, stream: torch.cuda.Stream):
        """
        从CPU加载参数，并使用指定的CUDA流进行异步复制到GPU。
        
        参数:
            cpu_mlp: 包含CPU上参数的字典（第一个专家）。
            cpu_mlp_expert1: 包含CPU上参数的字典（第二个专家）。
            stream: 用于数据传输的CUDA流。
        """
        # 从CPU加载参数（第一个专家）
        self.sparse_w1_cpu.copy_(cpu_mlp['w1'].data[:self.activenum, :])
        self.sparse_w2_cpu.copy_(cpu_mlp['w2'].data[:, :self.activenum])
        self.sparse_w3_cpu.copy_(cpu_mlp['w3'].data[:self.activenum, :])

        # 从CPU加载参数（第二个专家）
        self.sparse_w1_cpu_expert1.copy_(cpu_mlp_expert1['w1'].data[:self.activenum, :])
        self.sparse_w2_cpu_expert1.copy_(cpu_mlp_expert1['w2'].data[:, :self.activenum])
        self.sparse_w3_cpu_expert1.copy_(cpu_mlp_expert1['w3'].data[:self.activenum, :])

        # 异步复制到GPU
        with torch.cuda.stream(stream):
            self.w1_gpu.copy_(self.sparse_w1_cpu, non_blocking=True)
            self.w2_gpu.copy_(self.sparse_w2_cpu, non_blocking=True)
            self.w3_gpu.copy_(self.sparse_w3_cpu, non_blocking=True)

            # 第二个专家的异步复制
            self.w1_gpu_expert1.copy_(self.sparse_w1_cpu_expert1, non_blocking=True)
            self.w2_gpu_expert1.copy_(self.sparse_w2_cpu_expert1, non_blocking=True)
            self.w3_gpu_expert1.copy_(self.sparse_w3_cpu_expert1, non_blocking=True)

def convert_mixtral_to_cached_mlp(llm, dtype, sparsity=0.9):
    ### 其他部分存放在GPU上
    llm.model.embed_tokens.cuda()
    for i in range(len(llm.model.layers)):
        llm.model.layers[i].self_attn.cuda()
        llm.model.layers[i].input_layernorm.cuda()
        llm.model.layers[i].post_attention_layernorm.cuda()
        llm.model.layers[i].block_sparse_moe.gate.cuda()
    ### 第0层的专家存放在GPU上
    for j in range(len(llm.model.layers[0].block_sparse_moe.experts)):
        llm.model.layers[0].block_sparse_moe.experts[j].cuda()

    llm.model.norm.cuda()
    llm.lm_head.cuda()
    
    # 创建两个共享的CachedMLP实例
    buffer0 = CachedMLP(
        input_dim=llm.config.hidden_size,
        hidden_dim=llm.config.intermediate_size,
        dtype=dtype,
        sparsity=sparsity
    )
    buffer1 = CachedMLP(
        input_dim=llm.config.hidden_size,
        hidden_dim=llm.config.intermediate_size,
        dtype=dtype,
        sparsity=sparsity
    )
    cached_mlps = [buffer0, buffer1]
    
    for i, layer in enumerate(llm.model.layers):
        if i==0:
            continue
        # 将专家的forward方法替换为PipelineLLM管理的方式
        for j, expert in enumerate(layer.block_sparse_moe.experts):
            expert.cpu_mlp = {
                "w1": expert.w1.cpu().weight,
                "w2": expert.w2.cpu().weight,
                "w3": expert.w3.cpu().weight,
            }
    return llm, cached_mlps

class PipelineLLM:
    def __init__(self, llm, cached_mlps):
        """
        初始化 PipelineLLM，替换模型每一层的 forward 方法。
        
        参数:
            llm: 原始的大模型
            cached_mlps: 两个 CachedMLP 实例列表
        """
        self.llm = llm
        self.cached_mlps = cached_mlps  # [buffer0, buffer1]
        self.num_layers = len(llm.model.layers)
        self.lock = threading.Lock()
        self.use_buffer0 = True  # 标记当前使用哪个缓冲区

        # 创建两个共享的CUDA流
        self.stream0 = torch.cuda.Stream()
        self.stream1 = torch.cuda.Stream()

        # 初始化加载第一个和第二个层的参数
        self._load_layer(1, buffer_index=0,expert_ids=torch.tensor([0,1]))
        self._load_layer(1, buffer_index=1,expert_ids=torch.tensor([0,1]))
        self.top_k = 2
        self.activation = nn.SiLU()

        self._replace_forward_methods()
    
    def _load_layer(self, layer_idx, buffer_index, 
                    expert_ids, expert_weights=torch.tensor([0,0])):
        """
        加载指定层的参数到指定的缓冲区。
        
        参数:
            layer_idx: 层的索引
            buffer_index: 缓冲区的索引（0 或 1）
        """
        layer = self.llm.model.layers[layer_idx]
        expert0 = layer.block_sparse_moe.experts[expert_ids[0]]
        expert1 = layer.block_sparse_moe.experts[expert_ids[1]]
        if layer_idx == 1:
            print(expert_ids[0].data, expert_ids[1].data, '{:.3f}, {:.3f}'.format(expert_weights[0], expert_weights[1]))

        cpu_mlp = expert0.cpu_mlp
        cpu_mlp_expert1 = expert1.cpu_mlp
        buffer = self.cached_mlps[buffer_index]
        stream = self.stream0 if buffer_index == 0 else self.stream1

        buffer.load_expert_weights(expert_weights)
        # 异步加载参数
        buffer.load_from_cpu(cpu_mlp, cpu_mlp_expert1, stream)
    
    def _replace_forward_methods(self):
        """
        替换模型每一层的 forward 方法，添加参数预加载逻辑和注意力计算。
        """
        for i, layer in enumerate(self.llm.model.layers):
            def new_prefill_forward(hidden_states: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None,
                        position_ids: Optional[torch.LongTensor] = None,
                        past_key_value: Optional[Tuple[torch.Tensor]] = None,
                        output_attentions: Optional[bool] = False,
                        output_router_logits: Optional[bool] = False,
                        use_cache: Optional[bool] = False,
                        cache_position: Optional[torch.LongTensor] = None):
                """
                Prefill阶段的forward方法，处理seq_length > 1的情况
                """
                # Self Attention
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
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
                
                # 对于prefill阶段，仅将experts加载到GPU计算
                batch_size, sequence_length, hidden_dim = hidden_states.shape
                hidden_states = hidden_states.view(-1, hidden_dim)
                
                # 获取当前层的experts
                experts = layer.block_sparse_moe.experts
                
                # 将experts移动到GPU
                for expert in experts:
                    expert.to('cuda')
                
                # 在GPU上进行MoE计算（gate保持在CPU）
                final_hidden_states, router_logits = layer.block_sparse_moe(hidden_states)
                
                # 将计算结果reshape回原始形状
                final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
                
                # 计算完成后将experts移回CPU
                for expert in experts:
                    expert.to('cpu')
                
                hidden_states = residual + final_hidden_states

                outputs = (hidden_states,)
                if output_attentions:
                    outputs += (self_attn_weights,)
                if use_cache:
                    outputs += (present_key_value,)
                if output_router_logits:
                    outputs += (router_logits,)
                    
                return outputs
            def new_forward(hidden_states: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None,
                            position_ids: Optional[torch.LongTensor] = None,
                            past_key_value: Optional[Tuple[torch.Tensor]] = None,
                            output_attentions: Optional[bool] = False,
                            output_router_logits: Optional[bool] = False,
                            use_cache: Optional[bool] = False,
                            cache_position: Optional[torch.LongTensor] = None,
                            layer_idx=i):
                with self.lock:
                    # 选择当前使用的缓冲区
                    current_buffer = self.cached_mlps[0] if self.use_buffer0 else self.cached_mlps[1]
                    # current_stream = self.stream0 if self.use_buffer0 else self.stream1

                    # 切换缓冲区用于下一次
                    next_buffer_index = 1 if self.use_buffer0 else 0
                    # next_buffer = self.cached_mlps[next_buffer_index]
                    # next_stream = self.stream1 if self.use_buffer0 else self.stream0

                    next_layer_idx = layer_idx + 1

                    if next_layer_idx < self.num_layers:
                        # 预加载下一层的参数
                        next_layer = self.llm.model.layers[next_layer_idx]
                        router = next_layer.block_sparse_moe.gate

                        batch_size, sequence_length, hidden_dim = hidden_states.shape
                        hidden_states = hidden_states.view(-1, hidden_dim)
                        # router_logits: (batch * sequence_length, n_experts)
                        router_logits = router(hidden_states)
                        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

                        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
                        
                        self._load_layer(next_layer_idx, buffer_index=next_buffer_index, expert_ids =selected_experts[0], expert_weights=routing_weights[0])
                    
                    # 切换缓冲区
                    self.use_buffer0 = not self.use_buffer0

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
                    batch_size, sequence_length, hidden_dim = hidden_states.shape
                    hidden_states = hidden_states.view(-1, hidden_dim)
                    if layer_idx > 0:
                        ### 使用当前缓冲区进行 MLP 计算 ###
                        final_hidden_states = current_buffer(hidden_states)
                    else:
                        ### 根据router计算需要使用的专家 ###
                        cur_layer = self.llm.model.layers[layer_idx]
                        router = cur_layer.block_sparse_moe.gate
                        # router_logits: (batch * sequence_length, n_experts)
                        router_logits = router(hidden_states)

                        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                        # we cast back to the input dtype
                        routing_weights = routing_weights.to(hidden_states.dtype)

                        first_expert, second_expert = selected_experts[0][0], selected_experts[0][1]

                        final_hidden_states_expert0 = cur_layer.block_sparse_moe.experts[first_expert](hidden_states) * routing_weights[0][0]

                        final_hidden_states_expert1 = cur_layer.block_sparse_moe.experts[second_expert](hidden_states) * routing_weights[0][1]

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

            # 根据seq_length选择使用prefill还是decode forward
            def combined_forward(hidden_states: torch.Tensor, *args, **kwargs):
                batch_size, seq_len, _ = hidden_states.shape
                if seq_len > 1:  # prefill阶段
                    return new_prefill_forward(hidden_states, *args, **kwargs)
                else:  # decode阶段
                    return new_forward(hidden_states, *args, **kwargs)
                
            # 替换forward方法
            layer.forward = combined_forward

# # 将模型转换为使用CachedMLP的版本
# llm, cached_mlps = convert_mixtral_to_cached_mlp(llm, dtype, sparsity=0.8)

# # 创建流水线模型
# pipeline_llm = PipelineLLM(llm, cached_mlps).llm