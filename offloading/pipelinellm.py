from typing import Tuple, Optional
import torch
import torch.nn as nn
import threading
import json
import torch.nn.functional as F

class CachedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dtype, sparsity: float = 0.2):
        super(CachedMLP, self).__init__()
        self.sparsity = sparsity
        self.activenum = int((1 - sparsity) * hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        print("active neural num ", self.activenum)

        self.activation = nn.SiLU()

        #### 中间变量
        self.w3_result1 = None
        self.w3_result2 = None

        # 将GPU缓存张量改为列表存储
        self.w_gpu = [
            torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device='cuda:0') for _ in range(4)
        ]  # [w1_gpu, w2_gpu, w1_gpu_expert1, w2_gpu_expert1]

        # 将Pinned Memory缓冲区改为列表存储
        self.sparse_w_cpu = [
            torch.empty((self.activenum, self.input_dim), dtype=self.dtype, device='cpu').pin_memory() for _ in range(4)
        ]  # [sparse_w1_cpu, sparse_w2_cpu, sparse_w1_cpu_expert1, sparse_w2_cpu_expert1]

        self.expert0_weight = torch.tensor(0)
        self.expert1_weight = torch.tensor(0)

    def load_from_cpu(self, cpu_mlp, cpu_mlp_expert1, stream: torch.cuda.Stream, hidden_states):
        """
        从CPU加载参数，并使用指定的CUDA流进行异步复制到GPU。
        
        参数:
            cpu_mlp: 包含CPU上参数的字典（第一个专家）
            cpu_mlp_expert1: 包含CPU上参数的字典（第二个专家）。
            stream: 用于数据传输的CUDA流。
        """
        ### 根据up计算的结果进行稀疏化
        up_result1 = cpu_mlp['w3'](hidden_states)
        up_result2 = cpu_mlp_expert1['w3'](hidden_states)
        # 提取 up_result1 的值并计算 top-k 索引
        _, indices1 = torch.topk(up_result1, self.activenum, dim=1)  # 在第二个维度上取 top-k
        # 对 w1 进行索引操作
        self.w3_result1 = up_result1[: , indices1[0]]
        indices1 = indices1[0].cpu()

        _, indices2 = torch.topk(up_result2, self.activenum, dim=1)  # 在第二个维度上取 top-k
        self.w3_result2 = up_result2[: , indices2[0]]
        indices2 = indices2[0].cpu()  # 去除多余的维度，得到形状为 [k] 的索引张量

        # 使用列表索引更新CPU数据
        self.sparse_w_cpu[0].copy_(cpu_mlp['w1'].data[indices1, :])
        self.sparse_w_cpu[1].copy_(cpu_mlp['w2'].data[indices1, :])
        self.sparse_w_cpu[2].copy_(cpu_mlp_expert1['w1'].data[indices2, :])
        self.sparse_w_cpu[3].copy_(cpu_mlp_expert1['w2'].data[indices2, :])
        
        # 异步复制到GPU
        with torch.cuda.stream(stream):
            self.w_gpu[0].copy_(self.sparse_w_cpu[0], non_blocking=True)
            self.w_gpu[1].copy_(self.sparse_w_cpu[1], non_blocking=True)
            self.w_gpu[2].copy_(self.sparse_w_cpu[2], non_blocking=True)
            self.w_gpu[3].copy_(self.sparse_w_cpu[3], non_blocking=True)

    def load_expert_weights(self, expert_weights):
        self.expert0_weight = expert_weights[0]
        self.expert1_weight = expert_weights[1]

    def forward(self, hidden_states):
        """
        根据hidden_states， 分别计算两个专家的输出
        """
        w3_output = self.w3_result1
        w1_output = self.activation(torch.matmul(hidden_states, self.w_gpu[0].T))
        hidden_states_expert0 = torch.matmul(w1_output * w3_output, self.w_gpu[1])

        # 第二个专家的计算
        w3_output_expert1 = self.w3_result2
        w1_output_expert1 = self.activation(torch.matmul(hidden_states, self.w_gpu[2].T))
        hidden_states_expert1 = torch.matmul(w1_output_expert1 * w3_output_expert1, self.w_gpu[3])

        final_hidden_states = hidden_states_expert0 * self.expert0_weight + hidden_states_expert1 * self.expert1_weight
        return final_hidden_states
                        
def convert_mixtral_to_cached_mlp(llm, dtype, sparsity=0.9):
    ### 其他部分存放在GPU上
    llm.model.embed_tokens.cuda(0)
    for i in range(len(llm.model.layers)):
        llm.model.layers[i].self_attn.cuda(0)
        llm.model.layers[i].input_layernorm.cuda(0)
        llm.model.layers[i].post_attention_layernorm.cuda(0)
        llm.model.layers[i].block_sparse_moe.gate.cuda(0)
        for j in range(len(llm.model.layers[0].block_sparse_moe.experts)):
            llm.model.layers[i].block_sparse_moe.experts[j].w3.cuda(0)
    ### 第0层的专家存放在GPU上
    for j in range(len(llm.model.layers[0].block_sparse_moe.experts)):
        llm.model.layers[0].block_sparse_moe.experts[j].cuda(0)

    llm.model.norm.cuda(0)
    llm.lm_head.cuda(0)
    
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
                "w2": expert.w2.cpu().weight.T.contiguous(),
                "w3": expert.w3,
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

        self.top_k = 2
        self.activation = nn.SiLU()

        self._replace_forward_methods()

        # 用于统计时间的变量
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0

    def _load_layer(self, layer_idx, buffer_index, expert_ids, expert_weights,
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
        buffer = self.cached_mlps[buffer_index]
        stream = self.stream0 if buffer_index == 0 else self.stream1

        buffer.load_expert_weights(expert_weights)
        # 异步加载参数
        buffer.load_from_cpu(cpu_mlp, cpu_mlp_expert1, stream, hidden_states)

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
                with self.lock:
                    batch_size, sequence_length, hidden_dim = hidden_states.shape
                    
                    if sequence_length == 1:
                        #### decode phase ####
                        # 选择当前使用的缓冲区
                        current_buffer = self.cached_mlps[0] if self.use_buffer0 else self.cached_mlps[1]

                        next_buffer_index = 1 if self.use_buffer0 else 0

                        next_layer_idx = layer_idx + 1

                        if next_layer_idx < self.num_layers:
                            # 预加载下一层的参数
                            next_layer = self.llm.model.layers[next_layer_idx]
                            router = next_layer.block_sparse_moe.gate

                            # batch_size, sequence_length, hidden_dim = hidden_states.shape
                            hidden_states_flat = hidden_states.view(-1, hidden_dim)
                            # router_logits: (batch * sequence_length, n_experts)
                            router_logits = router(hidden_states_flat)

                            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

                            self._load_layer(
                                next_layer_idx,
                                buffer_index=next_buffer_index,
                                expert_ids=selected_experts[0],
                                expert_weights=routing_weights[0],
                                hidden_states=hidden_states_flat,
                            )

                            hidden_states = hidden_states_flat.reshape(batch_size, sequence_length, hidden_dim)

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
                    
                    if sequence_length > 1:
                        print("in prefill layer ", layer_idx)
                        # 对于prefill阶段，仅将experts加载到GPU计算
                        experts = layer.block_sparse_moe.experts

                        # 将experts移动到GPU
                        if layer_idx != 0:
                            for expert in experts:
                                expert.cuda(0)

                        # 在GPU上进行MoE计算（gate保持在CPU）
                        final_hidden_states, router_logits = layer.block_sparse_moe(hidden_states)

                        # 计算完成后将experts移回CPU
                        if layer_idx != 0:
                            for expert in experts:
                                expert.w1.to('cpu')
                                expert.w2.to('cpu')
                    else:
                        # batch_size, sequence_length, hidden_dim = hidden_states.shape
                        hidden_states_flat = hidden_states.view(-1, hidden_dim)
                        # print("in decode layer", layer_idx)
                        if layer_idx > 0:
                            ### 使用当前缓冲区进行 MLP 计算 ###
                            final_hidden_states = current_buffer(hidden_states_flat)
                        else:
                            ### 根据router计算需要使用的专家 ###
                            cur_layer = layer
                            router = cur_layer.block_sparse_moe.gate
                            # router_logits: (batch * sequence_length, n_experts)
                            router_logits = router(hidden_states_flat)

                            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                            # we cast back to the input dtype
                            routing_weights = routing_weights.to(hidden_states_flat.dtype)

                            first_expert, second_expert = selected_experts[0][0], selected_experts[0][1]

                            final_hidden_states_expert0 = cur_layer.block_sparse_moe.experts[first_expert](
                                hidden_states_flat) * routing_weights[0][0]

                            final_hidden_states_expert1 = cur_layer.block_sparse_moe.experts[second_expert](
                                hidden_states_flat) * routing_weights[0][1]

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