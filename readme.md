### 相关工作
1. Fast Inference of Mixture-of-Experts Language Models with Offloading
2. EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models
3. CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models
4. MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs  (offline & batch workloads)
5. MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

### 实验思路
两种预测方式进行结合（取并集）：
- 保留prefill阶段的一部分固定神经元的位置
- gatemulup预测器（根据i-1层）预测出来的位置【gamma】 : 4096->1024->14336

### 代码结构

* convert_llama.py： 用于MLP层（专家）的稀疏预测器结构
* saving/: 定制化生成 输入x和激活值 的【数据集】