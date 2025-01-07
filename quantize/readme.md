### 实验思路
将up进行int2量化，同时使用eora进行低秩恢复。

在80%稀疏后，进行LoRA微调恢复。

### 代码结构

* quevaluate2.py： 用于测试模型在下游任务的性能
* modeling_mixtral.py/: 加载MLP稀疏化的模型
* recover.py: LoRA微调的代码
* threshold.py: 统计稀疏阈值的代码