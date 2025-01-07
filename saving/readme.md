### 实验思路
按照平均数去统计silu(gate) 然后去乘上up，根据提前统计的阈值去进行稀疏

### 代码结构

* activation.py： 保存llama的激活值
* activation_mixtral.py： 保存mixtral的激活值
* modeling_llama_up.py/: 加载MLP稀疏化的llama模型
* modeling_mixtral_up.py/: 加载MLP稀疏化的mixtral模型
* mystatistics.py.py: 统计稀疏阈值的代码


(average.py, threshold.py是另一种统计方式，但是并不精准，准备替换/删除)