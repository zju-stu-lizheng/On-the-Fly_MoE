#!/bin/bash
sparsity_levels=(0.5 0.6 0.7 0.8 0.9)  # 定义多个 sparsity_level

# CUDA_VISIBLE_DEVICES=0,1,2 python mystatistics_phi.py
for sparsity_level in "${sparsity_levels[@]}"; do
    CUDA_VISIBLE_DEVICES=3,4,5 python mystatistics_llama.py  --sparsity_level $sparsity_level
    # CUDA_VISIBLE_DEVICES=2,3,7 python mystatistics_phi.py --sparsity_level $sparsity_level
    # CUDA_VISIBLE_DEVICES=2,3,4 python mystatistics_deepseek.py --sparsity_level $sparsity_level
done