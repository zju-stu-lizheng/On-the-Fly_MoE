#!/bin/bash
sparsity_levels=(0.5 0.6 0.7)  # 定义多个 sparsity_level


for sparsity_level in "${sparsity_levels[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6 python mystatistics.py --sparsity_level $sparsity_level
done