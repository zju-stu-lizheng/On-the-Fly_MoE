#!/bin/bash
sparsity_levels=(7)  # 定义多个 sparsity_level

for prefill_layers in "${sparsity_levels[@]}"; do
    CUDA_VISIBLE_DEVICES=5,1,2 python 1.py --prefill_layers $prefill_layers
done