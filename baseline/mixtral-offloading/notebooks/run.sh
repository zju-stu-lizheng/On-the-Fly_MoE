#!/bin/bash
# sparsity_levels=(2 3 4 5 6 7 8)  # 定义多个 sparsity_level

# for prefill_layers in "${sparsity_levels[@]}"; do
#     CUDA_VISIBLE_DEVICES=5 python eval-baseline.py --framework mixtral-offloading \
#     --quantized --offload_layers $prefill_layers
# done

CUDA_VISIBLE_DEVICES=5 python eval-baseline.py --framework deepspeed-mii \
    --quantized