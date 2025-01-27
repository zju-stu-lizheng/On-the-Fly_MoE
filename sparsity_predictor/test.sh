#### eval
CUDA_VISIBLE_DEVICES=3,4 python quevaluate.py \
--threshold_path 'chess_up_threshold' \
--sparsity_level 0.8 \
--task_name_list 'arc_challenge' 'openbookqa' 'winogrande' 'sciq' 'arc_easy' 'boolq' >> prediction.out


CUDA_VISIBLE_DEVICES=0,1 python quevaluate.py \
--threshold_path 'chess_up_threshold' \
--sparsity_level 0.8 \
--lora_path '/home/bcds/On-the-Fly_MoE_Inference/quantize/output/mixtral/90_20000_expert_atten/0' \
--task_name_list 'arc_challenge' 'openbookqa' 'winogrande' 'sciq' 'arc_easy' 'boolq' >> prediction_lora.out
# CUDA_VISIBLE_DEVICES=3,4 python quevaluate.py \
# --threshold_path 'chess_up_threshold' \
# --sparsity_level 0.8 \
# --task_name_list 'arc_challenge' 'openbookqa' 'winogrande' 'sciq' 'arc_easy' 'boolq' >> reuse_80_2.out