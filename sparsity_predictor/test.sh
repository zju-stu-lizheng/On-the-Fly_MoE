#### eval
CUDA_VISIBLE_DEVICES=3,4 python quevaluate.py \
--threshold_path 'chess_up_threshold' \
--sparsity_level 0.8 \
--task_name_list 'arc_challenge' 'openbookqa' 'winogrande' 'sciq' 'arc_easy' 'boolq' >> train_80.out