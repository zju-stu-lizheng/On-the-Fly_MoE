#### train
CUDA_VISIBLE_DEVICES=4,5,6 python finetune.py \
--model_save_path '/home/lz/On-the-Fly_MoE_Inference/quantize/saved/training/nohqq' \
--epoch 1 --training_steps 10000 --use_average
echo 'sparsity 80' > train_new.out
#### eval
CUDA_VISIBLE_DEVICES=4,5 python quevaluate.py \
--lora_path '/home/lz/On-the-Fly_MoE_Inference/quantize/saved/training/nohqq/checkpoint-1250' \
--threshold_path 'chess_up_threshold' --use_average \
--task_name_list 'arc_challenge' 'openbookqa' 'winogrande' 'sciq' 'arc_easy' >> train_new.out