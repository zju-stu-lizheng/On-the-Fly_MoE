CUDA_VISIBLE_DEVICES=1,2,3 python recover.py \
--model_save_path '/home/lz/On-the-Fly_MoE_Inference/quantize/saved/training/test' \
--epoch 2 --training_steps 3000
echo 'sparsity 80' > train_new.out
CUDA_VISIBLE_DEVICES=0,1 python quevaluate2.py \
--lora_path '/home/lz/On-the-Fly_MoE_Inference/quantize/saved/training/test/checkpoint-750_lora_combine.pt' \
--threshold_path 'chess_up_sparsity_threshold' >> train_new.out