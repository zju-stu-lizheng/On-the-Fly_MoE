import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,7"
# CUDA_VISIBLE_DEVICES=4,5,7 python finetune.py
from transformers import AutoTokenizer, BitsAndBytesConfig, AdamW
from modeling.modeling_mixtral import MixtralForCausalLM, set_th_sparsity
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
import functools
import mlflow

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    

def get_model_for_training(model_name, has_qkv=False, rank=128, use_lora=False):
    if use_lora:
        nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        ) # 量化参数
        
        nf8_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        model = MixtralForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            quantization_config=nf4_config,
            use_cache=False,
            # attn_implementation="flash_attention_2"
        )
        target_modules = ["w1", "w2", "w3",]
        if has_qkv: 
            target_modules += [
                "q_proj",
                "k_proj",
                # "v_proj",
                # "o_proj",
            ]
    
        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.01,
            r=rank,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM"
        )

        model = prepare_model_for_kbit_training(model) # 用来使得模型能够训练在4Bits精度
        model = get_peft_model(model, peft_config) 
    else:
        model = MixtralForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.float16,
            use_cache=False)
    
        for name, param in model.named_parameters():
            # # 指定冻结层
            if not any(nd in name for nd in ["model.layers.1.block_sparse_moe.gate.weight"]):
                param.requires_grad = False
            # if not any(nd in name for nd in ["o_proj", "q_proj", "k_proj", "v_proj", "input_layernorm", "post_attention_layernorm"]):
            #     # , "lm_head", "embed_tokens"
            #     param.requires_grad = True
            # else:
            #     # 将参数的 requires_grad 属性设置为False，即冻结该参数
            #     param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print_trainable_parameters(model)
    return model, tokenizer


def preprocess_data(batch, tokenizer):
    # 使用 tokenizer 将文本数据转换为模型输入
    inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    inputs["labels"] = inputs.input_ids.clone()
    return inputs


def get_fineweb_dataset(tokenizer, sample_num=24000, seed=42):
    # 加载fineweb数据集
    test_num = sample_num//10
    # fineweb = load_dataset("parquet", data_dir="./fineweb-edu/")
    fineweb = load_dataset("json", data_files="/home/zyx/moe/fineweb-edu/fineweb_edu_sample100000.json")
    ## 随机从fineweb中抽取sample_num条数据
    fineweb_train = fineweb['train'].train_test_split(test_size=test_num, seed=seed)
    train_data = fineweb_train['train'].select(range(sample_num))
    test_data = fineweb_train['test']

    fineweb_train_data = train_data.map(
        functools.partial(
        preprocess_data,
        tokenizer=tokenizer
    ), batched=True)
    fineweb_test_data = test_data.map(
        functools.partial(
        preprocess_data,
        tokenizer=tokenizer
    ), batched=True)
    return fineweb_train_data, fineweb_test_data


def train_moe(model_name, model_save_path, sparsity=0, has_qkv=False):
    sample_num = 500
    model, tokenizer = get_model_for_training(model_name, has_qkv=has_qkv, rank=64, use_lora=False)
    fineweb_train_data, fineweb_test_data = get_fineweb_dataset(tokenizer, sample_num=sample_num)

    print('model train is starting with sparsity:', sparsity)
    set_th_sparsity(sparsity, dataset='c4_base')
    
    learning_rate = 0.01
    micro_batch_size=2
    epochs=20
    ###### 不保存
    save_steps = 10000
    save_total_limit = 2
    optimizer=AdamW(filter(lambda p : p.requires_grad, model.parameters()),lr=learning_rate)
    linear_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(sample_num*epochs) // micro_batch_size)
    args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=epochs,
        # max_steps=opt.max_steps,
        # fp16=True,
        bf16=True,
        optim="adamw_torch",# paged_adamw_8bit
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,   ### 先设置成False
        group_by_length=False,
        logging_steps=10,
        eval_steps=100,
        save_strategy="steps",
        save_only_model=True,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=fineweb_train_data,
        eval_dataset=fineweb_test_data,
        args=args,
        optimizers=(optimizer, linear_scheduler),
        data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    # silence the warnings. re-enable for inference!
    model.config.use_cache = False
    with mlflow.start_run() as run:
        trainer.train()
    # trainer.model.save_pretrained(model_save_path)
    print('model train is finished')

if __name__ == '__main__':
    mlflow.set_experiment("mixtral-base expert")
    model_name = "/home/lz/workspace/llama2-7b/Mixtral-8x7B-v0.1"
    sparsity = 0
    model_save_path = f'./output/fineweb-{sparsity}-router'
    train_moe(model_name, model_save_path, sparsity=sparsity, has_qkv=True)
