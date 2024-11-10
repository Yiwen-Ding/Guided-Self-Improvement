#!/bin/bash
### Required variables
exp_name=${exp_name}
train_file=${train_file:-''}
engine=${engine:-''}
model_name_or_path=${model_name_or_path:-''}
tokenizer_name_or_path=${tokenizer_name_or_path:-''}
n_epochs=${n_epochs:-''}
iteration=${iteration}
model_dir=${model_dir:-''}

### Default variables
config_file="./default_config_deepspeed_ga2.yaml"

batch_size="4"
gradient_accumulation_steps="2"
max_input_length="1024"
num_workers="8"
learning_rate="1e-5"
weight_decay="0"
warmup_step="-100"
clip_grad_norm="1"
seed="42"
keep_num_ckpt='0'

logging_epoch_freq="1"
saving_epoch_freq="1"

logging_step_freq="10"

wandb_log="True"
wandb_project="self_improve"
wandb_run_name="${exp_name}"
#########

num_processes='8'

echo $(echo "template scripts!")

accelerate launch \
        --config_file "${config_file}" \
        --num_processes=${num_processes} \
        train.py \
        --model_name_or_path "${model_name_or_path}" \
        --tokenizer_name_or_path "${tokenizer_name_or_path}" \
        --train_file "${train_file}" \
        --model_dir "${model_dir}" \
        --batch_size "${batch_size}" \
        --n_epochs "${n_epochs}" \
        --num_workers "${num_workers}" \
        --learning_rate "${learning_rate}" \
        --weight_decay "${weight_decay}" \
        --warmup_step "${warmup_step}" \
        --clip_grad_norm "${clip_grad_norm}" \
        --logging_epoch_freq "${logging_epoch_freq}" \
        --saving_epoch_freq "${saving_epoch_freq}" \
        --logging_step_freq "${logging_step_freq}" \
        --seed "${seed}" \
        --max_input_length "${max_input_length}" \
        --gradient_accumulation_steps "${gradient_accumulation_steps}" \
        --keep_num_ckpt "${keep_num_ckpt}" \
        --wandb_log "${wandb_log}" \
        --wandb_project "${wandb_project}" \
        --wandb_run_name "${wandb_run_name}" \
        --engine "${engine}" \
        > "${model_dir}"/"sft_bsz_${batch_size}".log 2>&1