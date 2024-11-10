#!/bin/bash
### Required variables
dataset_name=mathinstruct
model=llama3
model_name_or_path="meta-llama/Meta-Llama-3-8B"
n_epochs=1
iteration=4
max_length=1024
teacher_model_name="llama3-70b-instruct"

# file path
eval_data_files=("./data/test/aqua_rat.json" "./data/test/gsm8k.json" "./data/test/math.json")
inference_data_files=("./data/inference/cot/aqua_rat.json" "./data/inference/cot/gsm8k.json" "./data/inference/cot/math.json")
datasets=("aqua" "gsm8k" "math")

held_out_datasets=("theoremqa" "mathqa" "svamp")
held_eval_data_files=("./data/test/theoremqa.json" "./data/test/mathqa.json" "./data/test/svamp.json")

# sample config
sample_num=8
min_correct_num=4
max_resample_num=8
# data merge config
include_original_data=false
include_pre_iter_data=false
mode="cot"

# output config
exp_name="teacher_help_sample_${sample_num}_correct_${min_correct_num}_max_${max_resample_num}_orig_${include_original_data}_preiter_${include_pre_iter_data}"
exp_output_dir="output/${model}/${exp_name}"
#########

numgpu=8

# iter train data
mkdir -p "${exp_output_dir}/iter_data/"
cp "./data/MathInstruct.json" "${exp_output_dir}/iter_data/${dataset_name}_iter_0.json"

# original iter train data
mkdir -p "${exp_output_dir}/iter_data_original/"
cp "./data/MathInstruct.json" "${exp_output_dir}/iter_data_original/${dataset_name}_iter_0.json"

for ((ITER = 0; ITER < iteration; ITER++))
do
        iter_output_dir="${exp_output_dir}/iter_${ITER}"
        mkdir -p "${iter_output_dir}"
        cur_train_file="${exp_output_dir}/iter_data/${dataset_name}_iter_${ITER}.json"

        cur_model_save_dir="${iter_output_dir}/model"

        # train
        exp_name=${exp_name} \
        train_file=${cur_train_file} \
        engine='nl' \
        model_name_or_path=${model_name_or_path} \
        tokenizer_name_or_path=${model_name_or_path} \
        n_epochs=${n_epochs} \
        iteration=${ITER} \
        model_dir=${iter_output_dir} \
        bash ./scripts/_template.sh

        for i in "${!datasets[@]}"; do
                cur_dataset_name=${datasets[$i]}
                eval_data_file=${eval_data_files[$i]}
                
                # Evaluation
                python3 eval.py \
                        --model_name_or_path ${cur_model_save_dir} \
                        --results_path "${iter_output_dir}/eval_data" \
                        --temperature 0.0 \
                        --dataset_name ${cur_dataset_name} \
                        --device_num ${numgpu} \
                        --eval_data_file  ${eval_data_file} \
                        --max_length ${max_length} \
                        --mode ${mode} \
                        > "${iter_output_dir}/eval_${cur_dataset_name}.log" 2>&1 
        done
        
        for i in "${!held_out_datasets[@]}"; do
                cur_dataset_name=${held_out_datasets[$i]}
                eval_data_file=${held_eval_data_files[$i]}
        
                # Evaluation
                python3 eval.py \
                        --model_name_or_path ${cur_model_save_dir} \
                        --results_path "${iter_output_dir}/eval_data" \
                        --temperature 0.0 \
                        --dataset_name ${cur_dataset_name} \
                        --device_num ${numgpu} \
                        --eval_data_file  ${eval_data_file} \
                        --max_length ${max_length} \
                        --mode ${mode} \
                        > "${iter_output_dir}/eval_${cur_dataset_name}.log" 2>&1 
        done    


        # Inference
        for i in "${!datasets[@]}"; do
                cur_dataset_name=${datasets[$i]}
                inference_data_file=${inference_data_files[$i]}

                python_script="inference_teacher_help.py" \
                model_path=${cur_model_save_dir} \
                teacher_model=${teacher_model_name} \
                results_path="${iter_output_dir}/inference_data" \
                temperature=0.7 \
                dataset_name=${cur_dataset_name} \
                min_correct_num=${min_correct_num} \
                device_num=${numgpu} \
                sample_num=${sample_num} \
                max_resample_num=${max_resample_num} \
                max_length=${max_length} \
                inference_data_file=${inference_data_file} \
                mode=${mode} \
                log_file="${iter_output_dir}/inference_${cur_dataset_name}.log" \
                bash ./scripts/_template_inference.sh 
        done

        cur_sampled_data_file="${iter_output_dir}/inference_data/results.json"
        next_train_file="${exp_output_dir}/iter_data/${dataset_name}_iter_$((ITER+1)).json"

        # Construct the arguments for generate_train_data.py
        generate_train_data_args=""
        if [ "$include_original_data" = true ]; then
                generate_train_data_args+=" --include_original_data"
        fi
        if [ "$include_pre_iter_data" = true ]; then
                generate_train_data_args+=" --include_pre_iter_data"
        fi

        python3 generate_train_data.py  \
                --original_data_path ${inference_data_files[0]} \
                --pre_iter_data_path ${cur_train_file} \
                --cur_sampled_data_path ${cur_sampled_data_file} \
                --next_iter_data_path ${next_train_file} \
                $generate_train_data_args \
                > "${iter_output_dir}/generate_train_data.log" 2>&1
done
