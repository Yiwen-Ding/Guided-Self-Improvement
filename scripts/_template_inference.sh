#!/bin/bash

# set python script path and max retry times
python_script=${python_script}
max_retries=10

# args
model_path=${model_path}
results_path=${results_path}
temperature=${temperature}
dataset_name=${dataset_name}
min_correct_num=${min_correct_num}
device_num=${device_num}
sample_num=${sample_num}
max_resample_num=${max_resample_num}
max_length=${max_length}
inference_data_file=${inference_data_file}
log_file=${log_file}
mode=${mode:-"cot"}
teacher_model=${teacher_model}

# Initialize retry count
retry_count=0

# 定义一个函数来运行Python脚本
run_python_script() {
    if [ -n "$teacher_model" ]; then
        python3 $python_script \
            --model_name_or_path $model_path \
            --results_path $results_path \
            --temperature $temperature \
            --dataset_name $dataset_name \
            --min_correct_num $min_correct_num \
            --device_num $device_num \
            --sample_num $sample_num \
            --max_resample_num $max_resample_num \
            --max_length $max_length \
            --inference_data_file $inference_data_file \
            --mode $mode \
            --teacher_model $teacher_model \
            >> "$log_file" 2>&1
    else
        python3 $python_script \
            --model_name_or_path $model_path \
            --results_path $results_path \
            --temperature $temperature \
            --dataset_name $dataset_name \
            --min_correct_num $min_correct_num \
            --device_num $device_num \
            --sample_num $sample_num \
            --max_resample_num $max_resample_num \
            --max_length $max_length \
            --inference_data_file $inference_data_file \
            --mode $mode \
            >> "$log_file" 2>&1
    fi
}

# continue retrying until the script runs successfully
while [ $retry_count -lt $max_retries ]; do
    # run script
    run_python_script
    
    # check if the script ran successfully (exit status 0 means success)
    if [ $? -eq 0 ]; then
        echo "Script ran successfully." >> "$log_file" 2>&1
        exit 0
    else
        echo "Script failed. Retrying... ($((retry_count+1))/$max_retries)" >> "$log_file" 2>&1
        retry_count=$((retry_count+1))
        
        sleep 5
    fi
done

# exit and display error message if max retry count is reached
echo "Script failed after $max_retries attempts." >> "$log_file" 2>&1
exit 1
