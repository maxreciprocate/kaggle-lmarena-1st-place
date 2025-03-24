#!/bin/bash
H=`hostname`
RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

export lr=${lr:-4e-6}
export use_all=${use_all:-False}
export model_path=${model_path:-"google/gemma-2-9b-it"}
export dataset_name=${dataset_name:-"new_train_0fold"}
export valid_name=${valid_name:-"new_valid_0fold"}
export formatting=${formatting:-"fmt5multirev"}
export scheduler=${scheduler:-"linear_min"}
export experiment=${experiment:-"train"}
export bs=${bs:-1}
export distillation=${distillation:-false}
export distillation_temp=${distillation_temp:-0.25}
export distillation_loss_mix=${distillation_loss_mix:-0.5}
export max_length=${max_length:-8192}
export gradient_accumulation_steps=${gradient_accumulation_steps:-1}
export arch=${arch:-"lm"}

accelerate launch --config_file zero3.yaml \
           --num_processes $((8 * $COUNT_NODE)) \
           --num_machines $COUNT_NODE \
           --machine_rank $RANK \
           --rdzv_backend static \
           --main_process_port $MASTER_PORT \
           --main_process_ip $MASTER_ADDR \
           --deepspeed_multinode_launcher standard \
           train.py --dataset_name $dataset_name --learning_rate $lr --model_path $model_path --formatting $formatting --per_device_train_batch_size $bs --valid_name $valid_name --output_dir $output_dir --use_all $use_all --experiment $experiment --distillation $distillation --distillation_temp $distillation_temp --distillation_loss_mix $distillation_loss_mix --scheduler $scheduler --max_length $max_length --gradient_accumulation_steps $gradient_accumulation_steps --arch $arch
