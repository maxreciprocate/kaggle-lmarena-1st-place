#!/bin/bash
#SBATCH --partition=p5
#SBATCH --gpus=2
#SBATCH --array=0-0%1

for arg in "$@"; do
    key=$(echo $arg | cut -f1 -d=)
    key_length=${#key}
    val="${arg:$key_length+1}"

    if [[ $val == *" "* ]]; then
        # Value contains spaces, treat as array
        IFS=' ' read -ra ARRAY_VAL <<< "$val"
        declare -a "$key"
        eval "$key=(${ARRAY_VAL[@]})"
        echo "$key=${ARRAY_VAL[@]}"
    else
        # Single value, not an array
        export "$key=$val"
    fi
done

python label.py --rank $SLURM_ARRAY_TASK_ID --world_size $SLURM_ARRAY_TASK_COUNT --model_path $model_path --data_name $data_name
