#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=13043
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

for arg in "$@"; do
    key=$(echo $arg | cut -f1 -d=)
    key_length=${#key}
    val="${arg:$key_length+1}"

    if [[ $val == *" "* ]]; then
        IFS=' ' read -ra ARRAY_VAL <<< "$val"
        declare -a "$key"
        eval "$key=(${ARRAY_VAL[@]})"
        echo "$key=${ARRAY_VAL[@]}"
    else
        export "$key=$val"
    fi
done

srun run.sh "$@"
