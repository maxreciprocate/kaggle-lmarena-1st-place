### Stage 1. Pretraining

```bash
# prepare data
python stage1/prepare_pretrain_data.py
```

```bash
# pretrain student
sbatch launch.slurm lr=4e-6 dataset_name=orpoufc90k model_path=Qwen/Qwen2.5-14B-Instruct max_length=8192 formatting=fmt5multirev

# pretrain teacher
sbatch --nodes 4 launch.slurm lr=4e-6 dataset_name=orpoufc90k model_path=Qwen/Qwen2.5-72B-Instruct max_length=6144 formatting=fmt5multirev
```

