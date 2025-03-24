### Stage 2. Teacher training

```bash
# prepare data
python stage2/prepare_teacher_data.py

# train 5 teachers starting from pretrained ckpt
sbatch --nodes 4 launch.slurm dataset_name=new_and_old_win_80k valid_name=new_valid_0fold model_path=ckpts/qwen72b-orpoufc90k-fmt5q formatting=fmt7_models_rev max_length=6144 arch=class-2layers
sbatch --nodes 4 launch.slurm dataset_name=new_and_old_win_80k valid_name=new_valid_1fold model_path=ckpts/qwen72b-orpoufc90k-fmt5q formatting=fmt7_models_rev max_length=6144 arch=class-2layers
sbatch --nodes 4 launch.slurm dataset_name=new_and_old_win_80k valid_name=new_valid_2fold model_path=ckpts/qwen72b-orpoufc90k-fmt5q formatting=fmt7_models_rev max_length=6144 arch=class-2layers
sbatch --nodes 4 launch.slurm dataset_name=new_and_old_win_80k valid_name=new_valid_3fold model_path=ckpts/qwen72b-orpoufc90k-fmt5q formatting=fmt7_models_rev max_length=6144 arch=class-2layers
sbatch --nodes 4 launch.slurm dataset_name=new_and_old_win_80k valid_name=new_valid_4fold model_path=ckpts/qwen72b-orpoufc90k-fmt5q formatting=fmt7_models_rev max_length=6144 arch=class-2layers
```

