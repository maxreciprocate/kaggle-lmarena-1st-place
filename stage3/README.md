### Stage 3. Distillation

#### Prepare data

```bash
python stage3/prepare_synth_data.py
```

Pseudolabel datasets:

```bash
# label new data out of fold
sbatch label.slurm model_path=ckpts/new_and_old_win_80k_196a_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_0fold data_name=new_valid_0fold
sbatch label.slurm model_path=ckpts/new_and_old_win_80k_212d_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_1fold data_name=new_valid_1fold
sbatch label.slurm model_path=ckpts/new_and_old_win_80k_4877_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_2fold data_name=new_valid_2fold
sbatch label.slurm model_path=ckpts/new_and_old_win_80k_5178_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_3fold data_name=new_valid_3fold
sbatch label.slurm model_path=ckpts/new_and_old_win_80k_203e_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_4fold data_name=new_valid_4fold

# label the rest with 5 models overlapping
bash label.sh lmsys-old-nturn-ties
bash label.sh synth50k
bash label.sh hf25k
bash label.sh ppe-10k
bash label.sh ppe-tie-5k
bash label.sh vibe-1k
```

Average pseudolabels:

```bash
python collect_labels.py
python prepare_student_data.py
```

#### Distillation

Train two students, one on full dataset, one excluded our synthetic data and nbroad's:

```bash
sbatch launch_two.slurm use_all=true distillation=true distillation_loss_mix=0.5 lr=2e-6 dataset_name=nopvhs_qwen_clean31 valid_name=new_valid_0fold_clean29 model_path=ckpts/default_orpoufc90k_bf15_Qwen2.5-14B-Instruct_lr4e-06_bs32_fmt5rev_new_valid_0fold_edb2 max_length=4096 formatting=fmt5multirev gradient_accumulation_steps=1 bs=2 arch=class

# exclude the last two synthetic datasets
sbatch launch_two.slurm use_all=true distillation=true distillation_loss_mix=0.5 lr=2e-6 dataset_name=nopv_qwen_clean31 valid_name=new_valid_0fold_clean29 model_path=ckpts/default_orpoufc90k_bf15_Qwen2.5-14B-Instruct_lr4e-06_bs32_fmt5rev_new_valid_0fold_edb2 max_length=4096 formatting=fmt5multirev gradient_accumulation_steps=1 bs=2 arch=class
```

#### Model preparation

Merge gate up:
```bash
python prepare_student.py -i ckpts/qwen14b_useall_nopvhs_clean31 -o ckpts/qwen14b_useall_nopvhs_clean31
python prepare_student.py -i ckpts/qwen14b_useall_nopv_clean31 -o ckpts/qwen14b_useall_nopv_clean31
```

Merge two models:
```bash
python merge_students.py
```
