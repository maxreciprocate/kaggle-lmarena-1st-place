#!/bin/bash
dataset=$1

sbatch label.slurm model_path=ckpts/new_and_old_win_80k_196a_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_0fold data_name=$dataset
sbatch label.slurm model_path=ckpts/new_and_old_win_80k_212d_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_1fold data_name=$dataset
sbatch label.slurm model_path=ckpts/new_and_old_win_80k_4877_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_2fold data_name=$dataset
sbatch label.slurm model_path=ckpts/new_and_old_win_80k_5178_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_3fold data_name=$dataset
sbatch label.slurm model_path=ckpts/new_and_old_win_80k_203e_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_4fold data_name=$dataset

