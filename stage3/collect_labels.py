import hashlib
import numpy as np
from datasets import Dataset, concatenate_datasets, Features, Value
from collections import Counter

def cast(dataset):
    new_features = dataset.features.copy()
    for col in dataset.features:
        if dataset.features[col].dtype == 'string':
            new_features[col] = Value('large_string')
    return dataset.cast(new_features)

def avg_from_nlabels(ds):
    logits = np.array([x['logits'] for x in ds]).mean(axis=0)
    d = ds[0].remove_columns(['logits'])
    d = d.add_column("logits", logits.tolist())
    if "winner" not in d.column_names:
        d = d.add_column("winner", ["model_a" if xs.argmax() == 0 else "model_b" for xs in logits])
    return d

def avg_from_swap(d):
    dorig = d.filter(lambda x: x["swap"] == False)
    dswap = d.filter(lambda x: x["swap"] == True)
    logits = np.array([x['logits'] for x in dorig])
    logits_swap = np.array([x['logits'][::-1] for x in dswap])
    logits_avg = (logits + logits_swap) / 2

    dorig = dorig.remove_columns(["a_win", "b_win", "logits", "swap"])
    dorig = dorig.add_column("logits", [x.tolist() for x in logits_avg])
    return dorig

# average ab, ba and concatenate since it's out of fold
dataset_name = "new"
paths = [
    "label_new_valid_0fold_new_and_old_win_80k_196a_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_0fold_19k.parquet",
    "label_new_valid_1fold_new_and_old_win_80k_212d_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_1fold_19k.parquet",
    "label_new_valid_2fold_new_and_old_win_80k_4877_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_2fold_19k.parquet",
    "label_new_valid_3fold_new_and_old_win_80k_5178_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_3fold_19k.parquet",
    "label_new_valid_4fold_new_and_old_win_80k_203e_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_4fold_19k.parquet",
]

ds = [Dataset.from_parquet(f"artifacts/label/{path}") for path in paths]
ds = [avg_from_swap(d) for d in ds]
d = concatenate_datasets(ds)
d.to_parquet(f"data/{dataset_name}-label-qwen.parquet")

# average ab, ba and then average 5 predictions from 5 models
dataset_name = "old"
old_paths = [
    "label_lmsys-80k-old-nturn-ties_new_and_old_win_80k_196a_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_0fold_165k.parquet",
    "label_lmsys-80k-old-nturn-ties_new_and_old_win_80k_203e_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_4fold_165k.parquet",
    "label_lmsys-80k-old-nturn-ties_new_and_old_win_80k_212d_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_1fold_165k.parquet",
    "label_lmsys-80k-old-nturn-ties_new_and_old_win_80k_4877_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_2fold_165k.parquet",
    "label_lmsys-80k-old-nturn-ties_new_and_old_win_80k_5178_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_3fold_165k.parquet",
]

ds = [Dataset.from_parquet(f"artifacts/label/{path}") for path in old_paths]
ds = [avg_from_swap(d) for d in ds]
d = concatenate_datasets(ds)
d.to_parquet(f"data/{dataset_name}-label-qwen.parquet")

dataset_name = "hf"
paths = [
    "label_hf25k_new_and_old_win_80k_196a_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_0fold_53k.parquet",
    "label_hf25k_new_and_old_win_80k_203e_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_4fold_53k.parquet",
    "label_hf25k_new_and_old_win_80k_212d_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_1fold_53k.parquet",
    "label_hf25k_new_and_old_win_80k_4877_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_2fold_53k.parquet",
    "label_hf25k_new_and_old_win_80k_5178_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_3fold_53k.parquet",
]

ds = [Dataset.from_parquet(f"artifacts/label/{path}") for path in old_paths]
ds = [avg_from_swap(d) for d in ds]
d = concatenate_datasets(ds)
d.to_parquet(f"data/{dataset_name}-label-qwen.parquet")

dataset_name = "synth"
paths = [
    "label_synth50k_new_and_old_win_80k_196a_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_0fold_100k.parquet",
    "label_synth50k_new_and_old_win_80k_203e_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_4fold_100k.parquet",
    "label_synth50k_new_and_old_win_80k_212d_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_1fold_100k.parquet",
    "label_synth50k_new_and_old_win_80k_4877_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_2fold_100k.parquet",
    "label_synth50k_new_and_old_win_80k_5178_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_3fold_100k.parquet",
]

ds = [Dataset.from_parquet(f"artifacts/label/{path}") for path in old_paths]
ds = [avg_from_swap(d) for d in ds]
d = concatenate_datasets(ds)
d.to_parquet(f"data/{dataset_name}-label-qwen.parquet")

dataset_name = "ppe"
paths = [
    "label_ppe-10k_new_and_old_win_80k_196a_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_0fold_20k.parquet",
    "label_ppe-10k_new_and_old_win_80k_203e_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_4fold_20k.parquet",
    "label_ppe-10k_new_and_old_win_80k_212d_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_1fold_20k.parquet",
    "label_ppe-10k_new_and_old_win_80k_4877_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_2fold_20k.parquet",
    "label_ppe-10k_new_and_old_win_80k_5178_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_3fold_20k.parquet",
]

ds = [Dataset.from_parquet(f"artifacts/label/{path}") for path in old_paths]
ds = [avg_from_swap(d) for d in ds]
d = concatenate_datasets(ds)
d.to_parquet(f"data/{dataset_name}-label-qwen.parquet")

dataset_name = "vibe"
paths = [
    "label_vibe-1k_new_and_old_win_80k_196a_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_0fold_3k.parquet",
    "label_vibe-1k_new_and_old_win_80k_203e_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_4fold_3k.parquet",
    "label_vibe-1k_new_and_old_win_80k_212d_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_1fold_3k.parquet",
    "label_vibe-1k_new_and_old_win_80k_4877_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_2fold_3k.parquet",
    "label_vibe-1k_new_and_old_win_80k_5178_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_3fold_3k.parquet",
]

ds = [Dataset.from_parquet(f"artifacts/label/{path}") for path in old_paths]
ds = [avg_from_swap(d) for d in ds]
d = concatenate_datasets(ds)
d.to_parquet(f"data/{dataset_name}-label-qwen.parquet")

dataset_name = "ppe-tie"
paths = [
    "label_ppe-tie-5k_new_and_old_win_80k_196a_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_0fold_12k.parquet",
    "label_ppe-tie-5k_new_and_old_win_80k_203e_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_4fold_12k.parquet",
    "label_ppe-tie-5k_new_and_old_win_80k_212d_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_1fold_12k.parquet",
    "label_ppe-tie-5k_new_and_old_win_80k_4877_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_2fold_12k.parquet",
    "label_ppe-tie-5k_new_and_old_win_80k_5178_qwen72b-orpoufc90k-8k-fmt5q_lr4e-06_bs32_fmt7_models_rev_new_valid_3fold_12k.parquet",
]

ds = [Dataset.from_parquet(f"artifacts/label/{path}") for path in old_paths]
ds = [avg_from_swap(d) for d in ds]
d = concatenate_datasets(ds)
d.to_parquet(f"data/{dataset_name}-label-qwen.parquet")

