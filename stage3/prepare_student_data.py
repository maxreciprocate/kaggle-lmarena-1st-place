from datasets import load_dataset, Dataset, concatenate_datasets
from collections import Counter
from datasets import Features, Value
from transformers import AutoTokenizer
import hashlib
import datasets
datasets.disable_progress_bar()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

def cast(dataset):
    new_features = dataset.features.copy()
    for col in dataset.features:
        try:
            if dataset.features[col].dtype == 'string':
                new_features[col] = Value('large_string')
        except:
            pass
    return dataset.cast(new_features)

def hash_prompt_responses(x):
    prompt = x['prompt'][:512]
    response_a = x['response_a'][:512]
    response_b = x['response_b'][:512]
    if response_a > response_b:
        response_a, response_b = response_b, response_a
    return {"id": hashlib.sha256(f"{prompt}{response_a}{response_b}".encode()).hexdigest()}

def remove_response_duplicates(d):
    print(f"Before removing a == b: {len(d)}")
    d = d.filter(lambda x: x['response_a'].strip() != x['response_b'].strip())
    print(f"After remove a == b: {len(d)}")
    return d

def remove_duplicates(d):
    df = d.to_pandas()
    print(f"Before removing duplicates: {len(df)}")
    df = df.drop_duplicates(subset=["id"], keep="first")
    print(f"After removing duplicates: {len(df)}")
    return Dataset.from_pandas(df, preserve_index=False)

def remove_prompt_duplicates(d):
    d = d.map(lambda x: {"prompt_lower": x["prompt"].lower()})
    df = d.to_pandas()
    print(f"Before removing prompt duplicates: {len(df)}")
    df = df.drop_duplicates(subset=["prompt_lower"], keep="first")
    print(f"After removing prompt duplicates: {len(df)}")
    d = Dataset.from_pandas(df, preserve_index=False)
    d = d.remove_columns("prompt_lower")
    return d

def balance_ab_winners_with_swap(d):
    print(f"Before balancing: {Counter(d['winner'])} {len(d)}")
    d_a = d.filter(lambda x: x['winner'] == 'model_a')
    d_b = d.filter(lambda x: x['winner'] == 'model_b')
    d_rest = d.filter(lambda x: x['winner'] not in ['model_a', 'model_b'])

    if len(d_b) > len(d_a):
        n_diff = (len(d_b) - len(d_a)) // 2
        d_b, d_b_to_swap = d_b.train_test_split(test_size=n_diff, seed=0).values()
        d_b_to_swap = d_b_to_swap.map(lambda x: {"winner": "model_a", "response_a": x["response_b"], "response_b": x["response_a"]})
        d_a = concatenate_datasets([d_a, d_b_to_swap])
    elif len(d_a) > len(d_b):
        n_diff = (len(d_a) - len(d_b)) // 2
        d_a, d_a_to_swap = d_a.train_test_split(test_size=n_diff, seed=0).values()
        d_a_to_swap = d_a_to_swap.map(lambda x: {"winner": "model_b", "response_a": x["response_b"], "response_b": x["response_a"]})
        d_b = concatenate_datasets([d_b, d_a_to_swap]).shuffle(seed=0)

    d = concatenate_datasets([d_a, d_b, d_rest])
    print(f"After balancing: {Counter(d['winner'])} {len(d)}")
    return d

root = "data/"
new = Dataset.from_parquet(root + "new-label-qwen.parquet")
old = Dataset.from_parquet(root + "old-label-qwen.parquet")
ppe = Dataset.from_parquet(root + "ppe-label-qwen.parquet")
vibe = Dataset.from_parquet(root + "vibe-label-qwen.parquet")
hf = Dataset.from_parquet(root + "hf-label-qwen.parquet")
synth = Dataset.from_parquet(root + "synth-label-qwen.parquet")

newoldppevibe = concatenate_datasets([cast(new), cast(old), cast(ppe), cast(vibe)])
newoldppevibe = remove_duplicates(newoldppevibe)
newoldppevibe = remove_response_duplicates(newoldppevibe)
newoldppevibe = balance_ab_winners_with_swap(newoldppevibe)

newoldppevibe.to_parquet(root + "nopv_qwen_clean31.parquet")
print(f"Writting to {root + 'nopv_qwen_clean31.parquet'}")

newoldppevibehfsynth = concatenate_datasets([cast(new), cast(old), cast(ppe), cast(vibe), cast(hf), cast(synth)])
newoldppevibehfsynth = remove_duplicates(newoldppevibehfsynth)
newoldppevibehfsynth = remove_response_duplicates(newoldppevibehfsynth)
newoldppevibehfsynth = balance_ab_winners_with_swap(newoldppevibehfsynth)

newoldppevibehfsynth.to_parquet(root + "nopvhs_qwen_clean31.parquet")
print(f"Writting to {root + 'nopvhs_qwen_clean31.parquet'}")

