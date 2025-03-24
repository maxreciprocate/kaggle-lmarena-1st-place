from datasets import load_dataset, Dataset, concatenate_datasets
from collections import Counter
from datasets import Features, Value
from transformers import AutoTokenizer
import hashlib
import datasets
import random
datasets.disable_progress_bar()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

def cast(dataset):
    new_features = dataset.features.copy()
    for col in dataset.features:
        if dataset.features[col].dtype == 'string':
            new_features[col] = Value('large_string')
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

def balance_ab_winners_with_swap(d):
    random.seed(0)
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

def add_length(x):
    x["len_p"] = len(tokenizer(x["prompt"]).input_ids)
    x["len_a"] = len(tokenizer(x["response_a"]).input_ids)
    x["len_b"] = len(tokenizer(x["response_b"]).input_ids)
    x["len"] = x["len_p"] + x["len_a"] + x["len_b"]
    x["longer_won"] = x["len_a"] > x["len_b"] and x["winner"] == "model_a" or x["len_b"] > x["len_a"] and x["winner"] == "model_b"
    return x

# ;;

dsynth50k = Dataset.from_parquet("reciprocate/kaggle-lmarena-synth-50k")
map_models = {
    'Nexusflow/Athene-V2-Chat': "athene-v2",
    'Qwen/Qwen2.5-72B-Instruct': "qwen2.5-72b-instruct",
    'meta-llama/Llama-3.1-405B-Instruct-FP8': "llama-3.1-405b-instruct-fp8",
    'meta-llama/Llama-3.3-70B-Instruct': "llama-3.3-70b-instruct",
}

def format_to_single(x):
    return {
        "prompt": x['prompt'].strip(),
        "response_a": x['response_a'].strip(),
        "response_b": x['response_b'].strip(),
        "model_a": map_models[x['model_a']],
        "model_b": map_models[x['model_b']],
        "winner": None
    }

dsynth50k = dsynth50k.map(format_to_single)
dsynth50k.add_column('source', ['synth50k'] * len(dsynth50k))
dsynth50k = dsynth50k.map(hash_prompt_responses)
dsynth50k = remove_response_duplicates(dsynth50k)
dsynth50k = remove_duplicates(dsynth50k)

dsynth50k.to_parquet("data/synth50k.parquet")

# ;; hf

d1 = Dataset.from_parquet("data/hf-open-models-v1.parquet")
d2 = Dataset.from_parquet("data/hf-open-models-v2.parquet")
d3 = Dataset.from_parquet("data/hf-open-models-v3.parquet")

dhf = concatenate_datasets([d1, d2, d3])

def format_to_single(x):
    return {
        "prompt": x['prompt'].strip(),
        "response_a": x['response_a'].strip(),
        "response_b": x['response_b'].strip(),
        "model_a": x['model_a'],
        "model_b": x['model_b'],
        "winner": None
    }

dhf = dhf.map(format_to_single)
dhf.add_column('source', ['hf'] * len(dhf))
dhf = dhf.remove_columns(["id", "__index_level_0__", "language"])
dhf = dhf.map(hash_prompt_responses)
dhf = remove_response_duplicates(dhf)
dhf = remove_duplicates(dhf)

dhf.to_parquet("data/hf25k.parquet")
