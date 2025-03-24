import random
from datasets import load_dataset
import hashlib

from datasets import load_dataset, Dataset, concatenate_datasets
from collections import Counter
from datasets import Features, Value
import hashlib
from transformers import AutoTokenizer

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

def split_5_fold(d):
    folds = [
        (
            [i for i in range(len(d)) if i % 5 != fold],
            [i for i in range(len(d)) if i % 5 == fold]
        )
        for fold in range(5)

    ]
    train, valid = [], []
    for train_idx, valid_idx in folds:
        d_train = d.select(train_idx)
        d_valid = d.select(valid_idx)
        train.append(d_train)
        valid.append(d_valid)
    return train, valid

def format_to_single(x):
    prompt = x['chosen'][0]['content']
    response_a = x['chosen'][1]['content']
    response_b = x['rejected'][1]['content']
    winner = 'model_a'
    n_turns = len(x['chosen']) // 2

    return {
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "winner": winner,
        "n_turns": n_turns
    }

# ;; ORPO-Mix-40k

dorpo = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train")
dorpo = dorpo.map(format_to_single)
dorpo = dorpo.filter(lambda x: x['n_turns'] == 1)
dorpo = dorpo.remove_columns(["chosen", "rejected", "question"])
dorpo = dorpo.filter(lambda x: x['prompt'].strip() != "" and x['response_a'].strip() != "" and x['response_b'].strip() != "")
dorpo = remove_response_duplicates(dorpo)
dorpo = dorpo.map(hash_prompt_responses)
dorpo = remove_duplicates(dorpo)
dorpo = balance_ab_winners_with_swap(dorpo)

# ;; UF-chinese

dufc = load_dataset("opencsg/UltraFeedback-chinese", data_files="ultrafeedback_zh_binarized_lowest.parquet", split="train")
def format_to_single(x):
    return {
        "prompt": x['instruction'],
        "response_a": x['chosen_response'],
        "response_b": x['rejected_response'],
        "winner": "model_a",
        "n_turns": 1,
        "source": x["source"]
    }

dufc = dufc.map(format_to_single)
dufc = dufc.remove_columns(["instruction", "chosen_response", "rejected_response", "__index_level_0__", "chosen_rating", "rejected_rating"])
dufc = dufc.filter(lambda x: x['prompt'].strip() != "" and x['response_a'].strip() != "" and x['response_b'].strip() != "")
dufc = remove_response_duplicates(dufc)
dufc = dufc.map(hash_prompt_responses)
dufc = remove_duplicates(dufc)
dufc = balance_ab_winners_with_swap(dufc)

;; ORPO-UFC-90K

orpoufc = concatenate_datasets([dorpo, dufc])
orpoufc = orpoufc.shuffle(seed=0)
orpoufc = remove_duplicates(orpoufc)
orpoufc = orpoufc.map(add_length)

orpoufc.to_parquet("data/orpoufc90k.parquet")

