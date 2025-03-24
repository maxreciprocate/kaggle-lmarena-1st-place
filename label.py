import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from time import time

import datasets
import pandas as pd
import torch
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

from format import formatting_map


def nhuman(num):
    return f"{round(num/1000)}k" if num >= 1000 else str(num)

import argparse

if __name__ == '__main__':
    stime = time.time()
    datasets.disable_caching()
    datasets.config.IN_MEMORY_MAX_SIZE = 125000 * 1024 * 1024
    num_proc = os.cpu_count() // 8

    parser = argparse.ArgumentParser(description="Labeling Script")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_name", type=str)
    args = parser.parse_args()

    model_path = args.model_path
    submit = False
    max_length = 6144
    batch_size = 2
    dtype = torch.bfloat16

    os.makedirs("artifacts/label", exist_ok=True)
    prefix = f"artifacts/label/label_{args.data_name}_{model_path.split('/')[-1]}"
    if os.path.exists(f"{prefix}_shard{args.rank}.parquet"):
        print(f"Already exists: {prefix}_shard{args.rank}.parquet")
        exit(0)
    print(prefix)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        print("Adding pad token")
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"

    def add_length(x):
        x["length"] = len(x["input_ids"])
        return x

    fmt_name = "fmt7_models"
    fmt = formatting_map[fmt_name]

    mbs = 1000
    data_swap = load_dataset("parquet", data_files=f"data/{args.data_name}.parquet", split="train")
    data_swap = data_swap.select(range(args.rank, len(data_swap), args.world_size))

    data_swap = data_swap.rename_columns({"response_a": "response_b", "response_b": "response_a", "model_a": "model_b", "model_b": "model_a"})
    if fmt_name == "fmt7_models":
        data_swap = data_swap.map(fmt, batched=True, input_columns=["prompt", "response_a", "response_b", "model_a", "model_b"], fn_kwargs={"max_length": max_length, "tokenizer": tokenizer}, batch_size=mbs, writer_batch_size=mbs, num_proc=num_proc)
    else:
        data_swap = data_swap.map(fmt, batched=True, input_columns=["prompt", "response_a", "response_b"], fn_kwargs={"max_length": max_length, "tokenizer": tokenizer}, batch_size=mbs, writer_batch_size=mbs, num_proc=num_proc)
    data_swap = data_swap.map(add_length, writer_batch_size=mbs, num_proc=num_proc)
    data_swap = data_swap.sort("length", reverse=True, writer_batch_size=mbs)

    data = load_dataset("parquet", data_files=f"data/{args.data_name}.parquet", split="train")
    data = data.select(range(args.rank, len(data), args.world_size))

    if fmt_name == "fmt7_models":
        data = data.map(fmt, batched=True, input_columns=["prompt", "response_a", "response_b", "model_a", "model_b"], fn_kwargs={"max_length": max_length, "tokenizer": tokenizer}, batch_size=mbs, writer_batch_size=mbs, num_proc=num_proc)
    else:
        data = data.map(fmt, batched=True, input_columns=["prompt", "response_a", "response_b"], fn_kwargs={"max_length": max_length, "tokenizer": tokenizer}, batch_size=mbs, writer_batch_size=mbs, num_proc=num_proc)
    data = data.map(add_length, writer_batch_size=mbs, batch_size=mbs, num_proc=num_proc)
    data = data.sort("length", reverse=True, writer_batch_size=mbs)

    print(f"Data length: {len(data)}")

    from models import (Gemma2ForSequenceClassificationPlus,
                        LlamaForSequenceClassificationPlus,
                        Qwen2ForSequenceClassificationPlus)
    if "llama" in model_path.lower():
        model_fn = LlamaForSequenceClassificationPlus
    if "qwen" in model_path.lower():
        model_fn = Qwen2ForSequenceClassificationPlus
    if "gemma" in model_path:
        model_fn = Gemma2ForSequenceClassificationPlus

    model = model_fn.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        num_labels=2,
        low_cpu_mem_usage=True,
        device_map='auto'
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()

    all_logits = []
    a_wins = []
    b_wins = []
    with torch.inference_mode():
        for input_ids in tqdm(batched(data["input_ids"], batch_size), total=math.ceil(len(data) / batch_size)):
            batch = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
            output = model(**batch.to(model.device))
            logits = output.logits.cpu()
            probs = logits.softmax(-1)
            a_win = probs[:, 0].tolist()
            b_win = probs[:, 1].tolist()
            all_logits.extend(logits.tolist())
            a_wins.extend(a_win)
            b_wins.extend(b_win)

    data = data.add_column("logits", all_logits)
    data = data.add_column("a_win", a_wins)
    data = data.add_column("b_win", b_wins)

    all_logits_swap = []
    a_wins_swap = []
    b_wins_swap = []
    with torch.inference_mode():
        for input_ids in tqdm(batched(data_swap["input_ids"], batch_size), total=math.ceil(len(data) / batch_size)):
            batch = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
            output = model(**batch.to(model.device))
            logits = output.logits.cpu()
            probs = logits.softmax(-1)
            a_win = probs[:, 0].tolist()
            b_win = probs[:, 1].tolist()
            a_wins_swap.extend(a_win)
            b_wins_swap.extend(b_win)
            all_logits_swap.extend(logits.tolist())

    data_swap = data_swap.add_column("a_win", a_wins_swap)
    data_swap = data_swap.add_column("b_win", b_wins_swap)
    data_swap = data_swap.add_column("logits", all_logits_swap)

    data = data.add_column("swap", [False] * len(data))
    data_swap = data_swap.add_column("swap", [True] * len(data_swap))

    data = concatenate_datasets([data, data_swap])
    data = data.remove_columns(["input_ids"])
    data.to_parquet(f"{prefix}_shard{args.rank}.parquet")

    def merge_shards(prefix_path):
        all_shards_present = all(os.path.exists(f"{prefix_path}_shard{i}.parquet") for i in range(args.world_size))

        if all_shards_present:
            print("All shards are present. Merging...")
            time.sleep(10)
            dfs = [Dataset.from_parquet(f"{prefix_path}_shard{i}.parquet") for i in range(args.world_size)]
            merged = concatenate_datasets(dfs)

            merged_path = f"{prefix_path}_{nhuman(len(merged))}.parquet"
            merged.to_parquet(merged_path)
            print(f"Merged dataset saved to {merged_path}")

            for i in range(args.world_size):
                os.remove(f"{prefix_path}_shard{i}.parquet")
            print("Individual shards removed")
        else:
            print("Not all shards are present yet. Skipping merge.")

    merge_shards(prefix)
    print(f"Took: {(time.time() - stime)/60:.0f}m")

