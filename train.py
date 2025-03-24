import argparse
import hashlib
import math
import os
import sys
from functools import partial
from itertools import batched
from time import time

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, disable_progress_bar
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          get_scheduler, set_seed)

from format import formatting_map
from models import Qwen2ForSequenceClassificationPlus

datasets.disable_caching()
disable_progress_bar()
set_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default="train", type=str)
parser.add_argument("--model_path", default="google/gemma-2-9b-it")
parser.add_argument("--max_length", default=8192, type=int)
parser.add_argument("--lora", default=False, type=bool)
parser.add_argument("--learning_rate", default=2e-6, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--per_device_train_batch_size", default=2, type=int)
parser.add_argument("--per_device_eval_batch_size", default=2, type=int)
parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
parser.add_argument("--datasets_dir", default="data", type=str)
parser.add_argument("--dataset_name", default="new_train_0fold", type=str)
parser.add_argument("--valid_name", default="new_valid_0fold", type=str)
parser.add_argument("--formatting", default="fmt5rev", type=str)
parser.add_argument("--use_all", type=lambda x: x.lower() == "true", default=False)
parser.add_argument("--output_dir", default="ckpts", type=str)
parser.add_argument("--eval_steps", default=1000, type=int)
parser.add_argument("--optim", default="adamw_torch", type=str)
parser.add_argument("--scheduler", default="linear", type=str)
parser.add_argument("--warmup_steps", default=50, type=int)
parser.add_argument("--distillation", type=lambda x: x.lower() == "true", default=False)
parser.add_argument("--distillation_temp", default=1.0, type=float)
parser.add_argument("--distillation_loss_mix", default=0.5, type=float)
parser.add_argument("--arch", default="class-2layers", type=str)
parser.add_argument("--testing", type=lambda x: x.lower() == "true", default=False)
args = parser.parse_args(args=[] if "__file__" not in globals() else sys.argv[1:])

def linear_with_min_lr(current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr: float, max_lr: float):
    """
    Linear schedule:
      1. Warm up from min_lr to max_lr over [0, num_warmup_steps].
      2. Then decay from max_lr back to min_lr over [num_warmup_steps, num_training_steps].
    """
    current_step = max(0, min(current_step, num_training_steps))

    # Warmup phase: from min_lr to max_lr
    if current_step < num_warmup_steps:
        ratio = current_step / max(1, num_warmup_steps)
        return min_lr + ratio * (max_lr - min_lr)
    # Decay phase: from max_lr back to min_lr
    else:
        # How far we are into the decay phase, from 0.0 to 1.0
        ratio = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max_lr - ratio * (max_lr - min_lr)

def main():
    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=args.gradient_accumulation_steps)
    print0 = accelerator.print

    use_all_str = "use_all_" if args.use_all else ""
    global_bs = args.per_device_train_batch_size*args.gradient_accumulation_steps*int(os.environ.get("WORLD_SIZE", "8"))
    experiment = args.experiment + "_" if args.experiment != "train" else ""
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_eos_token = True
    tokenizer.truncation_side = "left"
    if args.arch == "lm":
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer does not have pad token, please add pad token to tokenizer.")

    if args.arch == "class-2layers":
        elif 'qwen' in args.model_path.lower():
            print0("Using Qwen2ForSequenceClassificationPlus model")
            model = Qwen2ForSequenceClassificationPlus.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                num_labels=2
            )
    elif args.arch == "class":
        print0("Using default ForSequenceClassification model")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            num_labels=2
        )
    elif args.arch == "lm":
        print0("Using default ForCausalLM model")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        pos_token = tokenizer(" A", return_tensors="pt").input_ids.item()
        neg_token = tokenizer(" B", return_tensors="pt").input_ids.item()
        print0(f"Pos token: {pos_token}, Neg token: {neg_token}")

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.lora:
        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
            target_modules=find_all_linear_modules(model),
            modules_to_save=["score"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print0("Loading datasets...")
    train_ds = Dataset.from_parquet(f"{args.datasets_dir}/{args.dataset_name}.parquet")
    if args.distillation:
        assert "logits" in train_ds[0], f"No logits in dataset, please check you dataset {args.dataset_name}."
    valid_ds = Dataset.from_parquet(f"{args.datasets_dir}/{args.valid_name}.parquet")
    if "crit" in args.formatting:
        ppe_ds = Dataset.from_parquet(f"{args.datasets_dir}/ppe-crit-10k.parquet")
    else:
        ppe_ds = Dataset.from_parquet(f"{args.datasets_dir}/ppe-10k.parquet")

    if not args.use_all:
        # remove valid dataset from train
        valid_ids = set(valid_ds['id']) | set(ppe_ds['id'])
        train_ds = train_ds.filter(lambda x: x['id'] not in valid_ids)

    if args.testing:
        train_ds = train_ds.select(range(0, 100))
        valid_ds = valid_ds.select(range(0, 100))
        ppe_ds = ppe_ds.select(range(0, 100))

    hash_train = hashlib.sha256("".join(train_ds['id']).encode()).hexdigest()
    hash_valid = hashlib.sha256("".join(valid_ds['id']).encode()).hexdigest()
    hash_ppe = hashlib.sha256("".join(ppe_ds['id']).encode()).hexdigest()

    args.dataset_name = f"{args.dataset_name}_{hash_train[:4]}"
    args.valid_name = f"{args.valid_name}_{hash_valid[:4]}"
    args.ppe_name = f"ppe_{hash_ppe[:4]}"

    run_name = f"{experiment}{args.arch}{use_all_str}{args.dataset_name}_{args.model_path.split('/')[-1]}_lr{args.learning_rate}_{args.distillation_loss_mix}mix_bs{global_bs}_{args.formatting}_{args.valid_name}"
    max_length = args.max_length
    args.run_name = run_name
    args.output_dir = f"{args.output_dir}/{run_name}"

    print0(f"Running experiment: {run_name}")
    print0(f"Use_all: {args.use_all}")
    print0(f"Output directory: {args.output_dir}")

    if os.path.isdir(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        print0(f"Output directory non-empty, this experiment has likely run already, exiting...")
        return

    accelerator.init_trackers(project_name="whitefebruary", config=vars(args), init_kwargs={"wandb": {"name": run_name}})

    def format_ds(ds, max_length):
        if "crit" in args.formatting:
            ds = ds.map(formatting_map[args.formatting], num_proc=1, batched=args.formatting != "fmt0", input_columns=["prompt", "response_a", "response_b", "critique", "critique_swap"], fn_kwargs={"max_length": max_length, "tokenizer": tokenizer}, batch_size=10000)
        elif "models" in args.formatting:
            ds = ds.map(formatting_map[args.formatting], num_proc=1, batched=args.formatting != "fmt0", input_columns=["prompt", "response_a", "response_b", "model_a", "model_b"], fn_kwargs={"max_length": max_length, "tokenizer": tokenizer}, batch_size=10000)
        else:
            ds = ds.map(formatting_map[args.formatting], num_proc=1, batched=args.formatting != "fmt0", input_columns=["prompt", "response_a", "response_b"], fn_kwargs={"max_length": max_length, "tokenizer": tokenizer}, batch_size=10000)
        ds = ds.map(lambda x: {"labels": 0 if x['winner'] == 'model_a' else 1})
        return ds

    train_ds = format_ds(train_ds, max_length=max_length)
    valid_ds = format_ds(valid_ds, max_length=4096)
    ppe_ds = format_ds(ppe_ds, max_length=4096)

    print0(f"Length of train dataset: {len(train_ds)}")
    print0(f"Length of valid dataset: {len(valid_ds)}")

    def collate_fn(batch):
        return {k: [x[k] for x in batch] for k in batch[0].keys()}

    def collate_fn_incl_rev(batch):
        """Collate batch so that ab and ba for each sample are after each other, i.e., sample 1 ab, sample 1 ba, ..."""
        output = {k: [x[k] for x in batch for _ in range(2)] for k in batch[0].keys() if k not in ["input_ids", "labels"]}
        input_ids = [x for b in batch for x in b["input_ids"]]
        labels = [x["labels"] if not i else int(not(x["labels"])) for x in batch for i in range(2)]
        output["input_ids"] = input_ids
        output["labels"] = labels
        return output

    dataloader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, collate_fn=collate_fn_incl_rev, shuffle=True)
    eval_names = ["val", "ppe"]
    eval_dataloaders = [
        DataLoader(valid_ds, batch_size=args.per_device_eval_batch_size, collate_fn=collate_fn_incl_rev, shuffle=False),
        DataLoader(ppe_ds, batch_size=args.per_device_eval_batch_size, collate_fn=collate_fn_incl_rev, shuffle=False)
    ]

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, args.adam_beta2))
    model, opt, dataloader, *eval_dataloaders = accelerator.prepare(model, opt, dataloader, *eval_dataloaders)
    num_training_steps = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.scheduler == "linear_min":
        scheduler = LambdaLR(opt, partial(linear_with_min_lr, min_lr=0.1, max_lr=1.0, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps), last_epoch=-1)
    else:
        scheduler = get_scheduler(args.scheduler, optimizer=opt, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    scheduler = accelerator.prepare(scheduler)

    model.train()
    step = 0
    tbar = trange(math.ceil(len(dataloader) / args.gradient_accumulation_steps), disable=not accelerator.is_main_process)

    def evaluate(model):
        model.eval()

        for eval_name, eval_dataloader in zip(eval_names, eval_dataloaders):
            total_loss_local = 0.0
            total_correct_local = 0
            total_samples_local = 0
            if "rev" in args.formatting:
                 total_correct_local_ab = 0
                 total_correct_local_ba = 0

            for batch in tqdm(eval_dataloader, desc=f"Evaluating on {eval_name}", disable=not accelerator.is_main_process):
                with torch.no_grad():
                    padded = tokenizer.pad({"input_ids": batch["input_ids"]}, return_tensors="pt")
                    input_ids = padded["input_ids"].to(model.device)
                    attn_mask = padded["attention_mask"].to(model.device)
                    if args.arch == "lm":
                        hs = model.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True).hidden_states[-1][:, -1, :]
                        logits = model.lm_head(hs)
                        logits = logits[:, [pos_token, neg_token]]
                    else:
                        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
                    loss = nn.CrossEntropyLoss()(logits, torch.tensor(batch["labels"]).to(model.device))
                    probs = logits.cpu().softmax(-1)

                    if "rev" in args.formatting:
                         # get mean probs of ab and ba samples
                         probs_ab = probs[::2]
                         probs_ba = probs[1::2].flip(-1)  # probs for ba but flipped to have them at the same position as the ab setup
                         probs_mean = torch.stack([probs_ab, probs_ba]).mean(0)
                         preds_ab = probs_ab.argmax(-1).cpu().numpy()
                         preds_ba = probs_ba.argmax(-1).cpu().numpy()
                         preds = probs_mean.argmax(-1).cpu().numpy()
                         labels = np.array(batch["labels"][::2])

                         # for ab and ba here, mean is below
                         correct_ab = preds_ab == labels
                         correct_ba = preds_ba == labels

                         total_correct_local_ab += correct_ab.sum()
                         total_correct_local_ba += correct_ba.sum()
                    else:
                        preds = probs.argmax(-1).cpu().numpy()
                        labels = np.array(batch["labels"])

                    correct = preds == labels

                    total_samples_local += preds.shape[0]
                    total_loss_local += loss.item() * input_ids.shape[0]
                    total_correct_local += correct.sum()

            if "rev" in args.formatting:
                total_loss, total_correct, total_correct_ab, total_correct_ba, total_samples = [
                    accelerator.gather(torch.tensor(x).to(model.device)).sum().item()
                    for x in (total_loss_local, total_correct_local, total_correct_local_ab, total_correct_local_ba, total_samples_local)
                ]
            else:
                total_loss, total_correct, total_samples = [
                    accelerator.gather(torch.tensor(x).to(model.device)).sum().item()
                    for x in (total_loss_local, total_correct_local, total_samples_local)
                ]

            accuracy = total_correct / total_samples
            loss = total_loss / total_samples

            if "rev" in args.formatting:
                accuracy_ab = total_correct_ab / total_samples
                accuracy_ba = total_correct_ba / total_samples

                print0(f"[{eval_name}] acc: {accuracy:.4f}, acc ab: {accuracy_ab:.4f}, acc ba: {accuracy_ba:.4f}, loss: {loss:.4f}, total_samples: {total_samples}")
                accelerator.log({
                    f"eval/{eval_name}-acc": accuracy,
                    f"eval/{eval_name}-acc ab": accuracy_ab,
                    f"eval/{eval_name}-acc ba": accuracy_ba,
                    f"eval/{eval_name}-loss": loss
                }, step=step)
            else:
                print0(f"[{eval_name}] accuracy: {accuracy:.4f}, loss: {loss:.4f}, total_samples: {total_samples}")
                accelerator.log({
                    f"eval/{eval_name}-accuracy": accuracy,
                    f"eval/{eval_name}-loss": loss
                }, step=step)

        model.train()

    if args.distillation:
        divloss_fn = nn.KLDivLoss(reduction="batchmean")
        cosloss_fn = nn.CosineEmbeddingLoss()
        T = args.distillation_temp
        mix_factor = args.distillation_loss_mix

    for batch in dataloader:
        if step % args.eval_steps == 0 and step > 0 and accelerator.sync_gradients:
            evaluate(model)

        with accelerator.accumulate(model):
            stime = time()
            padded = tokenizer.pad({"input_ids": batch["input_ids"]}, return_tensors="pt")
            input_ids = padded["input_ids"].to(model.device)
            attn_mask = padded["attention_mask"].to(model.device)

            if args.arch == 'lm':
                hs = model.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True).hidden_states[-1][:, -1, :]
                logits = model.lm_head(hs)
                logits = logits[:, [pos_token, neg_token]]
            else:
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

            if args.distillation:
                if "rev" in args.formatting:
                    logit_targets = torch.tensor([logits if not i % 2 != 0 else logits[::-1] for i, logits in enumerate(batch["logits"])]).to(model.device)
                else:
                    logit_targets = torch.tensor(batch["logits"]).to(model.device)

                loss_ce = nn.CrossEntropyLoss()(logits, torch.tensor(batch["labels"]).to(model.device))
                loss_div = divloss_fn(F.log_softmax(logits / T, dim=1), F.softmax(logit_targets / T, dim=1))
                loss_cos = cosloss_fn(F.softmax(logits / T, dim=1), F.softmax(logit_targets / T, dim=1), torch.ones(logits.shape[0]).to(model.device))

                loss = (mix_factor * loss_ce + loss_div + loss_cos) / (mix_factor + 2.0)
            else:
                loss = nn.CrossEntropyLoss()(logits, torch.tensor(batch["labels"]).to(model.device))

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                opt.step()
                opt.zero_grad()
                scheduler.step()

                if args.distillation:
                    loss_div = accelerate.utils.reduce(loss_div).item()
                    loss_cos = accelerate.utils.reduce(loss_cos).item()
                    loss_ce = accelerate.utils.reduce(loss_ce).item()
                    loss = accelerate.utils.reduce(loss).item()
                    forward_time = time() - stime

                    tbar.update(1)
                    tbar.set_description(f"Loss: {loss:.4f}, loss_div: {loss_div:.4f}, loss_ce: {loss_ce:.4f}, loss_cos: {loss_cos:.4f}")
                    accelerator.log({"train/loss": loss, "train/loss_div": loss_div, "train/loss_ce": loss_ce, "train/loss_cos": loss_cos, "train/learning_rate": float(scheduler.get_last_lr()[0]), "train/forward_time": forward_time}, step=step)
                else:
                    loss = accelerate.utils.reduce(loss).item()
                    forward_time = time() - stime

                    tbar.update(1)
                    tbar.set_description(f"Loss: {loss:.4f}")
                    accelerator.log({"train/loss": loss, "train/learning_rate": float(scheduler.get_last_lr()[0]), "train/forward_time": forward_time}, step=step)
                step += 1

    evaluate(model)

    accelerator.unwrap_model(model).save_pretrained(
        args.output_dir,
        save_function=accelerator.save,
        is_main_process=accelerator.is_main_process,
        state_dict=accelerator.get_state_dict(model),
    )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        if os.path.exists(args.output_dir + "/model.safetensors.index.json") and os.path.exists(args.output_dir + "/model.safetensors"):
            os.remove(args.output_dir + "/model.safetensors")

    accelerator.print(f"Checkpoint: {args.output_dir}")
    accelerator.end_training()

if __name__ == "__main__":
    main()
