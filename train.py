# -*- coding: utf-8 -*-
import json
import os
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial

import torch
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import Dataset, DatasetDict
from src.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

tqdm = partial(tqdm, ncols=0, leave=False)

TIMEOUT = 2
instruction = None
cot_trigger = None
answer_trigger = None


def setup_cot():
    global instruction
    global cot_trigger
    global answer_trigger
    # Complete output is in this form: f'{instruction}{question.strip()}{cot_trigger}{answer_cot.strip()}'
    instruction = "Below is a math problem, please give a step-by-step answer.\n\n### Question:\n"
    cot_trigger = "\n\n### Your step-by-step answer:\n"
    return


def prepare_datasets_and_data_loaders(args, tokenizer):
    with accelerator.main_process_first():
        raw_dataset = DatasetDict(
            {
                "train": Dataset.from_list(json.load(open(args["train_file"], "r"))),
            }
        )
        accelerator.print("Raw data:", raw_dataset)
        setup_cot()
        accelerator.print("Using instruction:", instruction)
        accelerator.print("Using cot_trigger:", cot_trigger)
        accelerator.print("Using answer_trigger:", answer_trigger)

        def tokenize_fn(batch, args, tokenizer):
            new_batch = defaultdict(list)
            all_keys = list(batch.keys())
            for item_values in zip(*(batch[k] for k in all_keys)):
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                question, answer_value, answer_cot = (
                    item["question"],
                    item["answer_value"],
                    item.get("answer_cot", None),
                )
                question = question.strip()
                if answer_value is not None:
                    answer_value = answer_value.strip()

                if answer_cot is not None:
                    answer_cot = answer_cot.strip()

                input = f"{instruction}{question}{cot_trigger}"
                output = f"{answer_cot}"
                prefix_text = f"{instruction}{question}{cot_trigger}"

                input_encode = tokenizer(input, add_special_tokens=False)
                output_encode = tokenizer(output, add_special_tokens=False)
                prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

                input_ids = input_encode["input_ids"] + output_encode["input_ids"] + [tokenizer.eos_token_id]
                labels = (
                    [-100] * len(input_encode["input_ids"])
                    + output_encode["input_ids"]
                    + [tokenizer.eos_token_id]
                )
                attention_mask = [1] * len(input_ids)
                prefix = prefix_encode["input_ids"]
                prefix_attention_mask = prefix_encode["attention_mask"]

                # Truncation
                input_ids_max_length = len(input_ids)
                input_ids = input_ids[: args["max_input_length"]]
                labels = labels[: args["max_input_length"]]
                attention_mask = attention_mask[: args["max_input_length"]]
                prefix = prefix[: args["max_input_length"]]
                prefix_attention_mask = prefix_attention_mask[: args["max_input_length"]]

                ##
                new_batch["input_ids"].append(input_ids)
                new_batch["labels"].append(labels)
                new_batch["attention_mask"].append(attention_mask)
                new_batch["prefix"].append(prefix)
                new_batch["prefix_attention_mask"].append(prefix_attention_mask)
                ##
                new_batch["question"].append(question)
                new_batch["answer_cot"].append(answer_cot)
                new_batch["answer_value"].append(answer_value)
                new_batch["input_ids_max_length"].append(input_ids_max_length)

            return new_batch

        tokenized_dataset = DatasetDict(
            {
                mode: dataset.map(
                    tokenize_fn,
                    fn_kwargs={"args": args, "tokenizer": tokenizer},
                    batched=True,
                    remove_columns=dataset.column_names,
                    num_proc=8,
                    load_from_cache_file=False,
                )
                for mode, dataset in raw_dataset.items()
            }
        )
        accelerator.print("Processed data:", tokenized_dataset)
        for mode, dataset in tokenized_dataset.items():
            accelerator.print(mode, f"{mode}_input_ids_max_length", max(dataset["input_ids_max_length"]))

        if accelerator.is_main_process and args["wandb_log"]:
            wandb.config.update(
                {
                    "instruction": instruction,
                    "cot_trigger": cot_trigger,
                    "answer_trigger": answer_trigger,
                    "raw_dataset": str(raw_dataset),
                    "tokenized_dataset": str(tokenized_dataset),
                    "train_input_ids_max_length": max(tokenized_dataset["train"]["input_ids_max_length"]),
                }
            )

    def collate_fn(batch, args, tokenizer):
        max_input_length = max([len(item["input_ids"]) for item in batch])
        max_target_length = max([len(item["labels"]) for item in batch])
        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            # right pad
            input_ids.append(
                item["input_ids"] + [tokenizer.pad_token_id] * (max_input_length - len(item["input_ids"]))
            )
            attention_mask.append(
                item["attention_mask"] + [0] * (max_input_length - len(item["attention_mask"]))
            )
            labels.append(item["labels"] + [-100] * (max_target_length - len(item["labels"])))

        forward_kwargs = {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.BoolTensor(attention_mask),
            "labels": torch.LongTensor(labels),
        }
        return {"forward_kwargs": forward_kwargs}

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        pin_memory=True,
        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer),
    )

    return tokenized_dataset["train"], train_dataloader


def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        save_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    tokenizer.save_pretrained(save_path)
    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        most_recent_ckpts_paths.append(save_path)
        if args["keep_num_ckpt"] is not None and len(most_recent_ckpts_paths) > args["keep_num_ckpt"]:
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            shutil.rmtree(ckpt_to_be_removed)


def train_one_epoch(
    args,
    model,
    train_dataloader,
    optimizer,
    scheduler,
    global_step,
    prefix,
    epoch,
    best_eval_log_dict,
):
    clip_grad_norm = args.get("clip_grad_norm", None)
    logging_step_freq = args.get("logging_step_freq", None)
    model.train()
    epoch_result_dict = defaultdict(list)
    with tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        disable=not accelerator.is_main_process,
        desc="Train Loop",
    ) as t:
        for idx, batch in t:
            with accelerator.accumulate(model):
                output = model(**batch["forward_kwargs"])
                # Get some metrics
                loss = output[0]
                result_dict, extra = {}, None
                # Update
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()

            if accelerator.sync_gradients:
                global_step += 1
                # Step update metric
                epoch_result_dict["loss"].append(loss.item())
                for k, v in result_dict.items():
                    epoch_result_dict[k].append(v)

                # Step logging
                train_log_dict = {}
                if logging_step_freq is not None and global_step % logging_step_freq == 0:
                    train_log_dict = {
                        f"T.{k}": sum(v) / len(v) if isinstance(v, list) else v
                        for k, v in epoch_result_dict.items()
                    }

                if train_log_dict:
                    log_dict = {
                        "lr": scheduler.get_last_lr()[0],
                        **train_log_dict,
                        **best_eval_log_dict,
                    }
                    if accelerator.is_main_process and args["wandb_log"]:
                        wandb.log(log_dict, step=global_step)
                        log_dict = {"wandb": args["wandb_project"] + "|" + args["wandb_run_name"], **log_dict}
                    log_dict = {k: f"{v:.5g}" if isinstance(v, float) else v for k, v in log_dict.items()}
                    accelerator.print(f"{prefix}[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")

                # Keep only max_record items
                for k, v in epoch_result_dict.items():
                    if len(v) > 1:
                        epoch_result_dict[k] = v[-1:]

    # Metric summary:
    epoch_result_dict = {
        k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()
    }
    return epoch_result_dict, global_step


def main(args):
    set_seed(args["seed"] + accelerator.process_index)
    # os.environ["WANDB_MODE"] = "offline"
    if torch.distributed.get_rank() == 0 and args["wandb_log"]:
        wandb.init(project=args["wandb_project"], name=args["wandb_run_name"])
        wandb.config.update(args)

    tokenizer = AutoTokenizer.from_pretrained(args["tokenizer_name_or_path"], use_fast=True)
    # set pad_token_id to eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, train_dataloader = prepare_datasets_and_data_loaders(args, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args["model_name_or_path"], low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    accelerator.print(f"[Vocab size]: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    if accelerator.is_main_process and args["wandb_log"]:
        wandb.run.summary.update(
            {
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "vocab_size": len(tokenizer),
            }
        )

    n_epochs = args["n_epochs"]
    num_training_steps = (len(train_dataloader) // accelerator.num_processes * n_epochs) // args[
        "gradient_accumulation_steps"
    ]
    warmup_step = (
        args["warmup_step"]
        if args["warmup_step"] is not None and args["warmup_step"] >= 0
        else int(0.1 * num_training_steps)
    )
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": args["weight_decay"],
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )

    accelerator.print(
        f"***** Running training *****\n"
        f"  Num examples = {len(train_dataset)}\n"
        f"  Num Epochs = {n_epochs}\n"
        f"  Instantaneous batch size per device = {args['batch_size']}\n"
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args['batch_size']*accelerator.num_processes*args['gradient_accumulation_steps']}\n"
        f"  Total optimization steps = {num_training_steps}\n"
        f"  Warm up step: {warmup_step}\n"
        f"  Learning rate: {args['learning_rate']}\n"
    )
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0
    logging_epoch_freq = args["logging_epoch_freq"]
    saving_epoch_freq = args["saving_epoch_freq"]
    model_dir = args["model_dir"]
    best_eval_log_dict = {}
    os.makedirs(model_dir, exist_ok=True)
    most_recent_ckpts_paths = []
    with tqdm(range(1, n_epochs + 1), total=n_epochs, disable=False) as t:
        for epoch in t:
            kwargs = {
                "args": args,
                "model": model,
                "train_dataloader": train_dataloader,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "global_step": global_step,
                "prefix": "",
                "epoch": epoch,
                "best_eval_log_dict": best_eval_log_dict,
            }
            train_epoch_result_dict, global_step = train_one_epoch(**kwargs)

            train_log_dict = {}
            if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                train_log_dict = {
                    f"T.{k}": sum(v) / len(v) if isinstance(v, list) else v
                    for k, v in train_epoch_result_dict.items()
                }

            if train_log_dict:
                log_dict = {
                    "lr": scheduler.get_last_lr()[0],
                    **train_log_dict,
                    **best_eval_log_dict,
                }
                if accelerator.is_main_process and args["wandb_log"]:
                    wandb.log(log_dict, step=global_step)
                    log_dict = {"wandb": args["wandb_project"] + "|" + args["wandb_run_name"], **log_dict}
                log_dict = {k: f"{v:.5g}" if isinstance(v, float) else v for k, v in log_dict.items()}
                accelerator.print(f"[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")

            if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
                save_path = os.path.join(model_dir, f"model")
                do_checkpoint(args, model, tokenizer, save_path)
                if args["keep_num_ckpt"] > 0:
                    save_path = os.path.join(
                        args["model_dir"], f"global_step_{str(global_step)}_epoch_{epoch}"
                    )
                    do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

    return


if __name__ == "__main__":
    from transformers import HfArgumentParser

    NONE_INT = -100
    NONE_STR = "None"

    @dataclass
    class Arguments:
        model_name_or_path: str
        tokenizer_name_or_path: str
        model_dir: str
        train_file: str
        batch_size: int = field(default=4)
        n_epochs: int = field(default=40)
        num_workers: int = field(default=8)
        learning_rate: float = field(default=2e-5)
        weight_decay: float = field(default=1e-6)
        warmup_step: int = field(default=0)
        clip_grad_norm: float = field(default=1)
        logging_epoch_freq: int = field(default=1)
        saving_epoch_freq: int = field(default=1000)
        logging_step_freq: int = field(default=NONE_INT)
        seed: int = field(default=42)
        max_input_length: int = field(default=700)
        gradient_accumulation_steps: int = field(default=1)
        keep_num_ckpt: int = field(default=1)
        # wandb stuff
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default="tmp_anvfupsadfn")
        wandb_run_name: str = field(default="default_run_name")
        ###
        engine: str = field(default="python")

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k, v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    accelerator = Accelerator(
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))],
    )
    accelerator.print(args)
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)
