#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import csv
import logging
import math
import os
import random

import datasets
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version

from tensorboardX import SummaryWriter
import time

from dataset import SynseticDataDistill, Collator_synthetic_distill
from model import BertForSyntheticPretrain
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--synsetic_data_file",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--synsetic_data_dir",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--pseudo_labels_dir",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--logits_dir",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--distill_temp",
        type=float,
        default=1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.06, help="warm up steps in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--save_model", type=str, default=None, help="name of the final model")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--num_dataloader_worker",
        type=int,
        default=None,
        help="The number of processes to use for the dataloader.",
    )
    parser.add_argument(
        "--num_dataloader_worker_per_device",
        type=int,
        default=1,
        help="The number of processes to use for the dataloader.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="use mixed precision to accelerate training.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(fp16=args.use_fp16)

    # Set logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # set seeds
    if args.seed is not None:
        set_seed(args.seed)

    # prepare dataset
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, use_fast=True)
    train_dataset = SynseticDataDistill(tokenizer=tokenizer,
                                        data_dir=args.synsetic_data_dir,
                                        data_file=args.synsetic_data_file,
                                        pseudo_labels_dir=args.pseudo_labels_dir,
                                        max_seq_length=args.max_seq_length)
    if args.dataset == 'semeval':
        args.num_labels = 19
    elif args.dataset == 'tacred':
        args.num_labels = 42
    elif args.dataset == 'tacredrev':
        args.num_labels = 42
    elif args.dataset == 'retacred':
        args.num_labels = 40
    elif args.dataset == 'chemprot':
        args.num_labels = 13
    elif args.dataset == 'wiki80':
        args.num_labels = 80
    data_collator = Collator_synthetic_distill(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, num_workers=args.num_dataloader_worker_per_device
    )

    # init model
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = BertForSyntheticPretrain.from_pretrained(args.model_name_or_path, config=model_config, num_labels=args.num_labels, temp=args.distill_temp)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    warmup_steps = math.ceil(args.max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if accelerator.is_local_main_process:
        writer = SummaryWriter('output/log')
    completed_steps = 0
    time_log = time.time()
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            loss, logits = model(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 sub_start_marker_pos=batch['sub_start_marker_pos'],
                                 obj_start_marker_pos=batch['obj_start_marker_pos'],
                                 labels=batch["labels"])
            # labels_hard = batch["labels_hard"]
            loss = loss / args.gradient_accumulation_steps

            loss_log = float(loss.detach().cpu().numpy())
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            if accelerator.is_local_main_process:
                writer.add_scalar('loss', loss_log, completed_steps)
                writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], completed_steps)
            if completed_steps % 10 == 0:
                logger.info(f"steps: {completed_steps}/{args.max_train_steps},"
                            f" epoch: {epoch}/{args.num_train_epochs},"
                            f" lr: {lr_scheduler.get_last_lr()[0]:.2e},"
                            f" loss: {loss_log},"
                            f" efficiency: {10 / (time.time() - time_log):.2f}steps/s")
                time_log = time.time()
            if completed_steps >= args.max_train_steps:
                break
            if accelerator.is_local_main_process and completed_steps % args.save_steps == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                save_dir = os.path.join(args.save_model, str(completed_steps))
                unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
                tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()