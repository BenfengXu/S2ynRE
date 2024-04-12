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

from dataset import SynseticDataGPT2, Collator
from model import BertForRelationExtraction
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
        "--iteration",
        type=int,
        default=None,
        help="K-th iteration of Semi-supervised Learning.",
    )
    parser.add_argument(
        "--iteration_total",
        type=int,
        default=None,
        help="",
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
        "--do_distill",
        action="store_true",
        help="distill logits on distant data.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--distant_dir",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--distant_file",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--save_results_file",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--save_distilled",
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
        default=256,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
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

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, use_fast=True)

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

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = BertForRelationExtraction.from_pretrained(args.model_name_or_path, config=model_config, num_labels=args.num_labels)

    model.eval()
    distill_dataset = SynseticDataGPT2(tokenizer=tokenizer,
                                       data_dir=args.data_dir,
                                       data_file=args.train_file,
                                       max_seq_length=args.max_seq_length,
                                       iteration=args.iteration,
                                       iteration_total=args.iteration_total)
    data_collator = Collator(tokenizer=tokenizer)
    distill_dataloader = DataLoader(distill_dataset, shuffle=False, collate_fn=data_collator,
                                    batch_size=args.per_device_eval_batch_size)
    model, distill_dataloader = accelerator.prepare(model, distill_dataloader)
    logger.info(f"eval on {len(distill_dataloader) * args.per_device_eval_batch_size} examples.")
    logits_all = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(distill_dataloader), total=len(distill_dataloader)):
            loss, logits = model(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 sub_start_marker_pos=batch['sub_start_marker_pos'],
                                 obj_start_marker_pos=batch['obj_start_marker_pos'],
                                 labels=None)
            logits_all.append(logits.detach().cpu())
        logits_all = torch.cat(logits_all, dim=0)
    os.makedirs(args.save_distilled, exist_ok=True)
    torch.save(logits_all, os.path.join(args.save_distilled, os.path.basename(args.model_name_or_path) + '.pt'))


if __name__ == "__main__":
    main()