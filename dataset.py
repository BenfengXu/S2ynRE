import json
import codecs
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
import math

import torch
from torch.utils.data import Dataset, Sampler


class SemevalDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        data_file,
        max_seq_length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[unused0]", "[unused1]", "[unused2]", "[unused3]"]})
        self.data = []
        with open(os.path.join(data_dir, data_file), 'r') as f:
            read_lines = f.readlines()
            for line in read_lines:
                self.data.append(json.loads(line.strip()))
        self.rel2id = json.load(open(Path(data_dir).joinpath("rel2id.json")))
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def tokenize_list_of_words(self, list_of_words):
        tokens = []
        for word in list_of_words:
            tokens = tokens + self.tokenizer.tokenize(word)
        return tokens

    def __getitem__(self, index) -> Dict[str, Union[List[int], torch.Tensor]]:
        ins = self.data[index]
        sub_start, sub_end = ins['h']['pos']
        obj_start, obj_end = ins['t']['pos']

        input_tokens = [self.tokenizer.cls_token]
        input_tokens = input_tokens + self.tokenize_list_of_words(ins['token'][:sub_start])
        sub_start_marker_pos = len(input_tokens)
        input_tokens = input_tokens + self.tokenize_list_of_words(['[unused0]'] + ins['token'][sub_start:sub_end] + ['[unused1]'] + ins['token'][sub_end:obj_start])
        obj_start_marker_pos = len(input_tokens)
        input_tokens = input_tokens + self.tokenize_list_of_words(['[unused2]'] + ins['token'][obj_start:obj_end] + ['[unused3]'] + ins['token'][obj_end:])

        if len(input_tokens) >= self.max_seq_length - 1:
            input_tokens = input_tokens[:self.max_seq_length - 1]
            sub_start_marker_pos = sub_start_marker_pos if sub_start_marker_pos < self.max_seq_length - 1 else 0
            obj_start_marker_pos = sub_start_marker_pos if obj_start_marker_pos < self.max_seq_length - 1 else 0
        input_tokens = input_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        labels = self.rel2id[ins['relation']]

        return {
            "input_ids": input_ids,
            "sub_start_marker_pos": sub_start_marker_pos,
            "obj_start_marker_pos": obj_start_marker_pos,
            "labels": labels
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("You must implement this")


class TACREDDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        data_file,
        max_seq_length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[unused0]", "[unused1]", "[unused2]", "[unused3]"]})
        self.data = []
        with open(os.path.join(data_dir, data_file), 'r') as f:
            read_lines = f.readlines()
            for line in read_lines:
                self.data.append(json.loads(line.strip()))
        self.rel2id = json.load(open(Path(data_dir).joinpath("rel2id.json")))
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def tokenize_list_of_words(self, list_of_words):
        tokens = []
        for word in list_of_words:
            tokens = tokens + self.tokenizer.tokenize(word)
        return tokens

    def __getitem__(self, index) -> Dict[str, Union[List[int], torch.Tensor]]:
        ins = self.data[index]
        sub_start, sub_end = ins['h']['pos']
        obj_start, obj_end = ins['t']['pos']

        input_tokens = [self.tokenizer.cls_token]
        if sub_start <= obj_start:
            input_tokens = input_tokens + self.tokenize_list_of_words(ins['token'][:sub_start])
            sub_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(['[unused0]'] + ins['token'][sub_start:sub_end] + ['[unused1]'] + ins['token'][sub_end:obj_start])
            obj_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(['[unused2]'] + ins['token'][obj_start:obj_end] + ['[unused3]'] + ins['token'][obj_end:])
        else:
            input_tokens = input_tokens + self.tokenize_list_of_words(ins['token'][:obj_start])
            obj_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(['[unused2]'] + ins['token'][obj_start:obj_end] + ['[unused3]'] + ins['token'][obj_end:sub_start])
            sub_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(['[unused0]'] + ins['token'][sub_start:sub_end] + ['[unused1]'] + ins['token'][sub_end:])

        if len(input_tokens) >= self.max_seq_length - 1:
            input_tokens = input_tokens[:self.max_seq_length - 1]
            sub_start_marker_pos = sub_start_marker_pos if sub_start_marker_pos < self.max_seq_length - 1 else 0
            obj_start_marker_pos = obj_start_marker_pos if obj_start_marker_pos < self.max_seq_length - 1 else 0
        input_tokens = input_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        labels = self.rel2id[ins['relation']]

        return {
            "input_ids": input_ids,
            "sub_start_marker_pos": sub_start_marker_pos,
            "obj_start_marker_pos": obj_start_marker_pos,
            "labels": labels
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("You must implement this")


class Wiki80Dataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        data_file,
        max_seq_length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[unused0]", "[unused1]", "[unused2]", "[unused3]"]})
        self.data = []
        with open(os.path.join(data_dir, data_file), 'r') as f:
            read_lines = f.readlines()
            for line in read_lines:
                self.data.append(json.loads(line.strip()))
        self.rel2id = json.load(open(Path(data_dir).joinpath("rel2id.json")))
        self.id2rel = json.load(open(Path(data_dir).joinpath("id2rel.json")))
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def tokenize_list_of_words(self, list_of_words):
        tokens = []
        for word in list_of_words:
            tokens = tokens + self.tokenizer.tokenize(word)
        return tokens

    def __getitem__(self, index) -> Dict[str, Union[List[int], torch.Tensor]]:
        ins = self.data[index]
        sub_start, sub_end = ins['h']['pos']
        obj_start, obj_end = ins['t']['pos']

        input_tokens = [self.tokenizer.cls_token]
        if sub_start <= obj_start:
            input_tokens = input_tokens + self.tokenize_list_of_words(ins['token'][:sub_start])
            sub_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(['[unused0]'] + ins['token'][sub_start:sub_end] + ['[unused1]'] + ins['token'][sub_end:obj_start])
            obj_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(['[unused2]'] + ins['token'][obj_start:obj_end] + ['[unused3]'] + ins['token'][obj_end:])
        else:
            input_tokens = input_tokens + self.tokenize_list_of_words(ins['token'][:obj_start])
            obj_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(
                ['[unused0]'] + ins['token'][obj_start:obj_end] + ['[unused1]'] + ins['token'][obj_end:sub_start])
            sub_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(
                ['[unused2]'] + ins['token'][sub_start:sub_end] + ['[unused3]'] + ins['token'][sub_end:])

        if len(input_tokens) >= self.max_seq_length - 1:
            input_tokens = input_tokens[:self.max_seq_length - 1]
            sub_start_marker_pos = sub_start_marker_pos if sub_start_marker_pos < self.max_seq_length - 1 else 0
            obj_start_marker_pos = sub_start_marker_pos if obj_start_marker_pos < self.max_seq_length - 1 else 0
        input_tokens = input_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        labels = self.rel2id[ins['relation']]

        return {
            "input_ids": input_ids,
            "sub_start_marker_pos": sub_start_marker_pos,
            "obj_start_marker_pos": obj_start_marker_pos,
            "labels": labels
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("You must implement this")


class SynseticDataGPT2(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        data_file,
        max_seq_length,
        iteration=None,
        iteration_total=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[unused0]", "[unused1]", "[unused2]", "[unused3]"]})

        self.data = []
        with codecs.open(os.path.join(data_dir, data_file), 'r') as f:
            for line in f.readlines():
                self.data.append(json.loads(line.strip()))
        self.max_seq_length = max_seq_length

        if iteration is not None:
            per_iter = int(len(self.data) / iteration_total)
            self.data = self.data[: iteration * per_iter]
            # self.data = self.data[(iteration - 1) * per_iter: iteration * per_iter]

    def __len__(self):
        return len(self.data)

    def tokenize_list_of_words(self, list_of_words):
        tokens = []
        for word in list_of_words:
            tokens = tokens + self.tokenizer.tokenize(word)
        return tokens

    def __getitem__(self, index) -> Dict[str, Union[List[int], torch.Tensor]]:
        ins = self.data[index]
        sub_start, sub_end = ins['h']['pos']
        obj_start, obj_end = ins['t']['pos']

        input_tokens = [self.tokenizer.cls_token]
        if sub_start <= obj_start:
            input_tokens = input_tokens + self.tokenize_list_of_words(ins['token'][:sub_start])
            sub_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(['[unused0]'] + ins['token'][sub_start:sub_end] + ['[unused1]'] + ins['token'][sub_end:obj_start])
            obj_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(['[unused2]'] + ins['token'][obj_start:obj_end] + ['[unused3]'] + ins['token'][obj_end:])
        else:
            input_tokens = input_tokens + self.tokenize_list_of_words(ins['token'][:obj_start])
            obj_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(
                ['[unused0]'] + ins['token'][obj_start:obj_end] + ['[unused1]'] + ins['token'][obj_end:sub_start])
            sub_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(
                ['[unused2]'] + ins['token'][sub_start:sub_end] + ['[unused3]'] + ins['token'][sub_end:])

        if len(input_tokens) >= self.max_seq_length - 1:
            input_tokens = input_tokens[:self.max_seq_length - 1]
            sub_start_marker_pos = sub_start_marker_pos if sub_start_marker_pos < self.max_seq_length - 1 else 0
            obj_start_marker_pos = sub_start_marker_pos if obj_start_marker_pos < self.max_seq_length - 1 else 0
        input_tokens = input_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        return {
            "input_ids": input_ids,
            "sub_start_marker_pos": sub_start_marker_pos,
            "obj_start_marker_pos": obj_start_marker_pos
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("You must implement this")


class SynseticDataDistill(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        data_file,
        pseudo_labels_dir,
        max_seq_length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[unused0]", "[unused1]", "[unused2]", "[unused3]"]})

        self.data = []
        with codecs.open(os.path.join(data_dir, data_file), 'r') as f:
            for line in f.readlines():
                self.data.append(json.loads(line.strip()))
        self.max_seq_length = max_seq_length
        distilled_logits_files = os.listdir(pseudo_labels_dir)
        self.distilled_logits = []
        for i in range(len(distilled_logits_files)):
            self.distilled_logits.append(torch.load(os.path.join(pseudo_labels_dir, distilled_logits_files[i])))
        self.data = self.data[:len(self.distilled_logits[0])]
        # self.data, self.labels_hard, self.labels = self.distill_strategy(self.data, self.distilled_logits)
        self.data, self.labels = self.distill_strategy(self.data, self.distilled_logits)

    def __len__(self):
        return len(self.data)

    def tokenize_list_of_words(self, list_of_words):
        tokens = []
        for word in list_of_words:
            tokens = tokens + self.tokenizer.tokenize(word)
        return tokens

    # Multi-Teacher
    def distill_strategy(self, data, distilled_logits):
        return data, torch.stack(distilled_logits, dim=1)

    def __getitem__(self, index) -> Dict[str, Union[List[int], torch.Tensor]]:
        labels = self.labels[index]
        # labels_hard = self.labels_hard[index]
        ins = self.data[index]
        sub_start, sub_end = ins['h']['pos']
        obj_start, obj_end = ins['t']['pos']

        input_tokens = [self.tokenizer.cls_token]
        if sub_start <= obj_start:
            input_tokens = input_tokens + self.tokenize_list_of_words(ins['token'][:sub_start])
            sub_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(['[unused0]'] + ins['token'][sub_start:sub_end] + ['[unused1]'] + ins['token'][sub_end:obj_start])
            obj_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(['[unused2]'] + ins['token'][obj_start:obj_end] + ['[unused3]'] + ins['token'][obj_end:])
        else:
            input_tokens = input_tokens + self.tokenize_list_of_words(ins['token'][:obj_start])
            obj_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(
                ['[unused0]'] + ins['token'][obj_start:obj_end] + ['[unused1]'] + ins['token'][obj_end:sub_start])
            sub_start_marker_pos = len(input_tokens)
            input_tokens = input_tokens + self.tokenize_list_of_words(
                ['[unused2]'] + ins['token'][sub_start:sub_end] + ['[unused3]'] + ins['token'][sub_end:])

        if len(input_tokens) >= self.max_seq_length - 1:
            input_tokens = input_tokens[:self.max_seq_length - 1]
            sub_start_marker_pos = sub_start_marker_pos if sub_start_marker_pos < self.max_seq_length - 1 else 0
            obj_start_marker_pos = sub_start_marker_pos if obj_start_marker_pos < self.max_seq_length - 1 else 0
        input_tokens = input_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        return {
            "input_ids": input_ids,
            "sub_start_marker_pos": sub_start_marker_pos,
            "obj_start_marker_pos": obj_start_marker_pos,
            "labels": labels,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("You must implement this")


class DatasetForGPT(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        data_file,
        max_seq_length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        with open(os.path.join(data_dir, data_file), 'r') as f:
            read_lines = f.readlines()
            for line in read_lines:
                self.data.append(json.loads(line.strip()))
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, Union[List[int], torch.Tensor]]:
        ins = self.data[index]
        sub_start, sub_end = ins['h']['pos']
        obj_start, obj_end = ins['t']['pos']
        if sub_start <= obj_start:
            augmented_text = '<|endoftext|>' + ' '.join(ins['token'][:sub_start])\
                             + '[unused0]' + ' '.join(ins['token'][sub_start:sub_end])\
                             + '[unused1]' + ' '.join(ins['token'][sub_end:obj_start])\
                             + '[unused2]' + ' '.join(ins['token'][obj_start:obj_end])\
                             + '[unused3]' + ' '.join(ins['token'][obj_end:]) + '<|endoftext|>'
        else:
            augmented_text = '<|endoftext|>' + ' '.join(ins['token'][:obj_start]) \
                             + '[unused2]' + ' '.join(ins['token'][obj_start:obj_end]) \
                             + '[unused3]' + ' '.join(ins['token'][obj_end:sub_start]) \
                             + '[unused0]' + ' '.join(ins['token'][sub_start:sub_end]) \
                             + '[unused1]' + ' '.join(ins['token'][sub_end:]) + '<|endoftext|>'
        inputs = self.tokenizer(augmented_text, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'][0]
        inputs['attention_mask'] = inputs['attention_mask'][0]
        if len(inputs['input_ids']) > self.max_seq_length:
            inputs['input_ids'] = inputs['input_ids'][:self.max_seq_length]
            inputs['attention_mask'] = inputs['attention_mask'][:self.max_seq_length]

        return {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask']
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("You must implement this")



class Collator_synthetic_distill:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        labels = [feature['labels'] for feature in features]
        labels = torch.stack(labels).clone()
        for feature in features:
            del feature['labels']
        batch = self.tokenizer.pad(
            features,
            padding='longest',
            return_attention_mask=True,
            return_tensors='pt',
        )
        batch['labels'] = labels
        return batch


class CollatorForGPT:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        batch = self.tokenizer.pad(
            features,
            padding='longest',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return batch


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding='longest',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return batch