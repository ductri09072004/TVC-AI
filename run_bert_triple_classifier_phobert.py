
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import json
import os
import random
import sys
import shutil

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.optim import AdamW

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn import metrics

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

os.environ['CUDA_VISIBLE_DEVICES']= '0'
#torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                # Skip empty/noise lines
                if not line or all((c is None or str(c).strip() == "") for c in line):
                    continue
                lines.append([c.strip() if isinstance(c, str) else c for c in line])
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self):
        self.labels = set([])
        self.data_dir = None
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dataset_svo/train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dataset_svo/dev.txt")), "dev")

    def get_test_examples(self, data_dir):
      """See base class."""
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "dataset_svo/test.txt")), "test")

    def get_relations(self, data_dir):
        """Gets all relations in the knowledge graph."""
        rel_path = os.path.join(data_dir, "dataset_svo/relations.txt")
        if not os.path.exists(rel_path):
            return []
        with open(rel_path, 'r', encoding='utf-8') as f:
            relations = [line.strip() for line in f if line.strip()]
        return relations

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities (union of subjects/objects if needed)."""
        ent_path = os.path.join(data_dir, "dataset_svo/entities.txt")
        if os.path.exists(ent_path):
            with open(ent_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        # Fallback: merge subjects.txt and objects.txt
        entities = set()
        subj_path = os.path.join(data_dir, 'dataset_svo/subjects.txt')
        obj_path = os.path.join(data_dir, 'dataset_svo/objects.txt')
        if os.path.exists(subj_path):
            with open(subj_path, 'r', encoding='utf-8') as f:
                entities.update(line.strip() for line in f if line.strip())
        if os.path.exists(obj_path):
            with open(obj_path, 'r', encoding='utf-8') as f:
                entities.update(line.strip() for line in f if line.strip())
        return sorted(list(entities))

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "dataset_svo/train.txt"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dataset_svo/dev.txt"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "dataset_svo/test.txt"))

    def get_subject2text(self, data_dir):
        path = os.path.join(data_dir, 'dataset_svo/subject2text.txt')
        mapping = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            mapping[parts[0]] = '\t'.join(parts[1:])
                        elif len(parts) == 1:
                            mapping[parts[0]] = parts[0]
            except Exception as e:
                logger.warning(f"Error reading subject2text.txt: {e}")
        return mapping

    def get_object2text(self, data_dir):
        path = os.path.join(data_dir, 'dataset_svo/object2text.txt')
        mapping = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            mapping[parts[0]] = '\t'.join(parts[1:])
                        elif len(parts) == 1:
                            mapping[parts[0]] = parts[0]
            except Exception as e:
                logger.warning(f"Error reading object2text.txt: {e}")
        return mapping

    def get_entity2text(self, data_dir):
        """Gets entity to text mapping."""
        entity2text = {}
        entity2text_path = os.path.join(data_dir, "dataset_svo/entity2text.txt")
        subj2_path = os.path.join(data_dir, "dataset_svo/subject2text.txt")
        obj2_path = os.path.join(data_dir, "dataset_svo/object2text.txt")
        if os.path.exists(entity2text_path):
            try:
                with open(entity2text_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            entity, text = parts[0], '\t'.join(parts[1:])
                            entity2text[entity] = text
                        elif len(parts) == 1:
                            entity2text[parts[0]] = parts[0]
            except Exception as e:
                logger.warning(f"Error reading entity2text.txt: {e}")
        # Fallback: merge subject2text/object2text if present
        if not entity2text and (os.path.exists(subj2_path) or os.path.exists(obj2_path)):
            for p in [subj2_path, obj2_path]:
                if not os.path.exists(p):
                    continue
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                entity2text[parts[0]] = '\t'.join(parts[1:])
                            elif len(parts) == 1:
                                entity2text[parts[0]] = parts[0]
                except Exception as e:
                    logger.warning(f"Error reading fallback entity mapping {p}: {e}")
        if not entity2text:
            logger.warning(f"entity2text mapping not found in {data_dir}, using raw IDs as text")
        return entity2text

    def get_relation2text(self, data_dir):
        """Gets relation to text mapping."""
        relation2text = {}
        relation2text_path = os.path.join(data_dir, "dataset_svo/relation2text.txt")
        if os.path.exists(relation2text_path):
            try:
                with open(relation2text_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            relation, text = parts[0], '\t'.join(parts[1:])
                            relation2text[relation] = text
                        elif len(parts) == 1:
                            relation2text[parts[0]] = parts[0]
            except Exception as e:
                logger.warning(f"Error reading relation2text.txt: {e}")
        else:
            # Fallback: use relations.txt
            rels = self.get_relations(data_dir)
            for r in rels:
                relation2text[r] = r
        return relation2text

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and eval sets."""
        examples = []
        # Lấy entity2text và relation2text một lần (cache để tránh đọc nhiều lần)
        if self.data_dir:
            entity2text = self.get_entity2text(self.data_dir)
            # Role-specific mappings (preferred if available)
            subj2 = self.get_subject2text(self.data_dir)
            obj2 = self.get_object2text(self.data_dir)
            relation2text = self.get_relation2text(self.data_dir)
        else:
            entity2text = {}
            subj2 = {}
            obj2 = {}
            relation2text = {}
        
        for (i, line) in enumerate(lines):
            if len(line) < 4:
                continue  # Bỏ qua dòng không đủ cột
            guid = "%s-%s" % (set_type, i)
            
            # Chuyển đổi entity và relation sang text
            raw_head = line[0] if line[0] else ""
            raw_rel = line[1] if line[1] else ""
            raw_tail = line[2] if line[2] else ""

            # Bỏ qua header nếu vô tình lẫn trong file (subject, verb/relation, object, label)
            header_like = {raw_head.lower(), raw_rel.lower(), raw_tail.lower()}
            if {"subject", "s"} & header_like and {"verb", "relation", "v"} & header_like:
                continue

            # Prefer role-specific text; fallback to entity2text; fallback to raw
            head_text = subj2.get(raw_head) or entity2text.get(raw_head, raw_head)
            relation_text = relation2text.get(raw_rel, raw_rel)
            tail_text = obj2.get(raw_tail) or entity2text.get(raw_tail, raw_tail)
            
            # Kết hợp thành câu + làm sạch 'nan' và khoảng trắng
            def _clean(x: str) -> str:
                s = (x or '').strip()
                return '' if s.lower() == 'nan' else s
            head_text = _clean(head_text)
            relation_text = _clean(relation_text)
            tail_text = _clean(tail_text)

            text_a = f"{head_text} {relation_text} {tail_text}".strip()
            text_a = ' '.join([t for t in text_a.split() if t])
            if not text_a:
                continue  # Bỏ qua nếu không có text
            
            text_b = None
            # Chuẩn hóa nhãn về chuỗi "0"/"1"
            raw_label = line[3] if len(line) > 3 else "0"
            try:
                label = str(int(float(str(raw_label).strip())))
            except Exception:
                # Nếu header/không hợp lệ, bỏ qua dòng
                continue
            
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info = True):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list)}
    
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, weights: torch.Tensor = None) -> torch.Tensor:
    """Binary/multiclass focal loss on logits.
    logits: (N, C), targets: (N,) with class indices
    """
    log_prob = torch.nn.functional.log_softmax(logits, dim=1)
    prob = torch.exp(log_prob)
    n = logits.size(0)
    c = logits.size(1)
    one_hot = torch.zeros_like(logits).scatter_(1, targets.view(-1, 1), 1)
    pt = (one_hot * prob).sum(dim=1)
    loss = -((1 - pt) ** gamma) * (one_hot * log_prob).sum(dim=1)
    if weights is not None:
        class_w = weights.to(logits.device)
        w = (one_hot * class_w.view(1, -1)).sum(dim=1)
        loss = loss * w
    return loss.mean()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=3, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            # Save best model
            torch.save(model.state_dict(), path)
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese, vinai/phobert-base, vinai/phobert-large.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--patience',
                        type=int,
                        default=5,
                        help="Number of epochs to wait for improvement before early stopping (tăng để model học kỹ hơn)")
    parser.add_argument('--min_delta',
                        type=float,
                        default=0.0001,
                        help="Minimum change in the monitored quantity to qualify as an improvement (giảm để nhạy hơn)")
    # New: loss & regularization options
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'label_smoothing', 'focal'],
                        help='Loại loss: ce (mặc định), label_smoothing, focal')
    parser.add_argument('--label_smoothing', type=float, default=0.2, help='Hệ số label smoothing cho CE (tăng để giảm overfitting, default: 0.2)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma cho focal loss')
    parser.add_argument('--class_weights', type=str, default='', help='Trọng số lớp, ví dụ: "1.0,2.0" (cho [0,1])')
    # Regularization options để giảm overfitting
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate cho hidden layers và attention (default: 0.3, tăng để giảm overfitting)')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay cho optimizer (default: 0.1, tăng để giảm overfitting)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping max norm (default: 1.0)')
    # New: threshold tuning
    parser.add_argument('--tune_threshold', action='store_true', help='Tự động tìm ngưỡng tốt nhất trên Dev và lưu ra threshold.json')
    args = parser.parse_args()

    # Default data_dir to ./dataset if not provided or empty
    if not args.data_dir:
        args.data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'dataset'))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "kg": KGProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    processor.data_dir = args.data_dir
    label_list = processor.get_labels()
    num_labels = len(label_list)

    entity_list = processor.get_entities(args.data_dir)

    # Calculate number of training steps
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Load tokenizer and model với dropout tùy chỉnh để giảm overfitting
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    # Tạo config với dropout tùy chỉnh
    config = AutoConfig.from_pretrained(args.bert_model, num_labels=num_labels)
    config.hidden_dropout_prob = args.dropout
    config.attention_probs_dropout_prob = args.dropout
    if hasattr(config, 'classifier_dropout'):
        config.classifier_dropout = args.dropout
    
    logger.info(f"Using dropout: {args.dropout} (hidden_dropout_prob, attention_probs_dropout_prob)")
    
    # Không truyền num_labels vào from_pretrained vì đã có trong config rồi
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, config=config)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    logger.info(f"Using weight_decay: {args.weight_decay} for parameters with decay")
    t_total = num_train_optimization_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=int(t_total * args.warmup_proportion),
                                              num_training_steps=t_total)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, print_info = True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Tạo eval dataloader để dùng cho early stopping (dựa trên dev loss)
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, print_info=False)
        all_eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_eval_input_ids, all_eval_input_mask, all_eval_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, verbose=True)
        best_model_path = os.path.join(args.output_dir, 'best_model.pt')

        # Prepare loss function
        class_w_tensor = None
        if args.class_weights:
            try:
                ws = [float(x) for x in args.class_weights.split(',')]
                if len(ws) == num_labels:
                    class_w_tensor = torch.tensor(ws, dtype=torch.float32)
            except Exception:
                class_w_tensor = None
        # Đưa weight lên đúng device để tránh lỗi CPU/CUDA mismatch
        if class_w_tensor is not None:
            class_w_tensor = class_w_tensor.to(device)

        if args.loss_type == 'label_smoothing':
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_w_tensor)
        elif args.loss_type == 'ce':
            criterion = torch.nn.CrossEntropyLoss(weight=class_w_tensor)
        else:
            criterion = None  # focal uses custom function

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, label_ids = batch
                outputs = model(input_ids, attention_mask=input_mask)
                logits = outputs.logits
                if args.loss_type == 'focal':
                    loss = focal_loss(logits, label_ids, gamma=args.focal_gamma, weights=class_w_tensor)
                else:
                    loss = criterion(logits, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        lr_this_step = args.learning_rate * scheduler.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    # Gradient clipping để tránh gradient explosion và giúp training ổn định
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Calculate average loss for the epoch
            avg_loss = tr_loss / nb_tr_steps
            
            # Evaluate on dev set để early stopping dựa trên dev loss thay vì train loss
            # (giúp tránh overfitting tốt hơn)
            model.eval()
            eval_loss_sum = 0
            eval_steps = 0
            with torch.no_grad():
                for eval_batch in tqdm(eval_dataloader, desc="Eval during training"):
                    eval_batch = tuple(t.to(device) for t in eval_batch)
                    eval_input_ids, eval_input_mask, eval_label_ids = eval_batch
                    eval_outputs = model(eval_input_ids, attention_mask=eval_input_mask)
                    eval_logits = eval_outputs.logits
                    
                    if args.loss_type == 'focal':
                        eval_loss = focal_loss(eval_logits, eval_label_ids, gamma=args.focal_gamma, weights=class_w_tensor)
                    else:
                        eval_loss = criterion(eval_logits, eval_label_ids)
                    eval_loss_sum += eval_loss.mean().item()
                    eval_steps += 1
            
            avg_eval_loss = eval_loss_sum / eval_steps if eval_steps > 0 else avg_loss
            model.train()
            
            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Dev Loss = {avg_eval_loss:.4f}")
            
            # Early stopping check dựa trên dev loss (tốt hơn train loss)
            early_stopping(avg_eval_loss, model, best_model_path)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break

        # Load the best model
        model.load_state_dict(torch.load(best_model_path))
        logger.info("Loaded best model from %s", best_model_path)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # Save model and tokenizer
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, print_info = True)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits_list = []
        all_labels_list = []
        for input_ids, input_mask, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=input_mask)
                logits = outputs.logits
                # Use same loss as training for proper early stopping signal
                if args.loss_type == 'focal':
                    tmp_eval_loss = focal_loss(logits, label_ids, gamma=args.focal_gamma, weights=class_w_tensor)
                else:
                    tmp_eval_loss = (torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_w_tensor)
                                     if args.loss_type == 'label_smoothing'
                                     else torch.nn.CrossEntropyLoss(weight=class_w_tensor))(logits, label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            all_logits_list.append(logits)
            all_labels_list.append(label_ids)

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # Threshold tuning (optional)
        if args.tune_threshold:
            all_logits = np.concatenate(all_logits_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)
            probs_pos = _softmax(all_logits)[:, 1]
            best_t, best_f1 = 0.5, -1.0
            for t in np.linspace(0.3, 0.9, 61):
                preds = (probs_pos >= t).astype(int)
                f1 = f1_score(all_labels, preds, average='binary', zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            thr_path = os.path.join(args.output_dir, 'threshold.json')
            with open(thr_path, 'w') as f:
                json.dump({"best_threshold": round(float(best_t), 4), "dev_f1": round(float(best_f1), 4)}, f, indent=2)
            logger.info("Saved best threshold %.4f (F1=%.4f) to %s", best_t, best_f1, thr_path)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        predict_examples = processor.get_test_examples(args.data_dir)
        predict_features = convert_examples_to_features(predict_examples, label_list, args.max_seq_length, tokenizer, print_info = True)
        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in predict_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in predict_features], dtype=torch.long)
        predict_data = TensorDataset(all_input_ids, all_input_mask)
        # Run prediction for full data
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=args.eval_batch_size)

        model.eval()
        predict_results = []
        for input_ids, input_mask in tqdm(predict_dataloader, desc="Predicting"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=input_mask)
                logits = outputs.logits.detach().cpu().numpy()
                predict_results.append(logits)

        # Concatenate all results
        all_predictions = np.concatenate(predict_results, axis=0)
        
        output_predict_file = os.path.join(args.output_dir, "test_results.tsv")
        with open(output_predict_file, "w") as writer:
            num_written_lines = 0
            logger.info("***** Predict results *****")
            for prediction in all_predictions:
                probabilities = _softmax(prediction)
                output_line = "\t".join(str(class_probability) for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
            logger.info("  Num written lines = %d", num_written_lines)
            logger.info("  Num examples = %d", len(predict_examples))
            assert num_written_lines == len(predict_examples), "Number of written lines (%d) does not match number of examples (%d)" % (num_written_lines, len(predict_examples))

if __name__ == "__main__":
    main()
