from __future__ import absolute_import, division, print_function
import csv
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
# from prefetch_generator import BackgroundGenerator
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from SCmodel_test import Model
from tqdm import tqdm, trange
from itertools import cycle
import multiprocessing
from imblearn.over_sampling import RandomOverSampler
from transformers.data.data_collator import DataCollatorForLanguageModeling
import MMD as MMD
from parser.DFG import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from sklearn.model_selection import train_test_split
from parser.utils import (remove_comments_and_docstrings,
                          tree_to_token_index,
                          index_to_code_token,
                          tree_to_variable_index)
from tree_sitter import Language, Parser
from preproccessing import pre_proccessing, dataset_Peng_Qian, dataset_Peng_Qian_2

logger = logging.getLogger(__name__)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


lll = 2

dfg_function = {
    # 'python':DFG_python,
    'java': DFG_java,
    # 'ruby':DFG_ruby,
    # 'go':DFG_go,
    # 'php':DFG_php,
    # 'javascript':DFG_javascript
}

parsers = {}
parsers['java'] = Parser()
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


class Example(object):
    def __init__(self, label, source):
        self.label = label
        self.source = source


def read_examples(args=None):
    examples = []

    if args.SWCsType in ["TP", "BN", "DE", "EF", "UC", "RE", "OF", "SE"]:
        codes, Labels = dataset_Peng_Qian_2('../SmartContracts/Dataset',
                                          SWCsType=args.SWCsType)

    for code, label in zip(codes, Labels):
        examples.append(Example(label=label, source=code))

    return examples


class InputFeatures(object):
    def __init__(self, label, source_ids, source_mask, position_idx):
        self.label = label
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.position_idx = position_idx


def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


def convert_examples_to_features(examples, tokenizer, args):
    features = []
    for idx, example in tqdm(enumerate(examples), total=len(examples)):
        # source_tokens = tokenizer.tokenize(example.source)
        source_tokens, dfg = extract_dataflow(example.source, parser, 'java')
        source_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                         enumerate(source_tokens)]
        # source_tokens = [item for item in source_tokens if item != 'Ġ']
        source_tokens = [y for x in source_tokens for y in x][:args.max_source_length * lll - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length * lll - len(source_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example.label,
                source_ids,
                source_mask,
                position_idx
            )
        )
    return features


def main(trainname, testname, head, SWCType):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default="codebert-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default="Output", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default="dataset/train/java_train.jsonl", type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="codebert-base", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="codebert-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=512, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=16, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", default=True, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=False, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--n_gpu", default=1, action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=200000, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--head', default=head,
                        help="head")
    parser.add_argument('--SWCsType', default=SWCType, type=str,
                        help=["timestamp", "reentrancy", "integeroverflow", "delegatecall"])
    # print arguments
    args = parser.parse_args()
    if args.SWCsType in ["TP", "BN", "DE", "EF", "UC", "RE", "OF", "SE"]:
        args.num_train_epochs = 50
    logger.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    set_seed(args)
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    if head == "Original":
        from transformers import RobertaModel
        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
        config = config_class.from_pretrained("codebert-base")
        tokenizer = tokenizer_class.from_pretrained("codebert-base", maxlength=512)
        student = RobertaModel(config=config)
        student.load_state_dict(torch.load("codebert-base/pytorch_model" + ".bin"), strict=False)
        head = "Original"
    else:
        from transformer.modeling_roberta import RobertaModel
        head = args.head
        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
        config = config_class.from_pretrained("pretrain-solidity/h" + str(head) + "_l12_f3072_e768")
        tokenizer = tokenizer_class.from_pretrained("pretrain-solidity/h" + str(head) + "_l12_f3072_e768", maxlength=512)
        student = RobertaModel(config=config)
        if os.path.exists("pretrain-solidity/h" + str(head) + "_l12_f3072_e768/tencoder_" + str(head) + "_50000" + ".bin"):
            student.load_state_dict(torch.load("pretrain-solidity/h" + str(head) + "_l12_f3072_e768/tencoder_" + str(head) + "_50000" + ".bin"), strict=False)
        else:
            student.load_state_dict(torch.load("pretrain-solidity/h" + str(head) + "_l12_f3072_e768/tencoder_" + str(head) + "_10000" + ".bin"), strict=False)
    model = Model(student, config)
    model.to(device)

    data_examples = read_examples(args)
    data_features = convert_examples_to_features(data_examples, tokenizer, args)
    data_ids = torch.tensor([f.source_ids for f in data_features], dtype=torch.long)
    data_mask = torch.tensor([f.source_mask for f in data_features], dtype=torch.long)
    data_pos = torch.tensor([f.position_idx for f in data_features], dtype=torch.long)
    data_label = torch.tensor([f.label for f in data_features], dtype=torch.float)
    sample = torch.cat((data_ids, data_mask), dim=1)
    sample = torch.cat((sample, data_pos), dim=1)
    sample = np.array(sample)
    labels = np.array(data_label)
    train_data, test_data, train_labels, test_labels = train_test_split(sample, labels, test_size=0.3)

    sample = torch.tensor(train_data)
    train_labels = torch.tensor(train_labels)
    randover = RandomOverSampler()
    sample, train_labels = randover.fit_resample(X=sample, y=train_labels)

    train_ids = torch.tensor(sample[:, 0:args.max_source_length * lll])
    train_mask = torch.tensor(sample[:, args.max_source_length * lll:args.max_source_length * lll * 2])
    train_pos = torch.tensor(sample[:, args.max_source_length * lll * 2:args.max_source_length * lll * 3])
    train_label = torch.tensor(train_labels)

    train_data = TensorDataset(train_ids, train_mask, train_pos, train_label)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size // args.gradient_accumulation_steps, num_workers=0)

    sample = torch.tensor(test_data)
    test_ids = torch.tensor(sample[:, 0:args.max_source_length * lll])
    test_mask = torch.tensor(sample[:, args.max_source_length * lll:args.max_source_length * lll * 2])
    test_pos = torch.tensor(sample[:, args.max_source_length * lll * 2:args.max_source_length * lll * 3])
    test_label = torch.tensor(test_labels)

    test_data = TensorDataset(test_ids, test_mask, test_pos, test_label)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler,
                                 batch_size=args.train_batch_size // args.gradient_accumulation_steps, num_workers=0)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(0.9, 0.99))
    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    model.zero_grad()
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    global_step = 0
    EarlyStopping = False

    for idx in range(args.num_train_epochs):
        model.train()
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for step, batch in enumerate(bar):

            source_ids, source_mask, source_pos, source_label = [x.to(device) for x in batch]
            loss = model(source_ids, source_mask, source_pos, source_label)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                # avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

        model.eval()
        logits = []
        y_trues = []
        result = []
        if (idx + 1) % 5 == 0:
            eval_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
            for batch in eval_dataloader:
                source_ids, source_mask, source_pos, source_label = [x.to(device) for x in batch]
                with torch.no_grad():
                    logit = model(source_ids, source_mask, source_pos)
                logits.append(logit.cpu().detach().numpy())
                y_trues.append(source_label.cpu().detach().numpy())

            logits = np.concatenate(logits, 0)
            y_trues = np.concatenate(y_trues, 0)
            best_threshold = 0.5
            best_f1 = 0

            y_preds = logits[:] > best_threshold
            from sklearn.metrics import recall_score
            recall = recall_score(y_trues, y_preds)
            from sklearn.metrics import precision_score
            precision = precision_score(y_trues, y_preds)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_trues, y_preds)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_trues, y_preds)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_trues, logits)
            from sklearn.metrics import matthews_corrcoef
            mcc = matthews_corrcoef(y_trues, y_preds)
            result = {
                "train": 0,
                'test': 0,
                "eval_f1": float(f1),
                "eval_acc": float(acc),
                "eval_auc": float(auc),
                "eval_mcc": float(mcc),
                "eval_recall": float(recall),
                "eval_precision": float(precision),
                "eval_precision": float(precision),
                "eval_threshold": best_threshold,
                "epoch": idx+1,
                "Type": args.SWCsType,
                "heads": head

            }

            print("***** Eval results *****")
            print(trainname + '--' + testname)
            for key in sorted(result.keys()):
                print("  %s = %s", key, str(result[key]))
            df = pd.DataFrame(
                columns=['train', 'test', 'eval_f1', 'eval_acc', 'eval_auc', 'eval_mcc', 'eval_precision',
                         'eval_recall', 'epoch', 'head',
                         'Type', ]).from_dict(data=result, orient='index').T
            save_path = 'Output-solidity/SWCs_Pruning_' + str(idx + 1) + '.csv'
            # 判断文件是否存在
            if os.path.exists(save_path):
                df.to_csv(save_path, mode='a', header=False, index=False)
            else:
                df.to_csv(save_path, mode='w', index=False)
        if (idx + 1) % args.num_train_epochs == 0:
            df = pd.DataFrame(
                columns=['train', 'test', 'eval_f1', 'eval_acc', 'eval_auc', 'eval_mcc', 'eval_precision',
                         'eval_recall', 'epoch', 'head',
                         'Type', ]).from_dict(data=result, orient='index').T
            save_path = 'Output-solidity/SWCs_Pruning' + '.csv'
            # 判断文件是否存在
            if os.path.exists(save_path):
                df.to_csv(save_path, mode='a', header=False, index=False)
            else:
                df.to_csv(save_path, mode='w', index=False)
    return result


if __name__ == "__main__":
    SWCsType = ["TP", "BN", "DE", "EF", "UC", "RE", "OF", "SE"]
    heads = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    for head in heads:
        for SWCType in SWCsType:
            main(trainname=projectname[0] + '', testname=projectname[1] + '_OverSample')
            result = main(trainname=projectname[0] + '', testname=projectname[1][0], head=head, SWCType=SWCType)
