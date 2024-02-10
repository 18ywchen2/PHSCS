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
                          RobertaConfig, RobertaModel, RobertaTokenizer, BertModel)
from model_pretrain import Model
from tqdm import tqdm, trange
from itertools import cycle
import multiprocessing
from imblearn.over_sampling import RandomOverSampler
from transformers.data.data_collator import DataCollatorForLanguageModeling
from parser.DFG import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript

from parser.utils import (remove_comments_and_docstrings,
                          tree_to_token_index,
                          index_to_code_token,
                          tree_to_variable_index)
from tree_sitter import Language, Parser
from pruning_once import pruning
from preproccessing import pre_proccessing, pretrain_proccessing, pretrain_proccessing_DApp
logger = logging.getLogger(__name__)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

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


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class Example(object):
    def __init__(self, idx, source):
        self.idx = idx
        self.source = source


def read_examples():
    examples = []
    idx = 0

    codes = pretrain_proccessing_DApp('../SmartContracts/DAppSCAN-main/DAppSCAN-source')
    for code in codes:
        examples.append(Example(idx=idx, source=code))
        idx += 1
    codes = pretrain_proccessing("../SmartContracts/contract_dataset_ethereum")
    for code in codes:
        examples.append(Example(idx=idx, source=code))
        idx += 1
    codes = pretrain_proccessing("../SmartContracts/contract_dataset_github")
    for code in codes:
        examples.append(Example(idx=idx, source=code))
        idx += 1

    return examples


class InputFeatures(object):
    def __init__(self, example_id, source_ids, source_mask, code_to_code, idx):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.code_to_code = code_to_code
        self.idx = idx


def convert_examples_to_features(examples, tokenizer, stage=None):
    max_source_length = 256
    features = []
    parser = parsers['java']
    for example_index, example in tqdm(enumerate(examples), total=len(examples)):
        source_tokens, dfg = extract_dataflow(example.source, parser, 'java')
        code_to_code = []
        for item in dfg:
            if len(item[-1]) != 0:
                for comefroms in item[-1]:
                    code_to_code.append((item[1], comefroms))
        code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                       enumerate(source_tokens)]
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        c_t_c = []
        code_tokens = [y for x in code_tokens for y in x][:max_source_length - 2]
        for come, to in code_to_code:
            i, j = ori2cur_pos[come]
            k, l = ori2cur_pos[to]
            for m in range(i, j):
                for n in range(k, l):
                    if m < max_source_length - 2 and n < max_source_length - 2:
                        # if random.randint(0, 99) < 50:
                        c_t_c.append((m + 1, n + 1, 1))
                        while True:
                            a = random.randint(0, max_source_length - 2)
                            b = random.randint(0, max_source_length - 2)
                            flag1 = -1
                            flag2 = -1
                            for key, value in ori2cur_pos.items():
                                p, q = value
                                if p <= a < q:
                                    flag1 = key
                                if p <= b < q:
                                    flag2 = key
                            if (flag1, flag2) not in code_to_code:
                                c_t_c.append((a + 1, b + 1, 0))
                                break
                        # c_t_c.append((n+1, m+1))
        code_to_code = c_t_c
        code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        source_mask = [1] * (len(code_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
        features.append(
            InputFeatures(
                example_index,
                source_ids,
                source_mask,
                code_to_code,
                example.idx
            )
        )
    return features


def loaddata(tokenizer):
    datacollecter = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15, pad_to_multiple_of=1)
    train_examples = read_examples()
    train_features = convert_examples_to_features(train_examples, tokenizer, stage="train")
    input_mask = datacollecter([f.source_ids for f in train_features])
    all_source_ids_mask = torch.tensor(input_mask["input_ids"], dtype=torch.long)
    all_source_labels_mask = torch.tensor(input_mask["labels"], dtype=torch.long)
    all_source_labels_mask = torch.where(all_source_labels_mask == -100, 1, all_source_labels_mask)
    all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
    source_idx = torch.tensor([f.idx for f in train_features], dtype=torch.int)
    train_data = TensorDataset(all_source_ids, all_source_mask, all_source_ids_mask, all_source_labels_mask,
                               source_idx)
    code_to_code = []
    code_to_code += [f.code_to_code for f in train_features]
    train_dataloader = DataLoader(train_data, shuffle=True,
                                  batch_size=32,
                                  num_workers=0, drop_last=True)
    arr = np.array(code_to_code)
    np.save("code_to_code.npy", arr)
    torch.save(train_data.tensors, 'dataset.pth')
    return train_dataloader, code_to_code


def main(head, train_dataloader, code_to_code):
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
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
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
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=25000, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--compute_taylor', default=True,
                        help="compute_taylor")
    parser.add_argument('--head', default=head,
                        help="head")
    # print arguments
    args = parser.parse_args()

    logger.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    set_seed(args)
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']

    if args.do_train:
        if head == 12:
            config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
            config = config_class.from_pretrained("codebert-base")
            tokenizer = tokenizer_class.from_pretrained("codebert-base", maxlength=512)
            from transformers import RobertaModel
            model = RobertaModel(config=config)
            model.load_state_dict(torch.load("codebert-base/pytorch_model" + ".bin"), strict=False)
            model = Model(model, config)
            args.train_steps = 50000
            args.warmup_steps = 20000
            state_dict = torch.load("pretrain/tmodel_" + str(12) + "_150000.bin")
            mlm_weights = {
                '0.weight': state_dict['MLM.0.weight'],
                '2.weight': state_dict['MLM.2.weight'],
                '2.bias': state_dict['MLM.2.bias'],
                '3.weight': state_dict['MLM.3.weight']
            }
            df1_weights = {
                'weight': state_dict['df1.weight'],
            }
            df2_weights = {
                'weight': state_dict['df2.weight'],
            }
            model.MLM.load_state_dict(mlm_weights)
            model.df1.load_state_dict(df1_weights)
            model.df2.load_state_dict(df2_weights)
        else:
            from transformer.modeling_roberta import RobertaModel
            args.train_steps = 10000
            args.warmup_steps = 4000
            config = config_class.from_pretrained("pretrain-solidity/h" + str(head) + "_l12_f3072_e768")
            config.position_embedding_type = "absolute"
            tokenizer = tokenizer_class.from_pretrained("pretrain-solidity/h" + str(head) + "_l12_f3072_e768")
            model = RobertaModel(config)
            model.load_state_dict(torch.load("pretrain-solidity/h" + str(head) + "_l12_f3072_e768/pytorch_model" + ".bin"), strict=False)
            model = Model(model, config)
            if os.path.exists("pretrain/tmodel_" + str(head + 1) + "_10000.bin"):
                state_dict = torch.load("pretrain/tmodel_" + str(head + 1) + "_10000.bin")
            else:
                state_dict = torch.load("pretrain/tmodel_" + str(12) + "_150000.bin")
            mlm_weights = {
                '0.weight': state_dict['MLM.0.weight'],
                '2.weight': state_dict['MLM.2.weight'],
                '2.bias': state_dict['MLM.2.bias'],
                '3.weight': state_dict['MLM.3.weight']
            }
            df1_weights = {
                'weight': state_dict['df1.weight'],
            }
            df2_weights = {
                'weight': state_dict['df2.weight'],
            }
            model.MLM.load_state_dict(mlm_weights)
            model.df1.load_state_dict(df1_weights)
            model.df2.load_state_dict(df2_weights)
        model.to(device)

    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        num_train_optimization_steps = args.train_steps
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.compute_taylor:
            from AdamW_for_Taylor import AdamW
        else:
            from transformers import AdamW
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss, current_filename, current_step, total_step = 0, 0, 0, 0, 0, 1e6, 0, 0, 0
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)

        train_dataloader = cycle(train_dataloader)
        for step in bar:

           batch = next(train_dataloader)
           batch = tuple(t.to(device) for t in batch)
           source_ids, source_mask, all_source_ids_mask, all_source_labels_mask, idxs = batch
           loss, MLM_loss, DF_loss = model(source_ids=source_ids, source_mask=source_mask,
                                           all_source_ids_mask=all_source_ids_mask,
                                           all_source_labels_mask=all_source_labels_mask, idxs=idxs,
                                           code_to_code=code_to_code)
           current_step += 1
           if args.n_gpu > 1:
               loss = loss.mean()  # mean() to average on multi-gpu.
           tr_loss += loss.item()
           train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
           bar.set_description(
               "loss: {:.4f} lr: {:.2e} MLM: {:.2f} DF: {:.2f}".format(train_loss, optimizer.param_groups[1]["lr"],
                                                                       MLM_loss.item(), 0.1 * DF_loss.item()))
           with open(str(model.encoder.config.num_attention_heads) + "head.txt", "a") as file:
               file.write("step: {} loss: {:.4f} lr: {:.2e} MLM: {:.2f} DF: {:.2f}\n".format(step, train_loss,
                                                                                    optimizer.param_groups[1]["lr"],
                                                                                    MLM_loss.item(),
                                                                                    0.1 * DF_loss.item()))
           nb_tr_examples += source_ids.size(0)
           nb_tr_steps += 1
           loss.backward()
           if args.compute_taylor:
               optimizer.accumulate_grad()
           optimizer.step()
           optimizer.zero_grad()
           scheduler.step()
           global_step += 1

           if nb_tr_steps % args.train_steps == 0:
               model_to_save = model.encoder
               torch.save(model_to_save.state_dict(),
                          "pretrain/tencoder_" + str(model.encoder.config.num_attention_heads) + "_" + str(
                              nb_tr_steps) + ".bin")
               model_to_save = model
               torch.save(model_to_save.state_dict(),
                          "pretrain/tmodel_" + str(model.encoder.config.num_attention_heads) + "_" + str(
                              nb_tr_steps) + ".bin")

               if args.compute_taylor:
                   score = optimizer.get_taylor(args.train_steps)
                   score_dict = {}
                   modules2prune = []
                   for i in range(model.encoder.config.num_hidden_layers):
                       modules2prune += ['encoder.layer.%d.attention.self.query.weight' % i,
                                         'encoder.layer.%d.attention.self.key.weight' % i,
                                         'encoder.layer.%d.attention.self.value.weight' % i,
                                         'encoder.layer.%d.attention.output.dense.weight' % i,
                                         'encoder.layer.%d.intermediate.dense.weight' % i,
                                         'encoder.layer.%d.output.dense.weight' % i]
                   for name, param in list(model.encoder.named_parameters()):
                       if name in modules2prune:
                           cur_score = score[param]
                           score_dict[name] = cur_score.cpu()
                   torch.save(score_dict,
                              'pretrain/taylor_score/ttaylor' + str(model.encoder.config.num_attention_heads) + '.pkl')
    pruning(head-1)

if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("codebert-base", maxlength=512)
    # train_dataloader, code_to_code = loaddata(tokenizer)

    code_to_code = np.load('code_to_code.npy', allow_pickle=True)
    loaded_tensors = torch.load('dataset.pth')
    loaded_dataset = TensorDataset(*loaded_tensors)
    train_dataloader = DataLoader(loaded_dataset, shuffle=True,
                                  batch_size=32,
                                  num_workers=2, drop_last=True, pin_memory=True)
    for i in reversed(range(1)):
        main(-1, train_dataloader, code_to_code)
