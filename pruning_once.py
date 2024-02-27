import argparse, logging, os, sys, torch
from typing import Dict, Optional

import numpy as np
import torch.nn.utils.prune as prune

from transformer.configuration_bert_prun import BertConfigPrun
from transformer.modeling_prun import TinyBertForSequenceClassification as PrunTinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)

intermedia_modules, attn_modules = [], []
for i in range(12):
    intermedia_modules += ['encoder.layer.%d.intermediate.dense' % i, 'encoder.layer.%d.output.dense' % i]
    attn_modules += ['encoder.layer.%d.attention.self.query' % i, 'encoder.layer.%d.attention.self.key' % i, \
                     'encoder.layer.%d.attention.self.value' % i,
                     'encoder.layer.%d.attention.output.dense' % i]

# structured pruning prunes input or output neurons of a matrix
prune_out, prune_in = [], []
for i in range(12):
    prune_out += ['encoder.layer.%d.attention.self.query' % i, 'encoder.layer.%d.attention.self.key' % i, \
                  'encoder.layer.%d.attention.self.value' % i, 'encoder.layer.%d.intermediate.dense' % i]
    prune_in += ['encoder.layer.%d.attention.output.dense' % i, 'encoder.layer.%d.output.dense' % i]


class PruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, prun_ratio, score):
        self.prun_ratio = prun_ratio
        self.score = score

    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.prun_ratio, tensor_size)
        mask = default_mask.clone()
        topk = torch.topk(
            torch.abs(self.score).view(-1), k=nparams_toprune, largest=False
        )
        # topk will have .indices and .values
        mask.view(-1)[topk.indices] = 0
        return mask


def unstructured_pruning(module, name, prun_ratio, score):
    PruningMethod.apply(module, name, prun_ratio, score)
    return module


def Taylor_pruning_structured(model, prun_ratio, num_heads, keep_heads, taylor_path,
                              emb_hidden_dim, config):
    """
    Args:
        emb_hidden_dim[int]: Hidden size of embedding factorization. Choose from 128, 256, 512.
                                Do not factorize embedding if value==-1
    """
    # Counting scores
    taylor_dict = torch.load(taylor_path)
    intermedia_scores, attn_scores = [], []
    for i in range(len(model.encoder.layer)):
        score_inter_in = taylor_dict['encoder.layer.%d.intermediate.dense.weight' % i]
        score_inter_out = taylor_dict['encoder.layer.%d.output.dense.weight' % i]
        score_inter = score_inter_in.sum(1) + score_inter_out.sum(0)
        intermedia_scores.append(score_inter)

        score_attn_output = taylor_dict['encoder.layer.%d.attention.output.dense.weight' % i]
        score_attn = score_attn_output.sum(0)
        attn_score_chunks = torch.split(score_attn, 64)
        score_attn = torch.tensor([chunk.sum() for chunk in attn_score_chunks])
        attn_scores.append(score_attn)

    with torch.no_grad():
        layer_id = 0
        for name, module in model.named_modules():

            # Pruning Attention Heads
            if (name in attn_modules) and (not keep_heads == model.config.num_attention_heads):
                layer_id = int(name.split(".")[2])
                score_attn = attn_scores[layer_id]
                attn_size = module.weight.size(0) / float(num_heads) if name in prune_out \
                    else module.weight.size(1) / float(num_heads)
                _, indices = torch.topk(score_attn, keep_heads)
                if name in prune_out:
                    weight_chunks = torch.split(module.weight.data, int(attn_size), dim=0)
                    bias_chunks = torch.split(module.bias.data, int(attn_size))
                    module.bias.data = torch.cat([bias_chunks[i] for i in indices])
                    module.weight.data = torch.cat([weight_chunks[i] for i in indices], dim=0)
                elif name in prune_in:
                    weight_chunks = torch.split(module.weight.data, int(attn_size), dim=1)
                    module.weight.data = torch.cat([weight_chunks[i] for i in indices], dim=1)

    return model


def pruning(keep_heads):
    parser = argparse.ArgumentParser(description='pruning_one-step.py')
    # parser.add_argument('-model_path', default='../KD/models/bert_ft', type=str,
    #                     help="distill type")
    parser.add_argument('-output_dir', default='pretrain', type=str,
                        help="output dir")
    parser.add_argument('-keep_heads', type=int, default=keep_heads,
                        help="the number of attention heads to keep")
    parser.add_argument('-ffn_hidden_dim', type=int, default=256 * 12,
                        help="Hidden size of the FFN subnetworks.")
    parser.add_argument('-num_layers', type=int, default=12,
                        help="the number of layers of the pruned model")
    parser.add_argument('-emb_hidden_dim', type=int, default=768,
                        help="Hidden size of embedding factorization. \
                    Do not factorize embedding if value==-1")
    args = parser.parse_args()

    torch.manual_seed(0)
    from transformer.modeling_roberta import RobertaModel
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']

    print('Loading BERT...')

    config = config_class.from_pretrained("pretrain-solidity/h" + str(args.keep_heads + 1) + "_l12_f3072_e768")
    config.position_embedding_type = "absolute"
    tokenizer = tokenizer_class.from_pretrained("pretrain-solidity/h" + str(args.keep_heads + 1) + "_l12_f3072_e768")
    model = model_class.from_pretrained("pretrain-solidity/h" + str(args.keep_heads + 1) + "_l12_f3072_e768", config=config)
    if os.path.exists("pretrain/tencoder_" + str(args.keep_heads + 1) + "_10000.bin"):
        model.load_state_dict(torch.load("pretrain/tencoder_" + str(args.keep_heads + 1) + "_10000.bin"), strict=False)
    else:
        model.load_state_dict(torch.load("pretrain/tencoder_" + str(args.keep_heads + 1) + "_50000.bin"), strict=False)
    model.to('cuda')

    model.encoder.layer = torch.nn.ModuleList([model.encoder.layer[i] for i in range(args.num_layers)])
    config.prun_intermediate_size = 12 * 256
    config.emb_hidden_dim = 12 * 64
    if args.ffn_hidden_dim > config.prun_intermediate_size or \
            (args.emb_hidden_dim > config.emb_hidden_dim and config.emb_hidden_dim != -1):
        raise ValueError('Cannot prune the model to a larger size!')

    args.prun_ratio = args.ffn_hidden_dim / config.prun_intermediate_size
    print('Pruning to %d heads, %d layers, %d FFN hidden dim, %d emb hidden dim...' %
          (args.keep_heads, args.num_layers, args.ffn_hidden_dim, args.emb_hidden_dim))
    importance_dir = os.path.join("pretrain", 'taylor_score', 'ttaylor' + str(args.keep_heads + 1) + '.pkl')
    config.num_attention_heads = args.keep_heads + 1
    # new_config = BertConfigPrun(num_attention_heads=args.keep_heads,
    #                             prun_hidden_size=int(args.keep_heads * 64),
    #                             prun_intermediate_size=args.ffn_hidden_dim,
    #                             num_hidden_layers=args.num_layers,
    #                             emb_hidden_dim=args.emb_hidden_dim)
    model = Taylor_pruning_structured(model, args.prun_ratio, config.num_attention_heads,
                                      args.keep_heads, importance_dir,
                                      args.emb_hidden_dim, config)

    output_dir = os.path.join(args.output_dir, 'h%d_l%d_f%d_e%d'
                              % (args.keep_heads, args.num_layers, args.ffn_hidden_dim, args.emb_hidden_dim))

    print('Saving model to %s' % output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    config.num_attention_heads = args.keep_heads
    config.head_hidden_size = int(args.keep_heads*64)
    config.prun_intermediate_size = args.ffn_hidden_dim
    config.num_hidden_layers = args.num_layers
    config.emb_hidden_dim = args.emb_hidden_dim
    config.hidden_size = 768
    config.intermediate_size = args.ffn_hidden_dim

    config.save_pretrained(output_dir)
    tokenizer.save_vocabulary(output_dir)
    # model = PrunTinyBertForSequenceClassification.from_pretrained(output_dir)
    # torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    print("Number of parameters: %d" % sum([model.state_dict()[key].nelement() for key in model.state_dict()]))
    # print(model.state_dict().keys())




