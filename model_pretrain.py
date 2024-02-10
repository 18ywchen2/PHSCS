import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import os


class Model(nn.Module):

    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.MLM = nn.Sequential(nn.Linear(768, 768, bias=False),
                                 nn.ReLU(),
                                 nn.LayerNorm(768),
                                 nn.Linear(768, 50264, bias=False))
        self.df1 = nn.Linear(768, 64, bias=False)
        self.df2 = nn.Linear(768, 64, bias=False)
        for layer in [self.df1, self.df2]:
            layer.weight.data.normal_(mean=0.0, std=1.0)

    def forward(self, source_ids, source_mask, all_source_ids_mask=None, all_source_labels_mask=None, idxs=None,
                code_to_code=None):

        out = self.encoder(all_source_ids_mask, attention_mask=source_mask,
                           output_hidden_states=True)
        MLM_out = self.MLM(out.last_hidden_state)
        loss_mlm = nn.CrossEntropyLoss(ignore_index=1)
        pre = MLM_out.reshape(-1, 50264)
        label = source_ids.reshape(-1)
        loss = loss_mlm(pre, label)
        out_df = self.encoder(source_ids, attention_mask=source_mask,
                              output_hidden_states=True).last_hidden_state
        hidden1 = None
        df_label = []
        for i, idx in enumerate(idxs):
            c_t_c = code_to_code[idx]
            for j, k, l in c_t_c:
                if hidden1 is None:
                    hidden1 = self.df1(out_df[i, j: j + 1, :])
                    hidden2 = self.df2(out_df[i, k: k + 1, :])
                else:
                    hidden1 = torch.cat((hidden1, self.df1(out_df[i, j: j + 1, :])), dim=0)
                    hidden2 = torch.cat((hidden2, self.df2(out_df[i, k: k + 1, :])), dim=0)

                df_label.append(l)
        hidden1 = hidden1.unsqueeze(1)
        hidden2 = hidden2.unsqueeze(2)
        df_pre = torch.bmm(hidden1, hidden2).squeeze(1).squeeze(1)
        df_pre = nn.Sigmoid()(df_pre)
        df_label = torch.tensor(df_label).to("cuda")
        loss_dfg = nn.BCELoss(reduction="mean")
        df_loss = loss_dfg(df_pre, df_label.float())
        # return df_loss
        return loss + 0.1 * df_loss, loss, df_loss
