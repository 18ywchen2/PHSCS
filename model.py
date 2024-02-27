import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.config.max_source_length = 512
        self.linear1 = nn.Linear(config.hidden_size * 1, config.hidden_size * 1)
        self.linear2 = nn.Linear(config.hidden_size * 1, 1)
        self.tanh = nn.Tanh()
        self.maxpooling = nn.MaxPool2d((config.hidden_size, 1), stride=1)
        self.avgerpooling = nn.AvgPool2d((config.hidden_size, 1), stride=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        self.BatchNorm1d = nn.BatchNorm1d(config.hidden_size * 1)
        self.BatchNorm1d1 = nn.BatchNorm1d(config.hidden_size * 1)
        self.BatchNorm1d2 = nn.BatchNorm1d(config.hidden_size * 1)
        self.dense = nn.Linear(config.hidden_size * 1, config.hidden_size * 1)
        self.dense1 = nn.Linear(config.hidden_size * 2, config.hidden_size * 1)
        self.dense2 = nn.Linear(config.hidden_size * 1, config.hidden_size * 1)
        self.dense3 = nn.Linear(config.hidden_size * 1, int(config.hidden_size * 1))
        self.out_proj = nn.Linear(config.hidden_size * 1, 1)
        self.GRU = nn.GRU(input_size=config.hidden_size, hidden_size=384, num_layers=1, bias=False, dropout=0.1,
                          bidirectional=True)

    def forward(self, source_ids, source_mask, source_pos, source_label=None):
        bs, l = source_ids.size()
        inputs_ids = source_ids.unsqueeze(1).view(bs * 1, l)
        position_idx = source_pos.unsqueeze(1).view(bs * 1, l)
        inputs_embeddings = self.encoder.embeddings.word_embeddings(inputs_ids)
        with torch.no_grad():
            for i in range(int(source_ids.shape[1] / self.config.max_source_length)):
                if i == 0:
                    source_out = self.encoder(
                        inputs_embeds=inputs_embeddings[:,
                                      self.config.max_source_length * i: self.config.max_source_length * (i + 1)],
                        attention_mask=source_mask[:,
                                       self.config.max_source_length * i: self.config.max_source_length * (
                                               i + 1)], position_ids=position_idx[:,
                                                                     self.config.max_source_length * i: self.config.max_source_length * (
                                                                                 i + 1)],
                        token_type_ids=position_idx[:,
                                       self.config.max_source_length * i: self.config.max_source_length * (i + 1)].eq(
                            -1).long()).last_hidden_state[:, :, :]
                else:
                    source_out = torch.cat((source_out, self.encoder(
                        inputs_embeds=inputs_embeddings[:,
                                      self.config.max_source_length * i: self.config.max_source_length * (i + 1)],
                        attention_mask=source_mask[:,
                                       self.config.max_source_length * i: self.config.max_source_length * (
                                               i + 1)], position_ids=position_idx[:,
                                                                     self.config.max_source_length * 0: self.config.max_source_length * (
                                                                                 0 + 1)],
                        token_type_ids=position_idx[:,
                                       self.config.max_source_length * 0: self.config.max_source_length * (0 + 1)].eq(
                            -1).long()).last_hidden_state[:, :, :]), dim=1)

        for i in range(len(source_ids)):
            if i == 0:
                z = self.GRU(source_out[i][0:sum(source_mask[i]), :])[0].unsqueeze(0)
                x = z
                x = self.linear1(x)
                x = self.tanh(x)
                attn_weights = self.linear2(x).squeeze(2)
                softmax_attn_weights = torch.softmax(attn_weights, dim=1)
                x = torch.bmm(z.permute(0, 2, 1), softmax_attn_weights.unsqueeze(2)).squeeze(2)
                y = x
            else:
                z = self.GRU(source_out[i][0:sum(source_mask[i]), :])[0].unsqueeze(0)
                x = z
                x = self.linear1(x)
                x = self.tanh(x)
                attn_weights = self.linear2(x).squeeze(2)
                softmax_attn_weights = torch.softmax(attn_weights, dim=1)
                x = torch.bmm(z.permute(0, 2, 1), softmax_attn_weights.unsqueeze(2)).squeeze(2)
                y = torch.cat((y, x), dim=0)

        x = torch.tanh(y)
        x = self.dropout(x)
        x = self.BatchNorm1d(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.sigmoid(x)
        if source_label is not None:
            loss = nn.BCELoss()(x.squeeze(1), source_label)
            return loss
        else:
            return x.squeeze(1)
