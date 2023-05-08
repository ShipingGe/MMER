import copy
import math
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)  # shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class UMTransformerEncoder(Module):
    # Uni-modal transformer encoder.
    def __init__(self, num_layers, src_dim, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, activation='relu'):
        super(UMTransformerEncoder, self).__init__()

        # self.cls_token = nn.Parameter(torch.zeros([1, 1, d_model]))
        self.input_norm = nn.LayerNorm(src_dim)
        self.proj_layer = nn.Linear(src_dim, d_model)
        self.pe = PositionalEncoding(d_model)
        # self.pe = nn.Embedding(512, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_padding_mask):
        # proj original embs + cls_token + positional encoding
        src = self.input_norm(src)
        # src = torch.cat([self.cls_token.expand([src.shape[0], -1, -1]), self.proj_layer(src)], dim=1)
        src = self.proj_layer(src)

        # pos_ids = torch.arange(src_padding_mask.shape[1], device=src.device).unsqueeze(0).expand(src_padding_mask.shape[0], -1)
        # output = (src + self.pe(pos_ids)).permute(1, 0, 2)
        output = self.pe(src.permute(1, 0, 2))  # the required shape of nn.Transformer is (S, N, E)
        # src_padding_mask = torch.cat([torch.zeros([src_padding_mask.shape[0], 6], device=src.device).bool(), src_padding_mask], dim=1)
        # src_padding_mask = torch.cat([torch.zeros([src_padding_mask.shape[0], 1], device=src.device).bool(), src_padding_mask], dim=1)

        for mod in self.layers:
            output = mod(output, src_key_padding_mask=src_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output.permute(1, 0, 2)


class TMTransformerEncoder(Module):
    # triple modalities transformer encoder.
    def __init__(self, num_layers, src1_dim, src2_dim, src3_dim, d_model=256, nhead=4, dim_feedforward=1024,
                 dropout=0.1,
                 activation='relu'):
        super(TMTransformerEncoder, self).__init__()

        # self.linear1 = nn.Linear(src1_dim, d_model)
        # self.linear2 = nn.Linear(src2_dim, d_model)
        # self.linear3 = nn.Linear(src3_dim, d_model)

        # self.cls_token = nn.Parameter(torch.zeros([1, 1, d_model]))

        # modality encoding
        self.me = nn.Embedding(3, d_model)

        # positional encoding
        # self.pe = PositionalEncoding(d_model)
        self.pe = nn.Embedding(512, d_model)

        # self.me1 = nn.Parameter(torch.zeros([1, 1, d_model]))
        # self.me2 = nn.Parameter(torch.zeros([1, 1, d_model]))
        # self.me3 = nn.Parameter(torch.zeros([1, 1, d_model]))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.last_tf_layer = self.layers[-1]

        # self.last_tf_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src1, src1_padding_mask, src2, src2_padding_mask, src3, src3_padding_mask):
        # proj original embs + positional encoding + modality encoding
        modality_ids = torch.tensor(
            [[0] * src1_padding_mask.shape[1] + [1] * src2_padding_mask.shape[1] + [2] * src3_padding_mask.shape[1]] *
            src1.shape[0], device=src1.device)
        modality_embs = self.me(modality_ids)

        pos_ids = torch.arange(src1_padding_mask.shape[1] + src2_padding_mask.shape[1] + src3_padding_mask.shape[1],
                               device=src1.device).unsqueeze(0).expand(src1_padding_mask.shape[0], -1)
        pos_embs = self.pe(pos_ids)
        # src1 = self.linear1(src1)
        # src2 = self.linear2(src2)
        # src3 = self.linear3(src3)
        # output = self.pe((torch.cat([src1, src2, src3], dim=1) + modality_embs).permute(1, 0, 2))
        output = (torch.cat([src1, src2, src3], dim=1) + modality_embs + pos_embs).permute(1, 0, 2)
        # output = (torch.cat([self.cls_token.expand(src1_padding_mask.shape[0], -1, -1), src], dim=1) + pos_embs).permute(1, 0, 2)

        # output = torch.cat([self.cls_token.expand(-1, src1.shape[0], -1), src], dim=0) + self.pe(pos_ids).unsqueeze(1).expand(-1, src1_padding_mask.shape[0], -1)
        # src1 = self.linear1(src1).permute(1, 0, 2) + self.me1.expand(src1.shape[1], -1,
        #                                                                            -1)  # the required shape of nn.Transformer is (S, N, E)
        # src2 = self.linear2(src2).permute(1, 0, 2) + self.me2.expand(src2.shape[1], -1, -1)
        # src3 = self.linear3(src3).permute(1, 0, 2) + self.me2.expand(src3.shape[1], -1, -1)
        # output = self.pe(torch.cat([src1, src2, src3], dim=0))
        padding_mask = torch.cat([src1_padding_mask, src2_padding_mask, src3_padding_mask], dim=1)
        # output = self.pe(torch.cat([self.cls_token.expand(-1, src1.shape[1], -1), src1, src2, src3], dim=0))
        # output = self.pe(torch.cat([self.cls_token.expand(-1, src1.shape[0], -1), src], dim=0))
        # padding_mask = torch.cat([torch.zeros([src1_padding_mask.shape[0], 1], device=src1.device).bool(), src1_padding_mask, src2_padding_mask, src3_padding_mask], dim=1)

        # output = self.pe(torch.cat([prefix_tokens.permute(1, 0, 2), src1, src2, src3], dim=0))
        # padding_mask = torch.cat([torch.zeros([src1_padding_mask.shape[0], 6], device=src1.device).bool(), src1_padding_mask, src2_padding_mask, src3_padding_mask], dim=1)

        for mod in self.layers:
            output = mod(output, src_key_padding_mask=padding_mask)

        # output = self.last_tf_layer(output, src_key_padding_mask=padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output.permute(1, 0, 2)


class EmotionDecoder(nn.Module):
    def __init__(self,
                 num_layers=1,
                 d_model=256,
                 nhead=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation='relu'):
        super(EmotionDecoder, self).__init__()

        # self.emotion_queries = nn.Parameter(torch.zeros([6, 1, d_model]))
        self.emotion_queries = nn.Embedding(6, d_model)
        # positional encoding
        # self.pe = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(d_model, d_model),
                                        nn.LeakyReLU(0.1),
                                        nn.Linear(d_model, 1),
                                        nn.Sigmoid())

    def forward(self, memory, memory_padding_mask):
        # output = self.emotion_queries.weight.unsqueeze(1).expand(-1, memory.shape[0], -1)
        input_ids = torch.tensor(np.arange(6), device=memory.device).long()
        output = self.emotion_queries(input_ids).unsqueeze(1).expand(-1, memory_padding_mask.shape[0], -1)
        # output = self.emotion_queries.expand(-1, memory.shape[0], -1)
        memory = memory.permute(1, 0, 2)

        for mod in self.layers:
            output = mod(output, memory, memory_key_padding_mask=memory_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        logits = self.classifier(output).squeeze(-1).permute(1, 0)

        return logits
