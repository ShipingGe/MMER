import copy
import math
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from utils import GradReverse, TopK_custom, TransformerEncoderLayer, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
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


class UMTransformerEncoder(nn.Module):
    # Uni-modal transformer encoder.
    def __init__(self, num_layers, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-12):
        super(UMTransformerEncoder, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                                norm_first=False)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src):
        output = src.permute(1, 0, 2)

        for mod in self.layers:
            output = mod(output)

        if self.norm is not None:
            output = self.norm(output)

        return output.permute(1, 0, 2)


class CMTransformerEncoder(nn.Module):
    # Cross-modal transformer encoder.
    def __init__(self,
                 num_layers=1,
                 d_model=256,
                 nhead=4,
                 dim_feedforward=256,
                 dropout=0.2,
                 activation='relu',
                 layer_norm_eps=1e-5):
        super(CMTransformerEncoder, self).__init__()

        encoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                                norm_first=False)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src, memory):
        output = src.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        for mod in self.layers:
            output = mod(output, memory)

        if self.norm is not None:
            output = self.norm(output)

        return output.permute(1, 0, 2)


class TMTransformerEncoder(nn.Module):
    # triple modalities transformer encoder.
    def __init__(self, num_layers, d_model=256, nhead=4, dim_feedforward=1024,
                 dropout=0.1,
                 activation='relu',
                 layer_norm_eps=1e-12):
        super(TMTransformerEncoder, self).__init__()


        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                                norm_first=False)
        self.layers = _get_clones(encoder_layer, num_layers)

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src1, src2, src3):

        src = torch.cat([src1, src2, src3], dim=1)
        output = src.permute(1, 0, 2)

        for mod in self.layers:
            output = mod(output)

        if self.norm is not None:
            output = self.norm(output)

        return output.permute(1, 0, 2)


class QMTransformerEncoder(nn.Module):
    # quadra modalities transformer encoder.
    def __init__(self, num_layers, src1_dim, src2_dim, src3_dim, src4_dim, d_model=256, nhead=4, dim_feedforward=1024,
                 dropout=0.1,
                 activation='relu',
                 layer_norm_eps=1e-12):
        super(QMTransformerEncoder, self).__init__()

        # self.cls_token = nn.Parameter(torch.zeros([1, 1, d_model]))

        # modality encoding
        self.me = nn.Embedding(4, d_model)
        # positional encoding
        self.pe = nn.Embedding(512, d_model)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                                norm_first=False)
        self.layers = _get_clones(encoder_layer, num_layers)

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src1, src1_padding_mask, src2, src2_padding_mask, src3, src3_padding_mask, src4,
                src4_padding_mask):
        # proj original embs + positional encoding + modality encoding
        modality_ids = torch.tensor(
            [[0] * src1_padding_mask.shape[1] + [1] * src2_padding_mask.shape[1] + [2] * src3_padding_mask.shape[1] + [
                3] * src4_padding_mask.shape[1]] *
            src1.shape[0], device=src1.device)
        modality_embs = self.me(modality_ids)

        pos_ids = torch.arange(src1_padding_mask.shape[1] + src2_padding_mask.shape[1] + src3_padding_mask.shape[1] +
                               src4_padding_mask.shape[1],
                               device=src1.device).unsqueeze(0).expand(src1_padding_mask.shape[0], -1)
        pos_embs = self.pe(pos_ids)
        src = torch.cat([src1, src2, src3.unsqueeze(1), src4], dim=1)
        output = (src + modality_embs + pos_embs).permute(1, 0, 2)

        padding_mask = torch.cat([src1_padding_mask, src2_padding_mask, src3_padding_mask, src4_padding_mask], dim=1)

        for mod in self.layers:
            output = mod(output, src_key_padding_mask=padding_mask)

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
                 activation='relu',
                 layer_norm_eps=1e-5,
                 num_cls=6):
        super(EmotionDecoder, self).__init__()

        self.emotion_queries = nn.Embedding(num_cls, d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.num_cls = num_cls

        self.classifier = nn.Sequential(nn.Linear(d_model, 1),
                                        nn.Dropout(p=0.1))

    def forward(self, memory):

        input_ids = torch.tensor(np.arange(self.num_cls), device=memory.device).long()
        output = self.emotion_queries(input_ids).unsqueeze(1).expand(-1, memory.shape[0], -1)
        memory = memory.permute(1, 0, 2)

        for mod in self.layers:
            output = mod(output, memory)

        if self.norm is not None:
            output = self.norm(output)

        logits = self.classifier(output).squeeze(-1).permute(1, 0)

        logits = torch.sigmoid(logits)

        return logits


class MMERModel(nn.Module):
    def __init__(self,
                 num_encoder_layers=4,
                 num_decoder_layers=1,
                 src1_dim=300,
                 src2_dim=35,
                 src3_dim=74,
                 d_model=512,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation='relu',
                 layer_norm_eps=1e-5,
                 unaligned=False):
        super(MMERModel, self).__init__()

        self.src1_linear = nn.Sequential(nn.Linear(src1_dim, d_model),
                                         nn.LayerNorm(d_model),
                                         nn.Dropout(0.1))

        self.src2_linear = nn.Sequential(nn.Linear(src2_dim, d_model),
                                         nn.LayerNorm(d_model),
                                         nn.Dropout(0.1))

        self.src3_linear = nn.Sequential(nn.Linear(src3_dim, d_model),
                                         nn.LayerNorm(d_model),
                                         nn.Dropout(0.1))

        self.unaligned = unaligned

        self.text_base_encoder = UMTransformerEncoder(num_encoder_layers, d_model, nhead, dim_feedforward,
                                                      dropout, activation, layer_norm_eps)
        # self.text_base_encoder = LSTMAttnEncoder(d_model, nhead, dim_feedforward, num_encoder_layers)

        self.video_base_encoder = UMTransformerEncoder(num_encoder_layers, d_model, nhead, dim_feedforward,
                                                       dropout, activation, layer_norm_eps)

        self.audio_base_encoder = UMTransformerEncoder(num_encoder_layers, d_model, nhead, dim_feedforward,
                                                       dropout, activation, layer_norm_eps)

        self.tri_modal_encoder = TMTransformerEncoder(num_encoder_layers, d_model, nhead,
                                                      dim_feedforward, dropout, activation, layer_norm_eps)


        self.text_scorer_net = nn.Sequential(nn.Linear(d_model, 1),
                                             nn.Sigmoid())

        self.video_scorer_net = nn.Sequential(nn.Linear(d_model, 1),
                                              nn.Sigmoid())

        self.audio_scorer_net = nn.Sequential(nn.Linear(d_model, 1),
                                              nn.Sigmoid())

        self.emotion_decoder = EmotionDecoder(num_decoder_layers, d_model, nhead, dim_feedforward, dropout, activation,
                                              layer_norm_eps)

        self.topk = TopK_custom(k=24)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for name, param in self.named_parameters():
            if param.dim() > 1:
                xavier_uniform_(param)

    def forward(self, src1, src2, src3):
        src1 = self.src1_linear(src1)
        src2 = self.src2_linear(src2)
        src3 = self.src3_linear(src3)

        feats1 = self.text_base_encoder(src1)
        feats2 = self.video_base_encoder(src2)
        feats3 = self.audio_base_encoder(src3)

        s1 = self.text_scorer_net(feats1)
        s2 = self.video_scorer_net(feats2)
        s3 = self.audio_scorer_net(feats3)

        memory = self.tri_modal_encoder(feats1, feats2, feats3)

        logits = self.emotion_decoder(memory)


        scores1 = GradReverse.grad_reverse(s1.squeeze(-1), 0.1)
        scores2 = GradReverse.grad_reverse(s2.squeeze(-1), 0.1)
        scores3 = GradReverse.grad_reverse(s3.squeeze(-1), 0.1)

        src1_masked = (1 - self.topk(scores1).unsqueeze(-1)) * src1
        src2_masked = (1 - self.topk(scores2).unsqueeze(-1)) * src2
        src3_masked = (1 - self.topk(scores3).unsqueeze(-1)) * src3

        feats1_masked = self.text_base_encoder(src1_masked)
        feats2_masked = self.video_base_encoder(src2_masked)
        feats3_masked = self.audio_base_encoder(src3_masked)

        memory_masked = self.tri_modal_encoder(feats1_masked, feats2_masked, feats3_masked)

        logits_masked = self.emotion_decoder(memory_masked)

        return logits, logits_masked
