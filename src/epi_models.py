#!/usr/bin/env python3

import argparse, os, sys, time
#import warnings, json, gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# from performer_pytorch import SelfAttention

from typing import Dict, List


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransEPI(nn.Module):
    def __init__(self, in_dim: int, 
            cnn_channels: List[int], cnn_sizes: List[int], cnn_pool: List[int],
            enc_layers: int, num_heads: int, d_inner: int,
            da: int, r: int, att_C: float,
            fc: List[int], fc_dropout: float, seq_len: int=-1, pos_enc: bool=False,
            **kwargs):
        super(TransEPI, self).__init__()
        
        major, minor = torch.__version__.split('.')[:2]
        assert int(major) >= 1 and int(minor) >= 6, "PyTorch={}, while PyTorch>=1.6 is required".format(torch.__version__)
        if int(minor) < 9:
            self.transpose = True
        else:
            self.transpose = False

        if pos_enc:
            assert seq_len > 0
        
        self.cnn = nn.ModuleList()
        self.cnn.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_dim, 
                        out_channels=cnn_channels[0], 
                        kernel_size=cnn_sizes[0], 
                        padding=cnn_sizes[0] // 2),
                    nn.BatchNorm1d(cnn_channels[0]),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(cnn_pool[0])
                )
            )
        seq_len //= cnn_pool[0]
        for i in range(len(cnn_sizes) - 1):
            self.cnn.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=cnn_channels[i], 
                            out_channels=cnn_channels[i + 1], 
                            kernel_size=cnn_sizes[i + 1],
                            padding=cnn_sizes[i + 1] // 2),
                        nn.BatchNorm1d(cnn_channels[i + 1]),
                        nn.LeakyReLU(),
                        nn.MaxPool1d(cnn_pool[i + 1])
                )
            )
            seq_len //= cnn_pool[i + 1]

        if pos_enc:
            self.pos_enc = PositionalEncoding(d_hid=cnn_channels[-1], n_position=seq_len)
        else:
            self.pos_enc = None
        
        if not self.transpose:
            enc_layer = nn.TransformerEncoderLayer(
                    d_model=cnn_channels[-1],
                    nhead=num_heads,
                    dim_feedforward=d_inner,
                    batch_first=True
                )
        else:
             enc_layer = nn.TransformerEncoderLayer(
                    d_model=cnn_channels[-1],
                    nhead=num_heads,
                    dim_feedforward=d_inner,
                )
           
        self.encoder = nn.TransformerEncoder(
                enc_layer,
                num_layers=enc_layers
                )

        self.da = da
        self.r = r
        self.att_C = att_C
        self.att_first = nn.Linear(cnn_channels[-1], da)
        self.att_first.bias.data.fill_(0)
        self.att_second = nn.Linear(da, r)
        self.att_second.bias.data.fill_(0)

        if fc[-1] != 1:
            fc.append(1)
        self.fc = nn.ModuleList()
        self.fc.append(
                nn.Sequential(
                    nn.Dropout(p=fc_dropout),
                    nn.Linear(cnn_channels[-1] * 4, fc[0])
                )
            )

        for i in range(len(fc) - 1):
            self.fc.append(
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(fc[i], fc[i + 1])
                    )
                )
        self.fc.append(nn.Sigmoid())
        self.fc_dist = nn.Sequential(
                    nn.Linear(cnn_channels[-1] * 4, cnn_channels[-1]),
                    nn.ReLU(),
                    nn.Linear(cnn_channels[-1], 1)
                )


    def forward(self, feats, enh_idx, prom_idx, return_att=False):
        # feats: (B, D, S)
        if type(feats) is tuple:
            feats, length = feats
        else:
            length = None
        div = 1
        for cnn in  self.cnn:
            div *= cnn[-1].kernel_size
            enh_idx = torch.div(enh_idx, cnn[-1].kernel_size, rounding_mode="trunc")
            prom_idx = torch.div(prom_idx, cnn[-1].kernel_size, rounding_mode="trunc")
            feats = cnn(feats)
        feats = feats.transpose(1, 2) # -> (B, S, D)
        batch_size, seq_len, feat_dim = feats.size()
        if self.pos_enc is not None:
            feats = self.pos_enc(feats)
        if self.transpose:
            feats = feats.transpose(0, 1)
        feats = self.encoder(feats) # (B, S, D)
        if self.transpose:
            feats = feats.transpose(0, 1)

        out = torch.tanh(self.att_first(feats)) # (B, S, da)
        if length is not None:
            length = torch.div(length, div, rounding_mode="trunc")
            max_len = max(length)
            mask = torch.cat((
                [torch.cat((torch.ones(1, m, self.da), torch.zeros(1, max_len - m, self.da)), dim=1) for m in length]
            ), dim=0)
            assert mask.size() == out.size()
            out = out * mask.to(out.device)
            del mask
        out = F.softmax(self.att_second(out), 1) # (B, S, r)
        att = out.transpose(1, 2) # (B, r, S)
        del out
        seq_embed = torch.matmul(att, feats) # (B, r, D)
        # print(seq_embed.size())
        base_idx = seq_len * torch.arange(batch_size) # .to(feats.device)
        enh_idx = enh_idx.long().view(batch_size) + base_idx
        prom_idx = prom_idx.long().view(batch_size) + base_idx
        feats = feats.reshape(-1, feat_dim)
        seq_embed = torch.cat((
            feats[enh_idx, :].view(batch_size, -1), 
            feats[prom_idx, :].view(batch_size, -1),
            seq_embed.mean(dim=1).view(batch_size, -1),
            seq_embed.max(dim=1)[0].view(batch_size, -1)
        ), axis=1)
        del feats
        # feats = torch.cat((feats.max(dim=1)[0].squeeze(1), feats.mean(dim=1).squeeze(1)), dim=1)
        dists = self.fc_dist(seq_embed)

        for fc in self.fc:
            seq_embed = fc(seq_embed)

        if return_att:
            return seq_embed, dists, att
        else:
            del att
            return seq_embed

    def l2_matrix_norm(self, m):                                                                                        
        return torch.sum(torch.sum(torch.sum(m**2, 1), 1)**0.5).type(torch.cuda.DoubleTensor)



# class TransformerModule(nn.Module):
#     def __init__(self, in_dim: int, 
#             cnn_channels: List[int], cnn_sizes: List[int], cnn_pool: List[int],
#             enc_layers: int, num_heads: int, d_inner: int, **kwargs):
#         super(TransformerModule, self).__init__()
# 
#         self.cnn = nn.ModuleList()
#         self.cnn.append(
#                 nn.Sequential(
#                     nn.Conv1d(
#                         in_channels=in_dim, 
#                         out_channels=cnn_channels[0], 
#                         kernel_size=cnn_sizes[0], 
#                         padding=cnn_sizes[0] // 2),
#                     nn.BatchNorm1d(cnn_channels[0]),
#                     nn.LeakyReLU(),
#                     nn.MaxPool1d(cnn_pool[0])
#                 )
#             )
#         for i in range(len(cnn_sizes) - 1):
#             self.cnn.append(
#                     nn.Sequential(
#                         nn.Conv1d(
#                             in_channels=cnn_channels[i], 
#                             out_channels=cnn_channels[i + 1], 
#                             kernel_size=cnn_sizes[i + 1],
#                             padding=cnn_sizes[i + 1] // 2),
#                         nn.BatchNorm1d(cnn_channels[i + 1]),
#                         nn.LeakyReLU(),
#                         nn.MaxPool1d(cnn_pool[i + 1])
#                 )
#             )
# 
#         enc_layer = nn.TransformerEncoderLayer(
#                 d_model=cnn_channels[-1],
#                 nhead=num_heads,
#                 dim_feedforward=d_inner
#             )
#         self.encoder = nn.TransformerEncoder(
#                 enc_layer,
#                 num_layers=enc_layers
#                 )
# 
#     def forward(self, feats):
#         # feats: (B, D, S)
#         for cnn in  self.cnn:
#             feats = cnn(feats)
#         feats = feats.transpose(1, 2)
#         feats = self.encoder(feats)
#         return feats
# 
# 


# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, d_inner, n_head, dropout=0.1):
#         super(EncoderLayer, self).__init__()
#         self.slf_att = SelfAttention(dim=d_model, causal=True, heads=n_head)
#         self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)
# 
#     def forward(self, enc_input):
#         enc_input = self.slf_att(enc_input)
#         enc_input = self.pos_ffn(enc_input)
#         return enc_input
# 
# 
# class Encoder(nn.Module):
#     def __init__(self, n_layers, n_head, d_model, d_inner, dropout=0.1):
#         super(Encoder, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.layer_stack = nn.ModuleList(
#                 [EncoderLayer(d_model, d_inner, n_head, dropout=dropout) for _ in range(n_layers)]
#             )
#         self.layer_norm = nn.LayerNorm(d_model, eps=1E-6)
#         self.d_model = d_model
# 
#     def forward(self, seq):
#         seq = self.dropout(seq)
#         # seq = self.layer_norm(seq)
#         for layer in self.layer_stack:
#             seq = layer(seq)
#         return seq


# class PerformerModel(nn.Module):
#     def __init__(self, in_dim: int, 
#             cnn_channels: List[int], cnn_sizes: List[int], cnn_pool: List[int],
#             enc_layers: int, num_heads: int, d_inner: int,
#             fc: List[int], fc_dropout: float,
#             **kwargs):
#         super(PerformerModel, self).__init__()
# 
#         self.cnn = nn.ModuleList()
#         self.cnn.append(
#                 nn.Sequential(
#                     nn.Conv1d(
#                         in_channels=in_dim, 
#                         out_channels=cnn_channels[0], 
#                         kernel_size=cnn_sizes[0], 
#                         padding=cnn_sizes[0] // 2),
#                     nn.BatchNorm1d(cnn_channels[0]),
#                     nn.LeakyReLU(),
#                     nn.MaxPool1d(cnn_pool[0])
#                 )
#             )
#         for i in range(len(cnn_sizes) - 1):
#             self.cnn.append(
#                     nn.Sequential(
#                         nn.Conv1d(
#                             in_channels=cnn_channels[i], 
#                             out_channels=cnn_channels[i + 1], 
#                             kernel_size=cnn_sizes[i + 1],
#                             padding=cnn_sizes[i + 1] // 2),
#                         nn.BatchNorm1d(cnn_channels[i + 1]),
#                         nn.LeakyReLU(),
#                         nn.MaxPool1d(cnn_pool[i + 1])
#                 )
#             )
# 
#         self.performer_encoder = Encoder(
#                 n_layers=enc_layers, 
#                 d_model=cnn_channels[-1], 
#                 n_head=num_heads,
#                 d_inner=d_inner
#             )
# 
#         if fc[-1] != 1:
#             fc.append(1)
#         self.fc = nn.ModuleList()
#         self.fc.append(
#                 nn.Sequential(
#                     nn.Dropout(p=fc_dropout),
#                     nn.Linear(cnn_channels[-1] * 2, fc[0])
#                 )
#             )
#         for i in range(len(fc) - 1):
#             self.fc.append(
#                     nn.Sequential(
#                         nn.ReLU(),
#                         nn.Linear(fc[i], fc[i + 1])
#                     )
#                 )
#         self.fc.append(nn.Sigmoid())
# 
# 
#     def forward(self, feats):
#         # feats: (B, D, S)
#         for cnn in  self.cnn:
#             feats = cnn(feats)
#         feats = feats.transpose(1, 2)
#         feats = self.performer_encoder(feats)
#         feats = torch.cat((feats.max(dim=1)[0].squeeze(1), feats.mean(dim=1).squeeze(1)), dim=1)
#         for fc in self.fc:
#             feats = fc(feats)
#         return feats



def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument()

    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

