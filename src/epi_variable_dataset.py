#!/usr/bin/env python3

import argparse, os, sys, time
import warnings, json, gzip
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from misc_utils import hg19_chromsize, mm9_chromsize

import numpy as np

from typing import Dict, List, Union


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument()

    p.add_argument('--seed', type=int, default=2020)
    return p

class EPIDatasetVL(Dataset):
    def __init__(self, 
            datasets: Union[str, List], 
            feats_config: Dict[str, str], 
            feats_order: List[str], 
            # seq_len: int=2500000, 
            pad_len: int=500000, 
            bin_size: int=500, 
            use_mark: bool=False,
            # mask_neighbor=False,
            # mask_window=False,
            # mask_size=5,
            # sin_encoding=False,
            # rand_shift=False,
            buildver="hg19",
            **kwargs):
        super(EPIDatasetVL, self).__init__()

        if type(datasets) is str:
            self.datasets = [datasets]
        else:
            self.datasets = datasets
        if buildver == "hg19":
            print("hg19")
            self.chromsize = hg19_chromsize
        elif buildver == "mm9":
            print("mm9")
            self.chromsize = mm9_chromsize

        self.pad_len = int(pad_len)
        self.bin_size = int(bin_size)
        # assert self.seq_len % self.bin_size == 0, "{} / {}".format(self.seq_len, self.bin_size)
        # self.num_bins = seq_len // bin_size

        self.feats_order = list(feats_order)
        self.num_feats = len(feats_order)
        self.feats_config = json.load(open(feats_config))
        if "_location" in self.feats_config:
            location =self.feats_config["_location"] 
            del self.feats_config["_location"]
            for cell, assays in self.feats_config.items():
                for a, fn in assays.items():
                    self.feats_config[cell][a] = os.path.join(location, fn)

        self.feats = dict() # cell_name -> feature_name -> chrom > features (array)
        self.chrom_bins = {
                chrom: (length // bin_size) for chrom, length in self.chromsize.items()
                }

        self.samples = list()
        self.metainfo = {
                'label': list(), 
                'dist': list(), 
                'chrom': list(), 
                'cell': list(),
                'enh_name': list(),
                'prom_name': list(),
                'shift': list()
                }

        self.use_mark = use_mark

        self.load_datasets()
        self.feat_dim = len(self.feats_order) + 1
        if self.use_mark:
            self.feat_dim += 1

    def load_datasets(self):
        for fn in self.datasets:
            with open(fn) as infile:
                for l in infile:
                    label, dist, chrom, enh_start, enh_end, enh_name, \
                            _, prom_start, prom_end, prom_name = l.strip().split('\t')
                    cell = enh_name.split('|')[1]
                    strand = prom_name.split('|')[-1]

                    enh_coord = (int(enh_start) + int(enh_end)) // 2
                    tss_coord = (int(prom_start) + int(prom_end)) // 2

                    seq_begin = max(0, min(enh_coord, tss_coord) - self.pad_len)
                    seq_end = min(self.chromsize[chrom], max(enh_coord, tss_coord) + self.pad_len)
                    
                    enh_bin, prom_bin = enh_coord // self.bin_size, tss_coord // self.bin_size
                    start_bin, stop_bin = seq_begin // self.bin_size, seq_end // self.bin_size

                    left_pad_bin, right_pad_bin = 0, 0 # keep these for compatibility
                    shift = 0
                    self.samples.append((
                        start_bin + shift, stop_bin + shift, 
                        left_pad_bin, right_pad_bin, 
                        enh_bin, prom_bin, 
                        cell, chrom, np.log2(1 + 500000 / float(dist)),
                        int(label)
                    ))

                    self.metainfo['label'].append(int(label))
                    self.metainfo['dist'].append(float(dist))
                    self.metainfo['chrom'].append(chrom)
                    self.metainfo['cell'].append(cell)
                    self.metainfo['enh_name'].append(enh_name)
                    self.metainfo['prom_name'].append(prom_name)
                    self.metainfo['shift'].append(shift)

                    if cell not in self.feats:
                        self.feats[cell] = dict()
                        for feat in self.feats_order:
                            self.feats[cell][feat] = torch.load(self.feats_config[cell][feat])
        for k in self.metainfo:
            self.metainfo[k] = np.array(self.metainfo[k])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        start_bin, stop_bin, left_pad, right_pad, enh_bin, prom_bin, cell, chrom, dist, label = self.samples[idx]
        enh_idx = enh_bin - start_bin + left_pad
        prom_idx = prom_bin - start_bin + left_pad

        # print(self.samples[idx], self.metainfo["shift"][idx])
        try:
            ar = torch.zeros((0, stop_bin - start_bin))
        except RuntimeError as err:
            raise RuntimeError("{}\n{}".format((self.samples[idx]), err))
        # print(start_bin - left_pad, stop_bin + right_pad, enh_bin, prom_bin, enh_idx, prom_idx)
        for feat in self.feats_order:
            ar = torch.cat((ar, self.feats[cell][feat][chrom][start_bin:stop_bin].view(1, -1)), dim=0)
        ar = torch.cat((
            torch.zeros((self.num_feats, left_pad)),
            ar, 
            torch.zeros((self.num_feats, right_pad))
            ), dim=1)

        pos_enc = torch.arange(stop_bin - start_bin).view(1, -1)
        pos_enc = torch.cat((pos_enc - min(enh_idx, prom_idx), max(enh_idx, prom_idx) - pos_enc), dim=0)
        pos_enc = torch.arcsinh(pos_enc.min(dim=0)[0]/1000).view(1, -1)
        ar = torch.cat((torch.as_tensor(pos_enc, dtype=torch.float), ar), dim=0)
        

        if self.use_mark:
            mark = [0 for i in range(start_bin, stop_bin)]
            mark[enh_idx] = 1
            mark[enh_idx - 1] = 1
            mark[enh_idx + 1] = 1
            mark[prom_idx] = 1
            mark[prom_idx - 1] = 1
            mark[prom_idx + 1] = 1
            ar = torch.cat((
                torch.as_tensor(mark, dtype=torch.float).view(1, -1),
                ar
            ), dim=0)

        # ar: (dim, length)
        return ar, dist, enh_idx, prom_idx, label
    
    def collate(self, inputs):
        ar, dist, enh_idx, prom_idx, label = list(zip(*inputs))
        lengths = torch.as_tensor([_.size(1) for _ in ar], dtype=torch.int)
        ar = pad_sequence([_.T for _ in ar]) # (length, batch_size, dim)
        ar = ar.transpose(0, 1).transpose(1, 2)
        dist = torch.FloatTensor(np.array(dist))
        enh_idx = torch.IntTensor(np.array(enh_idx, dtype=int))
        prom_idx = torch.IntTensor(np.array(prom_idx, dtype=int))
        label = torch.FloatTensor(np.array(label))
        return (ar, lengths), dist, enh_idx, prom_idx, label.view(-1, 1)


    def sym_log(self, ar):
        sign = torch.sign(ar)
        ar = sign * torch.log10(1 + torch.abs(ar))
        return ar
    


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    np.random.seed(args.seed)

    all_data = EPIDataset(
            datasets=["../data/BENGI/GM12878.HiC-Benchmark.v3.tsv"],
            feats_config="../data/genomic_features/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            seq_len=2500000,
            bin_size=500,
            mask_window=True,
            mask_neighbor=True,
            sin_encoding=True,
            rand_shift=True,
            mask_size=101,
        )

    for i in range(0, len(all_data), 411):
        np.savetxt(
                "data_{}".format(i),
                all_data.__getitem__(i)[0].T,
                fmt="%.4f",
                header="{}\t{}\t{}\n{}".format(all_data.metainfo["label"][i], all_data.metainfo["enh_name"][i], all_data.metainfo["prom_name"][i], all_data.samples[i])
            )


#     batch_size = 16
#     data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, num_workers=8)
# 
#     # # import epi_models
#     # # model = epi_models.LstmAttModel(in_dim=6, 
#     # #         lstm_size=32, lstm_layer=2, lstm_dropout=0.2, 
#     # #         da=64, r=32,
#     # #         fc=[64, 32], fc_dropout=0.2)
#     # import epi_models
#     # model = epi_models.PerformerModel(
#     #         in_dim=6,
#     #         cnn_channels=[128],
#     #         cnn_sizes=[11],
#     #         cnn_pool=[5],
#     #         enc_layers=4,
#     #         num_heads=4,
#     #         d_inner=128,
#     #         fc=[32, 16],
#     #         fc_dropout=0.1
#     #     ).cuda()
#     for i, (feat, dist, label) in enumerate(data_loader):
#         print()
#         print(feat.size(), dist.size(), label.size())
#         # torch.save({'feat': feat, 'label': label}, "tmp.pt")
#         # feat = model(feat.cuda())
#         print(feat.size())
#         # for k in all_data.metainfo:
#         #     print(k, all_data.metainfo[k][i])
#         # if i > 200:
#         #     break
# 
