#!/usr/bin/env python3

import argparse, os, sys, time, tqdm
import logging, warnings, json, gzip, pickle
# warning = logging.warning
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

from collections import defaultdict, OrderedDict

import numpy as np
import epi_models
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr
from typing import Any, Dict, List, Union
from functools import partial
print = partial(print, flush=True)

HG19_CHROMSIZE = {"chr1": 249250621, "chr2": 243199373, 
        "chr3": 198022430, "chr4": 191154276, 
        "chr5": 180915260, "chr6": 171115067, 
        "chr7": 159138663, "chr8": 146364022, 
        "chr9": 141213431, "chr10": 135534747, 
        "chr11": 135006516, "chr12": 133851895, 
        "chr13": 115169878, "chr14": 107349540, 
        "chr15": 102531392, "chr16": 90354753, 
        "chr17": 81195210, "chr18": 78077248, 
        "chr19": 59128983, "chr20": 63025520, 
        "chr21": 48129895, "chr22": 51304566, 
        "chrX": 155270560, "chrY": 59373566,
        "chrM": 16569, "chrMT": 16569}
HG38_CHROMSIZE = {"chr1": 248956422, "chr2": 242193529,
        "chr3": 198295559, "chr4": 190214555,
        "chr5": 181538259, "chr6": 170805979,
        "chr7": 159345973, "chr8": 145138636,
        "chr9": 138394717, "chr10": 133797422,
        "chr11": 135086622, "chr12": 133275309,
        "chr13": 114364328, "chr14": 107043718,
        "chr15": 101991189, "chr16": 90338345,
        "chr17": 83257441, "chr18": 80373285,
        "chr19": 58617616, "chr20": 64444167,
        "chr21": 46709983, "chr22": 50818468,
        "chrX": 156040895, "chrY": 57227415,
        "chrM": 16569, "chrMT": 16569}
CHROM_SIZE_DICT = {'hg19': HG19_CHROMSIZE, 'GRCh37': HG19_CHROMSIZE, "hg38": HG38_CHROMSIZE, "GRCh38": HG38_CHROMSIZE}




class BasicBED(object):
    def __init__(self, input_file, bin_size=50000):
        self.input_file = input_file
        self.chroms = dict()
        self.bin_size = bin_size
        #self.parse_input()

    def intersect(self, chrom, start, end, gap=0):
        start, end = int(start) - gap, int(end) + gap
        if start >= end:
            warnings.warn("starat >= end: start={}, end={}".format(start, end))
        res = set()
        if chrom in self.chroms:
            for idx in range(start // self.bin_size, (end - 1) // self.bin_size + 1):
                if idx not in self.chroms[chrom]:
                    continue
                try:
                    for i_start, i_end, attr in self.chroms[chrom][idx]:
                        if i_start >= end or i_end <= start:
                            continue
                        res.add((i_start, i_end, attr))
                except BaseException as err:
                    print(self.chroms[chrom][idx])
                    exit("{}".format(err))
        res = sorted(list(res), key=lambda l:(l[0], l[1]))
        return res

    def sort(self, merge=False):
        for chrom in self.chroms:
            for idx in self.chroms[chrom]:
                self.chroms[chrom][idx] = \
                        sorted(self.chroms[chrom][idx], key=lambda l:(l[0], l[1]))

    def add_record(self, chrom, start, end, attrs=None, cut=False):
        start, end = int(start), int(end)
        if chrom not in self.chroms:
            self.chroms[chrom] = dict()
        for bin_idx in range(start // self.bin_size, (end - 1) // self.bin_size + 1):
            if bin_idx not in self.chroms[chrom]:
                self.chroms[chrom][bin_idx] = list()
            if cut:
                raise NotImplementedError
            else:
                self.chroms[chrom][bin_idx].append((start, end, attrs))


    def __str__(self):
        return "BasicBED(filename:{})".format(os.path.relpath(self.input_file))

    def parse_input(self):
        raise NotImplementedError
        ## with open(self.input_file) as infile:
        ##     for l in infile:
        ##         if l.startswith("#"):
        ##             continue
        ##         fields = l.strip('\n').split('\t')
        ##         chrom, start, end = fields[0:3]
        ##         self.add_record(chrom, start, end, attrs=fields[3:])
        # record format: (left, right, (XXX))
        # XXX: self defined attributes of interval [left, right)


class Targets(BasicBED):
    def __init__(self, input_file):
        super(Targets, self).__init__(input_file=input_file)
        self.parse_input()

    def parse_input(self):
        with open(self.input_file) as infile:
            for l in infile:
                if l.startswith("#"):
                    continue
                fields = l.strip('\n').split('\t')
                chrom, start, end = fields[0:3]
                self.add_record(chrom, start, end, attrs='\t'.join(fields[3:]))


class EPITestDataset(Dataset):
    def __init__(self, 
            datasets: str,  # loci of interest
            targets, # bed format
            cell, 
            feats_config: Dict[str, str], 
            feats_order: List[str], 
            seq_len: int=2500000, 
            bin_size: int=500, 
            buildver="hg19",
            shift_mutation=0,
            use_mark=False,
            **kwargs):
        super(EPITestDataset, self).__init__()
        
        self.datasets = datasets
        self.targets = Targets(targets)
        self.cell = cell

        self.use_mark = use_mark

        self.chrom_size = CHROM_SIZE_DICT[buildver]

        self.seq_len = int(seq_len)
        self.bin_size = int(bin_size)
        self.num_bins = seq_len // bin_size

        self.feats_order = list(feats_order)
        self.num_feats = len(feats_order)
        self.feats_config = OrderedDict(json.load(open(feats_config)))

        assert args.cell in self.feats_config.keys(), "Undefined cell type: {}".format(args.cell)

        self.feats = dict() # cell_name -> feature_name -> chrom > features (array)
        self.chrom_bins = {
                chrom: (length // bin_size) for chrom, length in self.chrom_size.items()
                }

        self.samples = list()
        self.metainfo = {
                'samples': list(), 
                'dist': list(), 
                'chrom': list(), 
                'cell': list(),
                'enh_name': list(),
                'prom_name': list()
                }
        
        # self.load_datasets()
        # self.load_test_dataset()
        self.make_candidate_pairs(self.datasets, shift_mutation=shift_mutation)
        self.feat_dim = len(self.feats_order) + 1
        if self.use_mark:
            self.feat_dim += 1

    def make_candidate_pairs(self, loi, loi_type="enhancer", search_range=1000000, shift_mutation=0):
        cell = self.cell
        with open(loi) as infile:
            for l in infile:
                fields = l.strip('\n').split('\t')
                chrom, loi_start, loi_end = fields[0:3]
                loi_start, loi_end = int(loi_start), int(loi_end)
                loi_start, loi_end = loi_start + shift_mutation, loi_end + shift_mutation
                attrs = '|'.join(fields[3:])
                attrs = "{}|shift={}".format(attrs, shift_mutation)

                loi_info = l.strip()
                for target_start, target_end, attrs in self.targets.intersect(chrom, max(0, loi_start - search_range), loi_end + search_range):
                    if loi_type == "enhancer":
                        enh_coord = (loi_start + loi_end) // 2
                        enh_name = "{}:{}-{}|{}".format(chrom, loi_start, loi_end, attrs)
                        tss_coord = (target_start + target_end) // 2
                        prom_name = "{}:{}-{}".format(chrom, target_start, target_end)
                    else:
                        tss_coord = (loi_start + loi_end) // 2
                        prom_name = "{}:{}-{}|{}".format(chrom, loi_start, loi_end, attrs)
                        enh_coord = (target_start + target_end) // 2
                        enh_name = "{}:{}-{}".format(chrom, target_start, target_end)
                    
                    dist = abs(tss_coord - enh_coord)
                    if dist < 35000:
                        continue

                    target_info = "{}\t{}\t{}\t{}".format(chrom, target_start, target_end, attrs)

                    self.metainfo["samples"].append("{}\t{}".format(loi_info, target_info))


                    seq_begin = (enh_coord + tss_coord) // 2 - self.seq_len // 2
                    seq_end = (enh_coord + tss_coord) // 2 + self.seq_len // 2

                    # enh_bin = (enh_coord - seq_begin) // self.bin_size
                    enh_bin = enh_coord // self.bin_size
                    # prom_bin = (tss_coord - seq_begin) // self.bin_size
                    prom_bin = tss_coord // self.bin_size
                    start_bin, stop_bin = seq_begin // self.bin_size, seq_end // self.bin_size

                    left_pad_bin, right_pad_bin = 0, 0
                    if start_bin < 0:
                        left_pad_bin = abs(start_bin)
                        start_bin = 0
                    if stop_bin > self.chrom_bins[chrom]: 
                        right_pad_bin = stop_bin - self.chrom_bins[chrom] 
                        stop_bin = self.chrom_bins[chrom]

                    self.samples.append((
                        start_bin, stop_bin, 
                        left_pad_bin, right_pad_bin, 
                        enh_bin, prom_bin, 
                        self.cell, chrom, np.log(1 + float(dist)),
                        -1              # no available label
                    ))

                    # self.metainfo['label'].append(int(label))
                    self.metainfo['dist'].append(float(dist))
                    self.metainfo['chrom'].append(chrom)
                    self.metainfo['cell'].append(cell)
                    self.metainfo['enh_name'].append(enh_name)
                    self.metainfo['prom_name'].append(prom_name)

                    if cell not in self.feats:
                        self.feats[cell] = dict()
                        for feat in self.feats_order:
                            self.feats[cell][feat] = torch.load(self.feats_config[cell][feat])
        for k in self.metainfo:
            self.metainfo[k] = np.array(self.metainfo[k])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        start_bin, stop_bin, left_bin, right_bin, enh_bin, prom_bin, cell, chrom, dist, label = self.samples[idx]
        enh_idx = enh_bin - start_bin + left_bin
        prom_idx = prom_bin - start_bin + left_bin
        ar = torch.zeros((0, stop_bin - start_bin))
        for feat in self.feats_order:
            ar = torch.cat((ar, self.feats[cell][feat][chrom][start_bin:stop_bin].view(1, -1)), dim=0)
        ar = torch.cat((
            torch.zeros((self.num_feats, left_bin)),
            ar, 
            torch.zeros((self.num_feats, right_bin))
            ), dim=1)

        pos_enc = torch.arange(self.num_bins).view(1, -1)
        pos_enc = torch.cat((pos_enc - min(enh_idx, prom_idx), max(enh_idx, prom_idx) - 1 - pos_enc), dim=0)
        pos_enc = self.sym_log(pos_enc.min(dim=0)[0])
        ar = torch.cat((torch.as_tensor(pos_enc, dtype=torch.float).view(1, -1), ar), dim=0)

        if self.use_mark:
            mark = [0 for i in range(self.num_bins)]
            mark[enh_idx] = 1
            mark[enh_idx- 1] = 1
            mark[enh_idx + 1] = 1
            mark[prom_idx] = 1
            mark[prom_idx - 1] = 1
            mark[prom_idx + 1] = 1
            ar = torch.cat((
                torch.as_tensor(mark, dtype=torch.float).view(1, -1),
                ar
            ), dim=0)

        return ar, torch.as_tensor([dist], dtype=torch.float), torch.as_tensor([enh_idx], dtype=torch.long), torch.as_tensor([prom_idx], dtype=torch.long), torch.as_tensor([label], dtype=torch.float)

    def sym_log(self, ar):
        sign = torch.sign(ar)
        ar = sign * torch.log10(1 + torch.abs(ar))
        return ar


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        'bed', 
        help="the region of interest, in BED4+ (e.g., variants)")
    p.add_argument(
        '-t', "--targets", 
        default="../data/annotation/gencode.v19.tss.bed",
        required=False, 
        help="in BED4+ format (enhancer/promoter annotation, etc)")
    p.add_argument(
        '-c', "--cell", 
        required=True, 
        help="cell type")
    p.add_argument('--gpu', default=-1, type=int, help="GPU ID, (-1 for CPU)")
    p.add_argument('--feature', 
            default="../data/genomic_data/CTCF_DNase_6histone.500.json", 
            help="feature configuration file (in json format)")
    p.add_argument("--shift-mutation", default=0, type=int, help="deprecated")
    p.add_argument("--batch-size", type=int, default=128, help="batch size")
    p.add_argument('-b', "--buildver", default="hg19", choices=("hg19", "hg38"), help="reference genome version")
    p.add_argument("--config", required=True, help="model configuration")
    p.add_argument(
        '-m', "--model", 
        required=True, 
        help="path to model file")
    p.add_argument("--num-workers", type=int, default=16, help="number of workers used in data loader")
    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)
    
    config = json.load(open(args.config))
    bin_size = config["data_opts"]["bin_size"]
    seq_len = config["data_opts"]["seq_len"]

    print("{}".format(config["data_opts"]), file=sys.stderr, flush=True)

    
    # config_fn = "/home/chenken/Documents/DeepEPI/data/genomic_features/CTCF_DNase_6histone.{}.json".format(bin_size)
    config_fn = args.feature

    all_data = EPITestDataset(
        args.bed, 
        targets=args.targets, 
        cell=args.cell, 
        feats_config=config_fn,
        feats_order=config["data_opts"]["feats_order"],
        shift_mutation=args.shift_mutation,
        seq_len=seq_len,
        bin_size=bin_size,
        use_mark=config["data_opts"]["use_mark"] if "use_mark" in config["data_opts"] else False
    )

    torch.save(all_data.__getitem__(0), "all_data.pt")

    # for i, sample in enumerate(all_data.samples):
    #     print([all_data.metainfo["dist"][i], all_data.metainfo["enh_name"][i], all_data.metainfo["prom_name"][i]], file=sys.stderr, flush=True)
    #     print('\t'.join([str(x) for x in sample]), file=sys.stderr, flush=True)

    print("{} samples".format(len(all_data)), file=sys.stderr, flush=True)
    print("feature size: {}".format(all_data.__getitem__(0)[0].size()), file=sys.stderr, flush=True)


    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')


    config["model_opts"]["in_dim"] = all_data.feat_dim
    config["model_opts"]["seq_len"] = all_data.seq_len // all_data.bin_size

    model_class = getattr(epi_models, config["model_opts"]["model"])
    model = model_class(**config["model_opts"]).to(device)
    model.load_state_dict(torch.load(args.model)["model_state_dict"])
    model.eval()
    print("{}".format(model), flush=True, file=sys.stderr)

    data_loader = DataLoader(all_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("{} samples".format(len(all_data)), file=sys.stderr, flush=True)
    print("{} batch".format(len(data_loader)), file=sys.stderr, flush=True)

    predictions = list()
    for inputs, dists, enh_idxs, prom_idxs, _ in tqdm.tqdm(data_loader):
        out = model(inputs.to(device), enh_idxs, prom_idxs).detach().cpu().numpy().reshape(-1)
        try:
            for o in out:
                predictions.append(o)
        except:
            warnings.warn("{}".format((inputs.size(), out)))
    
    for i, sample in enumerate(all_data.metainfo["samples"]):
        print("{:.5f}\t{}".format(predictions[i], sample))
