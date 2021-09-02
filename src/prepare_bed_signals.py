#!/usr/bin/env python3

import argparse, os, sys, time
import warnings, json, gzip

import numpy as np

import torch
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr
from misc_utils import hg19_chromsize, overlap_length

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('bed_config')
    p.add_argument('-p', '--prefix', required=True)
    p.add_argument('-b', '--bin-size', required=True, type=int)
    p.add_argument("--binary", action="store_true", help="Use 0/1 to represent the peaks")

    #p.add_argument('--seed', type=int, default=2020)
    return p


def prepare_bed_signals(bed, bin_size):
    chrom_signals = dict()
    for chrom, length in hg19_chromsize.items():
        num_bins = np.ceil(length / bin_size).astype(int) + 1
        chrom_signals[chrom] = [0 for i in range(num_bins)]
    with gzip.open(bed, 'rt') as infile:
        for l in infile:
            chrom, start, stop, _, _, _, sig_val = l.strip().split('\t')[0:7]
            begin, end = int(start) // bin_size, (int(stop) - 1) // bin_size
            if args.binary:
                sig_val = 1
            else:
                sig_val = np.arcsinh(float(sig_val))
            if sig_val > 10:
                warnings.warn(l)
            for i in range(begin, end + 1):
                if bin_size < 5000:
                    overlap = overlap_length(i * bin_size, (i + 1) * bin_size, start, stop)
                    chrom_signals[chrom][i] = max(overlap * sig_val / bin_size, chrom_signals[chrom][i])
                else:
                    warnings.warn("use max in stead of mean")
                    chrom_signals[chrom][i] = max(sig_val, chrom_signals[chrom][i])
    return chrom_signals


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

    bed_config = json.load(open(args.bed_config))

    folder = bed_config['location']
    for cell in bed_config['celltypes']:
        for assay, bed in bed_config['celltypes'][cell].items():
            fn = "{}_{}_{}{}.{}bp.pt".format(args.prefix, cell, assay, "-binary" if args.binary else "", args.bin_size)
            if os.path.exists(fn):
                continue
            bed_signals = prepare_bed_signals("{}/{}".format(folder, bed), bin_size=args.bin_size)
            for chrom in bed_signals:
                bed_signals[chrom] = torch.as_tensor(bed_signals[chrom], dtype=torch.float)
            torch.save(bed_signals, fn)

