#!/usr/bin/env python3

import argparse, os, sys, time
import warnings, json, gzip

import numpy as np

import pyBigWig

import torch
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr
from multiprocessing import Pool
from misc_utils import hg19_chromsize, overlap_length


def prepare_bw_signals(bigwig, bin_size, metainfo=None):
    chrom_signals = dict()
    for chrom, length in hg19_chromsize.items():
        num_bins = np.ceil(length / bin_size).astype(int) + 1
        chrom_signals[chrom] = [0 for i in range(num_bins)]
    with pyBigWig.open(bigwig) as bw:
        for chrom in chrom_signals:
            if chrom in {'chrM', 'chrMT', 'chrY'}:
                continue
            full_bin = hg19_chromsize[chrom] // bin_size
            if bin_size < 5000:
                full_sig = bw.stats(chrom, 0, full_bin * bin_size, nBins=full_bin, exact=True)
            else:
                warnings.warn("bin size=5000, using max to summarize")
                full_sig = bw.stats(chrom, 0, full_bin * bin_size, nBins=full_bin, exact=True, type="max")
            for i in range(full_bin):
                sig_val = full_sig[i]
                if sig_val is None:
                    sig_val = 0
                chrom_signals[chrom][i] = np.arcsinh(sig_val)
            if hg19_chromsize[chrom] > full_bin * bin_size:
                if bin_size < 5000:
                    sig_val = bw.stats(chrom, full_bin * bin_size, hg19_chromsize[chrom], exact=True)[0]
                else:
                    sig_val = bw.stats(chrom, full_bin * bin_size, hg19_chromsize[chrom], exact=True, type="max")[0]
                if sig_val is None:
                    sig_val = 0
                chrom_signals[chrom][full_bin] = np.arcsinh(sig_val)
            print("- summary: {}, len: {} min: {:.2f} median: {:.2f} max: {:.2f}".format(chrom, len(chrom_signals[chrom]), np.min(chrom_signals[chrom]), np.median(chrom_signals[chrom]), np.max(chrom_signals[chrom])))
    return chrom_signals, metainfo


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('bw_config')
    p.add_argument('-p', '--prefix', required=True)
    p.add_argument('-b', '--bin-size', required=True, type=int)
    p.add_argument('--preserve', action='store_true')
    p.add_argument("--threads", type=int, default=16)
    #p.add_argument('--seed', type=int, default=2020)
    return p



if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

    bw_config = json.load(open(args.bw_config))

    folder = bw_config['location']
    job_info = list()
    job_list = list()
    for cell in bw_config['celltypes']:
        for assay, bigwig in bw_config['celltypes'][cell].items():
            savename = "{}_{}_{}.{}bp.pt".format(args.prefix, cell, assay, args.bin_size)
            if args.preserve and os.path.exists(savename):
                print("- skip: {}".format(savename))
                continue
            if not os.path.exists("{}/{}".format(folder, bigwig)):
                warnings.warn("Missing file: {}/{}".format(folder, bigwig))
                continue
            job_list.append([
                "{}/{}".format(folder, bigwig), 
                args.bin_size, 
                {
                    "cell": cell, 
                    "assay": assay,
                    "savename": savename
                }
            ])
    with Pool(processes=args.threads) as pool:
        res = pool.starmap(prepare_bw_signals, job_list)
    
    for bw_signals, metainfo in res:
        savename = metainfo["savename"]
        for chrom in bw_signals:
            bw_signals[chrom] = torch.as_tensor(bw_signals[chrom], dtype=torch.float)
        torch.save(bw_signals, savename)

