#!/usr/bin/env python3


"""
product abadon form of 3Dpredictor
"""

import argparse, functools, glob, gzip, json, os, pickle, sys, time, h5py, pyBigWig
from warnings import warn
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.model_selection import GroupKFold, StratifiedKFold
import pyBigWig

import biock.biock as biock
from biock.biock import print


def parse_orient_bed(celltypes, chroms, starts, ends, peak_cache, assays, pool_size=1):
    samples_channels_length = None
    bin_num = (ends[0] - starts[0]) // pool_size
    for i, ct in enumerate(celltypes): # for each sample
        chrom, start, end = chroms[i], starts[i], ends[i]
        sample = None
        for a in assays:
            forward, reverse = [0 for i in range(end - start)], [0 for i in range(end - start)]
            peaks = peak_cache[ct][a]
            overlaps = peaks.intersect(chrom, start, end)
            for left, right, attrs in overlaps:
                left = max(0, left - start)
                right = min(end - start, right - start)
                enrich, strand = attrs[0:2]
                if strand == '+':
                    for i in range(left, right):
                        forward[i] = enrich
                else:
                    for i in range(left, right):
                        reverse[i] = enrich
            if pool_size > 1:
                forward = [np.max(forward[i * pool_size : (i + 1) * pool_size]) for i in range(bin_num)]
                reverse = [np.max(reverse[i * pool_size : (i + 1) * pool_size]) for i in range(bin_num)]
            forward, reverse = np.array(forward), np.array(reverse)
            assay_ar = np.array([forward, reverse]).astype(np.float16)
            if sample is None:
                sample = assay_ar.copy()
            else:
                sample = np.concatenate((sample, assay_ar), axis=0)
        if samples_channels_length is None:
            samples_channels_length = np.expand_dims(sample, axis=0)
        else:
            samples_channels_length = np.concatenate((samples_channels_length, 
                np.expand_dims(sample, axis=0)), axis=0)
    return samples_channels_length

# CTCF only %TODO: add more features
def prepare_loop_oriented_bed_peak(cell, dataset, outdir, oriented_bed_config, bin_size=5000, interval_len=3000000):
    print("Processing {} ... {}".format(dataset, time.asctime()))
    assert interval_len % bin_size == 0
    save_name = "{}/{}_3DPredictor.tsv.gz".format(outdir, cell)
    if os.path.exists(save_name):
        print("- {} exists, skipped.".format(save_name))
        return None
    # meta = np.load(metadata, allow_pickle=True)
    # e_loc = ((meta['e_start'] + meta['e_end']) // 2).squeeze()
    # p_loc = ((meta['p_start'] + meta['p_end']) // 2).squeeze()
    # ep_mid = (e_loc + p_loc) // 2
    df = pd.read_csv(dataset, header=None, delimiter='\t')
    dist = np.array(df[1])
    labels =np.array(df[0]) 
    chroms = np.array(df[2])
    e_start, e_end = np.array(df[3]), np.array(df[4])
    e_mid = ((e_start + e_end) / 2).astype(int)
    p_start, p_end = np.array(df[7]), np.array(df[8])
    p_mid = ((p_start + p_end) / 2).astype(int)
    ep_mid = ((e_mid + p_mid) / 2).astype(int)
    config = json.load(open(oriented_bed_config))
    bb = pyBigWig.open(os.path.join(config['location'], config['celltypes'][cell]['CTCF']))
    warn("Only use CTCF") # %XXX
    bin_num = interval_len // bin_size
    half_interal = interval_len // 2
    # TODO: parallel
    with gzip.open(save_name, 'wt') as out:
        out.write("label\tchrom\tdist")
        for i in range(bin_num):
            out.write("\tbin_{:04d}_+\tbin_{:04d}_-".format(i, i))
        for i, label in enumerate(labels):
            out.write("\n{:d}\t{:s}\t{:d}".format(label, chroms[i], int(dist[i])))
            left, right = ep_mid[i] - half_interal, ep_mid[i] + half_interal
            entries = bb.entries(chroms[i], max(left, 0), min(right, biock.hg19_chromsize[chroms[i]]))
            forward = [0 for t in range(bin_num)]
            reverse = [0 for t in range(bin_num)]
            if entries is not None:
                for l, r, attr in entries:
                    _, _, strand, signal, _, _, _ = attr.split('\t')
                    signal = float(signal)
                    if strand == '+':
                        for t in range(l - left, r - left):
                            if t >= interval_len:
                                break
                            forward[t // bin_size] = signal
                    else:
                        for t in range(l - left, r - left):
                            if t >= interval_len:
                                break
                            reverse[t // bin_size] = signal

            for t in range(bin_num):
                out.write("\t{:.3f}\t{:.3f}".format(forward[t], reverse[t]))
    bb.close()

GIMME_config = "/home/chenken/Documents/DeepEPI/data/genomic_features/gimme/gimme_new.json"
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('dataset')
    p.add_argument('-c', "--cell", required=True)
    p.add_argument('-od', "--outdir", required=True)
    p.add_argument('--bin-size', default=5000, type=int)
    #p.add_argument('--loop-bigwig', nargs='+', help="bigwig config for loop")
    p.add_argument('--bed_config', default=GIMME_config)
    p.add_argument('-t', "--nthreads", default=16, type=int)
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()

    biock.print_run_info(args)
    outdir = biock.make_directory(args.outdir)
    prepare_loop_oriented_bed_peak(args.cell, args.dataset, outdir, args.bed_config, bin_size=args.bin_size, interval_len=3000000)

