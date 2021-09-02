#!/usr/bin/env python3

import argparse, functools, glob, gzip, json, os, pickle, sys, time, pyBigWig
from warnings import warn
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.model_selection import GroupKFold, StratifiedKFold
import pyBigWig

import biock.biock as biock
from biock.biock import print


# def parse_narrowpeak(cell, dataset, outdir, narrowpeak_config, assay="H3K27ac"):
#     print("Processing {} ... {}".format(dataset, time.asctime()))
#     save_name = "{}/{}_3DPredictor.{}.tsv.gz".format(outdir, cell, assay)
#     if os.path.exists(save_name):
#         print("- {} exists, skipped.".format(save_name))
#         return None
#     # meta = np.load(dataset, allow_pickle=True)
#     df = pd.read_csv(dataset, header=None, delimiter='\t')
#     # e_loc = ((meta['e_start'] + meta['e_end']) // 2).squeeze()
#     # p_loc = ((meta['p_start'] + meta['p_end']) // 2).squeeze()
#     dist = np.array(df[1])
#     labels =np.array(df[0]) 
#     chroms = np.array(df[2])
#     e_start, e_end = np.array(df[3]), np.array(df[4])
#     e_loc = ((e_start + e_end) / 2).astype(int)
#     p_start, p_end = np.array(df[7]), np.array(df[8])
#     p_loc = ((p_start + p_end) / 2).astype(int)
#     config = json.load(open(narrowpeak_config))
#     print("File: {}", os.path.join(config['location'], config['celltypes'][cell]['H3K27ac']))
#     bb = pyBigWig.open(os.path.join(config['location'], config['celltypes'][cell]['H3K27ac']))
#     warn("Only use H3K27ac") # %XXX
#     n_total = len(dist)
#     # TODO: parallel
#     with gzip.open(save_name, 'wt') as out:
#         header = "label\tchrom\tdist\t{left}\t{right}\t{window}\t{loci1}\t{loci2}".format(
#                 left='\t'.join([
#                     "L1_H3K27ac", "L1_H3K27ac_dist", \
#                     "L2_H3K27ac", "L2_H3K27ac_dist", \
#                     "L3_H3K27ac", "L3_H3K27ac_dist", \
#                     "L4_H3K27ac", "L4_H3K27ac_dist"]),
#                 right='\t'.join([
#                     "R1_H3K27ac", "R1_H3K27ac_dist", \
#                     "R2_H3K27ac", "R2_H3K27ac_dist", \
#                     "R3_H3K27ac", "R3_H3K27ac_dist", \
#                     "R4_H3K27ac", "R4_H3K27ac_dist"]),
#                 window='\t'.join([
#                     "W_H3K27ac_sum", "W_H3K27ac_count", \
#                     "W1_H3K27ac", "W1_H3K27ac_d1", "W1_H3K27ac_d2", \
#                     "W2_H3K27ac", "W2_H3K27ac_d1", "W2_H3K27ac_d2", \
#                     "W3_H3K27ac", "W3_H3K27ac_d1", "W3_H3K27ac_d2", \
#                     "W1_H3K27ac", "W4_H3K27ac_d1", "W1_H3K27ac_d2", \
#                     "W5_H3K27ac", "W5_H3K27ac_d1", "W5_H3K27ac_d2", \
#                     "W6_H3K27ac", "W6_H3K27ac_d1", "W6_H3K27ac_d2", \
#                     "W7_H3K27ac", "W7_H3K27ac_d1", "W7_H3K27ac_d2", \
#                     "W8_H3K27ac", "W8_H3K27ac_d1", "W8_H3K27ac_d2"]),
#                 loci1="l1_H3K27ac",
#                 loci2="l2_H3K27ac")
#         out.write(header)
#         for i in range(n_total):
#             l1, l2 = min(e_loc[i], p_loc[i]), max(e_loc[i], p_loc[i])
#             peaks_l = bb.entries(chroms[i], max(l1 - 3000000, 0), max(l1 - 12500, 0))
#             l_features = ['0\t0' for t in range(4)]
#             if peaks_l is None:
#                 peaks_l = []
#             for idx, (l, r, attr) in enumerate(peaks_l[::-1][:4]):
#                 _, _, _, signal = attr.split('\t')[0:4]
#                 signal = float(signal)
#                 l_features[idx] = '\t'.join(["{:.3f}".format(signal), \
#                         "{:d}".format(abs(l1 - (l + r) // 2))])
#             peaks_r = bb.entries(chroms[i], min(l2 + 12500, biock.hg19_chromsize[chroms[i]]), min(l2 + 3000000, biock.hg19_chromsize[chroms[i]]))
#             r_features = ['0\t0' for t in range(4)]
#             if peaks_r is None:
#                 peaks_r = []
#             for idx, (l, r, attr) in enumerate(peaks_r[:4]):
#                 _, _, _, signal = attr.split('\t')[0:4]
#                 signal = float(signal)
#                 r_features[idx] = '\t'.join(["{:.3f}".format(signal), \
#                         "{:d}".format(abs(l2 - (l + r) // 2))])
#             peaks_w = bb.entries(chroms[i], l1 , l2 )
#             if peaks_w is None:
#                 peaks_w = []
#             else:
#                 tmp = list()
#                 for l, r, attr in peaks_w:
#                     _, _, _, signal = attr.split('\t')[0:4]
#                     signal = float(signal)
#                     tmp.append((signal, (l, r, attr)))
#                     sorted(tmp, key=lambda l:l[0], reverse=True)
#                 peaks_w = [t[1] for t in tmp]
#                 del tmp
#             w_h3k27ac_sum, w_h3k27ac_count = 0, len(peaks_w)
#             w_features = ['0\t0\t0' for t in range(8)]
#             for idx, (l, r, attr) in enumerate(peaks_w):
#                 _, _, _, signal = attr.split('\t')[0:4]
#                 signal = float(signal)
#                 w_h3k27ac_sum += signal
#                 if idx < 8:
#                     w_features[idx] = '\t'.join(["{:.3f}".format(signal), \
#                             "{:d}".format(abs((l + r) // 2 - l1)), \
#                             "{:d}".format(abs((l + r) // 2 - l2))])
#             peaks_l1 = bb.entries(chroms[i], l1 - 12500, l1 + 12500)
#             peaks_l2 = bb.entries(chroms[i], l2 - 12500, l2 + 12500)
#             if peaks_l1 is None:
#                 peaks_l1 = []
#             if peaks_l2 is None:
#                 peaks_l2 = []
#             l1_h3k27ac = 0
#             for l, r, attr in peaks_l1:
#                 _, _, _, signal = attr.split('\t')[0:4]
#                 signal = float(signal)
#                 l1_h3k27ac += signal
#             l2_h3k27ac = 0
#             for l, r, attr in peaks_l2:
#                 _, _, _, signal = attr.split('\t')[0:4]
#                 signal = float(signal)
#                 l2_h3k27ac += signal
#             out.write("\n{label}\t{chrom}\t{dist}\t{left}\t{right}\t{window}\t{loci1}\t{loci2}".format(
#                 label=labels[i], chrom=chroms[i], dist=dist[i], 
#                 left='\t'.join(l_features), \
#                 right='\t'.join(r_features), \
#                 window="{}\t{}\t{}".format(w_h3k27ac_sum, w_h3k27ac_count, '\t'.join(w_features)), \
#                 loci1="{:.3f}".format(l1_h3k27ac), \
#                 loci2="{:.3f}".format(l2_h3k27ac)))
#     bb.close()



# CTCF only %TODO: add more features
def parse_gimme_peaks(cell, dataset, output, oriented_gimme_config):
    print("Processing {} ... {}".format(dataset, time.asctime()))
    save_name = output
    if os.path.exists(save_name):
        print("- {} exists, skipped.".format(save_name))
        return None
    df = pd.read_csv(dataset, header=None, delimiter='\t')
    dist = np.array(df[1])
    labels =np.array(df[0]) 
    chroms = np.array(df[2])
    e_start, e_end = np.array(df[3]), np.array(df[4])
    e_loc = ((e_start + e_end) / 2).astype(int)
    p_start, p_end = np.array(df[7]), np.array(df[8])
    p_loc = ((p_start + p_end) / 2).astype(int)

    config = json.load(open(oriented_gimme_config))
    bb = pyBigWig.open(os.path.join(config['location'], config['celltypes'][cell]['CTCF']))
    warn("Only use CTCF") # %XXX
    n_total = len(dist)
    # TODO: parallel
    with gzip.open(save_name, 'wt') as out:
        header = "label\tchrom\tdist\t{left}\t{right}\t{window}\t{loci1}\t{loci2}".format(
                left='\t'.join(["L1_CTCF", "L1_CTCF+", "L1_CTCF-", "L1_CTCF_dist", \
                        "L2_CTCF", "L2_CTCF+", "L2_CTCF-", "L2_CTCF_dist", \
                        "L3_CTCF", "L3_CTCF+", "L3_CTCF-", "L3_CTCF_dist", \
                        "L4_CTCF", "L4_CTCF+", "L4_CTCF-", "L4_CTCF_dist"]),
                right='\t'.join(["R1_CTCF", "R1_CTCF+", "R1_CTCF-", "R1_CTCF_dist", \
                        "R2_CTCF", "R2_CTCF+", "R2_CTCF-", "R2_CTCF_dist", \
                        "R3_CTCF", "R3_CTCF+", "R3_CTCF-", "R3_CTCF_dist", \
                        "R4_CTCF", "R4_CTCF+", "R4_CTCF-", "R4_CTCF_dist"]),
                window='\t'.join(["W_CTCF_sum", "W_CTCF_count", \
                        "W1_CTCF", "W1_CTCF+", "W1_CTCF-", "W1_CTCF_d1", "W1_CTCF_d2", \
                        "W2_CTCF", "W2_CTCF+", "W2_CTCF-", "W2_CTCF_d1", "W2_CTCF_d2", \
                        "W3_CTCF", "W3_CTCF+", "W3_CTCF-", "W3_CTCF_d1", "W3_CTCF_d2", \
                        "W1_CTCF", "W1_CTCF+", "W1_CTCF-", "W4_CTCF_d1", "W1_CTCF_d2", \
                        "W5_CTCF", "W5_CTCF+", "W5_CTCF-", "W5_CTCF_d1", "W5_CTCF_d2", \
                        "W6_CTCF", "W6_CTCF+", "W6_CTCF-", "W6_CTCF_d1", "W6_CTCF_d2", \
                        "W7_CTCF", "W7_CTCF+", "W7_CTCF-", "W7_CTCF_d1", "W7_CTCF_d2", \
                        "W8_CTCF", "W8_CTCF+", "W8_CTCF-", "W8_CTCF_d1", "W8_CTCF_d2"]),
                loci1="l1_CTCF\tl1_CTCF+\tl1_CTCF-",
                loci2="l2_CTCF\tl2_CTCF+\tl2_CTCF-")
        out.write(header)
        for i in range(n_total):
            l1, l2 = min(e_loc[i], p_loc[i]), max(e_loc[i], p_loc[i])
            peaks_l = bb.entries(chroms[i], max(l1 - 3000000, 0), max(l1 - 12500, 0))
            l_features = ['0\t0\t0\t0' for t in range(4)]
            if peaks_l is None:
                peaks_l = []
            for idx, (l, r, attr) in enumerate(peaks_l[::-1][:4]):
                motif_val, _, strand, signal, _, _, _ = attr.split('\t')
                motif_val, signal = float(motif_val.split('_')[-1]), float(signal)
                l_features[idx] = '\t'.join(["{:.3f}".format(signal), \
                        "{:.3f}".format(motif_val) if strand == '+' else '0', \
                        "{:.3f}".format(motif_val) if strand == '-' else '0', \
                        "{:d}".format(abs(l1 - (l + r) // 2))])
            peaks_r = bb.entries(chroms[i], min(l2 + 12500, biock.hg19_chromsize[chroms[i]]), min(l2 + 3000000, biock.hg19_chromsize[chroms[i]]))
            r_features = ['0\t0\t0\t0' for t in range(4)]
            if peaks_r is None:
                peaks_r = []
            for idx, (l, r, attr) in enumerate(peaks_r[:4]):
                motif_val, _, strand, signal, _, _, _ = attr.split('\t')
                motif_val, signal = float(motif_val.split('_')[-1]), float(signal)
                r_features[idx] = '\t'.join(["{:.3f}".format(signal), \
                        "{:.3f}".format(motif_val) if strand == '+' else '0', \
                        "{:.3f}".format(motif_val) if strand == '-' else '0', \
                        "{:d}".format(abs(l2 - (l + r) // 2))])
            peaks_w = bb.entries(chroms[i], l1 , l2 )
            if peaks_w is None:
                peaks_w = []
            else:
                tmp = list()
                for l, r, attr in peaks_w:
                    motif_val, _, strand, signal, _, _, _ = attr.split('\t')
                    signal, motif_val = float(signal), motif_val.split('_')[-1]
                    tmp.append((signal, (l, r, attr)))
                    sorted(tmp, key=lambda l:l[0], reverse=True)
                peaks_w = [t[1] for t in tmp]
                del tmp
            w_ctcf_sum, w_ctcf_count = 0, len(peaks_w)
            w_features = ['0\t0\t0\t0\t0' for t in range(8)]
            for idx, (l, r, attr) in enumerate(peaks_w):
                motif_val, _, strand, signal, _, _, _ = attr.split('\t')
                signal, motif_val = float(signal), motif_val.split('_')[-1]
                w_ctcf_sum += signal
                if idx < 8:
                    w_features[idx] = '\t'.join(["{:.3f}".format(signal), \
                            "{:.3f}".format(float(motif_val)) if strand == '+' else '0', \
                            "{:.3f}".format(float(motif_val)) if strand == '-' else '0', \
                            "{:d}".format(abs((l + r) // 2 - l1)), \
                            "{:d}".format(abs((l + r) // 2 - l2))])
            peaks_l1 = bb.entries(chroms[i], l1 - 12500, l1 + 12500)
            peaks_l2 = bb.entries(chroms[i], l2 - 12500, l2 + 12500)
            if peaks_l1 is None:
                peaks_l1 = []
            if peaks_l2 is None:
                peaks_l2 = []
            l1_ctcf, l1_ctcf_p, l1_ctcf_n = 0, 0, 0
            for l, r, attr in peaks_l1:
                motif_val, _, strand, signal, _, _, _ = attr.split('\t')
                signal, motif_val = float(signal), float(motif_val.split('_')[-1])
                l1_ctcf += signal
                l1_ctcf_p += (motif_val if strand == '+' else 0)
                l1_ctcf_n += (motif_val if strand == '-' else 0)
            l2_ctcf, l2_ctcf_p, l2_ctcf_n = 0, 0, 0
            for l, r, attr in peaks_l2:
                motif_val, _, strand, signal, _, _, _ = attr.split('\t')
                signal, motif_val = float(signal), float(motif_val.split('_')[-1])
                l2_ctcf += signal
                l2_ctcf_p += (motif_val if strand == '+' else 0)
                l2_ctcf_n += (motif_val if strand == '-' else 0)
            out.write("\n{label}\t{chrom}\t{dist}\t{left}\t{right}\t{window}\t{loci1}\t{loci2}".format(
                label=labels[i], chrom=chroms[i], dist=dist[i], 
                left='\t'.join(l_features), \
                right='\t'.join(r_features), \
                window="{}\t{}\t{}".format(w_ctcf_sum, w_ctcf_count, '\t'.join(w_features)), \
                loci1="{:.3f}\t{:.3f}\t{:.3f}".format(l1_ctcf, l1_ctcf_p, l1_ctcf_n), \
                loci2="{:.3f}\t{:.3f}\t{:.3f}".format(l2_ctcf, l2_ctcf_p, l2_ctcf_n)))
    bb.close()


GIMME_config = "/home/chenken/Documents/DeepEPI/data/genomic_features/gimme/gimme_new.json"
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('dataset')
    p.add_argument('-c', "--cell", required=True)
    # p.add_argument('-od', "--outdir", required=True)
    p.add_argument('-o', required=True)
    p.add_argument('--bin-size', default=5000, type=int)
    #p.add_argument('--loop-bigwig', nargs='+', help="bigwig config for loop")
    p.add_argument('--gimme_config', default=GIMME_config) # "/data2/users/chenken/Documents/EPIGL/data/gimme/gimme_new.json")
    # p.add_argument('--np_config', default="/data2/users/chenken/Documents/EPIGL/data/bed_config/epigenetic.json")
    p.add_argument('-t', "--nthreads", default=16, type=int)
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()

    biock.print_run_info(args)
    # outdir = biock.make_directory(args.outdir)
    parse_gimme_peaks(args.cell, args.dataset, args.o, args.gimme_config)
    # parse_narrowpeak(args.cell, args.dataset, outdir, args.np_config, assay="H3K27ac")
