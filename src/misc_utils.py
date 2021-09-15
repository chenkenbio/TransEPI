#!/usr/bin/env python3

import argparse, os, sys, time
#import warnings, json, gzip

import numpy as np
import sklearn.metrics as metrics

hg19_chromsize = {"chr1": 249250621, "chr2": 243199373, 
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
hg38_chromsize = {"chr1": 248956422, "chr2": 242193529,
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

def overlap_length(x1, x2, y1, y2):
    """ [x1, x2), [y1, y2) """
    length = 0
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    if x2 <= y1:
        length = x2 - y1
    elif x1 <= y2:
        length = min(x2, y2) - max(x1, y1)
    else:
        length = y2 - x1
    return length


def split_np_array(ar, test_chroms=["chr20", "chr21", "chr22", "chrX"]):
    ar = np.array(ar).squeeze()
    assert len(ar.shape) == 1
    all_index = np.arange(0, ar.shape[0], 1).astype(int)
    in_test = np.isin(ar, test_chroms)
    train_idx = all_index[np.logical_not(in_test)]
    test_idx = all_index[in_test]
    return train_idx, test_idx


def pr_auc_score(y_true, y_prob):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(recall, precision)
    return aupr


def evaluator(y_true, y_prob, out_keys: list=None, **kwargs):
    results = dict()
    results["AUC"] = metrics.roc_auc_score(y_true, y_prob)
    results["AUPR"] = pr_auc_score(y_true, y_prob)
    results["F1"] = metrics.f1_score(y_true, y_prob.round().astype(int))

    if out_keys is not None:
        results_ = list()
        for k in out_keys:
            results_.append(results[k])
        results = results_
    return results


def count_unique_itmes(ar):
    results = dict()
    total = len(ar)
    for name, count in zip(*np.unique(ar, return_counts=True)):
        results[name] = round(count / total, 3)
    return results

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument()

    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

