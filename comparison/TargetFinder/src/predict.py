#!/usr/bin/env python3

import argparse, os, sys, time
import warnings, json, gzip, pickle
#from collections import defaultdict, OrderedDict

import numpy as np
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr
from typing import Any, Dict, List, Union
from functools import partial
print = partial(print, flush=True)

def split_np_array(ar, test_chroms):
    ar = np.array(ar).squeeze()
    assert len(ar.shape) == 1
    all_index = np.arange(0, ar.shape[0], 1).astype(int)
    in_test = np.isin(ar, test_chroms)
    train_idx = all_index[np.logical_not(in_test)]
    test_idx = all_index[in_test]
    return train_idx, test_idx


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('datasets', nargs='+')
    p.add_argument('-m', required=True)
    p.add_argument('-g', default='chrom', help="group")
    p.add_argument('-l', default='label', help="label")
    p.add_argument('-e', nargs='+',
            default=["label", "enh_name", "prom_name", "celltype", "chrom"], 
            help="label")
    p.add_argument('-p', required=True)
    # p.add_argument('-c', "--config", help="json format, parameters")
    p.add_argument("--threads", type=int, default=32)
    #p.add_argument('--seed', type=int, default=2020)
    return p

FOLD_SPLITS = {
        0: ["chr1", "chr10", "chr15", "chr21"],
        1: ["chr19", "chr3", "chr4", "chr7", "chrX"],
        2: ["chr13", "chr17", "chr2", "chr22", "chr9"],
        3: ["chr12", "chr14", "chr16", "chr18", "chr20"],
        4: ["chr11", "chr5", "chr6", "chr8"]
        }



if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

    with open(args.m, 'rb') as handle:
        models = pickle.load(handle)

    for fn in args.datasets:
        bn = os.path.basename(fn).replace(".tsv.gz", '')
        auc_list, aupr_list = list(), list()
        df = pd.read_csv(fn, delimiter=',', compression='infer')
        groups = np.array(df["chrom"])
        excluded_keys = set(args.e).union({args.g, args.l})
        feature_names = sorted(list(set(df.columns).difference(excluded_keys)))
    
        labels = np.array(df[args.l])
        features = np.array(df[feature_names])

        save_label = list()
        save_prob = list()
        for fold, m in enumerate(models):
            _, test_idx = split_np_array(groups, FOLD_SPLITS[fold])
            print(fold, np.unique(groups[test_idx]), max(test_idx), min(test_idx))
            pred = m.predict_proba(features[test_idx, :])[:, 1].squeeze()
            auc_list.append(roc_auc_score(labels[test_idx], pred))
            aupr_list.append(average_precision_score(labels[test_idx], pred))

            save_label.append(labels[test_idx].reshape(-1))
            save_prob.append(pred.reshape(-1))

        
        print("\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})".format(
            fn, 
            np.mean(auc_list), np.std(auc_list),
            np.mean(aupr_list), np.std(aupr_list)
            ))
        print("AUC:\t{}".format("\t".join(["{:.4f}".format(x) for x in auc_list])))
        print("AUPR:\t{}".format("\t".join(["{:.4f}".format(x) for x in aupr_list])))

        np.savetxt(
                "{}_{}.prediction.txt".format(args.p, bn),
                np.concatenate((
                        np.concatenate(save_label).reshape(-1, 1), 
                        np.concatenate(save_prob).reshape(-1, 1)
                    ), axis=1),
                delimiter='\t',
                header=fn
        )
