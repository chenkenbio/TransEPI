#!/usr/bin/env python3

import argparse, os, sys, time
import warnings, json, gzip, pickle
#from collections import defaultdict, OrderedDict

import numpy as np
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr
from typing import Any, Dict, List, Union
from functools import partial
print = partial(print, flush=True)


def pr_auc_score(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(recall, precision)
    return aupr



def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('datasets', nargs='+')
    p.add_argument('-g', default='chrom', help="group")
    p.add_argument('-l', default='label', help="label")
    p.add_argument('-e', nargs='+',
            default=["label", "enh_name", "prom_name", "celltype", "chrom"], 
            help="label")
    p.add_argument('-o', required=True)
    p.add_argument('-c', "--config", help="json format, parameters")
    p.add_argument("--threads", type=int, default=32)
    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)


    if args.config is not None:
        config = json.load(open(args.config))
    else:
        config = dict()
    config["use_label_encoder"] = False
    config["n_jobs"] = args.threads

    df = None
    for i, fn in enumerate(args.datasets):
        if i == 0:
            df = pd.read_csv(fn, delimiter=',', compression='infer')
        else:
            df = pd.concat((df, pd.read_csv(fn, delimiter=',', compression='infer')), axis=0)
    
    excluded_keys = set(args.e).union({args.g, args.l})
    feature_names = sorted(list(set(df.columns).difference(excluded_keys)))
    
    groups = np.array(df[args.g])
    labels = np.array(df[args.l]).astype(int)
    features = np.array(df[feature_names])

    print(features.shape)


    splitter = GroupKFold(n_splits=5)
    AUC_list, AUPR_list = list(), list()
    models = list()
    for i, (train_idx, valid_idx) in enumerate(splitter.split(features, groups=groups)):
        train_X, train_y = features[train_idx, :], labels[train_idx]
        valid_X, valid_y = features[valid_idx, :], labels[valid_idx]
        xgb_model = xgb.XGBClassifier(**config)
        xgb_model.fit(train_X, train_y)
        models.append(xgb_model)

        pred = xgb_model.predict_proba(valid_X)[:, 1].squeeze()
        AUC = roc_auc_score(valid_y, pred)
        AUPR = pr_auc_score(valid_y, pred)
        AUC_list.append(AUC)
        AUPR_list.append(AUPR)
        print("\nFold: {}\t{} samples".format(i, len(valid_idx)))
        print("  groups: {}".format(np.unique(groups[valid_idx], return_counts=True)))
        print("  AUC/AUPR:\t{:.4f}\t{:.4f}".format(AUC, AUPR))

    print("\nAUC/AUPR:\t{:.4f}({:.4f})\t{:.4f}({:.4f})".format(np.mean(AUC_list), np.std(AUC_list), np.mean(AUPR_list), np.std(AUPR_list)))

    with open(args.o, 'wb') as handle:
        pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
