#!/usr/bin/env python3

import argparse, os, sys, json, pickle, time
import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

sys.stdout.flush()

def aupr_score(y_true, probas_pred):
    precision, recall = precision_recall_curve(y_true, probas_pred)
    return auc(recall, precision)


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('datasets', nargs='+')
    p.add_argument('-g', default='chrom', help="group")
    p.add_argument('-l', default='label', help="label")
    p.add_argument('-e', nargs='+',
            default=["label", "celltype", "chrom"], 
            help="label")
    # p.add_argument('-p', required=True, help="output prefix")
    p.add_argument('-c', "--config", required=True, help="json format, parameters")
    p.add_argument('-n', type=int, default=30, help="Number of iterations")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()

    df = None
    for i, fn in enumerate(args.datasets):
        if i == 0:
            df = pd.read_csv(fn, delimiter='\t', compression='infer')
        else:
            df = pd.concat((df, pd.read_csv(fn, delimiter='\t', compression='infer')), axis=0)
    
    excluded_keys = set(args.e).union({args.g, args.l})
    feature_names = sorted(list(set(df.columns).difference(excluded_keys)))
    
    groups = np.array(df[args.g])
    labels = np.array(df[args.l]).astype(int)
    features = np.array(df[feature_names])

    config = json.load(open(args.config))

    # config["use_label_encoder"] = False

    print("## {}".format(time.asctime()))
    print("# command: {}".format(' '.join(sys.argv)))
    print("# args: {}".format(args))
    print("# config: {}".format(config))

    xgb_model = xgb.XGBClassifier(n_jobs=args.threads, use_label_encoder=False)

    splitter = GroupKFold(n_splits=5).split(X=features, y=labels, groups=groups)
    clf = RandomizedSearchCV(
        xgb_model, 
        param_distributions=config, 
        scoring="roc_auc",
        n_iter=args.n,
        cv=splitter,
        random_state=args.seed,
        verbose=3,
        n_jobs=5
    )

    search = clf.fit(features, labels)

    # print(search)
    # with open(args.o, 'wb') as handle:
    #     pickle.dump(
    #         {"clf": clf, "result": dict(search)}, 
    #         handle, 
    #         protocol=pickle.HIGHEST_PROTOCOL
    #     )
    results = dict()
    results["best_score"] = search.best_score_
    results["best_params"] = search.best_params_

    json.dump(results, fp=sys.stdout, indent=4)
