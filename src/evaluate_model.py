#!/usr/bin/env python3

import argparse, os, sys, time
import warnings, json, gzip
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset

import epi_models
import epi_dataset
import misc_utils


import functools
print = functools.partial(print, flush=True)


def model_summary(model):
    """
    model: pytorch model
    """
    import torch
    total_param = 0
    trainable_param = 0
    for i, p in enumerate(model.parameters()):
        num_p = torch.numel(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}



def predict(model: nn.Module, data_loader: DataLoader, device=torch.device('cuda')):
    model.eval()
    result, true_label = list(), list()
    for feats, _, enh_idxs, prom_idxs, labels in data_loader:
        feats, labels = feats.to(device), labels.to(device)
        # enh_idxs, prom_idxs = feats.to(device), prom_idxs.to(device)
        pred = model(feats, enh_idx=enh_idxs, prom_idx=prom_idxs)
        pred = pred.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        result.append(pred)
        true_label.append(labels)
    result = np.concatenate(result, axis=0)
    true_label = np.concatenate(true_label, axis=0)
    return (result.squeeze(), true_label.squeeze())


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-t', "--test-data", nargs='+', required=True, help="test dataset")
    p.add_argument('--test-chroms', nargs='+', required=False, default=['all'], help="chromosomes used for evaluation")
    p.add_argument('--gpu', default=-1, type=int, help="GPU ID, (-1 for CPU)")
    p.add_argument('--batch-size',  default=32, type=int, help="batch size")
    p.add_argument('--num-workers',  default=16, type=int, help="number of the processes used by data loader ")
    p.add_argument('-c', "--config", required=True, help="model configuration")
    p.add_argument('-m', "--model", required=True, help="path to trained model")
    p.add_argument('-p', "--prefix", required=True, help="prefix of output file")

    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

    if not torch.cuda.is_available():
        warnings.warn("GPU is not available")
        args.gpu = -1

    config = json.load(open(args.config))
    
    config["data_opts"]["datasets"] = args.test_data
    config["model_opts"]["rand_shift"] = False

    all_data = epi_dataset.EPIDataset(**config["data_opts"])

    config["model_opts"]["in_dim"] = all_data.feat_dim
    config["model_opts"]["seq_len"] = config["data_opts"]["seq_len"] // config["data_opts"]["bin_size"]

    labels = all_data.metainfo["label"]
    
    chroms = np.array(all_data.metainfo["chrom"])
    enh_names = np.array(all_data.metainfo["enh_name"])
    prom_names = np.array(all_data.metainfo["prom_name"])

    if args.test_chroms[0] != "all":
        config["train_opts"]["valid_chroms"] = args.test_chroms
        chroms = all_data.metainfo["chrom"]
        _, test_idx = misc_utils.split_np_array(chroms, test_chroms=config["train_opts"]["valid_chroms"])
        labels = labels[test_idx]
        all_data = Subset(all_data, indices=test_idx)
        chroms = chroms[test_idx]
        enh_names = enh_names[test_idx]
        prom_names = prom_names[test_idx]

    test_loader = DataLoader(
            all_data,
            shuffle=False, 
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')


    model_class = getattr(epi_models, config["model_opts"]["model"])
    model = model_class(**config["model_opts"]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device)["model_state_dict"])
    model.eval()

    pred, true = predict(model, test_loader, device)

    np.savetxt(
            "{}.prediction.txt".format(args.prefix),
            np.concatenate((
                true.reshape(-1, 1).astype(int).astype(str),
                pred.reshape(-1, 1).round(4).astype(str),
                chroms.reshape(-1, 1),
                enh_names.reshape(-1, 1),
                prom_names.reshape(-1, 1),
            ), axis=1),
            delimiter='\t',
            fmt="%s",
            comments="",
            header="##command: {}\n#true\tpred\tchrom\tenh_name\tprom_name".format(' '.join(sys.argv))
        )

    AUC, AUPR = misc_utils.evaluator(true, pred, out_keys=['AUC', 'AUPR'])

    label_counts = misc_utils.count_unique_itmes(labels)
    print("## datasets: {}".format(config["data_opts"]["datasets"]))
    print("## testing chroms: {}".format(args.test_chroms))
    print("## label classes: {}".format(label_counts))
    print("AUC:\t{:.4f}\nAUPR:\t{:.4f}({:.4f})".format(AUC, AUPR, AUPR / label_counts[1]))


