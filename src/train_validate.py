#!/usr/bin/env python3

import argparse, os, sys, time, shutil, tqdm
import warnings, json, gzip
import numpy as np
import copy
from sklearn.model_selection import GroupKFold

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

def split_train_valid_test(groups, train_keys, valid_keys, test_keys=None):
    """
    groups: length N, the number of samples
    train
    """
    assert isinstance(train_keys, list)
    assert isinstance(valid_keys, list)
    assert test_keys is None or isinstance(test_keys, list)
    index = np.arange(len(groups))
    train_idx = index[np.isin(groups, train_keys)]
    valid_idx = index[np.isin(groups, valid_keys)]
    if test_keys is not None:
        test_idx = index[np.isin(groups, test_keys)]
        return train_idx, valid_idx, test_idx
    else:
        return train_idx, valid_idx

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir

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
    result, true_label = None, None
    for feats, _, enh_idxs, prom_idxs, labels in data_loader:
        feats, labels = feats.to(device), labels.to(device)
        # enh_idxs, prom_idxs = enh_idxs.to(device), prom_idxs.to(device)
        pred = model(feats, enh_idx=enh_idxs, prom_idx=prom_idxs)
        pred = pred.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        if result is None:
            result = pred
            true_label = labels
        else:
            result = np.concatenate((result, pred), axis=0)
            true_label = np.concatenate((true_label, labels), axis=0)
    return (result.squeeze(), true_label.squeeze())


def train_validate_test(
        model, optimizer, 
        train_loader, valid_loader, test_loader, 
        num_epoch, patience, outdir, 
        checkpoint_prefix, device, use_scheduler=False) -> nn.Module:
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    wait = 0
    best_epoch, best_val_auc, best_val_aupr = -1, -1, -1

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    for epoch_idx in range(num_epoch):
        model.train()
        for feats, dists, enh_idxs, prom_idxs, labels in tqdm.tqdm(train_loader):
            feats, dists, labels = feats.to(device), dists.to(device), labels.to(device)
            if hasattr(model, "att_C"):
                pred, pred_dists, att = model(feats, enh_idxs, prom_idxs, return_att=True)
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1)).to(device)
                identity = Variable(identity.unsqueeze(0).expand(labels.size(0), att.size(1), att.size(1)))
                penal = model.l2_matrix_norm(torch.matmul(att, attT) - identity)
                loss = bce_loss(pred, labels) + (model.att_C * penal / labels.size(0)).type(torch.cuda.FloatTensor) +  mse_loss(dists, pred_dists)
                del penal, identity
            else:
                pred = model(feats, dists)
                loss = bce_loss(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()

        model.eval()
        valid_pred, valid_true = predict(model, valid_loader)
        val_AUC, val_AUPR = misc_utils.evaluator(valid_true, valid_pred, out_keys=["AUC", "AUPR"])
        print("\nvalid_result({})\t{:.4f}\t{:.4f}\t({})".format(epoch_idx, val_AUC, val_AUPR, time.asctime()))
        if val_AUC + val_AUPR > best_val_auc + best_val_aupr:
            wait = 0
            best_epoch, best_val_auc, best_val_aupr = epoch_idx, val_AUC, val_AUPR
            test_pred, test_true = predict(model, test_loader)
            np.savetxt(
                    "{}/test_result.{}.txt.gz".format(outdir, epoch_idx), 
                    X=np.concatenate((test_pred.reshape(-1, 1), test_true.reshape(-1, 1)), axis=1),
                    fmt="%.5f",
                    delimiter='\t'
                )
            test_AUC, test_AUPR = misc_utils.evaluator(test_true, test_pred, out_keys=["AUC", "AUPR"])
            print("Test_result\t{:.4f}\t{:.4f}\t({})".format(test_AUC, test_AUPR, time.asctime()))
            if use_scheduler:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    }, "{}/checkpoint.{}.pt".format(outdir, epoch_idx))
            else:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    }, "{}/checkpoint.{}.pt".format(outdir, epoch_idx))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopped ({})".format(time.asctime()))
                print("Best epoch/AUC/AUPR: {}\t{:.4f}\t{:.4f}".format(best_epoch, best_val_auc, best_val_aupr))
                break
            else:
                print("Wait{} ({})".format(wait, time.asctime()))


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
            '--train', 
            required=True, 
            nargs='+'
            )
    p.add_argument(
            '--valid', 
            required=True, 
            nargs='+'
        )
    p.add_argument(
            "--test", 
            nargs='+', 
            default=None,
            help="Optional test set"
        )
    p.add_argument('-b', "--batch-size", type=int, default=256)
    p.add_argument('-c', "--config", required=True)
    p.add_argument('-o', "--outdir", required=True)
    p.add_argument("--threads", default=32, type=int)
    p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = json.load(open(args.config))

    # all_data = epi_dataset.EPIDataset(**config["data_opts"])
    train_config = config.copy()
    train_config["data_opts"]["datasets"] = args.train
    train_config["data_opts"]["use_reverse"] = args.use_reverse
    train_config["data_opts"]["max_aug"] = args.aug_num
    train_data = epi_dataset.EPIDataset(
        **train_config["data_opts"]
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)

    if args.test is None:
        valid_test_config = copy.deepcopy(config)
        valid_test_config["data_opts"]["datasets"] = args.valid
        valid_test_data = epi_dataset.EPIDataset(
            **valid_test_config["data_opts"]
        )
        valid_idx, test_idx = split_train_valid_test(
                np.array(valid_test_data.metainfo["chrom"]), 
                train_keys=["chr{}".format(i).replace("23", "X") for i in range(1, 24, 2)],
                valid_keys=["chr{}".format(i) for i in range(2, 22, 2)]
            )
        valid_data = Subset(valid_test_data, indices=valid_idx)
        test_data = Subset(valid_test_data, indices=test_idx)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    else:
        valid_config = copy.deepcopy(config)
        valid_config["data_opts"]["datasets"] = args.valid
        valid_data = epi_dataset.EPIDataset(
            **valid_config["data_opts"]
        )
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

        test_config = copy.deepcopy(config)
        test_config["data_opts"]["datasets"] = args.test
        test_data = epi_dataset.EPIDataset(
            **test_config["data_opts"]
        )
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    config["model_opts"]["in_dim"] = train_data.feat_dim
    config["model_opts"]["seq_len"] = config["data_opts"]["seq_len"] // config["data_opts"]["bin_size"]

    print("##{}".format(time.asctime()))
    print("##command: {}".format(' '.join(sys.argv)))
    print("##args: {}".format(args))
    print("##config: {}".format(config))
    print("##sample size: {}".format(len(train_data)))
    print("## feature size: {}".format([v.size() for v in train_data.__getitem__(0)]))

    if args.gpu == -1:
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = "cuda"
    device = torch.device(device)

    model_class = getattr(epi_models, config["model_opts"]["model"])
    model = model_class(**config["model_opts"]).to(device)

    optimizer_params = {'lr': config["train_opts"]["learning_rate"], 'weight_decay': 0}
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    print(model)
    print(model_summary(model))
    print(optimizer)

    if not os.path.isdir(args.outdir):
        args.outdir = make_directory(args.outdir)

    train_validate_test(
            model, 
            optimizer, 
            train_loader, valid_loader, test_loader,
            num_epoch=config["train_opts"]["num_epoch"], 
            patience=config["train_opts"]["patience"], 
            outdir=args.outdir, 
            checkpoint_prefix="checkpoint",
            device=device,
            use_scheduler=config["train_opts"]["use_scheduler"]
        )

