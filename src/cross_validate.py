#!/usr/bin/env python3

import argparse, os, sys, time, shutil, tqdm
import warnings, json, gzip
import numpy as np
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
        # enh_idxs, prom_idxs = feats.to(device), prom_idxs.to(device)
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


def train_transformer_model(
        model_class, model_params, 
        optimizer_class, optimizer_params, 
        dataset, groups, n_folds, 
        num_epoch, patience, batch_size, num_workers,
        outdir, checkpoint_prefix, device, use_scheduler=False) -> nn.Module:
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    splitter = GroupKFold(n_splits=n_folds)

    wait = 0
    best_epoch, best_val_auc, best_val_aupr = -1, -1, -1
    epoch_results = {"AUC": list(), "AUPR": list()}
    # if use_scheduler:
    #     schedulers = [None for _ in range(n_folds)]
    for epoch_idx in range(num_epoch):
        ## begin CV
        epoch_results["AUC"].append([-999 for _ in range(n_folds)])
        epoch_results["AUPR"].append([-999 for _ in range(n_folds)])
        if epoch_idx == 0:
            print("Fold splits(validation): ")
            for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X=groups, groups=groups)):
                print("  - Fold{}: {}({})".format(fold_idx, len(valid_idx), misc_utils.count_unique_itmes(groups[valid_idx])))

        print("\nCV epoch: {}/{}\t({})".format(epoch_idx, num_epoch, time.asctime()))
        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X=groups, groups=groups)):
            train_loader = DataLoader(Subset(dataset, indices=train_idx), shuffle=True, batch_size=batch_size, num_workers=num_workers)
            sample_idx = np.random.permutation(train_idx)[0:1024]
            sample_loader = DataLoader(Subset(dataset, indices=sample_idx), shuffle=False, batch_size=batch_size, num_workers=num_workers)
            valid_loader = DataLoader(Subset(dataset, indices=valid_idx), shuffle=False, batch_size=batch_size, num_workers=num_workers)
            checkpoint = "{}/{}_fold{}.pt".format(outdir, checkpoint_prefix, fold_idx)
            if epoch_idx == 0:
                model = model_class(**model_params).to(device)
                optimizer = optimizer_class(model.parameters(), **optimizer_params)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)
            else:
                state_dict = torch.load(checkpoint)
                model.load_state_dict(state_dict["model_state_dict"])
                optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                scheduler.load_state_dict(state_dict["scheduler_state_dict"])

            model.train()
            for feats, dists, enh_idxs, prom_idxs, labels in tqdm.tqdm(train_loader):
                feats, dists, labels = feats.to(device), dists.to(device), labels.to(device)
                if hasattr(model, "att_C"):
                    pred, pred_dists, att = model(feats, return_att=True, enh_idx=enh_idxs, prom_idx=prom_idxs)
                    # pred = model(feats, enh_idx=enh_idxs, prom_idx=prom_idxs)
                    attT = att.transpose(1, 2)
                    identity = torch.eye(att.size(1)).to(device)
                    identity = Variable(identity.unsqueeze(0).expand(labels.size(0), att.size(1), att.size(1)))
                    penal = model.l2_matrix_norm(torch.matmul(att, attT) - identity)

                    loss = bce_loss(pred, labels) + (model.att_C * penal / labels.size(0)).type(torch.cuda.FloatTensor) + mse_loss(dists, pred_dists)
                    del penal, identity
                else:
                    pred = model(feats, enh_idx=enh_idxs, prom_idx=prom_idxs)
                    loss = bce_loss(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if use_scheduler:
                    scheduler.step()
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
                }, checkpoint)

            model.eval()
            train_pred, train_true = predict(model, sample_loader)
            tra_AUC, tra_AUPR = misc_utils.evaluator(train_true, train_pred, out_keys=["AUC", "AUPR"])
            valid_pred, valid_true = predict(model, valid_loader)
            val_AUC, val_AUPR = misc_utils.evaluator(valid_true, valid_pred, out_keys=["AUC", "AUPR"])
            print("  - Fold{}:train(AUC/AUPR)/vald(AUC/AUPR):\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t({})".format(fold_idx, tra_AUC, tra_AUPR, val_AUC, val_AUPR, time.asctime()))
            epoch_results["AUC"][-1][fold_idx] = val_AUC
            epoch_results["AUPR"][-1][fold_idx] = val_AUPR
        auc_mean, auc_std = np.mean(epoch_results["AUC"][-1]), np.std(epoch_results["AUC"][-1])
        aupr_mean, aupr_std = np.mean(epoch_results["AUPR"][-1]), np.std(epoch_results["AUPR"][-1])
        print("Epoch{:03d}(AUC/AUPR):\t{:.4f}({:.4f})\t{:.4f}({:.4f})".format(epoch_idx, auc_mean, auc_std, aupr_mean, aupr_std))
        if auc_mean >= best_val_auc and aupr_mean >= best_val_aupr:
            wait = 0
            best_epoch, best_val_auc, best_val_aupr = epoch_idx, auc_mean, aupr_mean
            print("Best epoch {}\t({})".format(best_epoch, time.asctime()))
            for i in range(n_folds):
                checkpoint = "{}/{}_fold{}.pt".format(outdir, checkpoint_prefix, i)
                shutil.copyfile(checkpoint, "{}.best_epoch{}.pt".format(checkpoint.replace('.pt', ''), best_epoch))
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
    p.add_argument('-c', "--config", required=True, help="Configuration file for training the model")
    p.add_argument('-o', "--outdir", required=True, help="Output directory")
    p.add_argument('--gpu', default=-1, type=int, help="GPU ID, (-1 for CPU)")
    p.add_argument('--seed', type=int, default=2020, help="Random seed")
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = json.load(open(args.config))

    print(config["data_opts"])
    all_data = epi_dataset.EPIDataset(**config["data_opts"])

    config["model_opts"]["in_dim"] = all_data.feat_dim
    config["model_opts"]["seq_len"] = config["data_opts"]["seq_len"] // config["data_opts"]["bin_size"]

    print("##{}".format(time.asctime()))
    print("##command: {}".format(' '.join(sys.argv)))
    print("##config: {}".format(config))
    print("##sample size: {}".format(len(all_data)))
    torch.save(all_data.__getitem__(0), "tmp.pt")
    # print("##feature: {}".format(all_data.__getitem__(0)[0].size())) #squeeze(0).mean(dim=1)))
    # print("##feature: {}".format(all_data.__getitem__(0)[0].mean(dim=1)))
    # print("##feature: {}".format(all_data.__getitem__(0)[0][:, 2400:2600].mean(dim=0)))

    chroms = all_data.metainfo["chrom"]


    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model_class = getattr(epi_models, config["model_opts"]["model"])
    model = model_class(**config["model_opts"])

    print(model)
    print(model_summary(model))
    del model
    optimizer_params = {'lr': config["train_opts"]["learning_rate"], 'weight_decay': 1e-8}

    if not os.path.isdir(args.outdir):
        args.outdir = make_directory(args.outdir)

    train_transformer_model(
            model_class=model_class, 
            model_params=config["model_opts"],
            optimizer_class=torch.optim.Adam, 
            optimizer_params=optimizer_params,
            dataset=all_data,
            groups=all_data.metainfo["chrom"],
            n_folds=5,
            num_epoch=config["train_opts"]["num_epoch"], 
            patience=config["train_opts"]["patience"], 
            batch_size=config["train_opts"]["batch_size"], 
            num_workers=config["train_opts"]["num_workers"],
            outdir=args.outdir, 
            checkpoint_prefix="checkpoint",
            device=device,
            use_scheduler=config["train_opts"]["use_scheduler"]
        )

