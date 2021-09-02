#!/bin/bash

# model=cv_models.pkl
# test_data="../data/BENGI.HeLa.CTCF-ChIAPET-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.HeLa.HiC-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.HeLa.RNAPII-ChIAPET-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.HMEC.HiC-Benchmark.v3.targetfinder_features.tsv"

model=GM12878-HeLa_cv_models.pkl
test_data="../data/BENGI.HMEC.HiC-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.IMR90.HiC-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.K562.HiC-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.NHEK.HiC-Benchmark.v3.targetfinder_features.tsv"

./predict.py \
    $test_data \
    -m $model -p `basename $model .pkl`
