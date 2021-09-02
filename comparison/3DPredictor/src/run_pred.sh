#!/bin/bash

# model=cv_models.pkl
# test_data="../data/BENGI.HeLa.CTCF-ChIAPET-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.HeLa.HiC-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.HeLa.RNAPII-ChIAPET-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.HMEC.HiC-Benchmark.v3.targetfinder_features.tsv"

model=GM12878-HeLa_cv_models.pkl
test_data="../data/BENGI/HMEC.HiC_3DPredictor.tsv.gz ../data/BENGI/IMR90.HiC_3DPredictor.tsv.gz ../data/BENGI/K562.HiC_3DPredictor.tsv.gz ../data/BENGI/NHEK.HiC_3DPredictor.tsv.gz"


# ../data/BENGI/GM12878.CTCF-ChIAPET_3DPredictor.tsv.gz
# ../data/BENGI/GM12878.HiC_3DPredictor.tsv.gz
# ../data/BENGI/GM12878.RNAPII-ChIAPET_3DPredictor.tsv.gz
# ../data/BENGI/HeLa.CTCF-ChIAPET_3DPredictor.tsv.gz
# ../data/BENGI/HeLa.HiC_3DPredictor.tsv.gz
# ../data/BENGI/HeLa.RNAPII-ChIAPET_3DPredictor.tsv.gz

./predict.py \
    $test_data \
    -m $model -p `basename $model .pkl`
