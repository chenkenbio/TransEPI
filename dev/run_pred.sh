#!/bin/bash

## cv

# BENGI_train="../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz ../data/BENGI/GM12878.HiC-Benchmark.v3.tsv.gz ../data/BENGI/GM12878.RNAPII-ChIAPET-Benchmark.v3.tsv.gz ../data/BENGI/HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz ../data/BENGI/HeLa.HiC-Benchmark.v3.tsv.gz ../data/BENGI/HeLa.RNAPII-ChIAPET-Benchmark.v3.tsv.gz"
# # Fold splits(validation): 
fold0="chr1 chr10 chr15 chr21"
fold1="chr19 chr3 chr4 chr7 chrX"
fold2="chr13 chr17 chr2 chr22 chr9"
fold3="chr12 chr14 chr16 chr18 chr20"
fold4="chr11 chr5 chr6 chr8"
# 
# 
# 
# ../../src/evaluate_model.py -t $BENGI_train --batch-size 128 -c "./config_500bp_07.json" -m ./config_500bp_07_cv/checkpoint_fold0.best_epoch1.pt -p results_BENGI-train.fold0 --test-chroms $fold0 & sleep 30
# ../../src/evaluate_model.py -t $BENGI_train --batch-size 128 -c "./config_500bp_07.json" -m ./config_500bp_07_cv/checkpoint_fold1.best_epoch1.pt -p results_BENGI-train.fold1 --test-chroms $fold1 & sleep 30
# ../../src/evaluate_model.py -t $BENGI_train --batch-size 128 -c "./config_500bp_07.json" -m ./config_500bp_07_cv/checkpoint_fold2.best_epoch1.pt -p results_BENGI-train.fold2 --test-chroms $fold2 & sleep 30
# ../../src/evaluate_model.py -t $BENGI_train --batch-size 128 -c "./config_500bp_07.json" -m ./config_500bp_07_cv/checkpoint_fold3.best_epoch1.pt -p results_BENGI-train.fold3 --test-chroms $fold3 & sleep 30
# ../../src/evaluate_model.py -t $BENGI_train --batch-size 128 -c "./config_500bp_07.json" -m ./config_500bp_07_cv/checkpoint_fold4.best_epoch1.pt -p results_BENGI-train.fold4 --test-chroms $fold4 & sleep 30
# wait
# exit 0 

for fn in ../data/BENGI/K562.HiC-Benchmark.v3.tsv.gz ../data/BENGI/NHEK.HiC-Benchmark.v3.tsv.gz ../data/BENGI/HMEC.HiC-Benchmark.v3.tsv.gz ../data/BENGI/IMR90.HiC-Benchmark.v3.tsv.gz; do
    echo $fn
    bn=`basename $fn .tsv`
    for fold in 0 1 2 3 4; do
        echo "- Fold $fold"
        if   [ $fold -eq 0 ]; then
            ../src/evaluate_model.py --batch-size 64 -t $fn -c "config_500bp.json" --test-chroms $fold0 -m  ../models/TransEPI_EPI_fold0.pt -p results_${bn}.fold${fold}_strict
        elif [ $fold -eq 1 ]; then
            ../src/evaluate_model.py --batch-size 64 -t $fn -c "config_500bp.json" --test-chroms $fold1 -m ../models/TransEPI_EPI_fold1.pt -p results_${bn}.fold${fold}_strict
        elif [ $fold -eq 2 ]; then
            ../src/evaluate_model.py --batch-size 64 -t $fn -c "config_500bp.json" --test-chroms $fold2 -m ../models/TransEPI_EPI_fold2.pt -p results_${bn}.fold${fold}_strict
        elif [ $fold -eq 3 ]; then
            ../src/evaluate_model.py --batch-size 64 -t $fn -c "config_500bp.json" --test-chroms $fold3 -m ../models/TransEPI_EPI_fold3.pt -p results_${bn}.fold${fold}_strict
        elif [ $fold -eq 4 ]; then
            ../src/evaluate_model.py --batch-size 64 -t $fn -c "config_500bp.json" --test-chroms $fold4 -m ../models/TransEPI_EPI_fold4.pt  -p results_${bn}.fold${fold}_strict
        fi

    done &> results_${bn}_strict.log &
    sleep 20
done
wait
# usage: evaluate_model.py [-h] -t TEST_DATA [TEST_DATA ...]
#                          [--test-chroms TEST_CHROMS [TEST_CHROMS ...]] -c
#                          CONFIG -m MODEL -p PREFIX
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   -t TEST_DATA [TEST_DATA ...], --test-data TEST_DATA [TEST_DATA ...]
#   --test-chroms TEST_CHROMS [TEST_CHROMS ...]
#   -c CONFIG, --config CONFIG
#   -m MODEL, --model MODEL
#   -p PREFIX, --prefix PREFIX


