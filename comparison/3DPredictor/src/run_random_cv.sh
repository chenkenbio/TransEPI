#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage: $0 config"
    exit 1
fi

config="$1"
prefix=`basename $config .json`

# bengi_train="../data/BENGI.GM12878.CTCF-ChIAPET-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.GM12878.HiC-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.GM12878.RNAPII-ChIAPET-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.IMR90.HiC-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.K562.HiC-Benchmark.v3.targetfinder_features.tsv ../data/BENGI.NHEK.HiC-Benchmark.v3.targetfinder_features.tsv"
bengi_train="../data/BENGI/GM12878.CTCF-ChIAPET_3DPredictor.tsv.gz ../data/BENGI/GM12878.HiC_3DPredictor.tsv.gz ../data/BENGI/GM12878.RNAPII-ChIAPET_3DPredictor.tsv.gz ../data/BENGI/HeLa.CTCF-ChIAPET_3DPredictor.tsv.gz ../data/BENGI/HeLa.HiC_3DPredictor.tsv.gz ../data/BENGI/HeLa.RNAPII-ChIAPET_3DPredictor.tsv.gz"


# ../data/BENGI.HMEC.HiC-Benchmark.v3.targetfinder_features.tsv

# ../data/Whalen.GM12878.HiC-Whalen.targetfinder_features.tsv
# ../data/Whalen.HeLa-S3.HiC-Whalen.targetfinder_features.tsv
# ../data/Whalen.HUVEC.HiC-Whalen.targetfinder_features.tsv
# ../data/Whalen.IMR90.HiC-Whalen.targetfinder_features.tsv
# ../data/Whalen.K562.HiC-Whalen.targetfinder_features.tsv
# ../data/Whalen.NHEK.HiC-Whalen.targetfinder_features.tsv


output="${prefix}_GM12878-HeLa"

log=${output}.cv.log

./random_grid_search.py \
    $bengi_train \
    -c cv_config.json \
    -n 50 \
    --threads 20 &> $log

# usage: random_grid_search.py [-h] [-g G] [-l L] [-e E [E ...]] -o O -c CONFIG
#                              [-n N] [--threads THREADS] [--seed SEED]
#                              datasets [datasets ...]
# 
# positional arguments:
#   datasets
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   -g G                  group (default: chrom)
#   -l L                  label (default: label)
#   -e E [E ...]          label (default: ['label', 'enh_name', 'prom_name',
#                         'celltype', 'chrom'])
#   -o O                  save name (default: None)
#   -c CONFIG, --config CONFIG
#                         json format, parameters (default: None)
#   -n N                  Number of iterations (default: 30)
#   --threads THREADS
#   --seed SEED
