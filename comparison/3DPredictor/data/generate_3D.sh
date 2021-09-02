#!/bin/bash

#./prepare_features.py --metadata ../../data/bengi/All-Pairs.Natural-Ratio/HeLa-S3.RNAPII-ChIAPET.1028/HeLa-S3_metadata.npz -c HeLa-S3 -od ./tmp
mkdir ./features_3D

for fn in ../../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv ../../data/BENGI/GM12878.HiC-Benchmark.v3.tsv ../../data/BENGI/GM12878.RNAPII-ChIAPET-Benchmark.v3.tsv ../../data/BENGI/HeLa.CTCF-ChIAPET-Benchmark.v3.tsv ../../data/BENGI/HeLa.HiC-Benchmark.v3.tsv ../../data/BENGI/HeLa.RNAPII-ChIAPET-Benchmark.v3.tsv ../../data/BENGI/HMEC.HiC-Benchmark.v3.tsv ../../data/BENGI/IMR90.HiC-Benchmark.v3.tsv ../../data/BENGI/K562.HiC-Benchmark.v3.tsv ../../data/BENGI/NHEK.HiC-Benchmark.v3.tsv; do
    bn=`basename $fn -Benchmark.v3.tsv`
    cell=`basename $fn | cut -d '.' -f 1`
    if [ $cell = "HeLa" ]; then
        cell="HeLa-S3"
    fi
    ./prepare_3dpredictor.py -c $cell -o ${bn}_3DPredictor.tsv.gz $fn &> ${bn}_3DPredictor.log &
done
wait

# usage: prepare_3dpredictor.py [-h] -c CELL -od OUTDIR [--bin-size BIN_SIZE]
#                               [--gimme_config GIMME_CONFIG] [-t NTHREADS]
#                               dataset
# prepare_3dpredictor.py: error: the following arguments are required: dataset, -c/--cell, -od/--outdir
