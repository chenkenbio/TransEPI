#!/bin/bash

# for fn in ../../../data/BENGI/GM12878.CHiC-Benchmark.v3.tsv ../../../data/BENGI/GM12878.CHiC-HiC-CTCF-RNAPII.merged.tsv ../../../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv ../../../data/BENGI/GM12878.HiC-Benchmark.v3.tsv ../../../data/BENGI/GM12878.RNAPII-ChIAPET-Benchmark.v3.tsv ../../../data/BENGI/HeLa.CTCF-ChIAPET-Benchmark.v3.tsv ../../../data/BENGI/HeLa.HiC-Benchmark.v3.tsv ../../../data/BENGI/HeLa.RNAPII-ChIAPET-Benchmark.v3.tsv ../../../data/BENGI/HMEC.HiC-Benchmark.v3.tsv ../../../data/BENGI/IMR90.HiC-Benchmark.v3.tsv ../../../data/BENGI/K562.HiC-Benchmark.v3.tsv ../../../data/BENGI/NHEK.HiC-Benchmark.v3.tsv; do
#     bn=`basename $fn .tsv`
#     ../src/prepare_targetfinder_features.py $fn -c "../../../data/genomic_features/targetfinder_features.json" > BENGI.${bn}.targetfinder_features.tsv 2> BENGI.${bn}.targetfinder_features.log &
# done
# wait 


for fn in ../../../data/Whalen/GM12878.HiC-Whalen.tsv ../../../data/Whalen/HeLa-S3.HiC-Whalen.tsv ../../../data/Whalen/HUVEC.HiC-Whalen.tsv ../../../data/Whalen/IMR90.HiC-Whalen.tsv ../../../data/Whalen/K562.HiC-Whalen.tsv ../../../data/Whalen/NHEK.HiC-Whalen.tsv; do
    bn=`basename $fn .tsv`
    ../src/prepare_targetfinder_features.py $fn -c "../../../data/genomic_features/targetfinder_features.json" > Whalen.${bn}.targetfinder_features.tsv 2> Whalen.${bn}.targetfinder_features.log &
done
wait 

