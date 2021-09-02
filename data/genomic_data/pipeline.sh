#!/bin/bash

## prepare bed features
../../src/prepare_bed_signals.py ./bed/CTCF_bed.json -p ./narrowPeak --bin-size 500 &
../../src/prepare_bed_signals.py ./bed/CTCF_bed.json -p ./narrowPeak --bin-size 1000 &
# ../../src/prepare_bed_signals.py ./bed/CTCF_bed.json -p ./narrowPeak --bin-size 5000 &

# ../../src/prepare_bed_signals.py ./bed/CTCF_bed.json --binary -p ./narrowPeak --bin-size 500 &
# ../../src/prepare_bed_signals.py ./bed/CTCF_bed.json --binary -p ./narrowPeak --bin-size 1000 &
# ../../src/prepare_bed_signals.py ./bed/CTCF_bed.json --binary -p ./narrowPeak --bin-size 5000 &


## prepare bigWig features
../../src/prepare_bw_signals.py ./bigwig/bw_6histone.json --preserve -p ./bigWig --bin-size 500 &
../../src/prepare_bw_signals.py ./bigwig/bw_6histone.json --preserve -p ./bigWig --bin-size 1000 &
# ../../src/prepare_bw_signals.py ./bigwig/bw_6histone.json --preserve -p ./bigWig --bin-size 5000 &
wait
