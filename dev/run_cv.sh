#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage: $0 config"
    exit 1
fi

config="$1"
run_name=`basename $config .json`
run_name="${run_name}_cv"

../src/cross_validate.py \
    --gpu 2 \
    -c $config \
    -o $run_name &> ${run_name}.log
