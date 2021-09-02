# TransEPI: Capturing large genomic contexts for accurately predicting enhancer-promoter interactions

TransEPI is a Transformer-based model for EPI prediction. 
This repository contains the scripts, data, and trained models for TransEPI.

![TransEPI](./figures/Figure1.svg)

# Requirements

* numpy
* scikit-learn
* PyTorch>=1.6.0

# Data preparation

## Genomic data
    - CTCF data in narrowPeak  
    - DNase-seq data in bigWig (p-value track)
    - H3K27ac ChIP-seq data in bigWig (p-value track)
    - H3K4me3 ChIP-seq data in bigWig (p-value track)
    - H3K4me1 ChIP-seq data in bigWig (p-value track)
The users should edit the json files in `config/` to specify the location of these genomic data.


# Usage

```bash
./run_deepepi.sh /path/to/query/file /path/to/output/directory
```

## Input

The input file should be formatted as:

```
##CELLTYPE [cell type]
##BUILDVER [build version]
#chrom enhancer promoter
chr10	49875920-49876712	50396056-50398056
chr10	49874816-49877816	49874816-49877816
```

build version: GRCh37/hg19/GRCh38/hg38


## Output
The output will be saved at `/path/to/output/directory/results.txt`

## Demo

```bash
./run_deepepi.sh demo/demo_data.tsv output/demo
```

## model
The model `model/GM12878_IMR90_K562_NHEK_chr1_19.json` was trained on samples from GM12878, IMR90, K562, and NHEK. Only the pairs on chr1-chr19 were used.

# Datasets

All the datasets used in this study are available at `data/dataset`


# Questions
For questions about the datasets and code, please contact [chenkenbio@gmail.com](mailto:chenkenbio@gmail.com).
