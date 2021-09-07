# TransEPI: Capturing large genomic contexts for accurately predicting enhancer-promoter interactions

TransEPI is a Transformer-based model for EPI prediction. 
This repository contains the scripts, data, and trained models for TransEPI.

![TransEPI](./figures/Figure1.svg)

# Requirements

* numpy
* scikit-learn
* PyTorch>=1.6.0

# Preparing genomic features

## Genomic data  
Edit the following feature data configuration files in json format:  
- CTCF narrowPeak configuration file: `./data/genomic_data/bed/CTCF_bed.json`  
- bigWig configuration file: `./data/genomic_data/bigwig/bw_6histone.json`  
- Prepared features (500bp): `./data/genomic_data/CTCF_DNase_6histone.500.json`  
- Prepared features (1000bp): `./data/genomic_data/CTCF_DNase_6histone.1000.json`  

or directly download the genomic features from [Dropbox](https://www.dropbox.com/s/nlj01rw3ffku7x1/TransEPI_processed_features.tar?dl=0)/[Synapse](https://www.synapse.org/#!Synapse:syn26156164).


## model

See `./models`

# Datasets

All the datasets used in this study are available at `data/BENGI` and `data/HiC-loops`

# Baseline models and features   

- TargetFinder: `./comparison/TargetFinder`   
- 3DPredictor: `./comparison/3DPredictor`  


# Questions
For questions about the datasets and code, please contact [chenkenbio@gmail.com](mailto:chenkenbio@gmail.com).
