# TransEPI: Capturing large genomic contexts for accurately predicting enhancer-promoter interactions

**The Supplementary Tables** of the manuscript are available at [Supp_Tables.xlsx](./paper/Supp_Tables.xlsx)

The implementation and the datasets for [Capturing large genomic contexts for accurately predicting enhancer-promoter interactions](https://www.biorxiv.org/content/10.1101/2021.09.04.458817v1).

TransEPI is a Transformer-based model for EPI prediction. 
This repository contains the scripts, data, and trained models for TransEPI.


![TransEPI](./figures/Figure1.svg)

# Requirements

* numpy
* scikit-learn
* PyTorch>=1.6.0

# Preparing genomic features

## Genomic data  
Download the genomic features from [Synapse:syn26156164](https://www.synapse.org/#!Synapse:syn26156164) and edit the feature configuration file `./data/genomic_data/CTCF_DNase_6histone.500.json`.

## model

See `./models`

# Datasets

All the datasets used in this study are available at `data/BENGI` and `data/HiC-loops`

# Baseline models and features   

- TargetFinder: `./comparison/TargetFinder`   
- 3DPredictor: `./comparison/3DPredictor`  


# Questions
For questions about the datasets and code, please contact [chenkenbio@gmail.com](mailto:chenkenbio@gmail.com).

# Citation

```
@article {Chen2021.09.04.458817,
	author = {Chen, Ken and Zhao, Huiying and Yang, Yuedong},
	title = {Capturing large genomic contexts for accurately predicting enhancer-promoter interactions},
	elocation-id = {2021.09.04.458817},
	year = {2021},
	doi = {10.1101/2021.09.04.458817},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/09/06/2021.09.04.458817},
	eprint = {https://www.biorxiv.org/content/early/2021/09/06/2021.09.04.458817.full.pdf},
	journal = {bioRxiv}
}
```
