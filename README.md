# TransEPI: Capturing large genomic contexts for accurately predicting enhancer-promoter interactions

The codes and datasets for [Capturing large genomic contexts for accurately predicting enhancer-promoter interactions](https://www.biorxiv.org/content/10.1101/2021.09.04.458817v1).

---  

***The Supplementary Tables*** of the manuscript are available at [Supp_Tables.xlsx](./paper/Supp_Tables.xlsx)

---

TransEPI is a Transformer-based model for EPI prediction. 
This repository contains the scripts, data, and trained models for TransEPI.


![TransEPI](./figures/Figure1.svg)

# Requirements  

* numpy  
* tqdm  
* scikit-learn  
* PyTorch>=1.9.0 (recommended) or PyTorch 1.6.0+  
* pyBigWig (optional, required by `prepare_bw_signals.py` for preparing features)


# Datasets  

All the datasets used in this study are available at [data/BENGI](data/BENGI) and [data/HiC-loops](data/HiC-loops).  


# Input features  

- Download the genomic features from [Synapse:syn26156164](https://www.synapse.org/#!Synapse:syn26156164) and edit the feature configuration file `./data/genomic_data/CTCF_DNase_6histone.500.json` to specifiy the location of the genomic feature files. *Absolute path is required!*  
- Or prepare features for other cell types using `src/prepare_bed_signals.py` and `src/prepare_bw_signals.py`. See `./data/genomic_data/pipeline.sh` for usage.  

# Scripts

## dataset & model
- `src/epi_dataset.py`  
- `src/epi_models.py`  

## cross validation & evaluation
- `src/cross_validate.py`  
- `src/evaluate_model.py`  

## finding target genes of non-coding mutations  
- `src/find_targets.py`  

## preparing genomic data
- `src/prepare_bed_signals.py`  
- `src/prepare_bw_signals.py`  


## how to use:  

Run the above scripts with `--help` to see the usage:  
```
$ ./src/cross_validate.py --help
usage: cross_validate.py [-h] -c CONFIG -o OUTDIR [--gpu GPU] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Configuration file for training the model (default:
                        None)
  -o OUTDIR, --outdir OUTDIR
                        Output directory (default: None)
  --gpu GPU             GPU ID, (-1 for CPU) (default: -1)
  --seed SEED           Random seed (default: 2020)
```

A demo configuration file is available at [models/TransEPI_EPI.json](models/TransEPI_EPI.json).


# Models

See [models](./models).  


# Reproducibility

To reproduce the major results shown in the manuscripts, see [dev/run_cv.sh](./dev/run_cv.sh) [dev/run_pred.sh](./dev/run_pred.sh).


# Baseline models and features   

- TargetFinder: [comparison/TargetFinder](./comparison/TargetFinder)   
- 3DPredictor: [comparison/3DPredictor](./comparison/3DPredictor)  


# Questions
For questions about the datasets and code, please contact [chenkenbio@gmail.com](mailto:chenkenbio@gmail.com) or create an issue.

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
