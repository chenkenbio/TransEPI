# TransEPI: Capturing large genomic contexts for accurately predicting enhancer-promoter interactions

The codes and datasets for [Capturing large genomic contexts for accurately predicting enhancer-promoter interactions](https://www.biorxiv.org/content/10.1101/2021.09.04.458817v1).

---  

***Supplementary Data*** of the manuscript are available at [](./paper/supplementary_data)

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



# Quick start

1. Clone the codes: 
```
git clone git@github.com:biomed-AI/TransEPI.git
```

2. Preparing input features

	* **The feature data used in our study**:
Download the genomic features from [Synapse:syn26156164](https://www.synapse.org/#!Synapse:syn26156164) and edit the feature configuration file `./data/genomic_data/CTCF_DNase_6histone.500.json` to specifiy the location of the genomic feature files. *Absolute path is required!*  

3. Preparing input files
The input to the TransEPI model should be formatted as:
	1. label: for datasets without known labels, set it to 0
	2. distance: the between the enhancer and the promoter
	3. e_chr: enhancer chromosome
	4. e_start: enhancer start
	5. e_end: enhancer end
	6. enhancer name: the name of the cell type should be placed in the second field of enhancer name: e.g.: chr5:317258-317610|GM12878|EH37E0762690. (shoule be separated by `|`)
	7. p_chr: promoter chromosome
	8. p_start: promoter start
	9. p_end: promoter end
	10. promoter name: the name of the cell type should be placed in the second field of promoter name: e.g.: chr5:317258-317610|GM12878|EH37E0762690. (shoule be separated by `|`)
	11. mask region (optional): the feature values in the mask regions will be masked (set to 0). e.g.: 889314-895314;317258-327258

The input files should be tab separated

Example:
```
1	572380.0	chr5	317258	317610	chr5:317258-317610|GM12878|EH37E0762690	chr5	889314	891314	chr5:889813-889814|GM12878|ENSG00000028310.13|ENST00000388890.4|-
0	100101.0	chr5	317258	317610	chr5:317258-317610|GM12878|EH37E0762690	chr5	216833	218833	chr5:217332-217333|GM12878|ENSG00000164366.3|ENST00000441693.2|-
```



4. Run the model
```
cd TransEPI/src
chmod +x ./evaluate_
python ./evaluate_model.py \
	-t ../data/BENGI/HMEC.HiC-Benchmark.v3.tsv.gz \
	-c ../models/TransEPI_EPI.json \
	-m ../models/TransEPI_EPI_fold0.pt \
	-p output
```



# Models

The trained models are available at [models](./models).  


# Reproducibility

To reproduce the major results shown in the manuscripts, see [dev/run_cv.sh](./dev/run_cv.sh) (cross validation) and [dev/run_pred.sh](./dev/run_pred.sh) (evaluation).


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
