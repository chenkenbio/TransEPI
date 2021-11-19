# TransEPI models

configuration: TransEPI_EPI.json  

- TransEPI models trained by 5-fold CV

| model | cell types in training data | validation chromosomes |  
| --- | --- | --- | 
| TransEPI_EPI_fold0.pt | GM12878, HeLa-S3 | chr1, chr10, chr15, chr21 |  
| TransEPI_EPI_fold1.pt | GM12878, HeLa-S3 | chr19, chr3, chr4, chr7, chrX |  
| TransEPI_EPI_fold2.pt | GM12878, HeLa-S3 | chr13, chr17, chr2, chr22, chr9 |  
| TransEPI_EPI_fold3.pt | GM12878, HeLa-S3 | chr12, chr14, chr16, chr18, chr20 |  
| TransEPI_EPI_fold4.pt | GM12878, HeLa-S3 | chr11, chr5, chr6, chr8 |  


- Final EPI model: `TransEPI_EPI_valHMEC.pt`  
    - cell types in training data: GM12878, IMR90, HeLa-S3, K562, NHEK

