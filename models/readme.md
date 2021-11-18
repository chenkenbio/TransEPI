# TransEPI models

configuration: TransEPI_EPI.json  

- TransEPI models trained on BENGI-train (CV)
cell types included in the training data: GM12878, HeLa-S3

| model | validation chromosomes |  
| --- | --- |  
| TransEPI_EPI_fold0.pt | chr1, chr10, chr15, chr21 |  
| TransEPI_EPI_fold1.pt | chr19, chr3, chr4, chr7, chrX |  
| TransEPI_EPI_fold2.pt | chr13, chr17, chr2, chr22, chr9 |  
| TransEPI_EPI_fold3.pt | chr12, chr14, chr16, chr18, chr20 |  
| TransEPI_EPI_fold4.pt | chr11, chr5, chr6, chr8 |  


- TransEPI models trained for Hi-C loop prediction: 

cell types included in the training data: GM12878, IMR90, HeLa-S3, K562, NHEK

model file: TransEPI_EPI_loop.pt

