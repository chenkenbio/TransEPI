#!/bin/bash

#./prepare_features.py --metadata ../../data/bengi/All-Pairs.Natural-Ratio/HeLa-S3.RNAPII-ChIAPET.1028/HeLa-S3_metadata.npz -c HeLa-S3 -od ./tmp

# ./prepare_features.py -c GM12878 --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/GM12878.CHiC.1028/GM12878_metadata.npz           -od ./features_5kb/GM12878.CHiC.1028           &> ./features_5kb/GM12878.CHiC.1028.log &
# ./prepare_features.py -c GM12878 --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/GM12878.CTCF-ChIAPET.1028/GM12878_metadata.npz   -od ./features_5kb/GM12878.CTCF-ChIAPET.1028   &> ./features_5kb/GM12878.CTCF-ChIAPET.1028.log &
# ./prepare_features.py -c GM12878 --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/GM12878.HiC.1028/GM12878_metadata.npz            -od ./features_5kb/GM12878.HiC.1028            &> ./features_5kb/GM12878.HiC.1028.log &
# ./prepare_features.py -c GM12878 --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/GM12878.RNAPII-ChIAPET.1028/GM12878_metadata.npz -od ./features_5kb/GM12878.RNAPII-ChIAPET.1028 &> ./features_5kb/GM12878.RNAPII-ChIAPET.1028.log &
# ./prepare_features.py -c HeLa-S3 --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/HeLa-S3.CTCF-ChIAPET.1028/HeLa-S3_metadata.npz   -od ./features_5kb/HeLa-S3.CTCF-ChIAPET.1028   &> ./features_5kb/HeLa-S3.CTCF-ChIAPET.1028.log &
# ./prepare_features.py -c HeLa-S3 --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/HeLa-S3.HiC.1028/HeLa-S3_metadata.npz            -od ./features_5kb/HeLa-S3.HiC.1028            &> ./features_5kb/HeLa-S3.HiC.1028.log &
# ./prepare_features.py -c HeLa-S3 --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/HeLa-S3.RNAPII-ChIAPET.1028/HeLa-S3_metadata.npz -od ./features_5kb/HeLa-S3.RNAPII-ChIAPET.1028 &> ./features_5kb/HeLa-S3.RNAPII-ChIAPET.1028.log &
# ./prepare_features.py -c HMEC    --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/HMEC.HiC.1028/HMEC_metadata.npz                  -od ./features_5kb/HMEC.HiC.1028               &> ./features_5kb/HMEC.HiC.1028.log &
# ./prepare_features.py -c IMR90   --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/IMR90.HiC.1028/IMR90_metadata.npz                -od ./features_5kb/IMR90.HiC.1028              &> ./features_5kb/IMR90.HiC.1028.log &
# ./prepare_features.py -c K562    --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/K562.HiC.1028/K562_metadata.npz                  -od ./features_5kb/K562.HiC.1028               &> ./features_5kb/K562.HiC.1028.log &
# ./prepare_features.py -c NHEK    --bin-size 5000 --metadata ../../data/bengi/All-Pairs.Natural-Ratio/NHEK.HiC.1028/NHEK_metadata.npz                  -od ./features_5kb/NHEK.HiC.1028               &> ./features_5kb/NHEK.HiC.1028.log &
# wait

./prepare_features.py -c HeLa-S3 --bin-size 5000 --metadata ../../data/targetfinder/HeLa-S3/HeLa-S3_metadata.npz   -od ./features_5kb/HeLa-S3.TF   &> ./features_5kb/HeLa-S3.TF.log &
./prepare_features.py -c HUVEC   --bin-size 5000 --metadata ../../data/targetfinder/HUVEC/HUVEC_metadata.npz       -od ./features_5kb/HUVEC.TF     &> ./features_5kb/HUVEC.TF.log &

./prepare_features.py -c HeLa-S3 --bin-size 25000 --metadata ../../data/targetfinder/HeLa-S3/HeLa-S3_metadata.npz   -od ./features_25kb/HeLa-S3.TF   &> ./features_25kb/HeLa-S3.TF.log &
./prepare_features.py -c HUVEC   --bin-size 25000 --metadata ../../data/targetfinder/HUVEC/HUVEC_metadata.npz       -od ./features_25kb/HUVEC.TF     &> ./features_25kb/HUVEC.TF.log &
wait
