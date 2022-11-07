#!/bin/bash
wget https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip -O dataset
unzip dataset.zip -d dataset
python scripts/train.py \
    --train-path dataset/SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-train.txt \
    --dev-path dataset/SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-dev.txt