#!/bin/bash
REQFILE="requirements.txt"
FILE="dataset.zip"
FOLDER="dataset"
ENGLISH_PATH="SemEval2018-Task1-all-data/English/E-c"

if [ -f "$REQFILE" ]; then
    pip install -r "$REQFILE"
elif [ -f "../$REQFILE"]; then
    pip install -r "../$REQFILE"
else
    echo 'There is no requirements file'
fi

if [ ! -f "$FILE" ]; then
    echo "Downloading $FILE"
    wget https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip -O dataset.zip
else
    echo "$FILE is already here, skipping download"
fi

if [ ! -d "$FOLDER" ]; then
    echo "Unzipping file to folder"
    unzip "$FILE" -d "$FOLDER"
else
    echo "$FOLDER is already here, skipping unzip"
fi

python scripts/train.py \
    --train-path "$FOLDER/$ENGLISH_PATH/2018-E-c-En-train.txt" \
    --dev-path "$FOLDER/$ENGLISH_PATH/2018-E-c-En-dev.txt"