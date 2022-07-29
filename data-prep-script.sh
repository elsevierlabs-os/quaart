#!/bin/bash

if [ ! -d "data/output" ]
then
    echo "Building directory structure and downloading data"
    mkdir -p "data/output/fqa_train/"
    mkdir -p "data/output/hands_train/"
    echo "Downloading FIGER data:"
    wget https://github.com/xiaoling/figer/blob/master/aaai/exp.label -P data/
    wget https://github.com/xiaoling/figer/blob/master/aaai/exp.txt -P data/
    echo "Downloading HAnDS data:"
    wget https://github.com/abhipec/HAnDS/blob/master/datasets/1k-WFB-g/1k-WFB-g_complete.json -P data/
    wget https://github.com/abhipec/HAnDS/blob/master/datasets/1k-WFB-g/fner_dev.json -P data/
    wget https://github.com/abhipec/HAnDS/blob/master/datasets/1k-WFB-g/fner_test.json -P data/
    cat data/fner_dev.json | jq -cs '.' > data/fner_dev_array.json
    cat data/fner_test.json | jq -cs '.' > data/fner_test_array.json
    cat data/fner_dev.json | jq -cs '.' > data/fner_dev_array.json
    python hands_gdown.py

    echo "Download complete"
else
    echo "data/output exists - delete to redownolad"
fi

python figerDataPrep.py
python handsDataPrep.py
