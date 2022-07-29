#!/bin/bash

if [ ! -d "data/output" ]
then
    echo "Building directory structure and downloading data"
    mkdir -p "data/output/fqa_train/"
    mkdir -p "data/output/hands_train/"
    echo "Downloading FIGER data:"
    wget https://raw.githubusercontent.com/xiaoling/figer/master/aaai/exp.label -P data/output/
    wget https://raw.githubusercontent.com/xiaoling/figer/master/aaai/exp.txt -P data/output/
    echo "Downloading HAnDS data:"
    wget https://raw.githubusercontent.com/abhipec/HAnDS/master/datasets/1k-WFB-g/1k-WFB-g_complete.json -P data/output/
    wget https://raw.githubusercontent.com/abhipec/HAnDS/master/datasets/1k-WFB-g/fner_dev.json -P data/output/
    wget https://raw.githubusercontent.com/abhipec/HAnDS/master/datasets/1k-WFB-g/fner_test.json -P data/output/
    cat data/output/fner_dev.json | jq -cs '.' > data/output/fner_dev_array.json
    cat data/output/fner_test.json | jq -cs '.' > data/output/fner_test_array.json
    cat data/output/1k-WFB-g_complete.json | jq -cs '.' > data/output/1k-WFB-g_complete_array.json
    python hands_gdown.py
    tar -xvfz data/output/HAnDS_figer_types_stage_one_state_two_sentences_stage_three_pp.tar.gz -C data/output/
    echo "Download complete"
else
    echo "data/output exists - delete to redownolad"
fi

python figerDataPrep.py
python handsDataPrep.py
