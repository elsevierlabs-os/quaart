#!/bin/bash

letters=("a" "b" "c")

for i in {1..9};
do
  for l in ${letters[@]};
  do
    echo figer/train/roberta-fqaretrain_$i$l
    python eval_script.py -m figer/predict/roberta-fqaretrain_fqadev_$i$l -g data/output/fqa_dev.json -o $1.csv;
    python eval_script.py -m ../july-reruns/roberta-fqaretrain_fqatest_$i$l -g data/output/fqa_test.json -o $1.csv;
  done;
done &> &> eval_figer_logs_$1.log

python finalResultsFiger.py -p $1
