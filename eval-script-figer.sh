#!/bin/bash

letters=("a" "b" "c")
#letters=("a" "b" "c" "d" "e")

for i in {0..9}; 
do
for letter in ${letters[@]};
do
echo ../july-reruns/roberta-fqaretrain_$i$letter\_fqadev112_nbest10
#-m ../input/roberta-fqaretrain_0a_fqatest112_nbest10 -g ../input/fqa_test_112.json -o cumulativeResults4.csv
#python eval_script.py -m /data/tmp/2022/figer/predict/roberta-fqaretrain_$i$letter\_fqadev112_nbest10 -g fqa_dev_112.json;
python eval_script.py -m ../july-reruns/roberta-fqaretrain_$i$letter\_fqadev112_nbest10 -g data/output/fqa_dev.json -o ../july-reruns/cumulativeResultsRev.csv;
python eval_script.py -m ../july-reruns/roberta-fqaretrain_$i$letter\_fqatest112_nbest10 -g data/output/fqa_test.json -o ../july-reruns/cumulativeResultsRev.csv;
done;
done &> ../july-reruns/eval_fqa_logs_rev.log
