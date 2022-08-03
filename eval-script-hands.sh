#!/bin/bash

for i in {0..6};
do
  echo hands/train/roberta_hands_train_gold_$i\a
  echo figerQA112
  python eval_script.py -m hands/predict/roberta_hands_train_figerall_$i\a -g data/output/figerQA.json -o $1.csv
  echo fqadev112
  python eval_script.py -m hands/predict/roberta_hands_train_figerdev_$i\a -g data/output/fqa_dev.json -o $1.csv
  echo handsall
  python eval_script_hands.py -m hands/predict/roberta_hands_train_handsall_$i\a -g data/output/handsEvalAllQA.json -o $1.csv
  echo handsdev
  python eval_script_hands.py -m hands/predict/roberta_hands_train_handsdev_$i\a -g data/output/handsDevQA.json -o $1.csv
done &> eval_hands_logs_$1.log

python finalResultsHands.py -p $1
