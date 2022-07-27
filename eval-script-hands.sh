#!/bin/bash

#letters=("a" "b" "c")
#letters=("a" "b" "c" "d" "e")

for i in {0..6};
do
echo ../july-reruns/roberta_hands_retrain_nines_$i\a
#-m ../input/roberta-fqaretrain_0a_fqatest112_nbest10 -g ../input/fqa_test_112.json -o cumulativeResults4.csv
#python eval_script.py -m /data/tmp/2022/figer/predict/roberta-fqaretrain_$i$letter\_fqadev112_nbest10 -g fqa_dev_112.json;
echo figerQA112
python eval_script.py -m ../july-reruns/roberta_hands_retrain_nines_$i\a_figerQA112_nbest10 -g data/output/figerQA.json -o ../july-reruns/cumulativeResultsHands.csv;
echo fqadev112
python eval_script.py -m ../july-reruns/roberta_hands_retrain_nines_$i\a_fqadev112_nbest10 -g data/output/fqa_dev.json -o ../july-reruns/cumulativeResultsHands.csv;
echo handsall
python eval_script_hands.py -m ../july-reruns/roberta_hands_retrain_nines_$i\a_handsall_nbest10 -g data/output/handsEvalAllQA.json -o ../july-reruns/cumulativeResultsHands.csv;
echo handsdev
python eval_script_hands.py -m ../july-reruns/roberta_hands_retrain_nines_$i\a_handsdev_nbest10 -g data/output/handsDevQA.json -o ../july-reruns/cumulativeResultsHands.csv;
#python eval_script.py -m ../july-reruns/roberta-fqaretrain_$i$letter\_fqatest112_nbest10 -g data/output/fqa_test.json -o ../july-reruns/cumulativeResultsRev.csv;
done &> ../july-reruns/eval_hands_logs_rev.log
