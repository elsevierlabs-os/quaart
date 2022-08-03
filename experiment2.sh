#!/bin/bash

letters=("a" "b" "c")

if [ ! -d "figer/train/" ]
then
  mkdir -p "figer/train/"
fi
if [ ! -d "figer/predict/" ]
then
  mkdir -p "figer/predict/"
fi

# Train Models
for i in {1..9};
do
  for l in ${letters[@]};
  do
    python run_qa.py --model_name_or_path "deepset/roberta-base-squad2" --train_file data/output/fqa_train/fqa_train_gold_$i$l.json --do_train --version_2_with_negative --output_dir figer/train/roberta-fqaretrain_$i$l --overwrite_output_dir --max_seq_length 512 2>&1 | tee figer/train/figertrain_$i$l.log
  done;
done

# Predict Squad Only

python run_qa.py --model_name_or_path "deepset/roberta-base-squad2" --test_file data/output/fqa_dev.json --do_predict --version_2_with_negative --output_dir figer/predict/roberta-fqaretrain_fqadev_0a --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee figer/predict/figer_pred_fqadev_0a.log;

python run_qa.py --model_name_or_path "deepset/roberta-base-squad2" --test_file data/output/fqa_test.json --do_predict --version_2_with_negative --output_dir figer/predict/roberta-fqaretrain_fqatest_0a --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee figer/predict/figer_pred_fqatest_0a.log;

# Predict remaining models
for i in {1..9};
do
  for l in ${letters[@]};
  do
    python run_qa.py --model_name_or_path figer/train/roberta-fqaretrain_$i$l --test_file data/output/fqa_dev.json --do_predict --version_2_with_negative --output_dir figer/predict/roberta-fqaretrain_fqadev_$i$l --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee figer/predict/figer_pred_fqadev_$i$l.log;

    python run_qa.py --model_name_or_path figer/train/roberta-fqaretrain_$i$l --test_file data/output/fqa_test.json --do_predict --version_2_with_negative --output_dir figer/predict/roberta-fqaretrain_fqatest_$i$l --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee figer/predict/figer_pred_fqatest_$i$l.log;
  done
done

# Eval everything
./eval-script-figer.sh $1
