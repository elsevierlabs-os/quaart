#!/bin/bash

if [ ! -d "hands/train/" ]
then
  mkdir -p "hands/train/"
fi
if [ ! -d "hands/predict/" ]
then
  mkdir -p "hands/predict/"
fi

# Train Models
for i in {1..6};
do
  python run_qa.py --model_name_or_path "deepset/roberta-base-squad2" --train_file data/output/hands_train/hands_train_gold_nines_$i\a.json --do_train --version_2_with_negative --output_dir hands/train/roberta_hands_train_gold_$i\a --overwrite_output_dir --overwrite_cache --max_seq_length 512  2>&1 | tee hands/train/handstrain_$i\a.log
done

# Predict Squad Only
python run_qa.py --model_name_or_path "deepset/roberta-base-squad2" --test_file data/output/handsDevQA.json --do_predict --version_2_with_negative --output_dir hands/predict/roberta_hands_train_handsdev_0a --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee hands/predict/hands_pred_handsdev_0a.log;

python run_qa.py --model_name_or_path "deepset/roberta-base-squad2" --test_file data/output/handsEvalAllQA.json --do_predict --version_2_with_negative --output_dir hands/predict/roberta_hands_train_handsall_0a --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee hands/predict/hands_pred_handsall_0a.log;

python run_qa.py --model_name_or_path "deepset/roberta-base-squad2" --test_file data/output/fqa_dev.json --do_predict --version_2_with_negative --output_dir hands/predict/roberta_hands_train_figerdev_0a --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee hands/predict/hands_pred_figerdev_0a.log;

python run_qa.py --model_name_or_path "deepset/roberta-base-squad2" --test_file data/output/figerQA.json --do_predict --version_2_with_negative --output_dir hands/predict/roberta_hands_train_figerall_0a --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee hands/predict/hands_pred_figerall_0a.log;


# Predict remaining models
for i in {1..6};
do
  python run_qa.py --model_name_or_path hands/train/roberta_hands_train_gold_$i\a --test_file data/output/handsDevQA.json --do_predict --version_2_with_negative --output_dir hands/predict/roberta_hands_train_handsdev_$i\a --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee hands/predict/hands_pred_handsdev_$i\a.log;

  python run_qa.py --model_name_or_path hands/train/roberta_hands_train_gold_$i\a --test_file data/output/handsEvalAllQA.json --do_predict --version_2_with_negative --output_dir hands/predict/roberta_hands_train_handsall_$i\a --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee hands/predict/hands_pred_handsall_$i\a.log;

  python run_qa.py --model_name_or_path hands/train/roberta_hands_train_gold_$i\a --test_file data/output/fqa_dev.json --do_predict --version_2_with_negative --output_dir hands/predict/roberta_hands_train_figerdev_$i\a --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee hands/predict/hands_pred_figerdev_$i\a.log;

  python run_qa.py --model_name_or_path hands/train/roberta_hands_train_gold_$i\a --test_file data/output/figerQA.json --do_predict --version_2_with_negative --output_dir hands/predict/roberta_hands_train_figerall_$i\a --overwrite_output_dir --max_seq_length 512 --n_best 10 2>&1 | tee hands/predict/hands_pred_figerall_$i\a.log;
done

# Eval everything
./eval-script-hands.sh $1
