#!/bin/bash


export GLUE_DIR=/path/to/data/glue_data/
export TASK_NAME=STS-B

bbc_pbert="/path/to/pbert/model/dir/"


CUDA_VISIBLE_DEVICES=0 python -u run_classifier.py   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME   --bert_model $bbc_pbert  --max_seq_length 128   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 1   --output_dir "/path/to/output/dir/"   
