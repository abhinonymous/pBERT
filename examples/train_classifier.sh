#!/bin/bash

#$ -l 'hostname=b1[12345678]*|c*,gpu=1' 
#$ -l mem_free=10G,ram_free=10G
#$ -cwd
#$ -M abhinavsingh282@gmail.com
#$ -e Ep1_stsb_pBERT_Ep0.5.errlog
#$ -o Ep1_stsb_pBERT_Ep0.5.log
#$ -N Ep1_stsb_pBERT_Ep0.5
#$ -pe smp 4
#$ -V
#$ -m eas



export GLUE_DIR=/export/c10/asingh/data/glue_data/glue_data/
export TASK_NAME=STS-B
conda activate huggingfaceBert_27

bbc_pb2_Ep0_5="/export/c10/asingh/models/bert/pb2_finetuned/7nearest-cased_1Epoch/pytorch_model_Ep0.50_extracted/"

#CUDA_VISIBLE_DEVICES=`free-gpu` python -u run_classifier.py   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME   --bert_model bert-base-cased --max_seq_length 128   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 1   --output_dir "/export/c10/asingh/models/stsb_glue/Ep1_stsb_pBERT_Ep0.5/"   
CUDA_VISIBLE_DEVICES=`free-gpu` python -u run_classifier.py   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME   --bert_model $bbc_pb2_Ep0_5  --max_seq_length 128   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 1   --output_dir "/export/c10/asingh/models/stsb_glue/Ep1_stsb_pBERT_Ep0.5/"   
