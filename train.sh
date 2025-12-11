#!/bin/bash

set -e

ENV_NAME="myenv"
MODEL_DIR="vit5-base"

echo "=== Cloning VietAI/vit5-base model ==="
git clone https://huggingface.co/VietAI/vit5-base $MODEL_DIR

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -y -n $ENV_NAME python=3.10

echo "=== Activating environment: $ENV_NAME ==="
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "=== Installing requirements... ==="
pip install -r requirements.txt

echo "=== Running training... ==="
python3 run_qa.py \
  --num_train_epochs 300 \
  --learning_rate 1e-5 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_name "$MODEL_DIR" \
  --data_dir "Dataset" \
  --checkpoint_dir "checkpoint" \
  --model_pth "model_epoch_2.pth" \
  --output_dir "Result" \
  --weight_decay 0.01 \
  --seed 42 \
  --max_doc_len 400 \
  --max_query_len 80 \
  --max_option_len 20 \
  --max_expl_len 200 \
  --num_labels 4 \
  --eval_start_epoch 100 \
  --eval_interval 5 \
  --do_train "True"

echo "=== Done! ==="
