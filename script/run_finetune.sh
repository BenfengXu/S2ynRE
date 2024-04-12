#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=False

# Config
DATASET=wiki80
DATASETDIR=wiki80
TRAINFILE=train_0.01.txt

FINETUNE_EPOCH=(5 10 20 40 80)
FINETUNE_LR=(5e-5)
FINETUNE_BS=(16)
FINETUNE_SEED=(42 1 2 3 4) # K_TEACHERS=5

ITERATION_K=1
INIT_CKPT=./bert_base_uncased_huggingface

# finetune, produce teacher model
for SEED in "${FINETUNE_SEED[@]}"; do
for BATCH_SIZE in "${FINETUNE_BS[@]}"; do # 16, 64 for semeval0.01, 16 for semevalfull
for LR in "${FINETUNE_LR[@]}"; do
for EPOCH in "${FINETUNE_EPOCH[@]}"; do
python3 run_re.py \
    --model_name_or_path ${INIT_CKPT} \
    --iteration ${ITERATION_K} \
    --dataset ${DATASET} \
    --data_dir ${DATASETDIR} \
    --train_file ${TRAINFILE} \
    --output_dir ./output/ \
    --save_model ./output/teacher_model \
    --max_seq_length 128 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --num_train_epochs ${EPOCH} \
    --learning_rate ${LR} \
    --seed ${SEED} \
    --use_fp16 \
    --num_dataloader_worker_per_device 4
done
done
done
done