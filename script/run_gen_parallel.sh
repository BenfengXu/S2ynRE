#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=False

# for MAX_TRAIN_STEP in 2048 1024 512 256; do # tacred
for MAX_TRAIN_STEP in 128 64; do # semeval

# Config
DATASET=semeval
SETTING=train_0.1.txt
TRAINFILE=${SETTING}
#MAX_TRAIN_STEP=30
GEN_MODEL=./GPT_XL
SYNSETICSAMPLEFILE=${DATASET}_${TRAINFILE}_step${MAX_TRAIN_STEP}

GEN_NUM_PER_GPU=30000
DATASETDIR=./${DATASET}
BATCH_SIZE=2  # 4 GPU
SEED=42
LR=1e-5

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=45947 --use_env run_generate.py \
    --model_name_or_path ${GEN_MODEL} \
    --dataset ${DATASET} \
    --data_dir ${DATASETDIR} \
    --train_file ${TRAINFILE} \
    --output_dir ./output/ \
    --save_model na \
    --max_seq_length 128 \
    --gen_num ${GEN_NUM_PER_GPU} \
    --synsetic_data_file ${SYNSETICSAMPLEFILE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --max_train_steps ${MAX_TRAIN_STEP} \
    --learning_rate ${LR} \
    --seed ${SEED} \
    --use_fp16 \
    --num_dataloader_worker_per_device 4

for device in 0 1 2 3; do
  cat ./output/synseticsamples/"${SYNSETICSAMPLEFILE}.${device}" >> ./output/synseticsamples/${SYNSETICSAMPLEFILE}
#   cat ./output/synseticsamples/"${SYNSETICSAMPLEFILE}.raw.${device}" >> ./output/synseticsamples/${SYNSETICSAMPLEFILE}.raw
done