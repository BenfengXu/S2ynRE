#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=False


# Config
DATASET=semeval
DATASETDIR=${DATASET}
SETTING=train_0.1
GEN_TRAIN_STEP=512
# 512 / 1024 / 2048 semeval
# 1024 / 2048 / 4096 tacred
INIT_PLM=./cp
# INIT_PLM=bert_base_uncased_huggingface

TRAINFILE=${SETTING}.txt
PSEUDOLABELSDIR=./output/pseudolabels/
SYNSETICSAMPLEDIR=./

if [[ "${DATASET}" == "tacredrev" ]]; then
    SYNSETICSAMPLEFILE="tacred_${TRAINFILE}_step${GEN_TRAIN_STEP}"
else
    SYNSETICSAMPLEFILE="${DATASET}_${TRAINFILE}_step${GEN_TRAIN_STEP}"
fi

# FINETUNE_EPOCH
if [[ "${DATASET}" == "semeval" ]] || [[ "$DATASET" == "chemprot" ]] || [[ "$DATASET" == "wiki80" ]]; then
  if [[ "${TRAINFILE}" == "train_0.01.txt" ]]; then
    FINETUNE_EPOCH=(40 80)
  elif [[ "${TRAINFILE}" == "train_0.1.txt" ]]; then
    FINETUNE_EPOCH=(10 20)
  else
    FINETUNE_EPOCH=(5 10)
  fi
else
  if [[ "${TRAINFILE}" == "train_0.01.txt" ]]; then
    FINETUNE_EPOCH=(20)
  elif [[ "${TRAINFILE}" == "train_0.1.txt" ]]; then
    FINETUNE_EPOCH=(5)
  else
    FINETUNE_EPOCH=(2)
  fi
fi
FINETUNE_LR=(5e-5)
FINETUNE_BS=(16)
FINETUNE_SEED=(42 1 2 3 4) # K_TEACHERS=5
TOTAL_ITERATION=10

# =======================================
# ===== prepare synthetic samples =======
# =======================================
head -n 100000 ${SYNSETICSAMPLEFILE} > ${SYNSETICSAMPLEFILE}.tmp && mv ${SYNSETICSAMPLEFILE}.tmp ${SYNSETICSAMPLEFILE}

# =========================
# ===== iteration 1 =======
# =========================
ITERATION_K=1
rm -rf ./output/teacher_model

# finetune, produce teacher model
for SEED in "${FINETUNE_SEED[@]}"; do
for BATCH_SIZE in "${FINETUNE_BS[@]}"; do # 16, 64 for semeval0.01, 16 for semevalfull
for LR in "${FINETUNE_LR[@]}"; do
for EPOCH in "${FINETUNE_EPOCH[@]}"; do
python3 run_re.py \
    --model_name_or_path ${INIT_PLM} \
    --iteration ${ITERATION_K} \
    --dataset ${DATASET} \
    --data_dir ${DATASETDIR} \
    --train_file ${TRAINFILE} \
    --synsetic_data_dir ${SYNSETICSAMPLEDIR} \
    --synsetic_data_file ${SYNSETICSAMPLEFILE} \
    --pseudo_labels_dir ${PSEUDOLABELSDIR} \
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

# filter best k teacher
python3 ./utils/keep_best_teacher_perseed.py --iteration ${ITERATION_K} --result_file=./output/results_all.csv --teacher_model_dir=./output/teacher_model

# Produce Pseudo Labels
for teacher in `ls ./output/teacher_model`; do
python3 run_distill.py \
    --model_name_or_path ./output/teacher_model/${teacher} \
    --dataset ${DATASET} \
    --iteration ${ITERATION_K} \
    --iteration_total ${TOTAL_ITERATION} \
    --save_distilled ${PSEUDOLABELSDIR} \
    --data_dir ${SYNSETICSAMPLEDIR} \
    --train_file ${SYNSETICSAMPLEFILE} \
    --max_seq_length 128
done


# =========================
# ===== iteration 2~K =====
# =========================
for ITERATION_K in $(seq 2 ${TOTAL_ITERATION}); do

# Pretraning (Intermediate Training) on Distilled Pseudo Labels
rm -rf ./output/pretrained_ckpt
BATCH_SIZE=64
LR=3e-5
SEED=42
MAXSTEPS=1501
TEMP=1

python3 run_synthetic_pretrain.py \
    --model_name_or_path ${INIT_PLM} \
    --dataset ${DATASET} \
    --synsetic_data_dir ${SYNSETICSAMPLEDIR} \
    --synsetic_data_file ${SYNSETICSAMPLEFILE} \
    --pseudo_labels_dir ${PSEUDOLABELSDIR} \
    --logits_dir ${PSEUDOLABELSDIR} \
    --output_dir ./output/ \
    --save_model ./output/pretrained_ckpt \
    --save_steps 500 \
    --max_seq_length 128 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --max_train_steps ${MAXSTEPS} \
    --learning_rate ${LR} \
    --distill_temp ${TEMP} \
    --seed ${SEED} \
    --use_fp16 \
    --num_dataloader_worker_per_device 4

# finetune, produce teacher model
rm -rf ./output/teacher_model
for PRETRAINED_CKPT in `ls ./output/pretrained_ckpt`; do
INIT_PRETRAINED=./output/pretrained_ckpt/${PRETRAINED_CKPT}
for SEED in "${FINETUNE_SEED[@]}"; do
for BATCH_SIZE in "${FINETUNE_BS[@]}"; do # 16, 64 for semeval0.01, 16 for semevalfull
for LR in "${FINETUNE_LR[@]}"; do
for EPOCH in "${FINETUNE_EPOCH[@]}"; do
python3 run_re.py \
    --model_name_or_path ${INIT_PRETRAINED} \
    --iteration ${ITERATION_K} \
    --dataset ${DATASET} \
    --data_dir ${DATASETDIR} \
    --train_file ${TRAINFILE} \
    --synsetic_data_dir ${SYNSETICSAMPLEDIR} \
    --synsetic_data_file ${SYNSETICSAMPLEFILE} \
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
done

python3 ./utils/keep_best_teacher_perseed.py --iteration ${ITERATION_K} --result_file=./output/results_all.csv --teacher_model_dir=./output/teacher_model

rm -rf ./output/pseudolabels/

for teacher in `ls ./output/teacher_model`; do

python3 run_distill.py \
    --model_name_or_path ./output/teacher_model/${teacher} \
    --dataset ${DATASET} \
    --iteration ${ITERATION_K} \
    --iteration_total ${TOTAL_ITERATION} \
    --save_distilled ${PSEUDOLABELSDIR} \
    --data_dir ${SYNSETICSAMPLEDIR} \
    --train_file ${SYNSETICSAMPLEFILE} \
    --max_seq_length 128

done

done # end iteration