#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=False


# Config
DATASET=chemprot
DATASETDIR=chemprot
TRAINFILE=train_0.01.txt
EP=1
GPT_EPOCH=${EP}
SYNSETICSAMPLEDIR=./output/synseticsamples/
SYNSETICSAMPLEFILE=synseticsamples.txt
PSEUDOLABELSDIR=./output/pseudolabels/
FINETUNE_EPOCH=(20 40)
FINETUNE_LR=(5e-5)
FINETUNE_BS=(16)
FINETUNE_SEED=(42 1 2 3 4) # K_TEACHERS=5
TOTAL_ITERATION=10


# =========================
# ===== Synthesis =======
# =========================
GENBATCH=10000
GENBATCHSIZE=100
GEN_MODEL=./gpt-2-pubmed-large
BATCH_SIZE=16
SEED=42
LR=3e-5

python3 run_generate.py \
    --model_name_or_path ${GEN_MODEL} \
    --dataset ${DATASET} \
    --data_dir ${DATASETDIR} \
    --train_file ${TRAINFILE} \
    --output_dir ./output/ \
    --save_model na \
    --max_seq_length 128 \
    --gen_batch $GENBATCH \
    --gen_batch_size $GENBATCHSIZE \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --num_train_epochs ${GPT_EPOCH} \
    --learning_rate ${LR} \
    --seed ${SEED} \
    --use_fp16 \
    --num_dataloader_worker_per_device 4



# =========================
# ===== iteration 1 =======
# =========================
ITERATION_K=1
rm -rf ./output/teacher_model
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
INIT_CKPT=./bert_base_uncased_huggingface
BATCH_SIZE=64
LR=3e-5
SEED=42
MAXSTEPS=1501
TEMP=1

python3 run_synthetic_pretrain.py \
    --model_name_or_path ${INIT_CKPT} \
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
INIT=./output/pretrained_ckpt/${PRETRAINED_CKPT}
for SEED in "${FINETUNE_SEED[@]}"; do
for BATCH_SIZE in "${FINETUNE_BS[@]}"; do # 16, 64 for semeval0.01, 16 for semevalfull
for LR in "${FINETUNE_LR[@]}"; do
for EPOCH in "${FINETUNE_EPOCH[@]}"; do
python3 run_re.py \
    --model_name_or_path ${INIT} \
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