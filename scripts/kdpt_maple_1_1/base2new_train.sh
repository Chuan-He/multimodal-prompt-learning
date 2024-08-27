#!/bin/bash

cd ../..

# custom config
DATA="../DATA"
TRAINER=KDPT_MAPLE_1_1

DATASET=$1
SEED=$2

CFG=test
SHOTS=16
LOADEP=10

#DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
COMMON_DIR=${DATASET}/${TRAINER}/${CFG}_shots_${SHOTS}/seed${SEED}
DIR=./B2N/train_base/${COMMON_DIR}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python trainer_1_1.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    python trainer_1_1.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi
