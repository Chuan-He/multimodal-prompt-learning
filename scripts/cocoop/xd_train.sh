#!/bin/bash

cd ../..

# custom config
DATA='../DATA'
TRAINER=CoCoOp

DATASET=imagenet
SEED=$1

CFG=cross
SHOTS=16

for SEED in 3
do
DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
#if [ -d "$DIR" ]; then
    #echo "Results are available in ${DIR}. Skip this job"
#else
    echo "Run this job and save the output to ${DIR}"

    python trainer.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
#fi
done