#!/bin/bash

cd ../..

# custom config
DATA="/home/data2/hechuan/multimodal-prompt-learning/DATA"
TRAINER=KgCoOp

DATASET=imagenet
SEED=$2

CFG=cross
SHOTS=16

for SEED in  2
do
DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python trainer.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
done