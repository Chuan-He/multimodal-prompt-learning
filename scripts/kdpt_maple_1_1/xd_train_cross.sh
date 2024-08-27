#!/bin/bash

cd ../..

# custom config
DATA="../DATA"
TRAINER=KDPT_MAPLE_1_1

DATASET=imagenet

CFG=cross
SHOTS=16

for SEED in 1 2 3
do
    DIR=DG/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python trainer_1_1_cross.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
done