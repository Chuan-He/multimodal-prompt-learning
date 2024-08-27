#!/bin/bash

cd ../..

# custom config
DATA="../DATA"
TRAINER=CoCoOp

CFG=test_b4
SHOTS=16

for DATASET in eurosat fgvc_aircraft dtd oxford_flowers food101 stanford_cars caltech101 oxford_pets ucf101 sun397 imagenet
do
    for SEED in 1 2 3
    do
        DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Resuming..."
            python trainer.py \
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
            python trainer.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
        fi
    done
done