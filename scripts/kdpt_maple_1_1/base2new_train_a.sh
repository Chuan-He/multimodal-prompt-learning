#!/bin/bash

cd ../..

# custom config
DATA="../DATA"
TRAINER=KDPT_MAPLE_1_1

CFG=test
SHOTS=16
LOADEP=10

for DATASET in imagenet #oxford_flowers food101 stanford_cars oxford_pets caltech101 sun397 #imagenet dtd eurosat fgvc_aircraft ucf101 
do
    for SEED in 1 2 3
    do
        #DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        COMMON_DIR=${DATASET}/${TRAINER}/${CFG}_shots_${SHOTS}/seed${SEED}
        DIRTRAIN=./B2N/train_base/${COMMON_DIR}
        DIRTEST=./B2N/test_new/${COMMON_DIR}

        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Resuming..."
            python trainer_1_1.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIRTRAIN} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base

            python trainer_1_1.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIRTEST} \
            --model-dir ${DIRTRAIN} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES new
        else
            echo "Run this job and save the output to ${DIR}"
            python trainer_1_1.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIRTRAIN} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base

            python trainer_1_1.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIRTEST} \
            --model-dir ${DIRTRAIN} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES new
        fi
    done
done