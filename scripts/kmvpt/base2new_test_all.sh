#!/bin/bash

cd ../..

# custom config
DATA="/home/data2/hechuan/multimodal-prompt-learning/DATA"
TRAINER=KMVPT

CFG=vit_b16_c2_ep6_batch8
SHOTS=16
LOADEP=6
SUB=new

for DATASET in {"eurosat","fgvc_aircraft","dtd","oxford_flowers","food101",'stanford_cars','caltech101','oxford_pets','ucf101','sun397'}
do
    for SEED in 1 2 3
    do
        COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
        DIR=output/base2new/test_${SUB}/${COMMON_DIR}
        if [ -d "$DIR" ]; then
            echo "Evaluating model"
            echo "Results are available in ${DIR}. Resuming..."

            python trainer.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB}

        else
            echo "Evaluating model"
            echo "Runing the first phase job and save the output to ${DIR}"

            python trainer.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB}
        fi
    done
done
