#!/bin/bash

cd ../..

# custom config
DATA="../DATA"
TRAINER=KDPT_MAPLE_2

CFG=test
SHOTS=16
LOADEP=8
SUB=new

for DATASET in fgvc_aircraft eurosat  #{"eurosat","fgvc_aircraft","dtd","oxford_flowers","food101",'stanford_cars','oxford_pets','ucf101','caltech101','sun397','imagenet'}
do
    for SEED in 1 2 3 4 5
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
