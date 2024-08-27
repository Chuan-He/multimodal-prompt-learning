#!/bin/bash

cd ../..

# custom config
DATA="../DATA"
TRAINER=KDPT_MAPLE_2

#DATASET=$1
#SEED=$2

CFG=cross
SHOTS=16

for DATASET in eurosat fgvc_aircraft ucf101 stanford_cars oxford_pets caltech101
do
for SEED in 2
do
DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
# if [ -d "$DIR" ]; then
#     echo "Results are available in ${DIR}. Skip this job"
# else
    echo "Run this job and save the output to ${DIR}"

    python trainer_cross.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 2 \
    --eval-only
# fi
done
done