#!/bin/bash

cd ../..

# custom config
DATA="../DATA"
TRAINER=KDPT_MAPLE

#DATASET=$1
#SEED=$2

CFG=cross
SHOTS=16

for DATASET in eurosat #fgvc_aircraft dtd oxford_flowers food101 stanford_cars oxford_pets ucf101 caltech101 sun397 imagenetv2 imagenet_sketch imagenet_a imagenet_r
do 
for SEED in 1
do
DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}

    python trainer.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 2 \
    --eval-only

done
done