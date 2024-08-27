#!/bin/bash

cd ../..

# custom config
DATA="DATA"
TRAINER=ZeroshotCLIP
#DATASET=$1
CFG=test_b4  # rn50, rn101, vit_b32 or vit_b16

for DATASET in eurosat fgvc_aircraft dtd oxford_flowers food101 stanford_cars caltech101 oxford_pets ucf101 sun397 imagenet
do 
    python trainer.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir output/${TRAINER}/${CFG}/${DATASET} \
    --eval-only \
    DATASET.SUBSAMPLE_CLASSES new \
    DATASET.NUM_SHOTS 16
done