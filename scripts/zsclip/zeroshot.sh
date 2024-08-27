#!/bin/bash

cd ../..

# custom config
DATA="DATA"
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=test_b8  # rn50, rn101, vit_b32 or vit_b16


python trainer.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES new \
DATASET.NUM_SHOTS 16
