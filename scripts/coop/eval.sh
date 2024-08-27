#!/bin/bash

cd ../..

# custom config
DATA='../DATA'
TRAINER=CoOp
SHOTS=16
NCTX=2
CSC=False
CTP=end

DATASET=$1
CFG=cross

for DATASET in imagenetv2 imagenet_sketch imagenet_a imagenet_r
do 
for SEED in 1 2 3
do
    python trainer.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --load-epoch 2 \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
done
done