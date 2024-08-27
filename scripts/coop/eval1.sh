#!/bin/bash

cd ../..

# custom config
DATA='../DATA'
TRAINER=CoOp
SHOTS=16
NCTX=2
CSC=False
CTP=end
SUB=new

#DATASET=$1
CFG='test_b4'

for DATASET in eurosat fgvc_aircraft dtd oxford_flowers food101 stanford_cars caltech101 oxford_pets ucf101 sun397 imagenet
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
        --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
        --load-epoch 10 \
        --eval-only \
        DATASET.SUBSAMPLE_CLASSES ${SUB} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
    done
done