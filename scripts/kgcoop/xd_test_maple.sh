#!/bin/bash

cd ../..

# custom config
DATA="/home/data2/hechuan/DATA"
TRAINER=KgCoOp

DATASET=$1
SEED=$2

CFG=cross
SHOTS=16

for DATASET in imagenetv2 imagenet_sketch imagenet_a imagenet_r
do 
for SEED in 1 2 3
do
DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

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
fi
done
done