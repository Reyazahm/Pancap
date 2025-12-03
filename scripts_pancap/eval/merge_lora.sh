#!/bin/bash

#
MODEL_NAME='pancapchain-lora'

TRAIN_CKPT="./checkpoints/"${MODEL_NAME}
PRETRAIN_CKPT="./checkpoints/ASMv2"
MERGED_CKPT=${TRAIN_CKPT}"-merge"

CUDA_VISIBLE_DEVISES=0 python scripts_pancap/eval/merge_lora_weights.py \
    --model-path ${TRAIN_CKPT} \
    --model-base ${PRETRAIN_CKPT} \
    --save-model-path ${MERGED_CKPT}

