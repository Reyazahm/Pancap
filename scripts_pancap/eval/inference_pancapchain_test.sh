#!/bin/bash

# 
DATA_ROOT="/home/kylin/datasets/"

NUM_GPUS=6
GPUS=("0" "1" "2" "3" "4" "5")

#
# model path
MODEL_PATH="./checkpoints/pancapchain-lora-merge"

QFILE=playground/data/pancap/sapancap_test_data_list.json

for INDEX in "${!GPUS[@]}"; do
    GPU_IDX=${GPUS[INDEX]}
    echo ${INDEX}
    echo ${GPU_IDX}
    AFILE=playground/data/pancap_test/${MODEL_PATH}/ 
    CUDA_VISIBLE_DEVICES=${GPU_IDX} python llava/eval/model_vqa_loader_pancapchain.py \
        --model-path ${MODEL_PATH} \
        --question-file ${QFILE} \
        --image-folder ${DATA_ROOT} \
        --answers-file ${AFILE} \
        --num-chunk ${NUM_GPUS} \
        --chunk-idx ${INDEX} \
        --temperature 0 \
        --max_new_tokens 1024 \
        --conv-mode vicuna_v1 &  
done

