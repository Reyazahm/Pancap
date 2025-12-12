#!/bin/bash

############################################################################################################################################### 
# 
# Note!
#   -  Please obtain the stage2-trained checkpoint from "https://huggingface.co/OpenGVLab/ASMv2" and place it to "./checkpoints/ASMv2"
# 
############################################################################################################################################### 
# 
# Paramters to be modified in this scripts
#   -  --include="localhost:0,1,2,4": use the GPUs 0, 1, 2, 4 to run code
#   -  PRETRAIN_PATH: default="./checkpoints/ASMv2", the path of the pretrained model
#   -  DATA_PATH: default="./playground/data/pancap/pancapchain_train_data.json", the annotation json file of the training set
#       - You can use "playground/data/pancap/check_filepath.py" to check if the file_path of each image is correct.
#   -  DATA_ROOT: the root_path of data (i.e., images), please modify it
#   -  SAVE_CKPT: default="./checkpoints/pancapchain-lora"
#   -  TRAIN_BZ: training batchsize, please modify it adaptively
#   -  ACCU_STEP: accumulation steps to increase the batchsize, please modify it adaptively, NUM_GPU * TRAIN_BZ * ACCU_STEP = 512
#   -  NUM_EPOCH: default=1, the number of epochs for training
# 
############################################################################################################################################### 
#       

PRETRAIN_PATH="./checkpoints/ASMv2"
DATA_PATH="./playground/data/pancap/sapancap_train_data_pancapchain.json"
DATA_ROOT="/home/kylin/datasets/"
SAVE_CKPT="./checkpoints/pancapchain-lora/"
TRAIN_BZ=4
ACCU_STEP=8
NUM_EPOCH=1

deepspeed --include="localhost:0,1,2,3" --master_port 31900 llava/train/train_mem.py \
    --deepspeed ./scripts_pancap/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path ${PRETRAIN_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${DATA_ROOT} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${SAVE_CKPT} \
    --num_train_epochs ${NUM_EPOCH} \
    --per_device_train_batch_size ${TRAIN_BZ} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${ACCU_STEP} \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 10000 \
    --save_total_limit 10000 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to tensorboard 

