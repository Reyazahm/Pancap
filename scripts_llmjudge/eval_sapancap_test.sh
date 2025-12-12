#!/bin/bash

############################################################################################################################################### 
# 
# Paramters to be modified in this scripts
#   -  export CUDA_VISIBLE_DEVICES=0,1,2,3,4: use the GPUs 0, 1, 2, 3, 4 to run code (one GPU for one process)
#   -  NUM_GPUS: you should set it as the number of gpus used accordingly
#   -  DATA_ROOT: the root_path of data (i.e., images), please modify it
#   -  SAVE_CKPT: default="pancapchain-lora-merge"
#   -  CACHE_PTH: path of cached temporal results (e.g., extracted content, generated questions)
#   -  GT_CAPTION: path of GT captions (of val and test sets)
# 
############################################################################################################################################### 
#       

NUM_GPUS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

DATA_ROOT="/home/kylin/datasets/"
SAVE_CKPT="pancapchain-lora-merge"
CACHE_PTH="./tmp/"
GT_CAPTION="playground/data/pancap/sapancap_test_gtcaption.json"
GT_CONTENT=${CACHE_PTH}"GT/content/sapancap_test_content.json"
GT_QUESTIONS=${CACHE_PTH}"GT/questions/sapancap_test_question.json"


############################################################################################################################################### 
# paths of cached results (usually, you may not modify this part)
PRED_DIR="playground/data/pancap_test/checkpoints/"${SAVE_CKPT}
PRED_CAPTION="playground/data/pancap_test/checkpoints/"${SAVE_CKPT}"-all-predictions.json"
PRED_CONTENT=${CACHE_PTH}${SAVE_CKPT}"/content/sapancap_test_content.json"
GT4PRED_QUESTIONS=${CACHE_PTH}${SAVE_CKPT}"/content/sapancap_test_gt4pred_question.json"
PRED_ANSWERS=${CACHE_PTH}${SAVE_CKPT}"/content/sapancap_test_pred_answer.json"
PRED_QUESTIONS=${CACHE_PTH}${SAVE_CKPT}"/content/sapancap_test_pred_question.json"
PRED4GT_QUESTIONS=${CACHE_PTH}${SAVE_CKPT}"/content/sapancap_test_pred4gt_question.json"
GT_ANSWERS=${CACHE_PTH}${SAVE_CKPT}"/content/sapancap_test_gt_answer.json"
############################################################################################################################################### 


# Step 0: aggregate model predictions, and transform response into GT format (from asv2 format)
python -W ignore scripts_llmjudge/agg_jsons_asv2togt.py \
    --data-root ${DATA_ROOT} --res-dir ${PRED_DIR}

############################################################################################################################################### 
# You can skip the following GT-related steps after the first evaluation, since the temporal results are cached in CACHE_PTH
# Step 1: extract semantic content from GT
python -W ignore pancapscore/extract_content.py --num-gpu ${NUM_GPUS} --response_key "gt_response" \
    --caption_json ${GT_CAPTION} --content_json ${GT_CONTENT} 

# Step 2: generate GT's questions from extracted content
python -W ignore pancapscore/generate_questions.py --num-gpu ${NUM_GPUS} --response_key "gt_response" \
    --caption_json ${GT_CAPTION} --content_json ${GT_CONTENT} \
    --question_json ${GT_QUESTIONS} 
############################################################################################################################################### 

# Step 3: extract semantic content from predictions
python -W ignore pancapscore/extract_content.py --num-gpu ${NUM_GPUS} --response_key "model_response" \
    --caption_json ${PRED_CAPTION} --content_json ${PRED_CONTENT}

# Step 4: entity instance matching
python -W ignore pancapscore/eval_tag_and_loc.py \
    --cand_file ${PRED_CONTENT} \
    --gt_file ${GT_QUESTIONS} \
    --save_file ${GT4PRED_QUESTIONS}

# Step 5: answer GT's quetions
python -W ignore pancapscore/answer_questions.py --num-gpu ${NUM_GPUS} --response_key "model_response" \
    --caption_json ${PRED_CAPTION} --question_json ${GT4PRED_QUESTIONS} \
    --result_json ${PRED_ANSWERS} 

# Step 6: generate predition's questions from extracted content
python -W ignore pancapscore/generate_questions.py --num-gpu ${NUM_GPUS} --response_key "model_response" \
    --caption_json ${PRED_CAPTION} --content_json ${PRED_CONTENT} \
    --question_json ${PRED_QUESTIONS} 

# Step 7: entity instance matching
python -W ignore pancapscore/eval_tag_and_loc.py \
    --cand_file ${GT_CONTENT} \
    --gt_file ${PRED_QUESTIONS} \
    --save_file ${PRED4GT_QUESTIONS}

# Step 8: answer predition's quetions 
python -W ignore pancapscore/answer_questions.py --num-gpu ${NUM_GPUS} --response_key "gt_response" \
    --caption_json ${GT_CAPTION} --question_json ${PRED4GT_QUESTIONS} \
    --result_json ${GT_ANSWERS}  


# Step 9: overall evaluation 
python -W ignore pancapscore/eval_alldims.py \
    --cand_file ${PRED_CONTENT} \
    --gt_file ${GT_QUESTIONS} \
    --gt4pred_answer_file ${PRED_ANSWERS} \
    --pred4gt_answer_file ${GT_ANSWERS}

