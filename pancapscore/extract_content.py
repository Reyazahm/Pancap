import os
import pdb
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
from multiprocessing import Process

#
# parse input parameters
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--num-gpu', type=int, default=1)
parser.add_argument('--caption_json', type=str)
parser.add_argument('--content_json', type=str)
parser.add_argument('--response_key', type=str, default='model_response')
args = parser.parse_args()

num_process = args.num_gpu
text_json = args.caption_json
result_json = args.content_json

file_path = Path(result_json)
dir_path = file_path.parent
dir_path.mkdir(parents=True, exist_ok=True)

prompt = "Extract all contents from the following paragraph. The extracted contents should cover all sentences of the paragraph. You should output the semantic tags of all entities and their spatial locations mentioned in the paragraph. In addition, according to the paragraph, you should detailedly describe attributes and the relationships between all pairs of entities, and you shoud output the global status of the image. Please output the mentioned contents by using multiple lines. \n" + \
"The semantic tags should include all entities in the image. The attributes should include color, shape, type, state, material, texture, text rendering, appearance, decoration and any other attributes of all entities. The relationships should include part-whole relation, spatial relation, scale relation and action relation between all pairs of entities. The global status should include feeling, camera viewpoint, and any other status. \n" + \
"\nNote that each line of your output should strictly follow a specific format: ID | Type -->> Tuple . \n" + \
"The output format should strictly follow six rules as below:\n" + \
"1. Different lines should have different values of ID. \n" + \
"2. For the line indicating the semantic tag of an entity instance, the outputted Tuple should consist of two elements. The first element is \"whole\" or \"part\", and the second element is a short text indicating semantic tag. \n" + \
"3. For the line indicating the location of an entity instance, the outputted Tuple should consist of two elements. The first element is the ID of the entity instance, and the second element is a text consisting of bounding box coordinates of the location. A mentioned location in the paragraph should correspond to ONLY  one entity instance in the image. \n" + \
"4. For the line indicating the attribute of an entity instance, the outputted Tuple should consist of two elements. The first element is the ID of the entity, and the second element is a sentence describing one type of attribute. Please extract comprehensive descriptions for each entity. Please use the original words whenever possible according to the paragraph, with only the entity name is replaced by the entity instance ID. \n" + \
"5. For the line indicating the relation between instances of entity, the outputted Tuple should consist of four elements. The first and second elements are the IDs of entity instances within the relation, and the third element is a sentence describing one type of relation. \n" + \
"6. For the line indicating global status of the image, the outputted Tuple should consist of one element. This element is a sentence describing one type of global status. \n" + \
"\nAn example is given as follows: " + \
"\nGiven a description " + \
"\"A blue motorcycle ([[0.50, 0.70, 0.83, 1.00]]) parked by a painted door of a big house ([[0.01, 0.40, 1.00, 0.98]]), with a clear and expansive sky ([0.00, 0.00, 1.00, 0.40]) overhead. The motorcyle has black wheels. The scene feels tranquil and timeless.\"" + \
", you should extract all semantic tags, spatial locations, attributes and relationship from the given description following the below format:\n" + \
"1 | entity -->> (whole, motocycle) \n" + \
"2 | entity -->> (whole, house) \n" + \
"3 | entity -->> (whole, sky) \n" + \
"4 | entity -->> (part, door)\n" + \
"5 | entity -->> (part, wheels)\n" + \
"6 | location -->> (entity 1, [0.50, 0.70, 0.83, 1.00]) \n" + \
"7 | location -->> (entity 2, [0.01, 0.40, 1.00, 0.98]) \n" + \
"8 | location -->> (entity 3, [0.00, 0.00, 1.00, 0.40]) \n" + \
"9 | attribute -->> (entity 1, \"entity 1 is blue\") \n" + \
"10 | attribute -->> (entity 2, \"entity 2 is big\") \n" + \
"11 | attribute -->> (entity 4, \"entity 4 is painted\") \n" + \
"12 | attribute -->> (entity 1, \"entity 1 is parked\") \n" + \
"13 | attribute -->> (entity 5, \"entity 5 is black\") \n" + \
"14 | attribute -->> (entity 3, \"entity 3 is clear and expansive\") \n" + \
"15 | relation -->> (entity 1, entity 4, \"entity 1 is next to entity 4\") \n" + \
"16 | relation -->> (entity 3, entity 1, \"entity 3 is above entity 1\") \n" + \
"17 | relation -->> (entity 3, entity 2, \"entity 3 is above entity 2\") \n" + \
"18 | relation -->> (entity 5, entity 1, \"entity 5 is a part of entity 1\") \n" + \
"19 | relation -->> (entity 4, entity 2, \"entity 4 is a part of entity 2\") \n" + \
"20 | relation -->> (entity 2, entity 1, \"entity 2 is bigger than entity 1\") \n" + \
"21 | global -->> (\"the scene feels tranquil and timeless\") \n" + \
"You should fully understand this example to learn the required way to extract information and the formats of outputted lines. \n" + \
"\nFor the spatial location, [x1, y1, x2, y2] denotes an extracted bounding box. Different lines are separated by '\n'. Each bounding box should exactly follow the orginal format in the paragraph, DO NOT change the number of bounding box coordinates in any case. \n" + \
"\nThe original paragraph is as follows: \n" + \
"{}"

def process_data(gpu_idx, data_list):
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": gpu_idx},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for resp in tqdm(data_list):
        # skip existing results
        if os.path.exists(result_json.replace('.json', '_{}.json'.format('_'.join(resp['image_path'][:-4].split('/')[-3:])))):
            continue
    
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt.format(resp[args.response_key])}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=8192
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
        res_dct = {'image_path': resp['image_path'], 'caption_file': text_json, 'all_content': response}
        with open(result_json.replace('.json', '_{}.json'.format('_'.join(resp['image_path'][:-4].split('/')[-3:]))), 'w') as f:
            json.dump(res_dct, f, indent=4)
    

print('Extracting all content...')

# load data_list
with open(text_json) as f:
    resp_list = json.load(f)
chunks = [resp_list[i:len(resp_list):num_process] for i in range(num_process)]
processes = []
for i, chunk in enumerate(chunks):
    process = Process(target=process_data, args=(i, chunk))
    processes.append(process)
    process.start()

print('All processes start...')
for process in processes:   # wait for all processes end
    process.join()
print('All processes finished!')

result_list = []
result_files = [result_json.replace('.json', '_{}.json'.format('_'.join(resp['image_path'][:-4].split('/')[-3:]))) for resp in resp_list]
for fname in result_files:
    assert os.path.exists(fname)
    with open(fname) as f:
        dct = json.load(f)
    result_list.append(dct)

with open(result_json, 'w') as f:
    json.dump(result_list, f, indent=4)


