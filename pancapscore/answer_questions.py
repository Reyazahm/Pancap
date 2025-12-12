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
parser.add_argument('--result_json', type=str)
parser.add_argument('--question_json', type=str)
parser.add_argument('--response_key', type=str)
args = parser.parse_args()
num_process = args.num_gpu
text_json = args.caption_json
result_json = args.result_json
question_json = args.question_json
response_key = args.response_key

file_path = Path(result_json)
dir_path = file_path.parent
dir_path.mkdir(parents=True, exist_ok=True)

prompt = "Answer questions one by one according to the paragraph. Please output the answers by using multiple lines. \n" + \
"\nNote that each line of your output should strictly follow a specific format: ID | Type -->> Tuple . \n" + \
"The output format should strictly follow four rules as below:\n" + \
"1. Different lines should have different values of ID as the questions. \n" + \
"2. For the line indicating the attribute of an entity instance, the outputted Tuple should consist of four elements. The first element indicates the type of attribute, the second element is the ID of the entity, the third element is the answer for the first question at the corresponding line, and the fourth element is the answer for the second question at the corresponding line. \n" + \
"3. For the line indicating the relation between instances of entity, the outputted Tuple should consist of four elements. The first element indicates the type of relation, the second and the third elements are the IDs of entity instances within the relation, the fourth element is the answer for the first question at the corresponding line, and the fifth element is the answer for the second question at the corresponding line. \n" + \
"4. For the line indicating global status of the image, the outputted Tuple should consist of two elements. The first element is the type of global status, the second element is the answer for the first question at the corresponding line, and the third element is the answer for the second question at the corresponding line. \n" + \
"\nAn example is given as follows: " + \
"\nGiven a paragraph " + \
"\"A blue motorcycle ([[0.50, 0.70, 0.83, 1.00]]) parked by a painted door of a big house ([[0.01, 0.40, 1.00, 0.98]]), with a clear and expansive sky ([0.00, 0.00, 1.00, 0.40]) overhead. The motorcyle has black wheels. The scene feels tranquil and timeless.\"" + \
" and a set of questions from this paragraph " + \
"\"" + \
"1 | entity -->> (whole, motocycle) \n" + \
"2 | entity -->> (whole, house) \n" + \
"3 | entity -->> (whole, sky) \n" + \
"4 | entity -->> (part, door)\n" + \
"5 | entity -->> (part, wheels)\n" + \
"6 | location -->> (entity 1, [0.50, 0.70, 0.83, 1.00]) \n" + \
"7 | location -->> (entity 2, [0.01, 0.40, 1.00, 0.98]) \n" + \
"8 | location -->> (entity 3, [0.00, 0.00, 1.00, 0.40]) \n" + \
"9 | attribute -->> (entity 1, \"Is entity 1 red?\", \"Is entity 1 blue?\") \n" + \
"10 | attribute -->> (entity 2, \"Is entity 2 small?\", \"Is entity 2 big?\") \n" + \
"11 | attribute -->> (entity 4, \"Is entity 4 painted?\", \"Is entity 4 clean with no painting?\") \n" + \
"12 | attribute -->> (entity 1, \"Is entity 1 running?\", \"Is entity 1 parked?\") \n" + \
"13 | attribute -->> (entity 5, \"Is entity 5 red?\", \"Is entity 5 black?\") \n" + \
"14 | attribute -->> (entity 3, \"Is entity 3 clear and expansive?\", \"Is entity 3 overcast?\") \n" + \
"15 | relation -->> (entity 1, entity 4, \"Is entity 1 far away from entity 4?\", \"Is entity 1 next to entity 4?\") \n" + \
"16 | relation -->> (entity 3, entity 1, \"Is entity 3 below entity 1?\", \"Is entity 3 under entity 1?\") \n" + \
"17 | relation -->> (entity 3, entity 2, \"Is entity 3 beyond entity 2?\", \"Is entity 3 above entity 2?\") \n" + \
"18 | relation -->> (entity 5, entity 1, \"Is entity 5 a part of entity 1?\", \"Does entity 5 contain entity 1?\") \n" + \
"19 | relation -->> (entity 4, entity 2, \"Is entity 2 a part of entity 4?\", \"Is entity 4 a part of entity 2?\") \n" + \
"20 | relation -->> (entity 2, entity 1, \"Is entity 2 bigger than entity 1?\", \"Is entity 2 smaller than entity 1?\") \n" + \
"21 | global -->> (\"Does the scene feel tranquil and timeless?\", \"Does the scene feel depressive?\") \n" + \
"\"" + \
", you should answer these lines of true or false questions about attributes, relationships or global status according to the given paragraph. Your answers should be outputted by multiple lines following the below format:\n" + \
"1 | entity -->> (whole, motocycle) \n" + \
"2 | entity -->> (whole, house) \n" + \
"3 | entity -->> (whole, sky) \n" + \
"4 | entity -->> (part, door)\n" + \
"5 | entity -->> (part, wheels)\n" + \
"6 | location -->> (entity 1, [0.50, 0.70, 0.83, 1.00]) \n" + \
"7 | location -->> (entity 2, [0.01, 0.40, 1.00, 0.98]) \n" + \
"8 | location -->> (entity 3, [0.00, 0.00, 1.00, 0.40]) \n" + \
"9 | attribute -->> (entity 1, \"No\", \"Yes\") \n" + \
"10 | attribute -->> (entity 2, \"No\", \"Yes\") \n" + \
"11 | attribute -->> (entity 4, \"Yes\", \"No\") \n" + \
"12 | attribute -->> (entity 1, \"No\", \"Yes\") \n" + \
"13 | attribute -->> (entity 5, \"No\", \"Yes\") \n" + \
"14 | attribute -->> (entity 3, \"Yes\", \"No\") \n" + \
"15 | relation -->> (entity 1, entity 4, \"No\", \"Yes\") \n" + \
"16 | relation -->> (entity 3, entity 1, \"No\", \"No\") \n" + \
"17 | relation -->> (entity 3, entity 2, \"Yes\", \"Yes\") \n" + \
"18 | relation -->> (entity 5, entity 1, \"Yes\", \"No\") \n" + \
"19 | relation -->> (entity 4, entity 2, \"No\", \"Yes\") \n" + \
"20 | relation -->> (entity 2, entity 1, \"Yes\", \"No\") \n" + \
"21 | global -->> (\"Yes\", \"No\") \n" + \
"\n" + \
"You should fully understand this example to learn the required way to answer questions and the formats of outputted lines. \n" + \
"\nThe paragraph is as follows: {}\n" + \
"And the set of questions is as follows: {}"

def process_data(gpu_idx, data_list, content_dct):
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
            {"role": "user", "content": prompt.format(resp[response_key], content_dct['/'.join(resp['image_path'].split('/')[-3:])])}
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
    
        res_dct = {'image_path': '/'.join(resp['image_path'].split('/')[-3:]), 'original_file': text_json, 'answers': response}
        with open(result_json.replace('.json', '_{}.json'.format('_'.join(resp['image_path'][:-4].split('/')[-3:]))), 'w') as f:
            json.dump(res_dct, f, indent=4)


print('Answering questions...')

# load data_list
with open(text_json) as f:
    resp_list = json.load(f)
with open(question_json) as f:
    content_list = json.load(f)
    content_dct = {}
    for item in content_list:
        content_dct['/'.join(item['image_path'].split('/')[-3:])] = item['questions']

chunks = [resp_list[i:len(resp_list):num_process] for i in range(num_process)]
processes = []
for i, chunk in enumerate(chunks):
    process = Process(target=process_data, args=(i, chunk, content_dct))
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


