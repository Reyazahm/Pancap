import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        # line = self.questions[index]
        # image_file = line["image"]
        image_file = self.questions[index]

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        return image_tensor

    def __len__(self):
        return len(self.questions)

#
def get_inputids(dataset, image_tensor, qs):
    if dataset.model_config.mm_use_im_start_end and dataset.model_config.mm_use_im_patch_token:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * dataset.model_config.num_query_tokens + DEFAULT_IM_END_TOKEN + '\n' + qs
    elif dataset.model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, dataset.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    return input_ids


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataset, data_loader


# by kunyu
def equal_box(btag1, btag2):
    if btag1[0] == btag2[0] and btag1[1] == btag2[1] \
        and btag1[2] == btag2[2] and btag1[3] == btag2[3]:
        return True
    return False

def extract_box(anno_text):
    bbox_list = anno_text.split('</box>, ')
    bbox_list[-1] = bbox_list[-1].replace('</box>', '')
    box_list = []
    equal_cnt = 0 
    for i, bbox in enumerate(bbox_list):
        bbox = bbox.replace('<box>', '')
        try:
            box = bbox.lstrip('[').rstrip(']')
            box = [int(b) for b in box.split(', ')]
            assert len(box) == 4
            flag = True
            for j in range(len(box_list)):
                if equal_box(box, box_list[j]):
                    flag = False
            if flag:
                box_list.append(box)
        except Exception as e:
            # print(f"A format error occurred")
            break
    return box_list

def deduplicate_box(text):
    boxs = extract_box(text)
    bstr_list = []
    for box in boxs:
        bstr_list.append('<box>[[{}, {}, {}, {}]]</box>'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3])))
    return ', '.join(bstr_list)

def equal_boxtag(btag1, btag2):
    if btag1[0] == btag2[0] and btag1[1][0] == btag2[1][0] and btag1[1][1] == btag2[1][1] \
        and btag1[1][2] == btag2[1][2] and btag1[1][3] == btag2[1][3]:
        return True
    return False

def extract_boxtag(anno_text):
    btag_list = anno_text.split('</box>, ')
    btag_list[-1] = btag_list[-1].replace('</box>', '')
    box_list = []
    tag_list = []
    equal_cnt = 0 
    for i, btag in enumerate(btag_list):
        tag = btag.split(' <box>')[0]
        try:
            box = btag.split(' <box>')[1].lstrip('[').rstrip(']')
            box = [int(b) for b in box.split(', ')]
            assert len(box) == 4
            flag = True
            for j in range(len(box_list)):
                if equal_boxtag((tag, box), (tag_list[j], box_list[j])):
                    flag = False
            if flag:
                tag_list.append(tag)
                box_list.append(box)
        except Exception as e:
            # print(f"A format error occurred")
            break
    return box_list, tag_list

def deduplicate_boxtag(text):
    boxs, tags = extract_boxtag(text)
    bstr_list = []
    for box, tag in zip(boxs, tags):
        bstr_list.append('{} <box>[[{}, {}, {}, {}]]</box>'.format(tag, int(box[0]), int(box[1]), int(box[2]), int(box[3])))
    return ', '.join(bstr_list)

def extract_boxtag_v2(anno_text, box_list, tag_list):
    if anno_text[:4] == 'None':
        return box_list, tag_list
    btag_list = anno_text.split('</box>, ')
    btag_list[-1] = btag_list[-1].replace('</box>', '')
    for i, btag in enumerate(btag_list):
        tag = btag.split(' <box>')[0]
        try:
            box = btag.split(' <box>')[1].lstrip('[').rstrip(']')
            box = [int(b) for b in box.split(', ')]
            assert len(box) == 4
            flag = True
            for j in range(len(box_list)):
                if equal_boxtag((tag, box), (tag_list[j], box_list[j])):
                    flag = False
            if flag:
                tag_list.append(tag)
                box_list.append(box)
        except Exception as e:
            print(f"A format error occurred")
            break
    return box_list, tag_list

def merge_and_deduplicate(base_text, more_text):
    base_boxs, base_tags = extract_boxtag_v2(base_text, [], [])
    boxs, tags = extract_boxtag_v2(more_text, base_boxs, base_tags)
    bstr_list = []
    for box, tag in zip(boxs, tags):
        bstr_list.append('{} <box>[[{}, {}, {}, {}]]</box>'.format(tag, int(box[0]), int(box[1]), int(box[2]), int(box[3])))
    return ', '.join(bstr_list)


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = json.load(open(os.path.expanduser(args.question_file)))

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:    # default: False
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    dataset, data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for idx, (image_tensor, img) in tqdm(enumerate(zip(data_loader, questions)), total=len(questions)):
        ans_name = '_'.join(img[:-4].split('/'))+'.json'
        if os.path.exists(os.path.join(answers_file, ans_name)):
            continue

        # stage1: entity localization 
        cur_prompt = "Please localize all entities in this image."
        # print(idx, cur_prompt)
        input_ids = get_inputids(dataset, image_tensor, cur_prompt).unsqueeze(0).to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        output1 = outputs.strip()
        # print(output1)
        output1 = deduplicate_box(output1)
        # print(output1)

        # stage2: semantic tags 
        cur_prompt = "Please specify the semantic tags of all entities based on their bounding boxes: {}".format(output1)
        # print(idx, cur_prompt)
        input_ids = get_inputids(dataset, image_tensor, cur_prompt).unsqueeze(0).to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        output2 = outputs.strip()
        # print(2, output2)
        output2 = deduplicate_boxtag(output2)
        # print(2, output2)

        # stage3: extra instance
        cur_prompt = "Please specify missing entities and their locations for this image based on these specified entities: {}".format(output2)
        # print(idx, cur_prompt)
        input_ids = get_inputids(dataset, image_tensor, cur_prompt).unsqueeze(0).to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        output3 = outputs.strip()
        # print(3, output3)
        output3 = merge_and_deduplicate(output2, output3)
        # print(3, output3)

        # stage4: pancap generation
        cur_prompt = "Please provide a hyper-detailed description for this image, including all entities, their locations, attributes, and relationships, as well as the global image status, based on boxes and tags: {}".format(output3)
        # print(idx, cur_prompt)
        input_ids = get_inputids(dataset, image_tensor, cur_prompt).unsqueeze(0).to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        # print()
        # print(outputs)

        ans_file = open(os.path.join(answers_file, ans_name), "w")
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "image": img, 
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="data_generation")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)

