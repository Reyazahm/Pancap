import os
import re
import pdb
import json
import copy
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--data-root', type=str)
parser.add_argument('--res-dir', type=str)
args = parser.parse_args()

data_root = args.data_root
dir_name = args.res_dir
resp_file = dir_name + '-all-predictions.json'

pattern = r'<box>\[\[(\d+), (\d+), (\d+), (\d+)\]\]</box>'
def asv2box2gtformat(box, width, height, org_size, new_size):
    if width > height:
        for i in range(4):
            box[i] *= width / new_size
        pad = (width-height)//2
        box[1] -= pad
        box[3] -= pad
    else:
        for i in range(4):
            box[i] *= height / new_size
        pad = (height-width)//2
        box[0] -= pad
        box[2] -= pad
    box = [box[0]/width*org_size, box[1]/height*org_size, box[2]/width*org_size, box[3]/height*org_size]
    new_box = []
    for b in box:
        if b < 0:
            b = 0
        if b > org_size:
            b = org_size
        new_box.append(b)
    return new_box


anno_list = []
json_list = sorted(os.listdir(dir_name))
for jpath in json_list:
    json_path = os.path.join(dir_name, jpath)
    with open(json_path) as f:
        dct = json.load(f)

    pth = dct['image']
    ans = dct['text']

    img_path = os.path.join(data_root, pth)
    with Image.open(img_path) as img:
        width, height = img.size

    matches = list(re.finditer(pattern, ans))
    new_ans = copy.deepcopy(ans)
    if len(matches) > 0:
        for mbox in matches:
            box = [float(mbox.group(1)), float(mbox.group(2)), float(mbox.group(3)), float(mbox.group(4))]
            box = asv2box2gtformat(box, width, height, 1.0, 1000)
            box_str = '[[{:.3f}, {:.3f}, {:.3f}, {:.3f}]]'.format(box[0], box[1], box[2], box[3])
            new_ans = new_ans.replace(mbox.group(0), box_str)

    anno_dct = {'image_path': pth, 'model_response': new_ans}
    anno_list.append(anno_dct)

assert not os.path.exists(resp_file), '{} exists'.format(resp_file)
with open(resp_file, 'w') as f:
    json.dump(anno_list, f, indent=4)
print('Aggregated results saved in "{}".'.format(resp_file))

