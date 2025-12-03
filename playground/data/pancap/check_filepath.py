import os
import json

root_pth = '/home/kylin/datasets/'
json_pth = 'sapancap_train_data_pancapchain.json'

with open(json_pth) as f:
    anno_list = json.load(f)

for anno in anno_list:
    img_pth = os.path.join(root_pth, anno['image'])
    if not os.path.exists(img_pth):
        print(img_pth, "not exists.")

