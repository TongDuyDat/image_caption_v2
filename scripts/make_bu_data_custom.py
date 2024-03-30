from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse
from pathlib import Path
from tqdm import tqdm
parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='data/flickr8k_talk_box', help='downloaded feature directory')
parser.add_argument('--output_dir', default='data/cocobu', help='output feature files')
parser.add_argument('--input_json', default='data/dataset_flickr8k.json', help="input json path")
args = parser.parse_args()


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# os.makedirs(args.output_dir+'_att')
# os.makedirs(args.output_dir+'_fc')
# os.makedirs(args.output_dir+'_box')

#load json file 
imgs = json.load(open(args.input_json, 'r'))
dataset = imgs["dataset"]
imgs = imgs['images']

npz_data = np.load(Path(os.path.join(args.downloaded_feats, str(imgs[0]["imgid"]))).with_suffix('.npz'), allow_pickle=True)
for key, value in npz_data.items():
    print(f'{key}: {value}')
for i, img in enumerate(tqdm(imgs)):
    item = {}
    filename = img["filename"].split(".")[0]
    image_id = img["imgid"]
    npz_data = np.load(Path(os.path.join(args.downloaded_feats, str(image_id))).with_suffix('.npz'), allow_pickle=True)
    item["image_id"] = image_id
    item["num_boxes"] = npz_data["num_bbox"]
    item['boxes'] = npz_data['bbox'].astype(np.float32)
    item['features'] = npz_data['features'].astype(np.float32)
    if 0 in item['boxes'].shape:
        print(item['boxes'].shape)
        print(filename)
    np.savez_compressed(os.path.join(args.output_dir+'_att', str(item['image_id'])), feat=item['features'])
    np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
    np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])

