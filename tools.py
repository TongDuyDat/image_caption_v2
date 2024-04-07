import json
import os
from pathlib import Path

import cv2

data_root = 'D:/NCKH/ImageCaption/Dataset/Flickr8k_Dataset'
val_file = "D:/NCKH/ImageCaption/Dataset/Flickr8k_text/Flickr_8k.devImages.txt"
def convert_to_coco_format(flickr8k_data):
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": []
    }

    image_id = 1
    annotation_id = 1
    with open(os.path.join(val_file), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    # return
    for image_info in flickr8k_data:
        file_name = image_info["filename"]
        if file_name not in lines:
            print(file_name)
            continue
        sentences = image_info["sentences"]
        sent_ids = image_info["sentids"]
        image_id = image_info['imgid']
        img = cv2.imread(str(Path(os.path.join(data_root, str(file_name))).with_suffix('.jpg')))
        h, w, _ = img.shape
        coco_image = {
            "id": image_id,
            "file_name": file_name,
            "width": w,
            "height": h,
            "date_captured": None,
            "license": None,
            "flickr_url": None
        }
        coco_data["images"].append(coco_image)

        for i, sent_id in enumerate(sent_ids):
            sentence = sentences[i]
            caption = sentence["raw"]
            coco_annotation = {
                "id": sent_id,
                "image_id": image_id,
                "caption": caption
            }
            coco_data["annotations"].append(coco_annotation)
            # annotation_id += 1

        # image_id += 1

    return coco_data

# Đọc dữ liệu Flickr8k từ file JSON
with open("data/dataset_flickr8k.json", "r") as f:
    flickr8k_data = json.load(f)["images"]
print(len(flickr8k_data))
# Chuyển đổi dữ liệu sang định dạng COCO
coco_data = convert_to_coco_format(flickr8k_data)

# Lưu dữ liệu COCO vào file JSON
with open("flickr_val_anots_data.json", "w") as f:
    json.dump(coco_data, f)