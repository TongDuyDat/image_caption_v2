import cv2
import numpy as np

# Path to the .npz file
npz_file_path = "data/flickr8k_talk_box/3767841911_6678052eb6.npz"
img = "D:/NCKH/ImageCaption/Dataset/Flickr8k_Dataset/3767841911_6678052eb6.jpg"
# Load the .npz file
npz_data = np.load(npz_file_path, allow_pickle=True)

# Iterate over keys and print corresponding values
print(npz_data)
for key in npz_data:
    print(key, ":", npz_data[key])
width, height = npz_data["image_w"], npz_data['image_h']
im0 = cv2.imread(img)  # Read image using OpenCV

for xywh in  npz_data["bbox"]: 
    x1 = int(xywh[0] -  (xywh[2])/2) 
    y1 = int(xywh[1] -  (xywh[3])/2)
    x2 = int(xywh[0] +  (xywh[2])/2)
    y2 = int(xywh[1] +  (xywh[3])/2)
    
    p1  = (x1, y1)               # point 1 of rectangle
    p2 = (x2, y2)   
    
    p1_ = (int(xywh[0]), int(xywh[1]))
    p2_ = (int(xywh[2]), int(xywh[3]))
    im0 = cv2.rectangle(im0, p1_, p2_, (0, 0, 255), 2, 1, 1)
cv2.imshow("Test", im0)
cv2.waitKey(0)
box = npz_data['bbox']/np.array([width, height, width,height])  # Normalize bbox coordinates (xmin, ymin, xmax, ymax)
print(box)