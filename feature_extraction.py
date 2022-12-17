#Reference Page - https://github.com/endrol/DR_GCN/blob/9ad1929910ed30c3a623c25ba0da0198bd1655f5/dr_gcn/surf_feature.py

import numpy as np
import os
import cv2
from tqdm import tqdm
import cmath

#Load images
fundus_img=next(os.walk('/home/sbx5057/Documents/COMP597/eyepacs_preprocess/eyepacs_preprocess'))[2]
fundus_img.sort()

#Resize each image in the fundus dataset
for img in tqdm(fundus_img):
    if (img.split('.')[-1] == 'jpeg'):
        img_s = cv2.imread('/home/sbx5057/Documents/COMP597/eyepacs_preprocess/eyepacs_preprocess/'+img,cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img_s, (672, 448), interpolation=cv2.INTER_CUBIC)
        
      # sift descriptor
        sift = cv2.xfeatures2d.SIFT_create(1000)
        kp, des = sift.detectAndCompute(resized_img, None)          

        #Continue if no descriptors
        if des is None:
          continue

        #First 20 elements from the descriptor array
        des = des[:20]
        np.save('/home/sbx5057/Documents/COMP597/eyepacs_preprocess/eyepacs_preprocess/sift_des/{}'.format(img.split('.')[0]), des)
        
