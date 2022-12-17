#Reference Page - https://github.com/endrol/DR_GCN/blob/9ad1929910ed30c3a623c25ba0da0198bd1655f5/dr_gcn/dr.py
import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *

labels = [0, 1, 2, 3, 4]


def pair_image_label(files, labels):
    images = []
    for i in range(len(files)):
        name = files.values[i]
        label = np.asarray(labels.values[i])
        label = label.astype(float)
        label = torch.from_numpy(label)
        label = torch.unsqueeze(label, 0)
        item = (name, label)
        images.append(item)
    return images

def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                label_med = (np.asarray(row[1:num_categories + 1]))[:1]

                labels = label_med.astype(float)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


class DRclassification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None,train_data=None,label_data=None):
        self.root = root
        self.set = set
        self.path_images = root
        self.transform = transform
        self.target_transform = target_transform

        file_csv = os.path.join(root, set+'.csv')

        self.classes = labels
        if train_data is not None:
            self.images = pair_image_label(train_data,label_data)
        else:
            self.images = read_object_labels_csv(file_csv)
       
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

        print('[dataset] EyePacs classification set = %s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

   
    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path+'.jpeg')).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path, self.inp), target
        

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
