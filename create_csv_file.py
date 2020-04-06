import csv
import os
from imutils import paths
import random

img_folder_ck = './dataset/'
img_name_ck_list = list(paths.list_images(img_folder_ck))
random.shuffle(img_name_ck_list)
num_img = len(img_name_ck_list)
num_train = int(num_img*70/100)

with open('data_train.csv', 'a') as csvfile:

    writer = csv.writer(csvfile)
    for i in range(0, num_train):

        img_path = img_name_ck_list[i]
         # write the training section
        if 'Anger' in img_path:
            cls_gt = 0
            writer.writerow([img_path, cls_gt])
        if 'Disgust' in img_path:
            cls_gt = 1
            writer.writerow([img_path, cls_gt])
        if 'Fear' in img_path:
            cls_gt = 2
            writer.writerow([img_path, cls_gt])
        if 'Happy' in img_path:
            cls_gt = 3
            writer.writerow([img_path, cls_gt])
        if 'Sad' in img_path:
            cls_gt = 4
            writer.writerow([img_path, cls_gt])
        if 'Surprise' in img_path:
            cls_gt = 5
            writer.writerow([img_path, cls_gt])
        if 'Contempt' in img_path:
            cls_gt = 6
            writer.writerow([img_path, cls_gt])

with open('data_test.csv', 'a') as csvfile:

    writer = csv.writer(csvfile)
    for i in range(num_train, num_img):

        img_path = img_name_ck_list[i]
         # write the testing section
        if 'Anger' in img_path:
            cls_gt = 0
            writer.writerow([img_path, cls_gt])
        if 'Disgust' in img_path:
            cls_gt = 1
            writer.writerow([img_path, cls_gt])
        if 'Fear' in img_path:
            cls_gt = 2
            writer.writerow([img_path, cls_gt])
        if 'Happy' in img_path:
            cls_gt = 3
            writer.writerow([img_path, cls_gt])
        if 'Sad' in img_path:
            cls_gt = 4
            writer.writerow([img_path, cls_gt])
        if 'Surprise' in img_path:
            cls_gt = 5
            writer.writerow([img_path, cls_gt])
        if 'Contempt' in img_path:
            cls_gt = 6
            writer.writerow([img_path, cls_gt])
