from data_utils import data_pull,unzip,file_move
import os

data_pull('https://www.kaggle.com/competitions/carvana-image-masking-challenge/data')
data_pull('https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri')

file_move('lgg-mri-segmentation','data/segmentation/lgg-mri-segmentation')
file_move('carvana-image-masking-challenge','data/mask/carvana-image-masking-challenge')

unzip('data/mask/carvana-image-masking-challenge/train.zip','data/mask')
unzip('data/mask/carvana-image-masking-challenge/train_masks.zip','data/mask')


