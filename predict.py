import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as albu
from albumentations import pytorch as AT
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import time

from dataset import CloudDataset
import utils
from loss import BCEDiceLoss, DiceLoss
from trainer import TrainEpoch, ValidEpoch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", DEVICE)

print("preparing data...")

path = './dataset/'
train = pd.read_csv(f'{path}/train.csv')
sub = pd.read_csv(f'{path}/sample_submission.csv')

n_train = len(os.listdir(f'{path}/train_images'))
n_test = len(os.listdir(f'{path}/test_images'))
print(f'There are {n_train} images in train dataset')
print(f'There are {n_test} images in test dataset')

train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

print("creating preprocessing module...")

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

print("creating data loader...")

num_workers = 0
bs = 16

train_dataset = CloudDataset(path = path, df=train, datatype='train', img_ids=train_ids, transforms = utils.get_training_augmentation(), preprocessing = utils.get_preprocessing(preprocessing_fn))
valid_dataset = CloudDataset(path = path, df=train, datatype='valid', img_ids=valid_ids, transforms = utils.get_validation_augmentation(), preprocessing = utils.get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

print("loading state dict...")

model_path = 'best_dice_model.pth'
model = torch.load(model_path)

print("setting loss and criterion...")

loss = BCEDiceLoss() # or DiceLoss()
metrics = [
    smp.utils.metrics.Fscore(threshold=0.9), # Fscore means Dice Coefficient
    smp.utils.metrics.IoU(threshold=0.9), # IoU means Jaccord Coefficient
]

print("setting trainer...")

valid_epoch = ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

print("start predicting!")
    
train_logs = valid_epoch.run(train_loader)
valid_logs = valid_epoch.run(valid_loader)

print("predicting ends.")
print('Train IoU Score: ', train_logs['iou_score'])
print('Train Dice Score:', train_logs['fscore'])
print('Valid IoU Score: ', valid_logs['iou_score'])
<<<<<<< HEAD
print('Valid Dice Score:', valid_logs['fscore'])
=======
print('Valid Dice Score:', valid_logs['fscore'])
>>>>>>> main
