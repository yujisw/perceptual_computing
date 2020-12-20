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

exp_name = time.strftime('%Y%m%d-%H%M%S')
wandb.init(project="perceptual_computing", entity='yujisw', name=exp_name)

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

num_workers = 4
bs = 16

train_dataset = CloudDataset(path = path, df=train, datatype='train', img_ids=train_ids, transforms = utils.get_training_augmentation(), preprocessing = utils.get_preprocessing(preprocessing_fn))
valid_dataset = CloudDataset(path = path, df=train, datatype='valid', img_ids=valid_ids, transforms = utils.get_validation_augmentation(), preprocessing = utils.get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

print("setting for training...")

ACTIVATION = None
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=4, 
    activation=ACTIVATION,
)
wandb.watch(model)

num_epochs = 50
logdir = "./logs/segmentation"

# model, criterion, optimizer
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-2}, 
    {'params': model.encoder.parameters(), 'lr': 1e-3},  
])
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
loss = BCEDiceLoss() # or DiceLoss()
metrics = [
    smp.utils.metrics.Fscore(threshold=0.5), # Fscore means Dice Coefficient
    smp.utils.metrics.IoU(threshold=0.5), # IoU means Jaccord Coefficient
]

train_epoch = TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

print("start training!")

max_score = 0

for i in range(0, num_epochs):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_iou_model.pth')
        print('Model saved!')

    if max_score < valid_logs['fscore']:
        max_score = valid_logs['fscore']
        torch.save(model, './best_dice_model.pth')
        print('Model saved!')
        
    # if i % 1 == 0:
    #     optimizer.param_groups[0]['lr'] = 1e-5
    #     print('Decrease decoder learning rate to 1e-5!')

    wandb.log({
        "Train Loss": train_logs[loss.__name__],
        "Train IOU": train_logs['iou_score'],
        "Train Dice": train_logs['fscore'],
        "Valid Loss": valid_logs[loss.__name__],
        "Valid IOU": valid_logs['iou_score'],
        "Valid Dice": valid_logs['fscore'],
        "max_score": max_score,
        })

print("training ends.")
print('max_score:', max_score)
