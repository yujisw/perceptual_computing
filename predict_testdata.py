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
import tqdm

from dataset import CloudDataset
import utils
from loss import BCEDiceLoss, DiceLoss
from trainer import TrainEpoch, ValidEpoch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", DEVICE)

print("creating preprocessing module...")
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

print("creating data loader...")
path = './dataset/'
num_workers = 0
bs = 1

sub = pd.read_csv(os.path.join(path,'sample_submission.csv'))
sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

test_dataset = CloudDataset(
    path = path,
    df=sub,
    datatype='test',
    img_ids=test_ids,
    transforms = utils.get_validation_augmentation(),
    preprocessing = utils.get_preprocessing(preprocessing_fn)
)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

print("loading state dict...")
model_path = 'best_dice_model.pth'
model = torch.load(model_path)

print("predicting...")

sigmoid = lambda x: 1 / (1 + np.exp(-x))
class_params = {0: (0.65, 10000), 1: (0.7, 10000), 2: (0.7, 10000), 3: (0.6, 10000)}

encoded_pixels = []
image_id = 0
for i, test_batch in enumerate(tqdm.tqdm(test_loader)):
    test_batch = {"features": test_batch[0].to(DEVICE)}
    output = model(test_batch["features"])
    for i, batch in enumerate(output):
        for probability in batch:
            
            probability = probability.cpu().detach().numpy()
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = utils.post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = utils.mask2rle(predict)
                encoded_pixels.append(r)
            image_id += 1

sub['EncodedPixels'] = encoded_pixels
sub.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)