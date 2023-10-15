
import os 
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from patchify import patchify
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from Dataset import SegmentationDataset
from torch.utils.data import DataLoader

import ssl # Needed for avoiding expired SSL certify related issues 
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SRC_PATH = 'src_data'
DST_FOLDERS = ['train', 'test', 'val']
EPOCHS = 50
BS = 4

def TrainTestSplit( src = SRC_PATH ): 
    
    for path_name, _, file_name in os.walk(src): 
        for f in file_name:
            if f != 'classes.json':

                path_split = os.path.split(path_name)
                tile_num = re.findall(r'\d+', path_split[0])[0]
                
                img_type =path_split[1]  # either 'masks' or 'images'
                
                # leave out tile 2, issues with color dim
                if tile_num == '3':
                    target_folder_imgs = 'val'
                    target_folder_masks = 'val'
                elif tile_num == '1':
                    target_folder_imgs = 'test'
                    target_folder_masks = 'test'
                elif tile_num in ['4', '5', '6', '7', '8']:
                    target_folder_imgs = 'train'
                    target_folder_masks = 'train'

                # copy all images
                src = os.path.join(path_name, f)
                file_name_wo_ext = Path(src).stem
                # check if file exists in images and masks
                img_file = f"{path_split[0]}\images\{file_name_wo_ext}.jpg"
                mask_file = f"{path_split[0]}\masks\{file_name_wo_ext}.png"
                
                if os.path.exists(img_file) and os.path.exists(mask_file):
                    if img_type == 'images':
                        dest = os.path.join(target_folder_imgs, img_type)
                        CreatePatches(src, dest)
                        
                    # copy all masks
                    if img_type == 'masks':
                        dest = os.path.join(target_folder_masks, img_type)
                        CreatePatches(src, dest)

def CreatePatches(src, dest):
    path_split = os.path.split(src)
    tile_num = re.findall(r'\d+', path_split[0])[0]
    image = Image.open(src)
    image = np.asarray(image)

    for folder in DST_FOLDERS: 
        if not os.path.exists(folder):
            folder_imgs = f"{folder}\images"
            folder_msks = f"{folder}\masks"
            os.makedirs(folder_imgs) if not os.path.exists(folder_imgs) else print('folder already exists')
            os.makedirs(folder_msks) if not os.path.exists(folder_msks) else print('folder already exists')

    if len(image.shape) > 2:  # only if color channel exists as well
        patches = patchify(image, (320, 320, 3), step=300)
        file_name_wo_ext = Path(src).stem
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, 0]
                patch = Image.fromarray(patch)
                num = i * patches.shape[1] + j
                patch.save(f"{dest}/{file_name_wo_ext}_tile_{tile_num}_patch_{num}.png")


if __name__ == '__main__':

    # Function for creating the files datasets if needed 
    # TrainTestSplit()

    # if images are already reorganized, instantiate datasets 
    train_ds = SegmentationDataset(path_name='train')
    train_dataloader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_ds = SegmentationDataset(path_name='val')
    val_dataloader = DataLoader(val_ds, batch_size=BS, shuffle=True)
    
    # instantiate model and define hyperparameter 
    model = smp.FPN(
        encoder_name='se_resnext50_32x4d', 
        encoder_weights='imagenet', 
        classes=6, 
        activation='sigmoid'
    )

    model.to(DEVICE)

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    # Training the model
    criterion = nn.CrossEntropyLoss()
    # criterion = smp.losses.DiceLoss(mode='multiclass')
    train_losses, val_losses = [], []

    for e in range(EPOCHS):
        model.train()
        running_train_loss, running_val_loss = 0, 0
        for i, data in enumerate(train_dataloader): 
            #training phase 
            image_i, mask_i = data
            image = image_i.to(DEVICE)
            mask = mask_i.to(DEVICE)

            # reset gradients
            optimizer.zero_grad()
            # forward
            output = model(image.float())

            #calc losses 
            train_loss = criterion(output.float(), mask.long())

            # back propagation 
            train_loss.backward()
            optimizer.step() #update weights

            running_train_loss += train_loss.item()
        train_losses.append(running_train_loss)

        # validation
        model.eval()
        with torch.no_grad(): # not need to estimate gradints while evaluation
            for i, data in enumerate(val_dataloader):
                image_i, mask_i = data
                image = image_i.to(DEVICE)
                mask = mask_i.to(DEVICE)

                #forward 
                output = model(image.float())

                #calc losses 
                val_loss = criterion(output.float(), mask.long())
                running_val_loss += val_loss
        val_losses.append(running_val_loss)

        print(f"Epoch: {e}: Train Loss: {running_train_loss}, Val Loss: {running_val_loss}")

    sns.lineplot(x = range(len(train_losses)), y = train_losses).set('Train Loss')
    sns.lineplot(x = range(len(val_losses)), y = val_losses).set('Val Loss')
    torch.save(model.state_dict(), f"models/FPN_epochs_{EPOCHS}_CEloss_statedict.pth")