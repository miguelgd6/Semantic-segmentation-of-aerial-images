import os 
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from patchify import patchify
from pathlib import Path
from PIL import Image
import cv2

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchmetrics 

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import ssl # Needed for avoiding expired SSL certify related issues 
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SRC_PATH = 'src_data'
DST_FOLDERS = ['train', 'test', 'val']
EPOCHS = 50
BS = 4

class SegmentationDataset(Dataset):
    
    """Create Semantic Segmentation Dataset. Read images, apply augmentations, and process transformations

    Args:
        Dataset (image): Aerial Drone Images
    """
    # CLASSES = {'building': 44, 'land': 91, 'road':172, 'vegetation':212, 'water':171, 'unlabeled':155}
    # CLASSES_KEYS = list(CLASSES.keys())
    
    def __init__(self, path_name) -> None:
        super().__init__()
        self.image_names = os.listdir(f"{path_name}/images")
        self.image_paths = [f"{path_name}/images/{i}" for i in self.image_names]
        self.masks_names = os.listdir(f"{path_name}/masks")
        self.masks_paths = [f"{path_name}/masks/{i}" for i in self.masks_names]
        
        # filter all images that do not exist in both folders
        self.img_stem = [Path(i).stem for i in self.image_paths]
        self.msk_stem = [Path(i).stem for i in self.masks_paths]
        self.img_msk_stem = set(self.img_stem) & set(self.msk_stem)
        self.image_paths = [i for i in self.image_paths if (Path(i).stem in self.img_msk_stem)]


    def convert_mask(self, mask):
        mask[mask == 155] = 0  # unlabeled
        mask[mask == 44] = 1  # building
        mask[mask == 91] = 2  # land
        mask[mask == 171] = 3  # water
        mask[mask == 172] = 4  # road
        mask[mask == 212] = 5  # vegetation
        return mask   

    def __len__(self):
        return len(self.img_msk_stem)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))  #structure: BS, C, H, W
        mask =  cv2.imread(self.masks_paths[index], 0)
        mask = self.convert_mask(mask)
        return image, mask
    
def DataPreparation( src = SRC_PATH ): 
    
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

def ModelTraining():

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

    torch.save(model.state_dict(), f"models/FPN_epochs_{EPOCHS}_CEloss_statedict.pth")
    
    # sns.lineplot(x = range(len(train_losses)), y = train_losses).set('Train Loss')
    # sns.lineplot(x = range(len(val_losses)), y = val_losses).set('Val Loss')

def ModelEvaluation(img_index): 

    # Dataset and Dataloader
    test_ds = SegmentationDataset(path_name='test')
    test_dataloader = DataLoader(test_ds, batch_size=45, shuffle=False)

    # Charging device in GPU 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Model setup
    model = smp.FPN(
        encoder_name='se_resnext50_32x4d', 
        encoder_weights='imagenet', 
        classes=6, 
        activation='sigmoid',
    )
    model.to(DEVICE)

    # load weights
    model.load_state_dict(torch.load('models/FPN_epochs_50_CEloss_statedict.pth'))

    # Model Evaluation
    pixel_accuracies = [] 
    intersection_over_unions = []
    metric_iou = torchmetrics.JaccardIndex(num_classes=6, task='multiclass').to(DEVICE)

    with torch.no_grad():
        for data in test_dataloader:

            imgs_test, masks_test = data
            imgs_test = imgs_test.to(DEVICE).float()
            masks_test = masks_test.to(DEVICE).to(torch.float32) 

            pred = model(imgs_test) 

            _, predicted = torch.max(pred, 1) 
            correct_pixels = (masks_test == predicted).sum().item()
            total_pixels = masks_test.size(1) * masks_test.size(2)
            pixel_accuracies.append(correct_pixels / total_pixels)
            iou = metric_iou(predicted.float(), masks_test).item()
            intersection_over_unions.append(iou)

    # Median Accuracy
    print(f"Median Pixel Accuracy: {np.median(pixel_accuracies) * 100 }")
    print(f"Median IoU: {np.median(intersection_over_unions) * 100 }")

    # showing specific image
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    for index, data in enumerate(test_dataloader):
        if index == img_index:
             image_test, mask = data
        
    with torch.no_grad():
        image_test = image_test.float().to(DEVICE)
        output = model(image_test)

    img_og = np.transpose(image_test[0, :, :, :].cpu().numpy(), (1, 2, 0))
    mask_og = mask[0, :, :]

    predicted_mask = output.cpu().squeeze().numpy()
    predicted_mask = predicted_mask.transpose((1, 2, 0))
    predicted_mask = predicted_mask.argmax(axis=2)
     
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('True and Predicted Mask')
    
    axs[0].imshow(mask_og)
    axs[1].imshow(predicted_mask)

    axs[0].set_title(f"reference mask:")
    axs[1].set_title(f"predicted mask:")
    
    plt.show()

    #
    # output_cpu = output_cpu.transpose((1, 2, 3, 0)) # transpose just reorganize dimensions 
    # output_cpu = output_cpu.argmax(axis=3) # adjust size to biggest element 

    # trick to cover all classes
    # use at least one pixel for each class for both images
    required_range = list(range(6))
    # output_cpu[:, 0] = 0
    # output_cpu[:, 1] = 1
    # output_cpu[:, 2] = 2
    # output_cpu[:, 3] = 3
    # output_cpu[:, 4] = 4
    # output_cpu[:, 5] = 5

    # mask[:, 0, 0] = 0
    # mask[:, 0, 1] = 1
    # mask[:, 0, 2] = 2
    # mask[:, 0, 3] = 3
    # mask[:, 0, 4] = 4
    
    # mask[:, 0, 5] = 5

    

def main(): 

    # Function for creating the files datasets if needed 
    # DataPreparation()
    
    # Function for model training if needed 
    # ModelTraining()

    # Function for model evaluation
    ModelEvaluation(33)

    

if __name__ == '__main__':
    main()
