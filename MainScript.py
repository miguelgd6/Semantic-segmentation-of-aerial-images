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
import albumentations as A
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import ssl # Needed for avoiding expired SSL certify related issues 
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SRC_PATH = "src_data"
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

def ModelTraining(train_path, val_path):

    # if images are already reorganized, instantiate datasets 
    train_ds = SegmentationDataset(path_name=train_path)
    train_dataloader = DataLoader(train_ds, batch_size=BS, shuffle=True)

    val_ds = SegmentationDataset(path_name=val_path)
    val_dataloader = DataLoader(val_ds, batch_size=BS, shuffle=True)
    
    # instantiate model and define hyperparameter 
    model = smp.FPN(
        encoder_name='inceptionresnetv2', 
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

    torch.save(model.state_dict(), f"models/FPN_epochs_{EPOCHS}_CEloss_statedict_{train_path}_inceptionresnetv2.pth")
    
    # sns.lineplot(x = range(len(train_losses)), y = train_losses).set('Train Loss')
    # sns.lineplot(x = range(len(val_losses)), y = val_losses).set('Val Loss')

def ModelEvaluation(ev_imgs, encoder, savedModelName): 

    # Dataset and Dataloader
    test_ds = SegmentationDataset(path_name='test')
    test_dataloader = DataLoader(test_ds, batch_size=45, shuffle=False)

    # Charging device in GPU 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Model setup
    model = smp.FPN(
        encoder_name=encoder, 
        encoder_weights='imagenet', 
        classes=6, 
        activation='sigmoid',
    )
    model.to(DEVICE)

    # load weights
    model.load_state_dict(torch.load(f'models/{savedModelName}'))

    # Model Evaluation
    pixel_accuracies = [] 
    accuracies = []
    intersection_over_unions = []
    dice_coefs = []

    metric_iou = torchmetrics.JaccardIndex(num_classes=6, task='multiclass').to(DEVICE)
    metric_dice = torchmetrics.Dice(num_classes=6, average='micro').to(DEVICE)
    metric_accuracy = torchmetrics.Accuracy(num_classes=6, task='multiclass').to(DEVICE)

    with torch.no_grad():
        for data in test_dataloader:

            imgs_test, masks_test = data
            imgs_test = imgs_test.to(DEVICE).float()
            masks_test = masks_test.to(DEVICE).to(torch.float32) 

            pred = model(imgs_test) 

            _, predicted = torch.max(pred, 1) 
            correct_pixels = (masks_test == predicted).sum().item()
            total_pixels = masks_test.size(1) * masks_test.size(2)
            iou = metric_iou(predicted.float(), masks_test).item()
            dice = metric_dice(predicted, masks_test.int()).item()
            accuracy = metric_accuracy(predicted, masks_test.int()).item()

            pixel_accuracies.append(correct_pixels / total_pixels)
            accuracies.append(accuracy)
            intersection_over_unions.append(iou)
            dice_coefs.append(dice)

    # Median Accuracy
    print(f"Median Pixel Accuracy: {np.median(pixel_accuracies) * 100 }")
    print(f"Median IoU: {np.median(intersection_over_unions) * 100 }")
    print(f"Median DICE: {np.median(dice_coefs) * 100 }")
    print(f"Median TM Accuracy: {np.median(accuracies) * 100 }")

    # Epochs 

    # showing specific images
    imgs_list = []
    aux_list = []

    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    for n in ev_imgs: 
        for index, data in enumerate(test_dataloader):
            if index == n:
                image_test, mask = data
        
        with torch.no_grad():
            image_test = image_test.float().to(DEVICE)
            output = model(image_test)

        img_og = np.transpose(image_test[0, :, :, :].cpu().numpy(), (1, 2, 0))
        mask_og = mask[0, :, :]

        predicted_mask = output.cpu().squeeze().numpy()
        predicted_mask = predicted_mask.transpose((1, 2, 0))
        predicted_mask = predicted_mask.argmax(axis=2)
        
        aux_list = [img_og, mask_og, predicted_mask]
        imgs_list.append(aux_list) 

    count = 0 
    for imgs in imgs_list:

        count += 1

        fig = plt.figure(figsize=(20,8))

        # ax1 = fig.add_subplot(1,3,1)
        # ax1.imshow(imgs[0])
        # ax1.set_title('Input Image', fontdict={'fontsize': 16, 'fontweight': 'medium'})
        # ax1.grid(False)

        ax2 = fig.add_subplot(1,3,1)
        ax2.set_title('Ground Truth Mask', fontdict={'fontsize': 16, 'fontweight': 'medium'})
        ax2.imshow(imgs[1]) # ax2.imshow(onehot_to_rgb(imgs[1],id2code))
        ax2.grid(False)

        ax3 = fig.add_subplot(1,3,2)
        ax3.set_title('Predicted Mask', fontdict={'fontsize': 16, 'fontweight': 'medium'})
        ax3.imshow(imgs[2]) # ax3.imshow(onehot_to_rgb(imgs[2],id2code))
        ax3.grid(False)

        saving_folder = f'images_predicted/{savedModelName}'

        if not os.path.exists(saving_folder): 
            os.makedirs(saving_folder)  # print('imgs folder already exists') 

        plt.savefig(f"{saving_folder}/prediction_{format(count)}.png", facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 200)
        #plt.show()

        

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
    
def augment(width, height):

    transform = A.Compose([
        A.RandomCrop(width=width, height=height, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=[60, 300], p=1.0, interpolation=cv2.INTER_NEAREST),
        A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.3], contrast_limit=0.2, p=1.0),
        A.OneOf([
            A.CLAHE (clip_limit=1.5, tile_grid_size=(8, 8), p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5),
        ], p=1.0),
    ], p=1.0)
    
    return transform

def augment_dataset(path, count):

    '''Function for data augmentation
        Input:
            count - total no. of images after augmentation = initial no. of images * count
        Output:
            writes augmented images (input images & segmentation masks) to the working directory
    '''
    
    # Taking training images and performing augmentation
    images_dir = f'./{path}/images/'
    masks_dir = f'./{path}/masks/'

    file_names = np.sort(os.listdir(masks_dir))
    file_names = np.char.split(file_names, '.')
    filenames = np.array([])

    for i in range(len(file_names)):
        filenames = np.append(filenames, file_names[i][0])

    # this is the kaggle augment function, probably the sizes of the images, as when pathing the sizes
    #transform_1 = augment(512, 512)
    #transform_2 = augment(480, 480)
    #transform_3 = augment(512, 512)
    #transform_4 = augment(800, 800)
    #transform_5 = augment(1024, 1024)
    #transform_6 = augment(800, 800)
    #transform_7 = augment(1600, 1600)
    #transform_8 = augment(1920, 1280)
    
    # data sized changed during training 
    transform_1 = augment(320, 320)
    transform_2 = augment(320, 320)
    transform_3 = augment(320, 320)
    transform_4 = augment(320, 320)
    transform_5 = augment(320, 320)
    transform_6 = augment(320, 320)
    transform_7 = augment(320, 320)
    transform_8 = augment(320, 320)

    i = 0
    for i in range(count):
        for file in filenames:
            tile = file.split('_')[4]
            img = cv2.imread(images_dir+file+'.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(masks_dir+file+'.png')
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            tile_ok = False 

            if tile == '1':
                transformed = transform_1(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                tile_ok = True
            elif tile =='2':
                transformed = transform_2(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                tile_ok = True
            elif tile =='3':
                transformed = transform_3(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                tile_ok = True
            elif tile =='4':
                transformed = transform_4(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                tile_ok = True
            elif tile =='5':
                transformed = transform_5(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                tile_ok = True
            elif tile =='6':
                transformed = transform_6(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                tile_ok = True
            elif tile =='7':
                transformed = transform_7(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                tile_ok = True
            elif tile =='8':
                transformed = transform_8(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                tile_ok = True

            if tile_ok == True: 
                folder_imgs = f'./augmented_{path}/images'
                folder_masks = f'./augmented_{path}/masks'

                if not os.path.exists(folder_imgs): 
                    os.makedirs(folder_imgs)  # print('imgs folder already exists')

                if not os.path.exists(folder_masks):
                    os.makedirs(folder_masks)  # print('masks folder already exists')

                cv2.imwrite(folder_imgs + '/aug_{}_'.format(str(i+1))+file+'.png',cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
                cv2.imwrite(folder_masks + '/aug_{}_'.format(str(i+1))+file+'.png',cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2RGB))

def main(): 

    # Function for creating patches and generating test-train-val Splits 
    # DataPreparation()
    
    #Funtion for train data augmentation
    # count total no. of images after augmentation = initial no. of images * count 
    # augment_dataset("val", count = 8)
    # augment_dataset("train", count = 8)

    # Function for model training if needed 
    # ModelTraining("augmented_train", "augmented_val") # only train for not augmented dataset

    # Function for model evaluation
    num_ev = np.arange(0, 35) #[31, 32, 33]
    ModelEvaluation(num_ev, "inceptionresnetv2","FPN_epochs_50_CEloss_statedict_augmented_train_inceptionresnetv2.pth")
    #ModelEvaluation(num_ev, "se_resnext50_32x4d","FPN_epochs_50_CEloss_statedict_augmented_train.pth")
    # ModelEvaluation(num_ev, "se_resnext50_32x4d","FPN_epochs_50_CEloss_statedict.pth")

    

if __name__ == '__main__':
    main()
