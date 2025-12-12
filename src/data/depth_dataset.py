import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageOps
import ast
import random

# Define a custom dataset class for loading pose estimation data
class RegressDataset(Dataset): #Create a class RegressionDataset that inherits from Dataset
    def __init__(self, rootpath, augment=False, random_ratio=0.5):
        self.rootpath = rootpath
        self.sample = []
        self.random_ratio = random_ratio
        
        all_files = os.listdir(rootpath)
        sorted_files = sorted(all_files)

        for folder in sorted_files:
            folder_path = os.path.join(rootpath, folder)
            if not os.path.isdir(folder_path):
                continue

            txt_path = os.path.join(folder_path, f"{folder}_depth.txt")

            with open(txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
            
                    img_name, depth_value = ast.literal_eval(line)
                    img_path = os.path.join(folder_path, img_name)
                    self.sample.append((img_path, float(depth_value)))

        # Transformations: 1. Base transforms (always applied) 2. Augmentation (if specified)

        self.augment = augment

       
        # 1. Base TRANSFORMS
       
        self.base_transform = T.Compose([
        
            # If augmentation was already resizing, we keep this for val/test
            T.Resize((224, 224)),

            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # 2. Augmentation TRANSFORMS

        
        self.augment_transform = T.Compose([
            # Slight translation (no rotation allowed)
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),   # 5% shift
            ),

            # Random crop → Resized later
            T.RandomResizedCrop(
                size=224,
                scale=(0.9, 1.0),         # Keep 90–100% of the robot
                ratio=(0.9, 1.1)
            ),

            # Realistic microscope brightness & contrast variation
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ])    

    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self, idx):
        img_path, depth_value = self.sample[idx]

        # Load image and apply EXIF orientation correction
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)  # Apply EXIF orientation if present
        image = image.convert('RGB')

        # Apply augmentation ONLY for training 
        if self.augment and random.random() <self.random_ratio:
            image = self.augment_transform(image)

        #  Always apply base transforms 
        image = self.base_transform(image)

        depth_tensor = torch.tensor(depth_value, dtype=torch.float32)

        return image, depth_tensor