import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageOps
import random

# Define a custom dataset class for loading pose estimation data
class ClassificationDataset(Dataset): #Create a class ClassificationDataset that inherits from Dataset
    def __init__(self, rootpath, augment=False, random_ratio=0.5):
        
        
        self.rootpath = rootpath
        self.sample = []
        self.p_dict = {}
        self.r_dict = {}
        self.random_ratio = random_ratio
        
        all_files =os.listdir(rootpath)
        sorted_files = sorted(all_files)

        for folder in sorted_files:
            folder_path = os.path.join(rootpath, folder)
            if not os.path.isdir(folder_path):
                continue
            
            p_label, r_label = folder.split('_')
            if p_label not in self.p_dict:
                self.p_dict[p_label] = len(self.p_dict)
            if r_label not in self.r_dict:
                self.r_dict[r_label] = len(self.r_dict)

            p_idx = self.p_dict[p_label]
            r_idx = self.r_dict[r_label]

            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_file)
                    self.sample.append((img_path, p_idx, r_idx))

       
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
                size=224,                 # Directly produce 224×224 crop
                scale=(0.9, 1.0),         # Keep 90–100% of the robot
                ratio=(0.9, 1.1)
            ),

            # Realistic microscope brightness & contrast variation
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ])

    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self, idx):
        img_path, p_idx, r_idx = self.sample[idx]

        # Load image and apply EXIF orientation correction RELEVANT for non-zero roll-angle images
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)  # Apply EXIF orientation if present 
        image = image.convert('RGB')

        # Apply augmentation ONLY for training 
        if self.augment and random.random() < self.random_ratio:
            image = self.augment_transform(image)

        #  Always apply base transforms 
        image = self.base_transform(image)

        labels = {
            "pitch": torch.tensor(p_idx, dtype=torch.long),
            "roll": torch.tensor(r_idx, dtype=torch.long)
        }
        return image, labels
    

if __name__ == "__main__":
    dataset = ClassificationDataset(rootpath='data/pose_data/train', augment=True)
    print(f"Dataset size: {len(dataset)} samples")
    img, labels = dataset[0]
    print(f"Image shape: {img.shape}, Labels: {labels}")
       