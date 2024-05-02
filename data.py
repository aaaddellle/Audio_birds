import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def show_images(images, n_max=6):
    fig, axs = plt.subplots(1, min(n_max, len(images)), figsize=(15, 15))
    for i, img in enumerate(images):
        if i >= n_max: break
        axs[i].imshow(img.permute(1, 2, 0))  
        axs[i].axis('off')
    plt.show()

class AudioToImageDataset(Dataset):
    def __init__(self, features_path, images_dir, transform=None):
        with open(features_path, 'r') as file:
            self.features = json.load(file)  
        self.images_dir = images_dir
        self.transform = transform
        self.image_files = []
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.image_files.append(os.path.join(root, file))
        self.num_images = len(self.image_files) 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
         
        feature = [float(i) for i in self.flatten(feature)]

        feature_array = np.array(feature, dtype=np.float32)  

      
        image_idx = idx % self.num_images
        img_name = self.image_files[image_idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        feature_tensor = torch.tensor(feature_array, dtype=torch.float32)

        sample = {'feature': feature_tensor, 'image': image}
        return sample

    def flatten(self, x):
        result = []
        for el in x:
            if isinstance(el, list):
                result.extend(self.flatten(el))
            else:
                result.append(el)
        return result
    
    
transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

dataset = AudioToImageDataset(features_path='C:\\Users\\Lenovo\\Desktop\\python\\AtoI\\features_all.json', images_dir='C:\\Users\\Lenovo\\Desktop\\python\\AtoI\\images', transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Iterate over a few batches to test
for i, batch in enumerate(dataloader):
    images = batch['image']
    features = batch['feature']
    print(f"Batch {i+1}")
    print(f"Feature shape: {features.shape}")
    print(f"Image shape: {images.shape}")

    if i == 0:  
        show_images(images)

    if i >= 2:  
        break
