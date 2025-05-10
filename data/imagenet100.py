import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageNet100Dataset(Dataset):
    """
    PyTorch dataset for ImageNet-100
    """
    def __init__(self, hf_dataset, split="train", transform=None):
        self.dataset = hf_dataset[split]
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
            
        # Ensure the image is RGB (3 channels)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

class MoCoDataset(Dataset):
    """
    Dataset wrapper that returns two augmented views of the same image for contrastive learning
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]
        
        # Ensure the image is PIL and RGB
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply two separate augmentations
        img_q = self.transform(img)
        img_k = self.transform(img)
        
        return (img_q, img_k), target

def load_imagenet100():
    """
    Load ImageNet-100 dataset from Hugging Face
    """
    print("Loading ImageNet-100 dataset...")
    return load_dataset("clane9/imagenet-100")

def get_moco_augmentations():
    """
    Returns the augmentations used for MoCo training
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # MoCo augmentation
    moco_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    return moco_transform

def get_evaluation_transform():
    """
    Returns the transformation for evaluation
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    return eval_transform

def setup_imagenet100_training(batch_size=256, num_workers=4):
    """
    Set up data loaders for ImageNet-100 training
    """
    # Load dataset
    hf_dataset = load_imagenet100()
    
    # Get MoCo augmentation
    moco_transform = get_moco_augmentations()
    
    # Create base dataset
    base_train_dataset = ImageNet100Dataset(hf_dataset, split="train", transform=None)
    
    # Create MoCo dataset with two augmentations
    train_dataset = MoCoDataset(base_train_dataset, moco_transform)
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader

def create_eval_dataloaders(hf_dataset=None, batch_size=256, num_workers=4):
    """
    Create train and validation dataloaders for evaluation
    """
    if hf_dataset is None:
        hf_dataset = load_imagenet100()
        
    eval_transform = get_evaluation_transform()
    
    # Create datasets
    train_dataset = ImageNet100Dataset(hf_dataset, split="train", transform=eval_transform)
    val_dataset = ImageNet100Dataset(hf_dataset, split="validation", transform=eval_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for feature extraction
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader