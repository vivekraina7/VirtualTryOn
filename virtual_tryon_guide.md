# Complete Virtual Try-On Training & Deployment Guide

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Model Evaluation](#model-evaluation)
6. [Model Deployment](#model-deployment)
7. [Inference Pipeline](#inference-pipeline)
8. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### 1. System Requirements
```bash
# Hardware Requirements
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4070 or better)
- RAM: 16GB+ system memory
- Storage: 50GB+ free space
- CUDA: 11.8 or 12.x compatible
```

### 2. Python Environment
```bash
# Create conda environment
conda create -n virtual_tryon python=3.9
conda activate virtual_tryon

# Install core dependencies
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python==4.8.0.74
pip install pillow==9.5.0
pip install numpy==1.24.3
pip install matplotlib==3.7.1
pip install scikit-image==0.21.0
pip install tensorboard==2.13.0
pip install tqdm==4.65.0
pip install albumentations==1.3.1
pip install einops==0.6.1
```

### 3. Additional Dependencies
```bash
# For pose estimation (if not pre-computed)
pip install mediapipe==0.10.1

# For advanced metrics
pip install lpips==0.1.4
pip install pytorch-fid==0.3.0

# For deployment
pip install flask==2.3.2
pip install gradio==3.35.2
pip install streamlit==1.25.0
```

---

## Data Preprocessing

### 1. Data Loading and Validation
```python
import os
import cv2
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VirtualTryOnDataset(Dataset):
    def __init__(self, data_root, pairs_file, transform=None, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        
        # Load pairs
        self.pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                person_img, cloth_img = line.strip().split()
                self.pairs.append((person_img, cloth_img))
    
    def __len__(self):
        return len(self.pairs)
    
    def load_pose_keypoints(self, pose_file):
        """Load OpenPose keypoints from JSON"""
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
        
        if 'people' in pose_data and len(pose_data['people']) > 0:
            keypoints = np.array(pose_data['people'][0]['pose_keypoints_2d'])
            keypoints = keypoints.reshape(-1, 3)  # (x, y, confidence)
            return keypoints
        return np.zeros((18, 3))  # Default empty pose
    
    def create_pose_map(self, keypoints, img_size=(256, 192)):
        """Create pose heatmap from keypoints"""
        pose_map = np.zeros((img_size[1], img_size[0], 18))
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.1:  # Confidence threshold
                x, y = int(x), int(y)
                if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
                    # Create gaussian heatmap
                    sigma = 2
                    for dy in range(-3*sigma, 3*sigma+1):
                        for dx in range(-3*sigma, 3*sigma+1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < img_size[0] and 0 <= ny < img_size[1]:
                                dist = dx*dx + dy*dy
                                pose_map[ny, nx, i] = np.exp(-dist / (2 * sigma * sigma))
        
        return pose_map
    
    def __getitem__(self, idx):
        person_name, cloth_name = self.pairs[idx]
        
        # Load images
        person_img = Image.open(os.path.join(self.data_root, self.mode, 'image', person_name)).convert('RGB')
        cloth_img = Image.open(os.path.join(self.data_root, self.mode, 'cloth', cloth_name)).convert('RGB')
        cloth_mask = Image.open(os.path.join(self.data_root, self.mode, 'cloth-mask', cloth_name)).convert('L')
        
        # Load pose data
        pose_file = os.path.join(self.data_root, self.mode, 'openpose_json', 
                                person_name.replace('.jpg', '_keypoints.json'))
        keypoints = self.load_pose_keypoints(pose_file)
        
        # Convert to numpy
        person_img = np.array(person_img)
        cloth_img = np.array(cloth_img)
        cloth_mask = np.array(cloth_mask)
        
        # Create pose map
        pose_map = self.create_pose_map(keypoints, (person_img.shape[1], person_img.shape[0]))
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=person_img,
                cloth=cloth_img,
                cloth_mask=cloth_mask,
                pose_map=pose_map
            )
            person_img = transformed['image']
            cloth_img = transformed['cloth']
            cloth_mask = transformed['cloth_mask']
            pose_map = transformed['pose_map']
        
        return {
            'person': person_img,
            'cloth': cloth_img,
            'cloth_mask': cloth_mask,
            'pose_map': pose_map,
            'person_name': person_name,
            'cloth_name': cloth_name
        }

# Data transforms
def get_transforms(img_size=(256, 192)):
    train_transform = A.Compose([
        A.Resize(img_size[1], img_size[0]),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'cloth': 'image', 'cloth_mask': 'mask', 'pose_map': 'image'})
    
    val_transform = A.Compose([
        A.Resize(img_size[1], img_size[0]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'cloth': 'image', 'cloth_mask': 'mask', 'pose_map': 'image'})
    
    return train_transform, val_transform
```

### 2. Data Validation Script
```python
def validate_dataset(data_root):
    """Validate dataset structure and files"""
    required_dirs = ['train', 'test']
    subdirs = ['cloth', 'cloth-mask', 'image', 'openpose_img', 'openpose_json']
    
    for split in required_dirs:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            print(f"Missing directory: {split_path}")
            return False
        
        for subdir in subdirs:
            subdir_path = os.path.join(split_path, subdir)
            if not os.path.exists(subdir_path):
                print(f"Missing directory: {subdir_path}")
                return False
            
            # Check if directory has files
            files = os.listdir(subdir_path)
            print(f"{subdir_path}: {len(files)} files")
    
    # Check pair files
    for pairs_file in ['train_pairs.txt', 'test_pairs.txt']:
        pairs_path = os.path.join(data_root, pairs_file)
        if not os.path.exists(pairs_path):
            print(f"Missing file: {pairs_path}")
            return False
        
        with open(pairs_path, 'r') as f:
            pairs = f.readlines()
            print(f"{pairs_file}: {len(pairs)} pairs")
    
    print("Dataset validation passed!")
    return True

# Run validation
validate_dataset('path/to/virtual_tryon_dataset')
```

---

## Model Architecture

### 1. VITON-Style Generator
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class GMM(nn.Module):
    """Geometric Matching Module"""
    def __init__(self):
        super(GMM, self).__init__()
        # Feature extractor
        self.extractionA = FeatureExtraction(22, ngf=64, n_layers=3)  # person + pose
        self.extractionB = FeatureExtraction(4, ngf=64, n_layers=3)   # cloth + mask
        
        # Correlation and regression
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression()
    
    def forward(self, person, cloth, cloth_mask, pose_map):
        # Combine inputs
        input_A = torch.cat([person, pose_map], dim=1)  # (B, 22, H, W)
        input_B = torch.cat([cloth, cloth_mask.unsqueeze(1)], dim=1)  # (B, 4, H, W)
        
        # Extract features
        feature_A = self.extractionA(input_A)
        feature_B = self.extractionB(input_B)
        
        # Compute correlation
        correlation = self.correlation(feature_A, feature_B)
        
        # Predict transformation
        theta = self.regression(correlation)
        
        return theta

class TryOnGenerator(nn.Module):
    """Main Try-On Generator"""
    def __init__(self, input_nc=25, output_nc=3, ngf=64, n_layers=6):
        super(TryOnGenerator, self).__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = input_nc
        for i in range(n_layers):
            out_ch = ngf * (2 ** i) if i < 4 else ngf * 8
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            in_ch = out_ch
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(ngf * 8) for _ in range(6)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                in_ch = ngf * 8
                out_ch = ngf * 8
            else:
                in_ch = ngf * 8 * 2 if i <= 3 else ngf * (2 ** (n_layers - i))
                out_ch = ngf * (2 ** (n_layers - i - 1)) if i < n_layers - 1 else output_nc
            
            if i == n_layers - 1:
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                    nn.Tanh()
                ))
            else:
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
    
    def forward(self, x):
        # Encoder
        encoder_features = []
        for layer in self.encoder:
            x = layer(x)
            encoder_features.append(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if i > 0:  # Skip connection
                x = torch.cat([x, encoder_features[-(i+1)]], dim=1)
            x = layer(x)
        
        return x

class VirtualTryOnModel(nn.Module):
    """Complete Virtual Try-On Model"""
    def __init__(self):
        super(VirtualTryOnModel, self).__init__()
        self.gmm = GMM()
        self.generator = TryOnGenerator()
    
    def warp_cloth(self, cloth, cloth_mask, theta):
        """Warp cloth using predicted transformation"""
        grid = F.affine_grid(theta, cloth.size(), align_corners=False)
        warped_cloth = F.grid_sample(cloth, grid, align_corners=False)
        warped_mask = F.grid_sample(cloth_mask.unsqueeze(1), grid, align_corners=False)
        return warped_cloth, warped_mask.squeeze(1)
    
    def forward(self, person, cloth, cloth_mask, pose_map):
        # Stage 1: Geometric Matching
        theta = self.gmm(person, cloth, cloth_mask, pose_map)
        warped_cloth, warped_mask = self.warp_cloth(cloth, cloth_mask, theta)
        
        # Stage 2: Try-On Generation
        generator_input = torch.cat([
            person,           # 3 channels
            warped_cloth,     # 3 channels
            warped_mask.unsqueeze(1),  # 1 channel
            pose_map          # 18 channels
        ], dim=1)  # Total: 25 channels
        
        try_on_result = self.generator(generator_input)
        
        return {
            'try_on': try_on_result,
            'warped_cloth': warped_cloth,
            'warped_mask': warped_mask,
            'theta': theta
        }
```

### 2. Discriminator for Adversarial Training
```python
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()
        
        layers = []
        in_ch = input_nc
        
        # First layer
        layers.append(nn.Conv2d(in_ch, ndf, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        in_ch = ndf
        
        # Intermediate layers
        for i in range(n_layers - 1):
            out_ch = min(ndf * (2 ** (i + 1)), 512)
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
        
        # Final layer
        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
```

---

## Training Pipeline

### 1. Loss Functions
```python
import torch.nn as nn
from torchvision.models import vgg19
import lpips

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        h_relu1_x = self.slice1(x)
        h_relu2_x = self.slice2(h_relu1_x)
        h_relu3_x = self.slice3(h_relu2_x)
        h_relu4_x = self.slice4(h_relu3_x)
        
        h_relu1_y = self.slice1(y)
        h_relu2_y = self.slice2(h_relu1_y)
        h_relu3_y = self.slice3(h_relu2_y)
        h_relu4_y = self.slice4(h_relu3_y)
        
        loss = (F.l1_loss(h_relu1_x, h_relu1_y) + 
                F.l1_loss(h_relu2_x, h_relu2_y) + 
                F.l1_loss(h_relu3_x, h_relu3_y) + 
                F.l1_loss(h_relu4_x, h_relu4_y))
        
        return loss

class TryOnLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_vgg=10.0, lambda_mask=1.0):
        super(TryOnLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss()
        self.lpips_loss = lpips.LPIPS(net='alex')
        
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_mask = lambda_mask
    
    def forward(self, pred, target, pred_mask=None, target_mask=None):
        # L1 loss
        l1 = self.l1_loss(pred, target)
        
        # VGG perceptual loss
        vgg = self.vgg_loss(pred, target)
        
        # LPIPS loss
        lpips_val = self.lpips_loss(pred, target).mean()
        
        total_loss = self.lambda_l1 * l1 + self.lambda_vgg * vgg + lpips_val
        
        # Mask loss if provided
        if pred_mask is not None and target_mask is not None:
            mask_loss = self.l1_loss(pred_mask, target_mask)
            total_loss += self.lambda_mask * mask_loss
        
        return {
            'total': total_loss,
            'l1': l1,
            'vgg': vgg,
            'lpips': lpips_val
        }
```

### 2. Training Script
```python
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, discriminator, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.model.parameters(), 
            lr=config['lr_g'], 
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=config['lr_d'], 
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.criterion_tryon = TryOnLoss()
        self.criterion_adv = nn.BCEWithLogitsLoss()
        
        # Tensorboard
        self.writer = SummaryWriter(config['log_dir'])
        
        # Checkpoints
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        self.model.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            person = batch['person'].to(self.device)
            cloth = batch['cloth'].to(self.device)
            cloth_mask = batch['cloth_mask'].to(self.device)
            pose_map = batch['pose_map'].to(self.device)
            
            batch_size = person.size(0)
            
            # Real and fake labels
            real_label = torch.ones(batch_size, 1, device=self.device)
            fake_label = torch.zeros(batch_size, 1, device=self.device)
            
            # =============== Train Generator ===============
            self.optimizer_G.zero_grad()
            
            # Generate try-on result
            outputs = self.model(person, cloth, cloth_mask, pose_map)
            try_on = outputs['try_on']
            
            # Generator losses
            g_losses = self.criterion_tryon(try_on, person)
            
            # Adversarial loss
            fake_pred = self.discriminator(try_on)
            g_adv_loss = self.criterion_adv(fake_pred, real_label)
            
            # Total generator loss
            g_loss = g_losses['total'] + 0.1 * g_adv_loss
            g_loss.backward()
            self.optimizer_G.step()
            
            # =============== Train Discriminator ===============
            self.optimizer_D.zero_grad()
            
            # Real images
            real_pred = self.discriminator(person)
            d_real_loss = self.criterion_adv(real_pred, real_label)
            
            # Fake images
            fake_pred = self.discriminator(try_on.detach())
            d_fake_loss = self.criterion_adv(fake_pred, fake_label)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            self.optimizer_D.step()
            
            # Update progress bar
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/G_Loss', g_loss.item(), step)
                self.writer.add_scalar('Train/D_Loss', d_loss.item(), step)
                self.writer.add_scalar('Train/L1_Loss', g_losses['l1'].item(), step)
                self.writer.add_scalar('Train/VGG_Loss', g_losses['vgg'].item(), step)
        
        return total_g_loss / len(self.train_loader), total_d_loss / len(self.train_loader)
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                person = batch['person'].to(self.device)
                cloth = batch['cloth'].to(self.device)
                cloth_mask = batch['cloth_mask'].to(self.device)
                pose_map = batch['pose_map'].to(self.device)
                
                outputs = self.model(person, cloth, cloth_mask, pose_map)
                try_on = outputs['try_on']
                
                losses = self.criterion_tryon(try_on, person)
                total_loss += losses['total'].item()
                
                # Save sample images
                if batch_idx == 0:
                    self.save_sample_images(person, cloth, try_on, epoch)
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return avg_loss
    
    def save_sample_images(self, person, cloth, try_on, epoch):
        """Save sample images for visualization"""
        import torchvision.utils as vutils
        
        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        person_denorm = person * std + mean
        cloth_denorm = cloth * std + mean
        try_on_denorm = try_on * std + mean
        
        # Clamp values
        person_denorm = torch.clamp(person_denorm, 0, 1)
        cloth_denorm = torch.clamp(cloth_denorm, 0, 1)
        try_on_denorm = torch.clamp(try_on_denorm, 0, 1)
        
        # Concatenate images
        comparison = torch.cat([person_denorm[:4], cloth_denorm[:4], try_on_denorm[:4]], dim=0)
        
        # Save grid
        save_path = os.path.join(self.checkpoint_dir, f'samples_epoch_{epoch}.png')
        vutils.save_image(comparison, save_path, nrow=4, normalize=False)
    
    def save_checkpoint(self, epoch, g_loss, d_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss
        }
        
        filename = f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pth'))
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            g_loss, d_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate(epoch)
            
            print(f'Epoch {epoch}: G_Loss={g_loss:.4f}, D_Loss={d_loss:.4f}, Val_Loss={val_loss:.4f}')
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, g_loss, d_loss, is_best)
        
        self.writer.close()

# Training configuration
config = {
    'lr_g': 0.0002,
    'lr_d': 0.0002,
    'batch_size': 8,
    'num_epochs': 100,
    'img_size': (256, 192),
    'log_dir': './logs',
    'checkpoint_dir': './checkpoints',
    'data_root': './virtual_tryon_dataset'
}

# Initialize everything
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    train_transform, val_transform = get_transforms(config['img_size'])
    
    train_dataset = VirtualTryOnDataset(
        config['data_root'], 
        os.path.join(config['data_root'], 'train_pairs.txt'),
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = VirtualTryOnDataset(
        config['data_root'],
        os.path.join(config['data_root'], 'test_pairs.txt'),
        transform=val_transform,
        mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Models
    model = VirtualTryOnModel()
    discriminator = Discriminator()
    
    # Trainer
    trainer = Trainer(model, discriminator, train_loader, val_loader, device, config)
    
    # Start training
    trainer.train(config['num_epochs'])

if __name__ == "__main__":
    main()
```

---

## Model Evaluation

### 1. Evaluation Metrics
```python
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

class EvaluationMetrics:
    def __init__(self, device):
        self.device = device
        self.lpips_net = lpips.LPIPS(net='alex').to(device)
    
    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        # Convert to numpy and ensure proper format
        img1_np = self.tensor_to_numpy(img1)
        img2_np = self.tensor_to_numpy(img2)
        
        ssim_values = []
        for i in range(img1_np.shape[0]):
            ssim_val = ssim(
                img1_np[i].transpose(1, 2, 0),
                img2_np[i].transpose(1, 2, 0),
                multichannel=True,
                data_range=1.0
            )
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        img1_np = self.tensor_to_numpy(img1)
        img2_np = self.tensor_to_numpy(img2)
        
        psnr_values = []
        for i in range(img1_np.shape[0]):
            psnr_val = psnr(
                img1_np[i].transpose(1, 2, 0),
                img2_np[i].transpose(1, 2, 0),
                data_range=1.0
            )
            psnr_values.append(psnr_val)
        
        return np.mean(psnr_values)
    
    def calculate_lpips(self, img1, img2):
        """Calculate LPIPS perceptual distance"""
        with torch.no_grad():
            lpips_val = self.lpips_net(img1, img2)
        return lpips_val.mean().item()
    
    def tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if tensor.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor.cpu().numpy()

def evaluate_model(model, test_loader, device, save_results=True):
    """Comprehensive model evaluation"""
    model.eval()
    metrics = EvaluationMetrics(device)
    
    total_ssim = 0
    total_psnr = 0
    total_lpips = 0
    num_batches = 0
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            person = batch['person'].to(device)
            cloth = batch['cloth'].to(device)
            cloth_mask = batch['cloth_mask'].to(device)
            pose_map = batch['pose_map'].to(device)
            
            # Generate try-on result
            outputs = model(person, cloth, cloth_mask, pose_map)
            try_on = outputs['try_on']
            
            # Calculate metrics
            ssim_val = metrics.calculate_ssim(try_on, person)
            psnr_val = metrics.calculate_psnr(try_on, person)
            lpips_val = metrics.calculate_lpips(try_on, person)
            
            total_ssim += ssim_val
            total_psnr += psnr_val
            total_lpips += lpips_val
            num_batches += 1
            
            # Store results
            if save_results:
                for i in range(person.size(0)):
                    results.append({
                        'person_name': batch['person_name'][i],
                        'cloth_name': batch['cloth_name'][i],
                        'ssim': ssim_val,
                        'psnr': psnr_val,
                        'lpips': lpips_val
                    })
            
            # Save sample images every 10 batches
            if batch_idx % 10 == 0 and save_results:
                save_evaluation_images(person, cloth, try_on, batch_idx)
    
    # Calculate averages
    avg_ssim = total_ssim / num_batches
    avg_psnr = total_psnr / num_batches
    avg_lpips = total_lpips / num_batches
    
    print(f"Evaluation Results:")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    
    return {
        'ssim': avg_ssim,
        'psnr': avg_psnr,
        'lpips': avg_lpips,
        'detailed_results': results
    }

def save_evaluation_images(person, cloth, try_on, batch_idx):
    """Save evaluation images for visual inspection"""
    import torchvision.utils as vutils
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(person.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(person.device)
    
    person_denorm = torch.clamp(person * std + mean, 0, 1)
    cloth_denorm = torch.clamp(cloth * std + mean, 0, 1)
    try_on_denorm = torch.clamp(try_on * std + mean, 0, 1)
    
    # Create comparison grid
    comparison = torch.cat([
        person_denorm[:4],
        cloth_denorm[:4],
        try_on_denorm[:4]
    ], dim=0)
    
    os.makedirs('./evaluation_results', exist_ok=True)
    save_path = f'./evaluation_results/eval_batch_{batch_idx}.png'
    vutils.save_image(comparison, save_path, nrow=4, normalize=False)
```

### 2. Model Testing Script
```python
def test_model(checkpoint_path, test_data_path):
    """Test saved model on new data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = VirtualTryOnModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    _, val_transform = get_transforms((256, 192))
    test_dataset = VirtualTryOnDataset(
        test_data_path,
        os.path.join(test_data_path, 'test_pairs.txt'),
        transform=val_transform,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Save results
    import json
    with open('./evaluation_results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# Usage
# results = test_model('./checkpoints/best_model.pth', './virtual_tryon_dataset')
```

---

## Model Deployment

### 1. Model Export and Optimization
```python
import torch
import torch.jit as jit
import onnx
import onnxruntime as ort

class ModelExporter:
    def __init__(self, model, input_shape=(1, 3, 192, 256)):
        self.model = model
        self.input_shape = input_shape
    
    def export_torchscript(self, save_path):
        """Export model to TorchScript"""
        self.model.eval()
        
        # Create dummy inputs
        dummy_person = torch.randn(self.input_shape)
        dummy_cloth = torch.randn(self.input_shape)
        dummy_mask = torch.randn(1, 1, 192, 256)
        dummy_pose = torch.randn(1, 18, 192, 256)
        
        # Trace the model
        traced_model = jit.trace(
            self.model,
            (dummy_person, dummy_cloth, dummy_mask, dummy_pose)
        )
        
        # Save
        traced_model.save(save_path)
        print(f"TorchScript model saved to {save_path}")
        
        return traced_model
    
    def export_onnx(self, save_path):
        """Export model to ONNX format"""
        self.model.eval()
        
        # Create dummy inputs
        dummy_person = torch.randn(self.input_shape)
        dummy_cloth = torch.randn(self.input_shape)
        dummy_mask = torch.randn(1, 1, 192, 256)
        dummy_pose = torch.randn(1, 18, 192, 256)
        
        # Export
        torch.onnx.export(
            self.model,
            (dummy_person, dummy_cloth, dummy_mask, dummy_pose),
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['person', 'cloth', 'cloth_mask', 'pose_map'],
            output_names=['try_on_result'],
            dynamic_axes={
                'person': {0: 'batch_size'},
                'cloth': {0: 'batch_size'},
                'cloth_mask': {0: 'batch_size'},
                'pose_map': {0: 'batch_size'},
                'try_on_result': {0: 'batch_size'}
            }
        )
        
        print(f"ONNX model saved to {save_path}")
    
    def test_onnx_model(self, onnx_path):
        """Test ONNX model inference"""
        session = ort.InferenceSession(onnx_path)
        
        # Create test inputs
        person = np.random.randn(1, 3, 192, 256).astype(np.float32)
        cloth = np.random.randn(1, 3, 192, 256).astype(np.float32)
        cloth_mask = np.random.randn(1, 1, 192, 256).astype(np.float32)
        pose_map = np.random.randn(1, 18, 192, 256).astype(np.float32)
        
        # Run inference
        outputs = session.run(
            None,
            {
                'person': person,
                'cloth': cloth,
                'cloth_mask': cloth_mask,
                'pose_map': pose_map
            }
        )
        
        print(f"ONNX inference successful. Output shape: {outputs[0].shape}")
        return outputs

# Export models
def export_trained_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = VirtualTryOnModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Export
    exporter = ModelExporter(model)
    
    # TorchScript
    exporter.export_torchscript('./models/virtual_tryon_traced.pt')
    
    # ONNX
    exporter.export_onnx('./models/virtual_tryon.onnx')
    exporter.test_onnx_model('./models/virtual_tryon.onnx')

# Usage
# export_trained_model('./checkpoints/best_model.pth')
```

### 2. FastAPI Deployment
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import mediapipe as mp

app = FastAPI(title="Virtual Try-On API", version="1.0.0")

class VirtualTryOnService:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = VirtualTryOnModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize pose estimator
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Transforms
        self.transform = A.Compose([
            A.Resize(192, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def extract_pose(self, image):
        """Extract pose keypoints from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        keypoints = np.zeros((33, 3))  # MediaPipe has 33 landmarks
        
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[i] = [
                    landmark.x * image.shape[1],
                    landmark.y * image.shape[0],
                    landmark.visibility
                ]
        
        return keypoints
    
    def create_pose_map(self, keypoints, img_size=(256, 192)):
        """Create pose heatmap from keypoints"""
        pose_map = np.zeros((img_size[1], img_size[0], 18))
        
        # Map MediaPipe keypoints to OpenPose format (simplified)
        mp_to_openpose = {
            0: 0,   # nose
            2: 14,  # left_eye
            5: 15,  # right_eye
            7: 16,  # left_ear
            8: 17,  # right_ear
            11: 5,  # left_shoulder
            12: 2,  # right_shoulder
            13: 6,  # left_elbow
            14: 3,  # right_elbow
            15: 7,  # left_wrist
            16: 4,  # right_wrist
            23: 8,  # left_hip
            24: 11, # right_hip
            25: 9,  # left_knee
            26: 12, # right_knee
            27: 10, # left_ankle
            28: 13  # right_ankle
        }
        
        for mp_idx, op_idx in mp_to_openpose.items():
            if mp_idx < len(keypoints) and keypoints[mp_idx][2] > 0.5:
                x, y, conf = keypoints[mp_idx]
                x, y = int(x), int(y)
                
                if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
                    # Create gaussian heatmap
                    sigma = 2
                    for dy in range(-3*sigma, 3*sigma+1):
                        for dx in range(-3*sigma, 3*sigma+1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < img_size[0] and 0 <= ny < img_size[1]:
                                dist = dx*dx + dy*dy
                                pose_map[ny, nx, op_idx] = np.exp(-dist / (2 * sigma * sigma))
        
        return pose_map
    
    def preprocess_image(self, image, is_cloth=False):
        """Preprocess image for model input"""
        # Resize
        image = cv2.resize(image, (256, 192))
        
        if is_cloth:
            # For cloth images, ensure clean background
            image = self.remove_background(image)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def remove_background(self, image):
        """Simple background removal for cloth images"""
        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask (assuming white/light background)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask
        result = image.copy()
        result[mask == 0] = [255, 255, 255]  # Set background to white
        
        return result
    
    def try_on(self, person_image, cloth_image):
        """Perform virtual try-on"""
        with torch.no_grad():
            # Extract pose
            keypoints = self.extract_pose(person_image)
            pose_map = self.create_pose_map(keypoints)
            
            # Preprocess images
            person_tensor = self.preprocess_image(person_image)
            cloth_tensor = self.preprocess_image(cloth_image, is_cloth=True)
            
            # Create cloth mask (simple thresholding)
            cloth_gray = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2GRAY)
            _, cloth_mask = cv2.threshold(cloth_gray, 240, 255, cv2.THRESH_BINARY_INV)
            cloth_mask = cv2.resize(cloth_mask, (256, 192))
            cloth_mask_tensor = torch.from_numpy(cloth_mask / 255.0).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Pose map tensor
            pose_map_tensor = torch.from_numpy(pose_map).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            # Model inference
            outputs = self.model(person_tensor, cloth_tensor, cloth_mask_tensor, pose_map_tensor)
            try_on_result = outputs['try_on']
            
            # Post-process result
            result_image = self.postprocess_output(try_on_result)
            
            return result_image
    
    def postprocess_output(self, tensor):
        """Convert model output to image"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image

# Initialize service
service = VirtualTryOnService('./checkpoints/best_model.pth')

@app.post("/try-on")
async def virtual_try_on(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...)
):
    try:
        # Read images
        person_bytes = await person_image.read()
        cloth_bytes = await cloth_image.read()
        
        # Convert to OpenCV format
        person_np = np.frombuffer(person_bytes, np.uint8)
        cloth_np = np.frombuffer(cloth_bytes, np.uint8)
        
        person_img = cv2.imdecode(person_np, cv2.IMREAD_COLOR)
        cloth_img = cv2.imdecode(cloth_np, cv2.IMREAD_COLOR)
        
        # Perform try-on
        result = service.try_on(person_img, cloth_img)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', result)
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Gradio Web Interface
```python
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image

class GradioInterface:
    def __init__(self, model_path):
        self.service = VirtualTryOnService(model_path)
    
    def process_images(self, person_image, cloth_image):
        """Process images for Gradio interface"""
        try:
            # Convert PIL to OpenCV
            person_cv = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGB2BGR)
            cloth_cv = cv2.cvtColor(np.array(cloth_image), cv2.COLOR_RGB2BGR)
            
            # Perform try-on
            result = self.service.try_on(person_cv, cloth_cv)
            
            # Convert back to PIL
            result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            
            return result_pil
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(title="Virtual Try-On") as interface:
            gr.Markdown("# Virtual Try-On System")
            gr.Markdown("Upload a person image and a clothing item to see the virtual try-on result!")
            
            with gr.Row():
                with gr.Column():
                    person_input = gr.Image(
                        type="pil",
                        label="Person Image",
                        info="Upload an image of a person"
                    )
                    cloth_input = gr.Image(
                        type="pil",
                        label="Clothing Item",
                        info="Upload an image of clothing"
                    )
                    submit_btn = gr.Button("Try On", variant="primary")
                
                with gr.Column():
                    result_output = gr.Image(
                        label="Try-On Result",
                        info="Virtual try-on result will appear here"
                    )
            
            # Examples
            gr.Examples(
                examples=[
                    ["./examples/person1.jpg", "./examples/shirt1.jpg"],
                    ["./examples/person2.jpg", "./examples/dress1.jpg"],
                ],
                inputs=[person_input, cloth_input],
                outputs=result_output,
                fn=self.process_images,
                cache_examples=True
            )
            
            submit_btn.click(
                fn=self.process_images,
                inputs=[person_input, cloth_input],
                outputs=result_output
            )
        
        return interface

# Launch Gradio app
def launch_gradio_app(model_path, share=False):
    app = GradioInterface(model_path)
    interface = app.create_interface()
    interface.launch(share=share, server_name="0.0.0.0", server_port=7860)

# Usage
# launch_gradio_app('./checkpoints/best_model.pth', share=True)
```

---

## Inference Pipeline

### 1. Single Image Inference
```python
class InferenceEngine:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load model
        self.model = VirtualTryOnModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize preprocessing
        self.transform = A.Compose([
            A.Resize(192, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print(f"Model loaded on {self.device}")
    
    def infer_single(self, person_path, cloth_path, output_path=None):
        """Perform inference on single image pair"""
        # Load images
        person_img = cv2.imread(person_path)
        cloth_img = cv2.imread(cloth_path)
        
        if person_img is None or cloth_img is None:
            raise ValueError("Could not load images")
        
        # Perform try-on
        result = self.try_on_inference(person_img, cloth_img)
        
        # Save result
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Result saved to {output_path}")
        
        return result
    
    def try_on_inference(self, person_img, cloth_img):
        """Core try-on inference logic"""
        with torch.no_grad():
            # Preprocess images
            person_tensor = self.preprocess_image(person_img)
            cloth_tensor = self.preprocess_image(cloth_img)
            
            # Extract pose (simplified - using dummy pose for now)
            pose_map_tensor = torch.zeros(1, 18, 192, 256).to(self.device)
            
            # Create cloth mask (simple thresholding)
            cloth_mask_tensor = self.create_cloth_mask(cloth_img)
            
            # Model inference
            outputs = self.model(person_tensor, cloth_tensor, cloth_mask_tensor, pose_map_tensor)
            result_tensor = outputs['try_on']
            
            # Post-process
            result_img = self.tensor_to_image(result_tensor)
            
            return result_img
    
    def preprocess_image(self, image):
        """Preprocess image for model"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (256, 192))
        
        transformed = self.transform(image=image_resized)
        tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return tensor
    
    def create_cloth_mask(self, cloth_img):
        """Create cloth mask"""
        gray = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.resize(mask, (256, 192))
        
        mask_tensor = torch.from_numpy(mask / 255.0).float()
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        return mask_tensor
    
    def tensor_to_image(self, tensor):
        """Convert tensor to image"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image

# Usage example
def run_inference_example():
    # Initialize engine
    engine = InferenceEngine('./checkpoints/best_model.pth')
    
    # Single inference
    result = engine.infer_single(
        './test_images/person.jpg',
        './test_images/shirt.jpg',
        './results/try_on_result.jpg'
    )
    
    print("Inference completed!")

# Batch inference
def batch_inference(model_path, input_pairs, output_dir):
    """Run inference on multiple image pairs"""
    engine = InferenceEngine(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (person_path, cloth_path) in enumerate(input_pairs):
        try:
            output_path = os.path.join(output_dir, f'result_{i:04d}.jpg')
            engine.infer_single(person_path, cloth_path, output_path)
            print(f"Processed {i+1}/{len(input_pairs)}: {output_path}")
        except Exception as e:
            print(f"Error processing pair {i}: {e}")

# Example usage
if __name__ == "__main__":
    # Define input pairs
    input_pairs = [
        ('./test_data/person1.jpg', './test_data/shirt1.jpg'),
        ('./test_data/person2.jpg', './test_data/dress1.jpg'),
        # Add more pairs...
    ]
    
    # Run batch inference
    batch_inference('./checkpoints/best_model.pth', input_pairs, './batch_results')
```

### 2. Real-time Webcam Interface
```python
import cv2
import torch
import numpy as np
from threading import Thread
import queue

class WebcamTryOn:
    def __init__(self, model_path, cloth_image_path):
        # Initialize model
        self.engine = InferenceEngine(model_path)
        
        # Load reference cloth
        self.cloth_img = cv2.imread(cloth_image_path)
        if self.cloth_img is None:
            raise ValueError(f"Could not load cloth image: {cloth_image_path}")
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Threading for real-time processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.processing = False
        
        print("Webcam try-on initialized")
    
    def capture_frames(self):
        """Capture frames from webcam"""
        while True:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            else:
                break
    
    def process_frames(self):
        """Process frames for try-on"""
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                try:
                    # Perform try-on
                    result = self.engine.try_on_inference(frame, self.cloth_img)
                    
                    # Resize result to match original frame
                    result_resized = cv2.resize(result, (frame.shape[1], frame.shape[0]))
                    
                    if not self.result_queue.full():
                        self.result_queue.put(result_resized)
                
                except Exception as e:
                    print(f"Processing error: {e}")
                    if not self.result_queue.full():
                        self.result_queue.put(frame)  # Fallback to original frame
    
    def run(self):
        """Run real-time try-on"""
        # Start threads
        capture_thread = Thread(target=self.capture_frames, daemon=True)
        process_thread = Thread(target=self.process_frames, daemon=True)
        
        capture_thread.start()
        process_thread.start()
        
        print("Press 'q' to quit, 'c' to change cloth, 's' to save current frame")
        
        while True:
            # Get original frame for display
            if not self.frame_queue.empty():
                current_frame = self.frame_queue.queue[-1]  # Latest frame
            else:
                ret, current_frame = self.cap.read()
                if not ret:
                    break
            
            # Get processed result
            if not self.result_queue.empty():
                result_frame = self.result_queue.get()
            else:
                result_frame = current_frame  # Fallback
            
            # Create side-by-side display
            display_frame = np.hstack([current_frame, result_frame])
            
            # Add text
            cv2.putText(display_frame, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Try-On", (current_frame.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Virtual Try-On', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current result
                timestamp = cv2.getTickCount()
                filename = f'tryon_result_{timestamp}.jpg'
                cv2.imwrite(filename, result_frame)
                print(f"Saved: {filename}")
            elif key == ord('c'):
                # Change cloth (placeholder - could implement file dialog)
                print("Cloth change not implemented in this demo")
        
        self.cap.release()
        cv2.destroyAllWindows()

# Usage
def run_webcam_tryon():
    try:
        webcam_app = WebcamTryOn(
            model_path='./checkpoints/best_model.pth',
            cloth_image_path='./demo_clothes/shirt.jpg'
        )
        webcam_app.run()
    except Exception as e:
        print(f"Error: {e}")

# run_webcam_tryon()
```

### 3. Mobile App Integration (using Kivy)
```python
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.label import Label
import cv2
import numpy as np

class VirtualTryOnApp(App):
    def __init__(self, model_path):
        super().__init__()
        self.engine = InferenceEngine(model_path)
        self.person_image_path = None
        self.cloth_image_path = None
    
    def build(self):
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title = Label(text='Virtual Try-On Mobile App', 
                     size_hint_y=None, height=50, font_size=24)
        main_layout.add_widget(title)
        
        # Image display area
        image_layout = BoxLayout(orientation='horizontal', spacing=10)
        
        self.person_img = Image(source='placeholder_person.jpg', size_hint_x=0.33)
        self.cloth_img = Image(source='placeholder_cloth.jpg', size_hint_x=0.33)
        self.result_img = Image(source='placeholder_result.jpg', size_hint_x=0.33)
        
        image_layout.add_widget(self.person_img)
        image_layout.add_widget(self.cloth_img)
        image_layout.add_widget(self.result_img)
        
        main_layout.add_widget(image_layout)
        
        # Buttons
        button_layout = BoxLayout(orientation='horizontal', 
                                 size_hint_y=None, height=60, spacing=10)
        
        select_person_btn = Button(text='Select Person')
        select_person_btn.bind(on_press=self.select_person_image)
        
        select_cloth_btn = Button(text='Select Cloth')
        select_cloth_btn.bind(on_press=self.select_cloth_image)
        
        try_on_btn = Button(text='Try On!')
        try_on_btn.bind(on_press=self.perform_try_on)
        
        button_layout.add_widget(select_person_btn)
        button_layout.add_widget(select_cloth_btn)
        button_layout.add_widget(try_on_btn)
        
        main_layout.add_widget(button_layout)
        
        return main_layout
    
    def select_person_image(self, instance):
        self.open_file_chooser('person')
    
    def select_cloth_image(self, instance):
        self.open_file_chooser('cloth')
    
    def open_file_chooser(self, image_type):
        content = BoxLayout(orientation='vertical')
        
        filechooser = FileChooserIconView(
            filters=['*.jpg', '*.jpeg', '*.png', '*.bmp']
        )
        content.add_widget(filechooser)
        
        button_layout = BoxLayout(size_hint_y=None, height=40, spacing=10)
        
        select_btn = Button(text='Select', size_hint_x=0.5)
        cancel_btn = Button(text='Cancel', size_hint_x=0.5)
        
        button_layout.add_widget(select_btn)
        button_layout.add_widget(cancel_btn)
        content.add_widget(button_layout)
        
        popup = Popup(title=f'Select {image_type} Image',
                     content=content, size_hint=(0.9, 0.9))
        
        def select_file(btn):
            if filechooser.selection:
                selected_file = filechooser.selection[0]
                if image_type == 'person':
                    self.person_image_path = selected_file
                    self.person_img.source = selected_file
                else:
                    self.cloth_image_path = selected_file
                    self.cloth_img.source = selected_file
            popup.dismiss()
        
        def cancel_selection(btn):
            popup.dismiss()
        
        select_btn.bind(on_press=select_file)
        cancel_btn.bind(on_press=cancel_selection)
        
        popup.open()
    
    def perform_try_on(self, instance):
        if not self.person_image_path or not self.cloth_image_path:
            self.show_error("Please select both person and cloth images")
            return
        
        try:
            # Perform inference
            result = self.engine.infer_single(
                self.person_image_path,
                self.cloth_image_path,
                './temp_result.jpg'
            )
            
            # Update result image
            self.result_img.source = './temp_result.jpg'
            self.result_img.reload()
            
        except Exception as e:
            self.show_error(f"Try-on failed: {str(e)}")
    
    def show_error(self, message):
        popup = Popup(title='Error',
                     content=Label(text=message),
                     size_hint=(0.8, 0.4))
        popup.open()

# Usage
def run_mobile_app():
    app = VirtualTryOnApp('./checkpoints/best_model.pth')
    app.run()

# Uncomment to run
# run_mobile_app()
```

---

## Troubleshooting

### 1. Common Issues and Solutions

#### GPU Memory Issues
```python
# Reduce batch size
config['batch_size'] = 4  # or even 2

# Enable gradient checkpointing
def enable_gradient_checkpointing(model):
    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True

# Clear cache regularly
torch.cuda.empty_cache()
```

#### Training Instability
```python
# Learning rate scheduling
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(person, cloth, cloth_mask, pose_map)
    loss = criterion(outputs['try_on'], person)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Data Loading Issues
```python
# Validate all file paths before training
def validate_data_files(data_root, pairs_file):
    missing_files = []
    
    with open(pairs_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                person_name, cloth_name = line.strip().split()
                
                # Check all required files
                files_to_check = [
                    ('image', person_name),
                    ('cloth', cloth_name),
                    ('cloth-mask', cloth_name),
                    ('openpose_json', person_name.replace('.jpg', '_keypoints.json'))
                ]
                
                for folder, filename in files_to_check:
                    filepath = os.path.join(data_root, folder, filename)
                    if not os.path.exists(filepath):
                        missing_files.append(f"Line {line_num}: {filepath}")
                        
            except ValueError:
                missing_files.append(f"Line {line_num}: Invalid format")
    
    if missing_files:
        print("Missing files found:")
        for missing in missing_files[:10]:  # Show first 10
            print(f"  {missing}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        return False
    
    return True

# Use before training
if not validate_data_files('./virtual_tryon_dataset/train', './virtual_tryon_dataset/train_pairs.txt'):
    print("Please fix missing files before training")
    exit(1)
```

### 2. Performance Optimization

#### Model Optimization
```python
# Quantization for deployment
def quantize_model(model, example_inputs):
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model

# TensorRT optimization (NVIDIA GPUs)
def optimize_with_tensorrt(onnx_path, engine_path):
    import tensorrt as trt
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    engine = builder.build_engine(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved to {engine_path}")
```

#### Data Pipeline Optimization
```python
# Optimized data loading
class OptimizedDataLoader:
    def __init__(self, dataset, batch_size, num_workers=4):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=2  # Prefetch batches
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
```

### 3. Debugging Tools

#### Visualization Utilities
```python
def visualize_training_progress(log_dir):
    """Create training progress visualization"""
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get scalar data
    g_loss = [(s.step, s.value) for s in event_acc.Scalars('Train/G_Loss')]
    d_loss = [(s.step, s.value) for s in event_acc.Scalars('Train/D_Loss')]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    steps, values = zip(*g_loss)
    ax1.plot(steps, values)
    ax1.set_title('Generator Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    
    steps, values = zip(*d_loss)
    ax2.plot(steps, values)
    ax2.set_title('Discriminator Loss')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('./training_progress.png')
    plt.show()

def debug_model_outputs(model, sample_batch, device):
    """Debug model intermediate outputs"""
    model.eval()
    
    with torch.no_grad():
        person = sample_batch['person'].to(device)
        cloth = sample_batch['cloth'].to(device)
        cloth_mask = sample_batch['cloth_mask'].to(device)
        pose_map = sample_batch['pose_map'].to(device)
        
        # Get intermediate outputs
        outputs = model(person, cloth, cloth_mask, pose_map)
        
        print("Model output shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        # Check for NaN or infinite values
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    print(f"WARNING: NaN values found in {key}")
                if torch.isinf(value).any():
                    print(f"WARNING: Infinite values found in {key}")
```

### 4. Deployment Checklist

```markdown
## Pre-deployment Checklist

### Model Validation
- [ ] Model converges during training
- [ ] Validation metrics are acceptable
- [ ] No NaN/infinite values in outputs
- [ ] Model works on sample test cases
- [ ] Memory usage is within limits

### Performance Testing
- [ ] Inference time meets requirements
- [ ] GPU/CPU utilization is optimal
- [ ] Model works with different image sizes
- [ ] Batch processing works correctly
- [ ] Real-time processing achievable

### Code Quality
- [ ] Error handling implemented
- [ ] Input validation added
- [ ] Logging configured
- [ ] Documentation complete
- [ ] Unit tests written

### Infrastructure
- [ ] Dependencies documented
- [ ] Environment setup automated
- [ ] Model versioning implemented
- [ ] Monitoring setup
- [ ] Backup strategy defined

### Security
- [ ] Input sanitization implemented
- [ ] Rate limiting configured
- [ ] Authentication added (if required)
- [ ] HTTPS enabled
- [ ] Data privacy compliance
```

---

## Conclusion

This comprehensive guide covers the complete pipeline for training and deploying a virtual try-on model. The system includes:

- **Data preprocessing** with proper validation and augmentation
- **Modern CNN architecture** with geometric matching and adversarial training
- **Robust training pipeline** with multiple loss functions and monitoring
- **Comprehensive evaluation** with standard metrics
- **Multiple deployment options** (API, web interface, mobile app)
- **Real-time inference** capabilities
- **Troubleshooting and optimization** guidelines

### Next Steps

1. **Start with data validation** to ensure your dataset is properly formatted
2. **Train a baseline model** with reduced complexity first
3. **Gradually increase model complexity** and training duration
4. **Implement evaluation pipeline** to track progress
5. **Deploy incrementally** starting with simple API endpoints
6. **Monitor performance** and iterate based on user feedback

### Performance Expectations

- **Training time**: 2-5 days on modern GPU (RTX 3080/4080)
- **Inference time**: 100-500ms per image (depending on hardware)
- **Model size**: 50-200MB (depending on architecture)
- **Accuracy**: 85-95% subjective quality on good test data

This system provides a solid foundation for building production-ready virtual try-on applications.