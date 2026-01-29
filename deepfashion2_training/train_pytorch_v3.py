#!/usr/bin/env python3
"""
DeepFashion2 Training Script - V3 BREAKTHROUGH VERSION
Target: 80%+ Accuracy with CutMix + ReduceLROnPlateau + Higher Resolution

Key Improvements over v2:
1. CutMix augmentation (proven to learn rare features better than MixUp alone)
2. ReduceLROnPlateau (auto-adjusts LR when stalled - better than fixed cosine)
3. 300x300 resolution (more detail for clothing patterns)
4. Higher label smoothing (0.15)
5. Combined MixUp + CutMix (alternating)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import json
import random

# Configuration - OPTIMIZED FOR BREAKTHROUGH
DATA_DIR = r"C:\Users\pdave\tf_train\data\deepfashion2"
BATCH_SIZE = 32  # Reduced for 300x300 images (VRAM)
EPOCHS = 100
LEARNING_RATE = 0.0001  # Higher initial LR with ReduceLROnPlateau
IMAGE_SIZE = 300  # INCREASED from 224 for more detail
NUM_CLASSES = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 20
LABEL_SMOOTHING = 0.15  # Increased from 0.1

CATEGORY_MAPPING = {
    0: {"name": "short_sleeve_top", "display": "T-Shirt / Short Sleeve Top"},
    1: {"name": "long_sleeve_top", "display": "Shirt / Long Sleeve Top"},
    2: {"name": "short_sleeve_outwear", "display": "Short Sleeve Jacket"},
    3: {"name": "long_sleeve_outwear", "display": "Jacket / Coat"},
    4: {"name": "vest", "display": "Vest"},
    5: {"name": "sling", "display": "Sling / Camisole"},
    6: {"name": "shorts", "display": "Shorts"},
    7: {"name": "trousers", "display": "Pants / Trousers"},
    8: {"name": "skirt", "display": "Skirt"},
    9: {"name": "short_sleeve_dress", "display": "Short Sleeve Dress"},
    10: {"name": "long_sleeve_dress", "display": "Long Sleeve Dress"},
    11: {"name": "vest_dress", "display": "Vest Dress"},
    12: {"name": "sling_dress", "display": "Sling / Slip Dress"},
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation - patches from other images"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation"""
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_acc, model):
        if self.best_score is None:
            self.best_score = val_acc
            self.save_checkpoint(model)
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}


def create_model():
    """Create EfficientNetB3 with custom classifier"""
    model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),  # Increased dropout
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    return model


def freeze_layers(model, freeze_ratio=1.0):
    """Freeze a portion of the model layers"""
    feature_layers = list(model.features.children())
    num_layers = len(feature_layers)
    num_freeze = int(num_layers * freeze_ratio)
    
    for i, layer in enumerate(feature_layers):
        for param in layer.parameters():
            param.requires_grad = i >= num_freeze
    
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def main():
    set_seed(42)
    
    print("=" * 70)
    print("  PrismStyle AI - V3 BREAKTHROUGH Training")
    print("  Target: 80%+ Accuracy with CutMix + ReduceLROnPlateau")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset path:    {DATA_DIR}")
    print(f"  Device:          {DEVICE}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  Image size:      {IMAGE_SIZE}x{IMAGE_SIZE} (INCREASED)")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"  Augmentation:    MixUp + CutMix (alternating)")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"\n[OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Strong data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\n[STEP 1] Loading datasets...")
    train_dataset = ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(DATA_DIR, 'validation'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"[OK] Training samples:   {len(train_dataset):,}")
    print(f"[OK] Validation samples: {len(val_dataset):,}")
    print(f"[OK] Classes: {train_dataset.classes}")

    print("\n[STEP 2] Creating EfficientNet-B3 model...")
    model = create_model()
    model = model.to(DEVICE)
    print(f"[OK] Model loaded on {DEVICE}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.02)
    
    # ReduceLROnPlateau - KEY IMPROVEMENT
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    os.makedirs(r'C:\Users\pdave\tf_train\models', exist_ok=True)
    best_acc = 0.0
    training_history = []

    print("\n[STEP 3] Starting training with CutMix + MixUp...")
    print("--- Phase 1: Training classifier only (epochs 1-15) ---")
    freeze_layers(model, freeze_ratio=1.0)

    for epoch in range(EPOCHS):
        # Progressive unfreezing
        if epoch == 15:
            print("\n--- Phase 2: Unfreezing top 40% ---")
            freeze_layers(model, freeze_ratio=0.6)
        elif epoch == 35:
            print("\n--- Phase 3: Unfreezing top 70% ---")
            freeze_layers(model, freeze_ratio=0.3)
        elif epoch == 55:
            print("\n--- Phase 4: Fine-tuning all layers ---")
            freeze_layers(model, freeze_ratio=0.0)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        train_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Alternate between CutMix and MixUp
            if random.random() > 0.5:
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=1.0)
            else:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = mixed_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (lam * predicted.eq(labels_a).sum().item() + 
                            (1 - lam) * predicted.eq(labels_b).sum().item())
            
            train_bar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                if not torch.isnan(loss):
                    val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # ReduceLROnPlateau step
        scheduler.step(val_acc)
        
        training_history.append({
            'epoch': epoch + 1, 'train_loss': train_loss/len(train_loader),
            'train_acc': train_acc, 'val_loss': val_loss/len(val_loader),
            'val_acc': val_acc, 'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss/len(val_loader):.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc, 'category_mapping': CATEGORY_MAPPING,
            }, r'C:\Users\pdave\tf_train\models\best_model_v3.pth')
            print(f"  [SAVED] New best accuracy: {best_acc:.2f}%")
        
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs.")
            model.load_state_dict(early_stopping.best_model)
            break

    print("\n" + "=" * 70)
    print(f"  Training Complete! Best Accuracy: {best_acc:.2f}%")
    print("=" * 70)

    torch.save(model.state_dict(), r'C:\Users\pdave\tf_train\models\final_model_v3.pth')
    with open(r'C:\Users\pdave\tf_train\models\training_history_v3.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n[OK] Model and history saved")
    
    if best_acc >= 80:
        print(f"\n SUCCESS! Model achieved {best_acc:.2f}% (Target: 80%+)")
    else:
        print(f"\n Training complete. Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
