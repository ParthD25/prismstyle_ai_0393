#!/usr/bin/env python3
"""
DeepFashion2 Training Script - V4 BREAKTHROUGH
Target: 80%+ Accuracy with ConvNeXt + Class Balancing + TTA

KEY FIXES from v3:
1. ConvNeXt-Base backbone (better than EfficientNet for fashion)
2. Weighted sampling for class imbalance
3. OneCycleLR scheduler (proven better than ReduceLROnPlateau)
4. Test-Time Augmentation (TTA) for validation
5. Per-class accuracy tracking to identify weak classes
6. Focal Loss for imbalanced data
7. AutoAugment instead of manual augmentation
8. 384x384 resolution for more detail
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = r"C:\Users\pdave\tf_train\data\deepfashion2"
BATCH_SIZE = 24  # Reduced for 384x384 + ConvNeXt
EPOCHS = 60
MAX_LR = 0.0003  # OneCycleLR max
IMAGE_SIZE = 384  # Larger for more detail
NUM_CLASSES = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 15
NUM_TTA = 5  # Test-time augmentation passes

CATEGORIES = [
    "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear",
    "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"
]


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance - focuses on hard examples"""
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                   label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_class_weights(dataset):
    """Calculate class weights for imbalanced dataset"""
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    total = len(targets)
    
    print("\n[INFO] Class distribution:")
    for i, name in enumerate(CATEGORIES):
        count = class_counts.get(i, 0)
        pct = 100 * count / total
        print(f"  {i:2d}. {name:25s}: {count:6d} ({pct:5.2f}%)")
    
    # Calculate weights (inverse frequency)
    weights = []
    for i in range(NUM_CLASSES):
        count = class_counts.get(i, 1)
        weight = total / (NUM_CLASSES * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def get_weighted_sampler(dataset):
    """Create weighted sampler for balanced batches"""
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    
    # Weight per sample (inverse class frequency)
    sample_weights = []
    for target in targets:
        weight = 1.0 / class_counts[target]
        sample_weights.append(weight)
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def create_model():
    """Create ConvNeXt-Base model (better than EfficientNet for fashion)"""
    # ConvNeXt-Base: 88M params, excellent accuracy/speed tradeoff
    model = models.convnext_base(weights='IMAGENET1K_V1')
    
    # Replace classifier
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    
    return model


def create_efficientnetv2_model():
    """Alternative: EfficientNetV2-M (if ConvNeXt doesn't work)"""
    model = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    return model


def freeze_backbone(model, freeze_ratio=1.0):
    """Freeze backbone layers for transfer learning"""
    # ConvNeXt has features.0-7 (8 stages)
    stages = list(model.features.children())
    num_stages = len(stages)
    freeze_until = int(num_stages * freeze_ratio)
    
    for i, stage in enumerate(stages):
        for param in stage.parameters():
            param.requires_grad = i >= freeze_until
    
    # Always train classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def get_transforms(train=True):
    """Get transforms with AutoAugment for training"""
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),  # Proven augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def get_tta_transforms():
    """Test-Time Augmentation transforms"""
    return [
        # Original
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Slight rotation
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Scale up
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 40, IMAGE_SIZE + 40)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Scale down
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE - 20, IMAGE_SIZE - 20)),
            transforms.Pad(10),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]


def validate_with_tta(model, val_loader, device, use_tta=True):
    """Validate with Test-Time Augmentation"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            labels = labels.to(device)
            
            if use_tta and NUM_TTA > 1:
                # TTA: Average predictions from multiple augmentations
                batch_preds = []
                for _ in range(NUM_TTA):
                    aug_inputs = inputs.to(device)
                    # Add small random noise for diversity
                    aug_inputs = aug_inputs + torch.randn_like(aug_inputs) * 0.02
                    with autocast(device_type='cuda'):
                        outputs = model(aug_inputs)
                    batch_preds.append(F.softmax(outputs, dim=1))
                
                # Average predictions
                avg_preds = torch.stack(batch_preds).mean(dim=0)
                _, predicted = avg_preds.max(1)
            else:
                inputs = inputs.to(device)
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    
    # Per-class accuracy
    class_acc = {}
    for i, name in enumerate(CATEGORIES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc[name] = 100.0 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
        else:
            class_acc[name] = 0.0
    
    return accuracy, class_acc


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 70)
    print("  PrismStyle AI - V4 BREAKTHROUGH Training")
    print("  Target: 80%+ with ConvNeXt + Class Balancing + TTA")
    print("=" * 70)
    print(f"\nKey improvements:")
    print(f"  - ConvNeXt-Base backbone (better than EfficientNet)")
    print(f"  - Weighted sampling for class imbalance")
    print(f"  - AutoAugment (proven ImageNet augmentation)")
    print(f"  - Test-Time Augmentation ({NUM_TTA} passes)")
    print(f"  - Focal Loss for hard examples")
    print(f"  - {IMAGE_SIZE}x{IMAGE_SIZE} resolution")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"\n[OK] GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"     VRAM: {vram:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Load datasets
    print("\n[STEP 1] Loading datasets...")
    train_dataset = ImageFolder(os.path.join(DATA_DIR, 'train'), transform=get_transforms(train=True))
    val_dataset = ImageFolder(os.path.join(DATA_DIR, 'validation'), transform=get_transforms(train=False))
    
    # Get class weights for loss function
    class_weights = get_class_weights(train_dataset).to(DEVICE)
    
    # Create weighted sampler for balanced batches
    sampler = get_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"\n[OK] Training: {len(train_dataset):,} samples")
    print(f"[OK] Validation: {len(val_dataset):,} samples")

    # Create model
    print("\n[STEP 2] Creating ConvNeXt-Base model...")
    model = create_model()
    model = model.to(DEVICE)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Parameters: {params:,}")

    # Loss and optimizer
    criterion = FocalLoss(alpha=1, gamma=2, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.05)
    
    # OneCycleLR - proven to work better than ReduceLROnPlateau
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    scaler = GradScaler()
    os.makedirs(r'C:\Users\pdave\tf_train\models', exist_ok=True)
    
    best_acc = 0.0
    best_acc_tta = 0.0
    no_improve = 0
    history = []

    print("\n[STEP 3] Training with progressive unfreezing...")
    print("\n--- Phase 1: Classifier only (epochs 1-8) ---")
    freeze_backbone(model, freeze_ratio=1.0)

    for epoch in range(EPOCHS):
        # Progressive unfreezing
        if epoch == 8:
            print("\n--- Phase 2: Top 50% unfrozen ---")
            freeze_backbone(model, freeze_ratio=0.5)
        elif epoch == 20:
            print("\n--- Phase 3: Top 75% unfrozen ---")
            freeze_backbone(model, freeze_ratio=0.25)
        elif epoch == 35:
            print("\n--- Phase 4: All layers ---")
            freeze_backbone(model, freeze_ratio=0.0)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{train_loss/train_total*BATCH_SIZE:.4f}',
                'acc': f'{100*train_correct/train_total:.1f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        train_acc = 100 * train_correct / train_total
        
        # Validation without TTA (fast)
        val_acc, class_accs = validate_with_tta(model, val_loader, DEVICE, use_tta=False)
        
        # Validation WITH TTA (slower but more accurate)
        val_acc_tta, _ = validate_with_tta(model, val_loader, DEVICE, use_tta=True)
        
        history.append({
            'epoch': epoch + 1, 'train_acc': train_acc,
            'val_acc': val_acc, 'val_acc_tta': val_acc_tta,
            'lr': scheduler.get_last_lr()[0]
        })
        
        print(f"\nResults:")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc:   {val_acc:.2f}% (no TTA) | {val_acc_tta:.2f}% (with TTA)")
        
        # Show per-class accuracy
        print("\n  Per-class accuracy:")
        weak_classes = []
        for name, acc in sorted(class_accs.items(), key=lambda x: x[1]):
            status = "⚠️" if acc < 70 else "✓"
            print(f"    {status} {name:25s}: {acc:5.1f}%")
            if acc < 70:
                weak_classes.append(name)
        
        if weak_classes:
            print(f"\n  [FOCUS] Weak classes: {', '.join(weak_classes)}")
        
        # Save best model
        if val_acc_tta > best_acc_tta:
            best_acc_tta = val_acc_tta
            best_acc = val_acc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_acc_tta': val_acc_tta,
                'class_accuracies': class_accs,
            }, r'C:\Users\pdave\tf_train\models\best_model_v4.pth')
            print(f"\n  [SAVED] Best: {best_acc:.2f}% (TTA: {best_acc_tta:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs")
                break

    print("\n" + "=" * 70)
    print(f"  Training Complete!")
    print(f"  Best Val Accuracy: {best_acc:.2f}% (with TTA: {best_acc_tta:.2f}%)")
    print("=" * 70)
    
    # Save final artifacts
    with open(r'C:\Users\pdave\tf_train\models\training_history_v4.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Convert to ONNX for Core ML
    print("\n[STEP 4] Exporting to ONNX...")
    model.eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    torch.onnx.export(
        model, dummy,
        r'C:\Users\pdave\tf_train\models\clothing_classifier_v4.onnx',
        export_params=True, opset_version=14,
        input_names=['image'], output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}}
    )
    print("[OK] ONNX exported")
    
    if best_acc_tta >= 80:
        print(f"\n✅ SUCCESS! Achieved {best_acc_tta:.2f}% (Target: 80%+)")
    else:
        print(f"\n⚠️ Achieved {best_acc_tta:.2f}%. Try:")
        print("   - Train longer (increase EPOCHS)")
        print("   - Use EfficientNetV2-L instead")
        print("   - Check weak classes and add more data")


if __name__ == "__main__":
    main()
