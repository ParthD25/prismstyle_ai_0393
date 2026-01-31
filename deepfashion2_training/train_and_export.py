#!/usr/bin/env python3
"""
PrismStyle AI - Complete Training & Export Pipeline
Run this on your PC with GPU for best performance.

REQUIREMENTS:
    pip install torch torchvision onnx onnxruntime coremltools tensorflow tqdm numpy

USAGE:
    python train_and_export.py --data_dir /path/to/deepfashion2 --output_dir ./models

This will:
1. Train a ConvNeXt-Base model on DeepFashion2
2. Export to ONNX (for Android/iOS via onnxruntime)
3. Export to TFLite (for Android fallback)
4. Export to CoreML (for iOS native performance)
"""

import os
import sys
import argparse
import json
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
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CATEGORIES = [
    "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear",
    "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"
]

APP_CATEGORY_MAPPING = {
    "short_sleeve_top": "Tops",
    "long_sleeve_top": "Tops",
    "short_sleeve_outwear": "Tops",
    "long_sleeve_outwear": "Tops",
    "vest": "Tops",
    "sling": "Tops",
    "shorts": "Bottoms",
    "trousers": "Bottoms",
    "skirt": "Bottoms",
    "short_sleeve_dress": "Dresses",
    "long_sleeve_dress": "Dresses",
    "vest_dress": "Dresses",
    "sling_dress": "Dresses"
}

CONFIG = {
    "batch_size": 32,       # RTX 5060 Ti can handle 32 easily
    "epochs": 60,
    "max_lr": 0.0003,
    "image_size": 224,      # Mobile-friendly, use 384 for higher accuracy
    "num_classes": 13,
    "patience": 15,
    "num_tta": 5,
    # Hardware-specific (RTX 5060 Ti, 64GB RAM, AMD 9090X)
    "num_workers": 8,       # More workers for fast CPU
    "pin_memory": True,
    "use_amp": True,        # Mixed precision for faster training
}

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
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


def create_model(num_classes=13):
    """Create ConvNeXt-Base model optimized for fashion"""
    model = models.convnext_base(weights='IMAGENET1K_V1')
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model


def create_efficientnet_model(num_classes=13):
    """Alternative lighter model for faster inference"""
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(num_features, num_classes)
    )
    return model


# ============================================================================
# DATA LOADING
# ============================================================================

def get_transforms(train=True, image_size=224):
    """Get transforms with AutoAugment for training"""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def get_weighted_sampler(dataset):
    """Create weighted sampler for balanced batches"""
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    
    sample_weights = []
    for target in targets:
        weight = 1.0 / class_counts[target]
        sample_weights.append(weight)
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
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
            'loss': f'{train_loss/train_total*CONFIG["batch_size"]:.4f}',
            'acc': f'{100*train_correct/train_total:.1f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    return train_loss / len(train_loader), 100 * train_correct / train_total


def validate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
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


def train_model(data_dir, output_dir, epochs=None):
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or CONFIG["epochs"]
    
    print("=" * 70)
    print("  PrismStyle AI - Fashion Classifier Training")
    print("=" * 70)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load datasets
    print("\n[STEP 1] Loading datasets...")
    train_dataset = ImageFolder(
        os.path.join(data_dir, 'train'), 
        transform=get_transforms(train=True, image_size=CONFIG["image_size"])
    )
    val_dataset = ImageFolder(
        os.path.join(data_dir, 'validation'), 
        transform=get_transforms(train=False, image_size=CONFIG["image_size"])
    )
    
    sampler = get_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], sampler=sampler,
        num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"], drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"] * 2, shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"]
    )
    
    print(f"Training: {len(train_dataset):,} samples")
    print(f"Validation: {len(val_dataset):,} samples")
    
    # Create model
    print("\n[STEP 2] Creating model...")
    model = create_model(CONFIG["num_classes"])
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    # Loss, optimizer, scheduler
    criterion = FocalLoss(alpha=1, gamma=2, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["max_lr"], weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["max_lr"], epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos'
    )
    scaler = GradScaler()
    
    os.makedirs(output_dir, exist_ok=True)
    
    best_acc = 0.0
    no_improve = 0
    
    print("\n[STEP 3] Training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        val_acc, class_accs = validate(model, val_loader, device)
        
        print(f"\nTrain Acc: {train_acc:.2f}%  |  Val Acc: {val_acc:.2f}%")
        
        # Show weak classes
        weak = [k for k, v in class_accs.items() if v < 70]
        if weak:
            print(f"Weak classes: {', '.join(weak)}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_accuracies': class_accs,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"✓ Saved best model ({best_acc:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\n✓ Training complete! Best accuracy: {best_acc:.2f}%")
    return model, best_acc


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_onnx(model, output_path, image_size=224):
    """Export model to ONNX format"""
    print("\n[EXPORT] Converting to ONNX...")
    model.eval()
    model.cpu()
    
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ ONNX exported: {output_path} ({size_mb:.1f} MB)")


def export_to_tflite(model, output_path, image_size=224):
    """Export model to TFLite format via ONNX"""
    print("\n[EXPORT] Converting to TFLite...")
    
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
        
        # First export to ONNX
        temp_onnx = output_path.replace('.tflite', '_temp.onnx')
        model.eval()
        model.cpu()
        dummy_input = torch.randn(1, 3, image_size, image_size)
        
        torch.onnx.export(
            model, dummy_input, temp_onnx,
            export_params=True, opset_version=14,
            input_names=['image'], output_names=['logits']
        )
        
        # Convert ONNX to TensorFlow
        onnx_model = onnx.load(temp_onnx)
        tf_rep = prepare(onnx_model)
        tf_model_path = output_path.replace('.tflite', '_tf')
        tf_rep.export_graph(tf_model_path)
        
        # Convert TensorFlow to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Cleanup
        os.remove(temp_onnx)
        import shutil
        shutil.rmtree(tf_model_path, ignore_errors=True)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ TFLite exported: {output_path} ({size_mb:.1f} MB)")
        
    except ImportError as e:
        print(f"⚠ TFLite export requires additional packages: {e}")
        print("  Install with: pip install tensorflow onnx-tf")


def export_to_coreml(model, output_path, image_size=224):
    """Export model to CoreML format for iOS"""
    print("\n[EXPORT] Converting to CoreML...")
    
    try:
        import coremltools as ct
        
        model.eval()
        model.cpu()
        
        # Trace the model
        example_input = torch.randn(1, 3, image_size, image_size)
        traced_model = torch.jit.trace(model, example_input)
        
        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.ImageType(
                name="image",
                shape=(1, 3, image_size, image_size),
                scale=1/255.0,
                bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225]
            )],
            classifier_config=ct.ClassifierConfig(CATEGORIES),
            minimum_deployment_target=ct.target.iOS15
        )
        
        # Add metadata
        mlmodel.author = "PrismStyle AI"
        mlmodel.short_description = "Fashion clothing classifier"
        mlmodel.version = "1.0.0"
        
        mlmodel.save(output_path)
        
        print(f"✓ CoreML exported: {output_path}")
        
    except ImportError:
        print("⚠ CoreML export requires coremltools: pip install coremltools")
    except Exception as e:
        print(f"⚠ CoreML export failed: {e}")


def save_model_config(output_dir):
    """Save model configuration JSON"""
    config = {
        "model_name": "PrismStyle Fashion Classifier",
        "version": "1.0.0",
        "description": "Multi-category fashion clothing classifier",
        "input_size": CONFIG["image_size"],
        "num_classes": CONFIG["num_classes"],
        "categories": CATEGORIES,
        "app_category_mapping": APP_CATEGORY_MAPPING,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "model_files": {
            "onnx": "clothing_classifier.onnx",
            "tflite": "clothing_classifier.tflite",
            "coreml": "clothing_classifier.mlpackage"
        }
    }
    
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Config saved: {config_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PrismStyle AI Training Pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to DeepFashion2 dataset (with train/validation folders)')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: 60)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only export existing model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to existing model checkpoint for export')
    parser.add_argument('--quick', action='store_true',
                        help='Quick training mode (5 epochs, smaller batch)')
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        if args.epochs is None:
            args.epochs = 5
        CONFIG['batch_size'] = 16
        CONFIG['patience'] = 3
        print("🚀 Quick training mode enabled")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.skip_training and args.model_path:
        # Load existing model
        print(f"Loading model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model = create_model(CONFIG["num_classes"])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with accuracy: {checkpoint.get('val_acc', 'Unknown')}%")
    else:
        # Train new model
        model, accuracy = train_model(args.data_dir, args.output_dir, args.epochs)
        
        # Load best checkpoint
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Export to all formats
    print("\n" + "=" * 70)
    print("  EXPORTING MODELS")
    print("=" * 70)
    
    export_to_onnx(
        model,
        os.path.join(args.output_dir, 'clothing_classifier.onnx'),
        CONFIG["image_size"]
    )
    
    export_to_tflite(
        model,
        os.path.join(args.output_dir, 'clothing_classifier.tflite'),
        CONFIG["image_size"]
    )
    
    export_to_coreml(
        model,
        os.path.join(args.output_dir, 'clothing_classifier.mlpackage'),
        CONFIG["image_size"]
    )
    
    save_model_config(args.output_dir)
    
    print("\n" + "=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"\nModels saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Copy the following files to your Flutter project:")
    print("   - clothing_classifier.onnx -> assets/models/")
    print("   - clothing_classifier.tflite -> assets/models/")
    print("   - model_config.json -> assets/models/")
    print("   - clothing_classifier.mlpackage -> ios/Runner/ (for CoreML)")
    print("\n2. Run 'flutter pub get' to refresh assets")
    print("3. Build and test on device")


if __name__ == "__main__":
    main()
