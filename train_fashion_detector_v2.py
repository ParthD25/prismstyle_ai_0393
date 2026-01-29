"""
Fashion Detector Training v2 - Using DeepFashion2 Ground Truth
==============================================================

This script trains a lightweight fashion detector directly on DeepFashion2
ground truth annotations (bounding boxes + categories).

Model Architecture:
- Backbone: EfficientNet-B0 (pretrained)
- Neck: Feature Pyramid Network (FPN)  
- Head: Anchor-free detection head

This is more reliable than GroundingDINO pseudo-labeling since we use
actual ground truth annotations.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import box_iou, nms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import timm

# Fashion categories (DeepFashion2)
FASHION_CATEGORIES = [
    'short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 
    'long_sleeve_outwear', 'vest', 'sling', 'shorts', 'trousers', 
    'skirt', 'short_sleeve_dress', 'long_sleeve_dress', 'vest_dress', 
    'sling_dress'
]

NUM_CLASSES = len(FASHION_CATEGORIES)


class DeepFashion2Dataset(Dataset):
    """
    DeepFashion2 dataset loader using ground truth annotations.
    Each annotation contains bounding boxes and category IDs.
    """
    def __init__(self, root_dir, split='train', transform=None, max_samples=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Handle nested folder structure: root/split/split/image and root/split/split/annos
        base_path = self.root_dir / split / split
        if not base_path.exists():
            base_path = self.root_dir / split
        
        self.img_dir = base_path / 'image'
        self.ann_dir = base_path / 'annos'
        
        if not self.ann_dir.exists():
            raise ValueError(f"Annotations directory not found: {self.ann_dir}")
        
        # Load all annotation files
        self.samples = []
        ann_files = sorted(self.ann_dir.glob('*.json'))
        
        if max_samples:
            ann_files = ann_files[:max_samples]
        
        print(f"[Dataset] Loading {split} annotations from {self.ann_dir}")
        
        errors = 0
        for ann_file in tqdm(ann_files, desc=f"Loading {split}"):
            try:
                with open(ann_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        errors += 1
                        continue
                    data = json.loads(content)
                
                # Get image filename
                img_name = ann_file.stem + '.jpg'
                img_path = self.img_dir / img_name
                
                if not img_path.exists():
                    continue
                
                # Extract all items (item1, item2, etc.)
                boxes = []
                labels = []
                
                for key, value in data.items():
                    if key.startswith('item') and isinstance(value, dict):
                        category_id = value.get('category_id', 0)
                        bbox = value.get('bounding_box', [])
                        
                        if 1 <= category_id <= 13 and len(bbox) == 4:
                            # bbox format: [x1, y1, x2, y2]
                            boxes.append(bbox)
                            labels.append(category_id - 1)  # 0-indexed
                
                if boxes:
                    self.samples.append({
                        'image_path': str(img_path),
                        'boxes': boxes,
                        'labels': labels
                    })
            except (json.JSONDecodeError, Exception) as e:
                errors += 1
                continue
        
        if errors > 0:
            print(f"[Dataset] Skipped {errors} files with errors")
        
        print(f"[Dataset] Loaded {len(self.samples)} samples with {sum(len(s['boxes']) for s in self.samples)} total objects")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        orig_w, orig_h = image.size
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Scale boxes to match transformed image size (224x224)
        boxes = torch.tensor(sample['boxes'], dtype=torch.float32)
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (224.0 / orig_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (224.0 / orig_h)
            # Clamp to image bounds
            boxes = boxes.clamp(min=0, max=224)
        
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        
        return image, boxes, labels


def collate_fn(batch):
    """Custom collate function for variable-length boxes"""
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, boxes, labels


class FashionDetector(nn.Module):
    """
    Lightweight fashion detector with EfficientNet backbone.
    Uses anchor-free detection similar to FCOS/CenterNet.
    """
    def __init__(self, num_classes=NUM_CLASSES, backbone='efficientnet_b0'):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        
        # Get feature dimensions from backbone
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy)
        
        # Use last 3 feature maps for FPN
        self.feat_channels = [f.shape[1] for f in features[-3:]]
        self.fpn_out_channels = 128
        
        # FPN layers
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, self.fpn_out_channels, 1) for c in self.feat_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1)
            for _ in self.feat_channels
        ])
        
        # Detection head
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1),
            nn.BatchNorm2d(self.fpn_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_out_channels, num_classes, 1)
        )
        
        self.reg_head = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1),
            nn.BatchNorm2d(self.fpn_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_out_channels, 4, 1)  # l, t, r, b distances
        )
        
        self.centerness_head = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1),
            nn.BatchNorm2d(self.fpn_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_out_channels, 1, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.cls_head, self.reg_head, self.centerness_head]:
            for layer in m:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # Extract backbone features
        features = self.backbone(x)[-3:]  # Last 3 feature maps
        
        # FPN forward
        fpn_features = []
        last_feat = None
        
        for i in range(len(features) - 1, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            
            if last_feat is not None:
                # Upsample and add
                last_feat = F.interpolate(last_feat, size=lateral.shape[-2:], mode='nearest')
                lateral = lateral + last_feat
            
            fpn_feat = self.fpn_convs[i](lateral)
            fpn_features.insert(0, fpn_feat)
            last_feat = lateral
        
        # Use middle scale for detection (good balance)
        feat = fpn_features[1]
        
        # Detection heads
        cls_logits = self.cls_head(feat)
        reg_pred = self.reg_head(feat)
        centerness = self.centerness_head(feat)
        
        return {
            'cls_logits': cls_logits,
            'reg_pred': reg_pred,
            'centerness': centerness,
            'feat_size': feat.shape[-2:]
        }


def compute_targets(boxes_batch, labels_batch, feat_size, img_size=224):
    """
    Compute training targets for FCOS-style detection.
    """
    batch_size = len(boxes_batch)
    H, W = feat_size
    stride = img_size // H
    
    device = boxes_batch[0].device if len(boxes_batch[0]) > 0 else 'cpu'
    
    # Output targets
    cls_targets = torch.zeros(batch_size, H, W, dtype=torch.long, device=device)
    reg_targets = torch.zeros(batch_size, 4, H, W, device=device)
    centerness_targets = torch.zeros(batch_size, 1, H, W, device=device)
    
    for batch_idx in range(batch_size):
        boxes = boxes_batch[batch_idx]
        labels = labels_batch[batch_idx]
        
        if len(boxes) == 0:
            continue
        
        # Create grid of point locations
        for h in range(H):
            for w in range(W):
                # Point location (center of cell)
                px = (w + 0.5) * stride
                py = (h + 0.5) * stride
                
                # Check which boxes contain this point
                for box_idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.tolist()
                    
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        # Point is inside box
                        l_dist = px - x1
                        t_dist = py - y1
                        r_dist = x2 - px
                        b_dist = y2 - py
                        
                        # Compute centerness (avoid division by zero)
                        lr_ratio = min(l_dist, r_dist) / (max(l_dist, r_dist) + 1e-6)
                        tb_ratio = min(t_dist, b_dist) / (max(t_dist, b_dist) + 1e-6)
                        centerness = float(np.sqrt(lr_ratio * tb_ratio))
                        
                        # Use highest centerness box if multiple
                        if centerness > centerness_targets[batch_idx, 0, h, w].item():
                            cls_targets[batch_idx, h, w] = labels[box_idx].item() + 1  # +1 for background=0
                            reg_targets[batch_idx, :, h, w] = torch.tensor([l_dist, t_dist, r_dist, b_dist], device=device)
                            centerness_targets[batch_idx, 0, h, w] = centerness
    
    return cls_targets, reg_targets, centerness_targets


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for classification with class imbalance.
    """
    num_classes = pred.shape[1]
    
    # Flatten predictions and targets
    pred = pred.permute(0, 2, 3, 1).reshape(-1, num_classes)
    target = target.reshape(-1)
    
    # One-hot encode targets (background = class 0)
    target_one_hot = F.one_hot(target, num_classes + 1)[:, 1:]  # Skip background
    
    # Compute probabilities
    prob = torch.sigmoid(pred)
    
    # Focal loss
    pt = prob * target_one_hot + (1 - prob) * (1 - target_one_hot)
    focal_weight = (alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)) * (1 - pt).pow(gamma)
    
    loss = F.binary_cross_entropy_with_logits(pred, target_one_hot.float(), reduction='none')
    loss = (focal_weight * loss).sum(dim=1).mean()
    
    return loss


def train_epoch(model, dataloader, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, boxes_batch, labels_batch in pbar:
        images = images.to(device)
        boxes_batch = [b.to(device) for b in boxes_batch]
        labels_batch = [l.to(device) for l in labels_batch]
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(images)
            
            # Compute targets
            cls_targets, reg_targets, centerness_targets = compute_targets(
                boxes_batch, labels_batch, outputs['feat_size']
            )
            
            # Classification loss (focal loss)
            cls_loss = focal_loss(outputs['cls_logits'], cls_targets)
            
            # Regression loss (only on positive locations)
            pos_mask = cls_targets > 0
            if pos_mask.any():
                reg_pred = outputs['reg_pred'].permute(0, 2, 3, 1)[pos_mask]
                reg_target = reg_targets.permute(0, 2, 3, 1)[pos_mask]
                reg_loss = F.smooth_l1_loss(reg_pred, reg_target)
                
                # Centerness loss
                cent_pred = outputs['centerness'].permute(0, 2, 3, 1)[pos_mask]
                cent_target = centerness_targets.permute(0, 2, 3, 1)[pos_mask]
                cent_loss = F.binary_cross_entropy_with_logits(cent_pred, cent_target)
            else:
                reg_loss = torch.tensor(0.0, device=device)
                cent_loss = torch.tensor(0.0, device=device)
            
            loss = cls_loss + reg_loss + cent_loss
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'reg': f'{reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0:.4f}'
        })
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'cls_loss': total_cls_loss / n,
        'reg_loss': total_reg_loss / n
    }


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    for images, boxes_batch, labels_batch in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        boxes_batch = [b.to(device) for b in boxes_batch]
        labels_batch = [l.to(device) for l in labels_batch]
        
        outputs = model(images)
        
        cls_targets, reg_targets, centerness_targets = compute_targets(
            boxes_batch, labels_batch, outputs['feat_size']
        )
        
        cls_loss = focal_loss(outputs['cls_logits'], cls_targets)
        
        pos_mask = cls_targets > 0
        if pos_mask.any():
            reg_pred = outputs['reg_pred'].permute(0, 2, 3, 1)[pos_mask]
            reg_target = reg_targets.permute(0, 2, 3, 1)[pos_mask]
            reg_loss = F.smooth_l1_loss(reg_pred, reg_target)
        else:
            reg_loss = 0.0
        
        loss = cls_loss.item() + (reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss)
        total_loss += loss
    
    return total_loss / len(dataloader)


def export_to_onnx(model, output_path, device='cpu'):
    """Export model to ONNX format"""
    
    # Create a wrapper that only returns tensors (not dict with integers)
    class ONNXWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            outputs = self.model(x)
            # Return only tensor outputs (not feat_size which contains ints)
            return outputs['cls_logits'], outputs['reg_pred'], outputs['centerness']
    
    wrapper = ONNXWrapper(model)
    wrapper.eval()
    wrapper = wrapper.to(device)
    
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Use legacy exporter for better compatibility
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['cls_logits', 'reg_pred', 'centerness'],
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter
        dynamic_axes={
            'image': {0: 'batch_size'},
            'cls_logits': {0: 'batch_size'},
            'reg_pred': {0: 'batch_size'},
            'centerness': {0: 'batch_size'}
        }
    )
    
    print(f"[Export] Model saved to: {output_path}")
    print(f"[Export] File size: {os.path.getsize(output_path) / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Train Fashion Detector on DeepFashion2')
    parser.add_argument('--data_root', type=str, 
                        default='deepfashion2_training/data/deepfashion2',
                        help='Path to DeepFashion2 dataset')
    parser.add_argument('--output_dir', type=str, 
                        default='trained_models/fashion_detector_v2',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_train', type=int, default=5000, help='Max training samples')
    parser.add_argument('--max_val', type=int, default=1000, help='Max validation samples')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', 
                        help='Backbone model')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🎯 Fashion Detector Training (v2 - Ground Truth)")
    print("=" * 60)
    print(f"[Config] Device: {device}")
    print(f"[Config] Backbone: {args.backbone}")
    print(f"[Config] Epochs: {args.epochs}")
    print(f"[Config] Batch size: {args.batch_size}")
    print(f"[Config] Learning rate: {args.lr}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("\n" + "=" * 60)
    print("📂 Loading Datasets")
    print("=" * 60)
    
    train_dataset = DeepFashion2Dataset(
        args.data_root, split='train', transform=transform, max_samples=args.max_train
    )
    
    # Try to load validation set, fall back to train split if not available
    try:
        val_dataset = DeepFashion2Dataset(
            args.data_root, split='validation', transform=transform, max_samples=args.max_val
        )
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty")
    except (ValueError, Exception) as e:
        print(f"[Warning] Could not load validation set: {e}")
        print("[Warning] Using last 10% of training data for validation")
        # Split training data for validation
        total = len(train_dataset.samples)
        val_size = int(total * 0.1)
        val_dataset = DeepFashion2Dataset.__new__(DeepFashion2Dataset)
        val_dataset.transform = transform
        val_dataset.samples = train_dataset.samples[-val_size:]
        train_dataset.samples = train_dataset.samples[:-val_size]
        print(f"[Dataset] Train: {len(train_dataset.samples)} samples, Val: {len(val_dataset.samples)} samples")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    
    # Create model
    print("\n" + "=" * 60)
    print("🔧 Creating Model")
    print("=" * 60)
    
    model = FashionDetector(num_classes=NUM_CLASSES, backbone=args.backbone)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params / 1e6:.2f}M")
    print(f"[Model] Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Training loop
    print("\n" + "=" * 60)
    print("🚀 Starting Training")
    print("=" * 60)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, scaler)
        history['train_loss'].append(train_metrics['loss'])
        
        # Validate
        val_loss = validate(model, val_loader, device)
        history['val_loss'].append(val_loss)
        
        # Update scheduler
        scheduler.step()
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {train_metrics['loss']:.4f} (cls: {train_metrics['cls_loss']:.4f}, reg: {train_metrics['reg_loss']:.4f})")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'categories': FASHION_CATEGORIES
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"   ✅ Best model saved! (Val Loss: {val_loss:.4f})")
        
        # Save checkpoint every epoch
        torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth'))
    
    # Export to ONNX
    print("\n" + "=" * 60)
    print("📦 Exporting to ONNX")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    onnx_path = os.path.join(args.output_dir, 'fashion_detector.onnx')
    export_to_onnx(model, onnx_path, device='cpu')
    
    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {args.output_dir}")
    print(f"   ONNX exported to: {onnx_path}")


if __name__ == '__main__':
    main()
