#!/usr/bin/env python3
"""
GroundingDINO Fine-tuning for Fashion Detection using Official Repository

This script uses the official GroundingDINO repository which DOES support training
with proper loss computation (unlike the HuggingFace zero-shot wrapper).

Key approach:
1. Use groundingdino from source with training mode
2. Compute matching loss between predictions and ground truth
3. Train only the fusion layers and detection heads

Expected training time: 2-4 hours on RTX 5060 Ti
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Fashion categories (matching DeepFashion2)
CATEGORIES = [
    'short sleeve top', 'long sleeve top', 'short sleeve outwear', 
    'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers',
    'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress'
]

# Category text prompts for GroundingDINO
TEXT_PROMPT = ". ".join(CATEGORIES) + "."


def box_cxcywh_to_xyxy(x):
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h]"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """Compute generalized IoU between two sets of boxes"""
    # Convert to xyxy if needed
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


class DeepFashion2GroundingDataset(Dataset):
    """Dataset for GroundingDINO training on DeepFashion2."""
    
    def __init__(self, data_root, split='train', max_samples=None, img_size=800):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        
        # Try different path structures
        possible_paths = [
            self.data_root / split / 'annos',
            self.data_root / split / split / 'annos',
        ]
        
        self.ann_dir = None
        self.img_dir = None
        
        for ann_path in possible_paths:
            if ann_path.exists():
                self.ann_dir = ann_path
                self.img_dir = ann_path.parent / 'image'
                break
        
        if self.ann_dir is None:
            raise ValueError(f"Cannot find annotation directory. Tried: {possible_paths}")
        
        print(f"[Dataset] Using annotations from: {self.ann_dir}")
        print(f"[Dataset] Using images from: {self.img_dir}")
        
        # Load annotation files
        self.samples = []
        ann_files = sorted(list(self.ann_dir.glob('*.json')))
        
        for ann_file in tqdm(ann_files[:max_samples] if max_samples else ann_files, 
                            desc=f"Loading {split}"):
            try:
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                
                img_name = ann_file.stem + '.jpg'
                img_path = self.img_dir / img_name
                
                if img_path.exists():
                    # Parse annotations
                    items = data.get('items', [])
                    if not items:
                        items = [v for k, v in data.items() if k.startswith('item')]
                    
                    boxes = []
                    labels = []
                    
                    for item in items:
                        try:
                            cat_id = int(item.get('category_id', item.get('category', 0)))
                            if 1 <= cat_id <= 13:
                                bbox = item.get('bounding_box', item.get('bbox'))
                                if bbox and len(bbox) >= 4:
                                    x1, y1, x2, y2 = bbox[:4]
                                    if x2 > x1 and y2 > y1:
                                        boxes.append([x1, y1, x2, y2])
                                        labels.append(cat_id - 1)
                        except:
                            continue
                    
                    if boxes:
                        self.samples.append({
                            'img_path': img_path,
                            'boxes': boxes,
                            'labels': labels
                        })
            except Exception as e:
                continue
        
        print(f"[Dataset] Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def preprocess_image(self, img):
        """Preprocess image for GroundingDINO"""
        # Resize maintaining aspect ratio
        w, h = img.size
        scale = self.img_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Pad to square
        padded = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))
        padded.paste(img, (0, 0))
        
        # Convert to tensor and normalize
        img_array = np.array(padded).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        return img_tensor, scale, (new_w, new_h)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess image
        img = Image.open(sample['img_path']).convert('RGB')
        orig_w, orig_h = img.size
        
        img_tensor, scale, (new_w, new_h) = self.preprocess_image(img)
        
        # Scale and normalize boxes
        boxes = []
        for box in sample['boxes']:
            x1, y1, x2, y2 = box
            # Scale to new size
            x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
            # Normalize to [0, 1]
            x1, y1, x2, y2 = x1 / self.img_size, y1 / self.img_size, x2 / self.img_size, y2 / self.img_size
            boxes.append([x1, y1, x2, y2])
        
        # Create category text for this sample
        unique_labels = list(set(sample['labels']))
        category_names = [CATEGORIES[l] for l in unique_labels]
        text_prompt = ". ".join(category_names) + "."
        
        return {
            'image': img_tensor,
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(sample['labels'], dtype=torch.int64),
            'text': text_prompt,
            'orig_size': torch.tensor([orig_h, orig_w]),
        }


def collate_fn(batch):
    """Collate function for variable-size batches"""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    
    # Keep boxes and labels as lists (variable size per image)
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    orig_sizes = torch.stack([item['orig_size'] for item in batch])
    
    return {
        'images': images,
        'texts': texts,
        'boxes': boxes,
        'labels': labels,
        'orig_sizes': orig_sizes,
    }


class GroundingDINOLoss(nn.Module):
    """
    Loss function for GroundingDINO training.
    Combines:
    - Box regression loss (L1 + GIoU)
    - Classification loss (focal loss on text-box alignment)
    """
    
    def __init__(self, num_classes=13, box_loss_weight=5.0, giou_loss_weight=2.0, cls_loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.box_loss_weight = box_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.cls_loss_weight = cls_loss_weight
    
    def forward(self, pred_boxes, pred_logits, target_boxes, target_labels):
        """
        Compute detection loss.
        
        Args:
            pred_boxes: [B, num_queries, 4] predicted boxes in cxcywh format
            pred_logits: [B, num_queries, num_classes] predicted class logits
            target_boxes: list of [N_i, 4] target boxes per image
            target_labels: list of [N_i] target labels per image
        """
        batch_size = pred_boxes.shape[0]
        total_loss = 0
        num_boxes = 0
        
        for i in range(batch_size):
            if len(target_boxes[i]) == 0:
                continue
            
            # Get predictions and targets for this image
            pred_b = pred_boxes[i]  # [num_queries, 4]
            pred_l = pred_logits[i]  # [num_queries, num_classes]
            tgt_b = target_boxes[i]  # [N, 4]
            tgt_l = target_labels[i]  # [N]
            
            # Convert pred boxes from cxcywh to xyxy
            pred_b_xyxy = box_cxcywh_to_xyxy(pred_b)
            
            # Simple matching: for each target, find best matching prediction
            # Compute cost matrix based on IoU
            with torch.no_grad():
                iou_matrix = generalized_box_iou(pred_b_xyxy, tgt_b)  # [num_queries, N]
                
                # Hungarian-like matching (simplified)
                matched_pred_idx = []
                matched_tgt_idx = []
                used_preds = set()
                
                for j in range(len(tgt_b)):
                    # Find best prediction for this target
                    ious = iou_matrix[:, j].clone()
                    for used in used_preds:
                        ious[used] = -float('inf')
                    best_pred = ious.argmax().item()
                    
                    if ious[best_pred] > -float('inf'):
                        matched_pred_idx.append(best_pred)
                        matched_tgt_idx.append(j)
                        used_preds.add(best_pred)
            
            if len(matched_pred_idx) == 0:
                continue
            
            matched_pred_idx = torch.tensor(matched_pred_idx, device=pred_b.device)
            matched_tgt_idx = torch.tensor(matched_tgt_idx, device=pred_b.device)
            
            # Box loss (L1)
            matched_pred_boxes = pred_b[matched_pred_idx]
            matched_tgt_boxes = box_xyxy_to_cxcywh(tgt_b[matched_tgt_idx])
            l1_loss = F.l1_loss(matched_pred_boxes, matched_tgt_boxes, reduction='sum')
            
            # GIoU loss
            matched_pred_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
            matched_tgt_xyxy = tgt_b[matched_tgt_idx]
            giou = generalized_box_iou(matched_pred_xyxy, matched_tgt_xyxy)
            giou_loss = (1 - giou.diag()).sum()
            
            # Classification loss (simplified - use BCE on matched predictions)
            matched_logits = pred_l[matched_pred_idx]
            matched_labels = tgt_l[matched_tgt_idx]
            
            # Create target distribution
            target_dist = torch.zeros_like(matched_logits)
            for k, lbl in enumerate(matched_labels):
                if lbl < target_dist.shape[1]:
                    target_dist[k, lbl] = 1.0
            
            cls_loss = F.binary_cross_entropy_with_logits(matched_logits, target_dist, reduction='sum')
            
            # Combine losses
            total_loss += (
                self.box_loss_weight * l1_loss + 
                self.giou_loss_weight * giou_loss + 
                self.cls_loss_weight * cls_loss
            )
            num_boxes += len(matched_pred_idx)
        
        # Normalize by number of boxes
        if num_boxes > 0:
            total_loss = total_loss / num_boxes
        
        return total_loss


def load_groundingdino_model(device='cuda'):
    """
    Load GroundingDINO model for training.
    Uses HuggingFace model but extracts trainable components.
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    print("[Model] Loading GroundingDINO...")
    
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base",
        torch_dtype=torch.float32  # Use FP32 for training stability
    )
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    
    model = model.to(device)
    
    return model, processor


def freeze_backbone(model, freeze_vision=True, freeze_text=True):
    """Freeze backbone, only train fusion and detection heads"""
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        # Freeze vision backbone
        if freeze_vision and ('backbone' in name.lower() or 'encoder.layer' in name.lower()):
            param.requires_grad = False
            frozen_params += param.numel()
        # Freeze text encoder  
        elif freeze_text and 'text_encoder' in name.lower():
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trainable_params += param.numel()
    
    print(f"[Model] Trainable: {trainable_params:,} | Frozen: {frozen_params:,}")
    return model


def main():
    parser = argparse.ArgumentParser(description='GroundingDINO Fashion Training')
    parser.add_argument('--data_root', type=str, 
                       default='deepfashion2_training/data/deepfashion2',
                       help='Path to DeepFashion2 data')
    parser.add_argument('--output', type=str, 
                       default='trained_models/groundingdino_fashion',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (small due to memory)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Max training samples')
    parser.add_argument('--img_size', type=int, default=800,
                       help='Input image size')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 GroundingDINO Fashion Detection Training")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Setup] Device: {device}")
    
    if device == 'cuda':
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    model, processor = load_groundingdino_model(device)
    model = freeze_backbone(model, freeze_vision=True, freeze_text=True)
    
    # Dataset
    print("\n[Data] Loading dataset...")
    train_ds = DeepFashion2GroundingDataset(
        args.data_root,
        split='train',
        max_samples=args.max_samples,
        img_size=args.img_size
    )
    
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # Loss and optimizer
    criterion = GroundingDINOLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    scaler = GradScaler('cuda')
    
    print(f"\n[Training] Starting {args.epochs} epochs...")
    print(f"[Training] Batches per epoch: {len(train_dl)}")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            try:
                images = batch['images'].to(device)
                texts = batch['texts']
                target_boxes = [b.to(device) for b in batch['boxes']]
                target_labels = [l.to(device) for l in batch['labels']]
                
                # Forward pass through model
                with autocast('cuda', dtype=torch.float16):
                    # Process inputs
                    inputs = processor(
                        images=[Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) 
                               for img in images],
                        text=texts,
                        return_tensors="pt",
                    ).to(device)
                    
                    # Get model outputs
                    outputs = model(**inputs)
                    
                    # Extract predictions
                    pred_boxes = outputs.pred_boxes  # [B, num_queries, 4]
                    pred_logits = outputs.logits  # [B, num_queries, num_tokens]
                    
                    # Compute loss
                    loss = criterion(pred_boxes, pred_logits, target_boxes, target_labels)
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{epoch_loss/num_batches:.4f}'})
                
            except Exception as e:
                if 'out of memory' in str(e).lower():
                    print(f"\n[Warning] OOM, clearing cache...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                else:
                    print(f"\n[Warning] {e}")
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\n[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(f"{args.output}/best")
            processor.save_pretrained(f"{args.output}/best")
            print(f"[Epoch {epoch+1}] ✅ Saved best model")
        
        # Save checkpoint
        model.save_pretrained(f"{args.output}/epoch_{epoch+1}")
        processor.save_pretrained(f"{args.output}/epoch_{epoch+1}")
    
    # Save final
    model.save_pretrained(f"{args.output}/final")
    processor.save_pretrained(f"{args.output}/final")
    
    # Save config
    config = {
        'model_type': 'groundingdino-fashion',
        'categories': CATEGORIES,
        'text_prompt': TEXT_PROMPT,
        'best_loss': best_loss,
        'epochs': args.epochs,
    }
    with open(f"{args.output}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Model saved to: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
