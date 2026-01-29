#!/usr/bin/env python3
"""
DETR Fashion Detection Training
Fine-tune DETR on DeepFashion2 for fashion item detection.

This approach uses DETR (DEtection TRansformer) which:
- Supports actual supervised training with loss
- Works well with fashion categories
- Can be fine-tuned efficiently

Expected training time: 2-4 hours on RTX 5060 Ti
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Install dependencies if needed
try:
    from transformers import (
        DetrForObjectDetection,
        DetrImageProcessor,
        get_cosine_schedule_with_warmup
    )
except ImportError:
    print("Installing transformers...")
    os.system(f"{sys.executable} -m pip install transformers>=4.35.0")
    from transformers import (
        DetrForObjectDetection,
        DetrImageProcessor,
        get_cosine_schedule_with_warmup
    )

# Fashion categories (matching DeepFashion2)
CATEGORIES = [
    'short sleeve top', 'long sleeve top', 'short sleeve outwear', 
    'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers',
    'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress'
]

ID2LABEL = {i: cat for i, cat in enumerate(CATEGORIES)}
LABEL2ID = {cat: i for i, cat in enumerate(CATEGORIES)}


class DeepFashion2Dataset(Dataset):
    """Dataset for DETR training on DeepFashion2."""
    
    def __init__(self, data_root, split='train', processor=None, max_samples=None):
        self.data_root = Path(data_root)
        self.split = split
        self.processor = processor
        
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
                                    # DeepFashion2 bbox format: [x1, y1, x2, y2]
                                    x1, y1, x2, y2 = bbox[:4]
                                    if x2 > x1 and y2 > y1:  # Valid box
                                        boxes.append([x1, y1, x2, y2])
                                        labels.append(cat_id - 1)  # 0-indexed
                        except:
                            continue
                    
                    if boxes:  # Only add samples with valid annotations
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
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['img_path']).convert('RGB')
        width, height = img.size
        
        # Convert boxes to COCO format (relative coordinates)
        boxes = []
        for box in sample['boxes']:
            x1, y1, x2, y2 = box
            # Convert to relative [cx, cy, w, h] format
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            boxes.append([cx, cy, w, h])
        
        labels = sample['labels']
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'class_labels': torch.tensor(labels, dtype=torch.int64),
        }
        
        return {
            'image': img,  # Return PIL image for batch processing
            'labels': target
        }


def collate_fn_with_processor(processor):
    """Create a collate function with access to the processor for proper padding."""
    def collate_fn(batch):
        """Custom collate function for DETR with proper padding."""
        images = [item['image'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Use processor to handle variable-size images with proper padding
        encoding = processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding.get('pixel_mask', None),
            'labels': labels
        }
    return collate_fn


def main():
    parser = argparse.ArgumentParser(description='DETR Fashion Detection Training')
    parser.add_argument('--data_root', type=str, 
                       default='deepfashion2_training/data/deepfashion2',
                       help='Path to DeepFashion2 data')
    parser.add_argument('--output', type=str, 
                       default='trained_models/detr_fashion',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Limit training samples (None for all)')
    parser.add_argument('--val_samples', type=int, default=1000,
                       help='Validation samples')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 DETR Fashion Detection Training")
    print("=" * 60)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Setup] Using device: {device}")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[Setup] GPU: {gpu_name}")
        print(f"[Setup] VRAM: {vram:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load model and processor
    print(f"\n[Model] Loading DETR...")
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True  # Required for different num_classes
    )
    model = model.to(device)
    
    # Freeze backbone for faster training
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Prepare datasets
    print("\n[Data] Preparing datasets...")
    
    train_ds = DeepFashion2Dataset(
        args.data_root, 
        split='train',
        processor=processor,
        max_samples=args.max_samples
    )
    
    try:
        val_ds = DeepFashion2Dataset(
            args.data_root, 
            split='validation',
            processor=processor,
            max_samples=args.val_samples
        )
    except ValueError:
        print("[Data] No validation set found, using last 10% of training data")
        val_size = len(train_ds) // 10
        train_ds.samples, val_samples = train_ds.samples[:-val_size], train_ds.samples[-val_size:]
        val_ds = train_ds.__class__.__new__(train_ds.__class__)
        val_ds.processor = processor
        val_ds.samples = val_samples
    
    # Create collate function with processor for proper padding
    collate_fn = collate_fn_with_processor(processor)
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    val_dl = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    ) if len(val_ds) > 0 else None
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        weight_decay=1e-4
    )
    
    total_steps = args.epochs * len(train_dl)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(500, total_steps // 10),
        num_training_steps=total_steps
    )
    
    scaler = GradScaler('cuda')
    
    # Training info
    print(f"\n[Training] Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Training samples: {len(train_ds)}")
    print(f"  - Validation samples: {len(val_ds) if val_ds else 0}")
    print(f"  - Steps per epoch: {len(train_dl)}")
    print(f"  - Learning rate: {args.lr}")
    print("-" * 60)
    
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            try:
                pixel_values = batch['pixel_values'].to(device)
                pixel_mask = batch['pixel_mask'].to(device) if batch['pixel_mask'] is not None else None
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]
                
                with autocast('cuda', dtype=torch.float16):
                    outputs = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{epoch_loss/num_batches:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
            except Exception as e:
                if 'out of memory' in str(e).lower():
                    print(f"\n[Warning] OOM, clearing cache...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                else:
                    print(f"\n[Warning] Error: {e}")
                continue
        
        avg_train_loss = epoch_loss / max(num_batches, 1)
        history['train_loss'].append(avg_train_loss)
        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_dl:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dl, desc="Validating", leave=False):
                    try:
                        pixel_values = batch['pixel_values'].to(device)
                        pixel_mask = batch['pixel_mask'].to(device) if batch['pixel_mask'] is not None else None
                        labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]
                        
                        outputs = model(
                            pixel_values=pixel_values,
                            pixel_mask=pixel_mask,
                            labels=labels
                        )
                        val_loss += outputs.loss.item()
                        val_batches += 1
                    except:
                        continue
            
            avg_val_loss = val_loss / max(val_batches, 1)
            history['val_loss'].append(avg_val_loss)
            print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")
            current_loss = avg_val_loss
        else:
            current_loss = avg_train_loss
        
        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            model.save_pretrained(f"{args.output}/best")
            processor.save_pretrained(f"{args.output}/best")
            print(f"[Epoch {epoch+1}] ✅ Saved best model (loss: {best_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            model.save_pretrained(f"{args.output}/epoch_{epoch+1}")
            processor.save_pretrained(f"{args.output}/epoch_{epoch+1}")
    
    # Save final model
    model.save_pretrained(f"{args.output}/final")
    processor.save_pretrained(f"{args.output}/final")
    
    # Save config
    config = {
        'model_type': 'detr-fashion',
        'categories': CATEGORIES,
        'id2label': ID2LABEL,
        'label2id': LABEL2ID,
        'best_loss': best_loss,
        'epochs': args.epochs,
        'history': history
    }
    
    with open(f"{args.output}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"  - Best Loss: {best_loss:.4f}")
    print(f"  - Model saved to: {args.output}")
    
    if device == 'cuda':
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  - Peak GPU memory: {max_mem:.2f} GB")


if __name__ == '__main__':
    main()
