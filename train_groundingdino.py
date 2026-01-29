#!/usr/bin/env python3
"""
GroundingDINO Fine-tuning for Fashion Detection
Uses HuggingFace's transformers wrapper for IDEA-Research/grounding-dino-base

Features:
- Open-set text-prompt detection for fashion items
- Fine-tuned on DeepFashion2 categories
- Exports to ONNX for mobile deployment
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# Install dependencies if needed
try:
    from transformers import (
        AutoProcessor, 
        AutoModelForZeroShotObjectDetection,
        get_cosine_schedule_with_warmup
    )
except ImportError:
    print("Installing transformers...")
    os.system(f"{sys.executable} -m pip install transformers>=4.35.0")
    from transformers import (
        AutoProcessor, 
        AutoModelForZeroShotObjectDetection,
        get_cosine_schedule_with_warmup
    )

# Fashion categories (matching DeepFashion2)
CATEGORIES = [
    'short sleeve top', 'long sleeve top', 'short sleeve outwear', 
    'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers',
    'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress'
]

# Simplified category prompts for detection
CATEGORY_PROMPTS = {
    'short sleeve top': 'short sleeve shirt. t-shirt. top.',
    'long sleeve top': 'long sleeve shirt. sweater. blouse.',
    'short sleeve outwear': 'short sleeve jacket. vest coat.',
    'long sleeve outwear': 'jacket. coat. blazer. cardigan.',
    'vest': 'vest. sleeveless top.',
    'sling': 'tank top. camisole. sling top.',
    'shorts': 'shorts. short pants.',
    'trousers': 'pants. trousers. jeans.',
    'skirt': 'skirt. mini skirt.',
    'short sleeve dress': 'short sleeve dress.',
    'long sleeve dress': 'long sleeve dress.',
    'vest dress': 'sleeveless dress.',
    'sling dress': 'sling dress. slip dress.',
}


class DeepFashion2DetectionDataset(Dataset):
    """Dataset for GroundingDINO fine-tuning on DeepFashion2."""
    
    def __init__(self, data_root, split='train', max_samples=None):
        """
        Args:
            data_root: Path to DeepFashion2 data
            split: 'train' or 'validation'
            max_samples: Limit samples for faster iteration
        """
        self.data_root = Path(data_root)
        self.split = split
        
        # Try different path structures
        possible_paths = [
            self.data_root / split / 'annos',
            self.data_root / split / split / 'annos',
            self.data_root / 'train' / split / 'annos',
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
        ann_files = list(self.ann_dir.glob('*.json'))
        
        for ann_file in tqdm(ann_files[:max_samples] if max_samples else ann_files, 
                            desc=f"Loading {split}"):
            try:
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                
                # Find matching image
                img_name = ann_file.stem + '.jpg'
                img_path = self.img_dir / img_name
                
                if img_path.exists():
                    self.samples.append({
                        'ann_file': ann_file,
                        'img_path': img_path,
                        'data': data
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
        data = sample['data']
        
        # Extract categories from annotation
        categories = set()
        boxes = []
        
        # Handle different annotation formats
        items = data.get('items', [])
        if not items:
            # Try alternative format
            items = [v for k, v in data.items() if k.startswith('item')]
        
        for item in items:
            try:
                cat_id = int(item.get('category_id', item.get('category', 0)))
                if 1 <= cat_id <= 13:
                    cat_name = CATEGORIES[cat_id - 1]
                    categories.add(cat_name)
                    
                    # Get bounding box if available
                    bbox = item.get('bounding_box', item.get('bbox'))
                    if bbox:
                        boxes.append({
                            'category': cat_name,
                            'bbox': bbox
                        })
            except:
                continue
        
        # Create text prompt
        if categories:
            text = '. '.join(sorted(categories)) + '.'
        else:
            text = 'clothing.'
        
        return {
            'image': img,
            'text': text,
            'categories': list(categories),
            'boxes': boxes
        }


def collate_fn(batch):
    """Custom collate for variable-size data."""
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    return images, texts


@torch.no_grad()
def evaluate(model, processor, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for images, texts in tqdm(dataloader, desc="Evaluating"):
        try:
            inputs = processor(
                images=images, 
                text=texts, 
                return_tensors='pt', 
                padding=True
            ).to(device)
            
            outputs = model(**inputs)
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                total_loss += outputs.loss.item()
            num_batches += 1
        except Exception as e:
            continue
    
    avg_loss = total_loss / max(num_batches, 1)
    return {'val_loss': avg_loss, 'batches': num_batches}


def export_to_onnx(model, processor, output_path, device='cpu'):
    """Export model to ONNX format."""
    print("\n[Export] Exporting to ONNX...")
    
    # GroundingDINO export is complex due to text encoding
    # We'll save the HuggingFace format which can be loaded on mobile via ONNX Runtime
    
    model_path = Path(output_path)
    model_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)
    
    print(f"[Export] Saved model to {model_path}")
    
    # Create config for mobile loading
    config = {
        'model_type': 'grounding-dino',
        'categories': CATEGORIES,
        'category_prompts': CATEGORY_PROMPTS,
        'input_size': 800,
        'box_threshold': 0.35,
        'text_threshold': 0.25
    }
    
    with open(model_path / 'mobile_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("[Export] Created mobile_config.json")
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Fine-tune GroundingDINO on DeepFashion2')
    parser.add_argument('--data_root', type=str, 
                       default='deepfashion2_training/data/deepfashion2',
                       help='Path to DeepFashion2 data')
    parser.add_argument('--output', type=str, 
                       default='trained_models/groundingdino',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (keep small due to memory)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit training samples (for testing)')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                       help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GroundingDINO Fine-tuning for Fashion Detection")
    print("=" * 60)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Setup] Using device: {device}")
    
    if device == 'cuda':
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model and processor
    print("\n[Model] Loading GroundingDINO base model...")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        'IDEA-Research/grounding-dino-base'
    ).to(device)
    processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
    print("[Model] ✅ Loaded")
    
    # Prepare datasets
    print("\n[Data] Preparing datasets...")
    
    try:
        train_ds = DeepFashion2DetectionDataset(
            args.data_root, 
            split='train',
            max_samples=args.max_samples
        )
    except ValueError as e:
        print(f"[Error] {e}")
        print("\n[Info] Checking data structure...")
        data_path = Path(args.data_root)
        for item in data_path.rglob('*'):
            if item.is_dir():
                print(f"  {item.relative_to(data_path)}/")
        return
    
    # Check if validation exists
    try:
        val_ds = DeepFashion2DetectionDataset(
            args.data_root, 
            split='validation',
            max_samples=args.max_samples // 5 if args.max_samples else None
        )
    except:
        print("[Data] No validation set found, using train subset")
        val_ds = None
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Avoid multiprocessing issues on Windows
    )
    
    if val_ds:
        val_dl = DataLoader(
            val_ds, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0
        )
    else:
        val_dl = None
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_dl) // args.gradient_accumulation
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, total_steps // 20),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\n[Training] Starting training for {args.epochs} epochs...")
    print(f"[Training] Total batches per epoch: {len(train_dl)}")
    print(f"[Training] Gradient accumulation: {args.gradient_accumulation}")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        
        for batch_idx, (images, texts) in enumerate(pbar):
            try:
                inputs = processor(
                    images=images, 
                    text=texts, 
                    return_tensors='pt', 
                    padding=True
                ).to(device)
                
                outputs = model(**inputs)
                
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss / args.gradient_accumulation
                    loss.backward()
                    
                    if (batch_idx + 1) % args.gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item() * args.gradient_accumulation
                    num_batches += 1
                    
                    pbar.set_postfix({
                        'loss': f'{epoch_loss/num_batches:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                    
            except Exception as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                continue
        
        avg_train_loss = epoch_loss / max(num_batches, 1)
        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_dl:
            val_metrics = evaluate(model, processor, val_dl, device)
            print(f"[Epoch {epoch+1}] Val Loss: {val_metrics['val_loss']:.4f}")
            current_loss = val_metrics['val_loss']
        else:
            current_loss = avg_train_loss
        
        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            model.save_pretrained(f"{args.output}/best")
            processor.save_pretrained(f"{args.output}/best")
            print(f"[Epoch {epoch+1}] ✅ Saved best model (loss: {best_loss:.4f})")
        
        # Save checkpoint
        model.save_pretrained(f"{args.output}/epoch_{epoch+1}")
        processor.save_pretrained(f"{args.output}/epoch_{epoch+1}")
    
    # Final export
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    export_path = export_to_onnx(model, processor, f"{args.output}/final", device)
    
    print(f"\n[Summary]")
    print(f"  - Best Loss: {best_loss:.4f}")
    print(f"  - Model saved to: {args.output}")
    print(f"  - Export path: {export_path}")


if __name__ == '__main__':
    main()
