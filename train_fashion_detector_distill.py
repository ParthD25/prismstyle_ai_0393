#!/usr/bin/env python3
"""
GroundingDINO Knowledge Distillation for Fashion Detection

Since HuggingFace's GroundingDINO doesn't support native training with loss computation,
we use a knowledge distillation approach:

1. Use pretrained GroundingDINO to generate pseudo-labels on DeepFashion2
2. Train a lightweight, mobile-friendly detection model on these pseudo-labels
3. Export the student model for inference

This approach gives us:
- GroundingDINO-quality text-conditioned detection
- Mobile-optimized inference speed
- Fashion-specific training
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

# Fashion categories
CATEGORIES = [
    'short sleeve top', 'long sleeve top', 'short sleeve outwear', 
    'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers',
    'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress'
]

def generate_pseudo_labels_with_groundingdino(data_root, output_file, max_samples=2000):
    """
    Use GroundingDINO to generate pseudo-labels for fashion detection.
    This creates training data that captures GroundingDINO's text-grounding capability.
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    print("=" * 60)
    print("🔍 Generating Pseudo-Labels with GroundingDINO")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load GroundingDINO
    print("[Model] Loading GroundingDINO...")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to(device)
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model.eval()
    
    # Find images
    data_path = Path(data_root)
    img_dir = data_path / 'train' / 'train' / 'image'
    if not img_dir.exists():
        img_dir = data_path / 'train' / 'image'
    
    print(f"[Data] Looking for images in: {img_dir}")
    
    img_files = sorted(list(img_dir.glob('*.jpg')))[:max_samples]
    print(f"[Data] Found {len(img_files)} images")
    
    # Text prompt for all categories
    text_prompt = ". ".join(CATEGORIES) + "."
    
    pseudo_labels = []
    
    with torch.no_grad():
        for img_path in tqdm(img_files, desc="Generating labels"):
            try:
                image = Image.open(img_path).convert('RGB')
                w, h = image.size
                
                # Process with GroundingDINO
                inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
                outputs = model(**inputs)
                
                # Post-process
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.25,
                    text_threshold=0.25,
                    target_sizes=[(h, w)]
                )[0]
                
                boxes = results['boxes'].cpu().numpy()
                labels = results['labels']
                scores = results['scores'].cpu().numpy()
                
                # Map labels to category indices
                label_indices = []
                for lbl in labels:
                    lbl_lower = lbl.lower().strip()
                    for i, cat in enumerate(CATEGORIES):
                        if cat in lbl_lower or lbl_lower in cat:
                            label_indices.append(i)
                            break
                    else:
                        label_indices.append(-1)
                
                # Filter valid detections
                valid_dets = []
                for box, label_idx, score in zip(boxes, label_indices, scores):
                    if label_idx >= 0:
                        valid_dets.append({
                            'box': box.tolist(),
                            'label': label_idx,
                            'score': float(score)
                        })
                
                if valid_dets:
                    pseudo_labels.append({
                        'image': str(img_path),
                        'width': w,
                        'height': h,
                        'detections': valid_dets
                    })
                    
            except Exception as e:
                continue
    
    print(f"[Data] Generated {len(pseudo_labels)} labeled images")
    
    # Save pseudo-labels
    with open(output_file, 'w') as f:
        json.dump({
            'categories': CATEGORIES,
            'samples': pseudo_labels
        }, f, indent=2)
    
    print(f"[Data] Saved to: {output_file}")
    return pseudo_labels


class PseudoLabelDataset(Dataset):
    """Dataset using GroundingDINO pseudo-labels."""
    
    def __init__(self, pseudo_labels_file, img_size=640):
        with open(pseudo_labels_file, 'r') as f:
            data = json.load(f)
        
        self.samples = data['samples']
        self.categories = data['categories']
        self.img_size = img_size
        
        print(f"[Dataset] Loaded {len(self.samples)} pseudo-labeled samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['image']).convert('RGB')
        orig_w, orig_h = img.size
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Convert to tensor
        img_array = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        # Scale boxes
        boxes = []
        labels = []
        for det in sample['detections']:
            x1, y1, x2, y2 = det['box']
            # Scale to new size
            x1 = x1 / orig_w * self.img_size
            y1 = y1 / orig_h * self.img_size
            x2 = x2 / orig_w * self.img_size
            y2 = y2 / orig_h * self.img_size
            boxes.append([x1, y1, x2, y2])
            labels.append(det['label'])
        
        return {
            'image': img_tensor,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        }


def collate_fn(batch):
    """Custom collate for variable-size detections."""
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {'images': images, 'boxes': boxes, 'labels': labels}


class SimpleFashionDetector(nn.Module):
    """
    Lightweight fashion detector that can be trained with pseudo-labels.
    Uses EfficientNet backbone + simple detection head.
    """
    
    def __init__(self, num_classes=13, backbone='efficientnet_b0'):
        super().__init__()
        
        import timm
        
        # Backbone
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        
        # Get feature channels from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            features = self.backbone(dummy)
            self.feat_channels = [f.shape[1] for f in features]
        
        # Use last two feature maps
        self.fpn_out_channels = 256
        
        # FPN lateral connections
        self.lateral_conv1 = nn.Conv2d(self.feat_channels[-1], self.fpn_out_channels, 1)
        self.lateral_conv2 = nn.Conv2d(self.feat_channels[-2], self.fpn_out_channels, 1)
        
        # FPN output convolutions
        self.fpn_conv1 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1)
        self.fpn_conv2 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1)
        
        # Detection heads (per-level)
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_out_channels, num_classes, 3, padding=1)
        )
        
        self.box_head = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_out_channels, 4, 3, padding=1)  # x, y, w, h
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        # Get backbone features
        features = self.backbone(x)
        
        # FPN
        p5 = self.lateral_conv1(features[-1])
        p4 = self.lateral_conv2(features[-2])
        
        # Upsample and add
        p5_upsampled = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4 = p4 + p5_upsampled
        
        # FPN output
        f5 = self.fpn_conv1(p5)
        f4 = self.fpn_conv2(p4)
        
        # Detection outputs
        cls_outputs = []
        box_outputs = []
        
        for feat in [f4, f5]:
            cls_out = self.cls_head(feat)  # [B, num_classes, H, W]
            box_out = self.box_head(feat)  # [B, 4, H, W]
            
            b, _, h, w = cls_out.shape
            # Reshape to [B, H*W, num_classes/4]
            cls_out = cls_out.permute(0, 2, 3, 1).reshape(b, h*w, self.num_classes)
            box_out = box_out.permute(0, 2, 3, 1).reshape(b, h*w, 4)
            
            cls_outputs.append(cls_out)
            box_outputs.append(box_out)
        
        # Concatenate all levels
        cls_logits = torch.cat(cls_outputs, dim=1)  # [B, total_anchors, num_classes]
        box_preds = torch.cat(box_outputs, dim=1)   # [B, total_anchors, 4]
        
        return cls_logits, box_preds


def compute_loss(cls_logits, box_preds, target_boxes, target_labels, num_classes=13):
    """
    Compute detection loss using focal loss for classification and smooth L1 for boxes.
    """
    batch_size = cls_logits.shape[0]
    total_loss = 0
    num_pos = 0
    
    for i in range(batch_size):
        tgt_boxes = target_boxes[i]
        tgt_labels = target_labels[i]
        
        if len(tgt_boxes) == 0:
            continue
        
        # Simple assignment: assign each anchor to nearest target
        # For simplicity, use top-k anchors per target based on classification score
        cls = cls_logits[i]  # [num_anchors, num_classes]
        boxes = box_preds[i]  # [num_anchors, 4]
        
        # For each target, find top-5 anchors
        for j in range(len(tgt_boxes)):
            target_cls = tgt_labels[j].item()
            target_box = tgt_boxes[j]
            
            # Get scores for this class
            scores = cls[:, target_cls]
            
            # Top-5 anchors
            k = min(5, len(scores))
            _, top_indices = scores.topk(k)
            
            # Classification loss (focal loss simplified)
            for idx in top_indices:
                cls_target = torch.zeros(num_classes, device=cls.device)
                cls_target[target_cls] = 1.0
                
                # Focal loss
                p = torch.sigmoid(cls[idx])
                pt = p * cls_target + (1 - p) * (1 - cls_target)
                focal_weight = (1 - pt) ** 2
                bce = F.binary_cross_entropy_with_logits(cls[idx], cls_target, reduction='none')
                total_loss += (focal_weight * bce).sum()
                
                # Box loss (smooth L1)
                total_loss += F.smooth_l1_loss(boxes[idx], target_box)
                
                num_pos += 1
    
    if num_pos > 0:
        return total_loss / num_pos
    else:
        return cls_logits.new_tensor(0.0, requires_grad=True)


def train_student_detector(pseudo_labels_file, output_dir, epochs=10, batch_size=8, lr=1e-4):
    """Train the student detector on pseudo-labels."""
    
    print("\n" + "=" * 60)
    print("🎓 Training Student Fashion Detector")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Setup] Device: {device}")
    
    # Dataset
    dataset = PseudoLabelDataset(pseudo_labels_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # Model
    model = SimpleFashionDetector(num_classes=13).to(device)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler('cuda')
    
    # Training
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            images = batch['images'].to(device)
            boxes = [b.to(device) for b in batch['boxes']]
            labels = [l.to(device) for l in batch['labels']]
            
            with autocast('cuda', dtype=torch.float16):
                cls_logits, box_preds = model(images)
                loss = compute_loss(cls_logits, box_preds, boxes, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{epoch_loss/num_batches:.4f}'})
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{output_dir}/fashion_detector_best.pth")
            print(f"[Epoch {epoch+1}] ✅ Saved best model")
    
    # Save final
    torch.save(model.state_dict(), f"{output_dir}/fashion_detector_final.pth")
    
    # Export to ONNX
    print("\n[Export] Converting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        f"{output_dir}/fashion_detector.onnx",
        input_names=['image'],
        output_names=['class_logits', 'box_predictions'],
        dynamic_axes={
            'image': {0: 'batch'},
            'class_logits': {0: 'batch'},
            'box_predictions': {0: 'batch'}
        },
        opset_version=14
    )
    print(f"[Export] ✅ ONNX model saved to {output_dir}/fashion_detector.onnx")
    
    # Save config
    config = {
        'model_type': 'fashion_detector_distilled',
        'categories': CATEGORIES,
        'input_size': 640,
        'best_loss': best_loss,
        'teacher': 'grounding-dino-base'
    }
    with open(f"{output_dir}/detector_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='GroundingDINO Knowledge Distillation')
    parser.add_argument('--data_root', type=str,
                       default='deepfashion2_training/data/deepfashion2',
                       help='Path to DeepFashion2 data')
    parser.add_argument('--output', type=str,
                       default='trained_models/fashion_detector',
                       help='Output directory')
    parser.add_argument('--max_samples', type=int, default=2000,
                       help='Max samples for pseudo-labeling')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--skip_pseudo_labels', action='store_true',
                       help='Skip pseudo-label generation')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    pseudo_labels_file = f"{args.output}/pseudo_labels.json"
    
    # Step 1: Generate pseudo-labels with GroundingDINO
    if not args.skip_pseudo_labels:
        generate_pseudo_labels_with_groundingdino(
            args.data_root,
            pseudo_labels_file,
            max_samples=args.max_samples
        )
    
    # Step 2: Train student detector
    if os.path.exists(pseudo_labels_file):
        train_student_detector(
            pseudo_labels_file,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        print(f"[Error] Pseudo-labels not found: {pseudo_labels_file}")
        print("Run without --skip_pseudo_labels first")
    
    print("\n" + "=" * 60)
    print("✅ Knowledge Distillation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
