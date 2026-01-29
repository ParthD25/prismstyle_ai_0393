#!/usr/bin/env python
"""
PrismStyle AI - OpenCLIP Training Script
Fine-tune OpenCLIP (ViT-B-32) on DeepFashion2 for fashion-aware embeddings.

Usage:
    python run_openclip_training.py

Requirements:
    pip install torch torchvision open_clip_torch faiss-cpu tqdm matplotlib pillow numpy
"""

import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    # Paths
    'data_root': './deepfashion2_training/data/deepfashion2',
    'wardrobe_dir': './assets/wardrobe_sample',
    'output_dir': './trained_models/openclip',
    
    # Model
    'model_name': 'ViT-B-32',
    'pretrained': 'laion2b_s34b_b79k',
    
    # Training
    'batch_size': 32,  # Reduce if OOM
    'learning_rate': 5e-6,
    'epochs': 10,
    'num_workers': 0,  # Set to 0 for Windows compatibility
    
    # Validation
    'val_every': 1,
}

CATEGORIES = [
    'short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear',
    'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress',
    'long sleeve dress', 'vest dress', 'sling dress'
]


# ============================================================================
# Dataset
# ============================================================================
class DF2CLIPDataset(Dataset):
    """DeepFashion2 dataset for CLIP contrastive training"""
    
    def __init__(self, root, split='train', preprocess=None):
        # Handle nested folder structure: root/split/split/image
        base_path = os.path.join(root, split, split)
        if not os.path.exists(base_path):
            base_path = os.path.join(root, split)
        
        self.img_dir = os.path.join(base_path, 'image')
        self.ann_dir = os.path.join(base_path, 'annos')
        self.preprocess = preprocess
        
        if not os.path.exists(self.ann_dir):
            print(f"⚠️ Annotations not found at: {self.ann_dir}")
            self.files = []
        else:
            # Only include annotations that have corresponding images
            all_annos = [f for f in os.listdir(self.ann_dir) if f.endswith('.json')]
            self.files = []
            for f in all_annos:
                img_name = f.replace('.json', '.jpg')
                if os.path.exists(os.path.join(self.img_dir, img_name)):
                    self.files.append(f)
        
        print(f"📂 Loaded {len(self.files)} samples from {split} (with images)")
        print(f"   Image dir: {self.img_dir}")
        print(f"   Anno dir: {self.ann_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        ann_path = os.path.join(self.ann_dir, self.files[idx])
        
        try:
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            
            # Get image filename (use annotation filename as fallback)
            img_name = self.files[idx].replace('.json', '.jpg')
            img_path = os.path.join(self.img_dir, img_name)
            
            # Extract categories
            items = ann.get('items', []) or [v for k, v in ann.items() if k.startswith('item')]
            cats = []
            for it in items:
                if isinstance(it, dict):
                    cid = int(it.get('category_id', it.get('category', 0)))
                    if 1 <= cid <= 13:
                        cats.append(CATEGORIES[cid - 1])
            
            text = ', '.join(sorted(set(cats))) if cats else 'clothing'
            img = Image.open(img_path).convert('RGB')
            
            # Always preprocess to tensor
            if self.preprocess:
                img = self.preprocess(img)
            
            return img, text
            
        except Exception as e:
            print(f"⚠️ Error loading {ann_path}: {e}")
            # Return a placeholder tensor
            img = Image.new('RGB', (224, 224), color='gray')
            if self.preprocess:
                img = self.preprocess(img)
            return img, 'clothing'


# ============================================================================
# Training Functions
# ============================================================================
def recall_at_k(image_feats, text_feats, ks=(1, 5, 10)):
    """Compute Recall@K for image-text retrieval"""
    sims = image_feats @ text_feats.T
    ranks = np.argsort(-sims, axis=1)
    gt = np.arange(sims.shape[0])
    recalls = {}
    for k in ks:
        hit = (ranks[:, :k] == gt[:, None]).any(axis=1).mean()
        recalls[f"R@{k}"] = float(hit)
    return recalls


def export_onnx(model, output_path, device):
    """Export CLIP image encoder to ONNX"""
    class CLIPImageEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model.encode_image(x)
    
    model.eval()
    encoder = CLIPImageEncoder(model)
    dummy = torch.randn(1, 3, 224, 224, device=device)
    
    torch.onnx.export(
        encoder,
        dummy,
        output_path,
        input_names=['image'],
        output_names=['embedding'],
        opset_version=14,
        do_constant_folding=True,
        dynamic_axes={
            'image': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    )
    print(f"✅ ONNX exported to: {output_path}")


def build_wardrobe_index(model, preprocess, wardrobe_dir, output_dir, device):
    """Build FAISS index for wardrobe images"""
    try:
        import faiss
    except ImportError:
        print("⚠️ faiss not installed. Skipping wardrobe indexing.")
        return
    
    # Find images
    paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        paths.extend(glob.glob(os.path.join(wardrobe_dir, '**', ext), recursive=True))
    
    if not paths:
        print(f"⚠️ No images found in {wardrobe_dir}")
        return
    
    paths = sorted(list(set(paths)))
    print(f"📂 Found {len(paths)} wardrobe images")
    
    # Compute embeddings
    model.eval()
    embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="Embedding wardrobe"):
            batch_paths = paths[i:i+batch_size]
            batch_imgs = []
            for p in batch_paths:
                try:
                    img = preprocess(Image.open(p).convert('RGB'))
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"⚠️ Error loading {p}: {e}")
                    batch_imgs.append(preprocess(Image.new('RGB', (224, 224))))
            
            batch_tensor = torch.stack(batch_imgs).to(device)
            feats = model.encode_image(batch_tensor)
            feats = F.normalize(feats, dim=-1)
            embeddings.append(feats.cpu().numpy().astype('float32'))
    
    embeddings = np.concatenate(embeddings, axis=0)
    
    # Save
    index_dir = os.path.join(output_dir, 'index')
    os.makedirs(index_dir, exist_ok=True)
    
    np.save(os.path.join(index_dir, 'embeddings.npy'), embeddings)
    with open(os.path.join(index_dir, 'paths.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(paths))
    
    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, 'index.faiss'))
    
    print(f"✅ Wardrobe index saved to: {index_dir}")


# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    print("=" * 60)
    print("🚀 PrismStyle AI - OpenCLIP Training")
    print("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load model
    print(f"\n🔄 Loading OpenCLIP {CONFIG['model_name']}...")
    try:
        import open_clip
    except ImportError:
        print("❌ open_clip not installed. Run: pip install open_clip_torch")
        return
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        CONFIG['model_name'],
        pretrained=CONFIG['pretrained']
    )
    tokenizer = open_clip.get_tokenizer(CONFIG['model_name'])
    model = model.to(device)
    print("✅ Model loaded")
    
    # Create datasets - pass preprocess function
    print(f"\n📂 Loading datasets from: {CONFIG['data_root']}")
    print("   Scanning for images with annotations (this may take a minute)...")
    train_ds = DF2CLIPDataset(CONFIG['data_root'], 'train', preprocess=preprocess)
    val_ds = DF2CLIPDataset(CONFIG['data_root'], 'validation', preprocess=preprocess)
    
    if len(train_ds) == 0:
        print("❌ No training samples found. Check data_root path.")
        return
    
    train_dl = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers']
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )
    
    print(f"   Train batches: {len(train_dl)}")
    print(f"   Val batches: {len(val_dl)}")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    total_steps = len(train_dl) * CONFIG['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training history
    history = {'train_loss': [], 'val_recall': []}
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"🚀 Starting Training for {CONFIG['epochs']} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for imgs, texts in pbar:
            # imgs are already preprocessed tensors from dataset
            imgs_tensor = imgs.to(device)
            tokens = tokenizer(list(texts)).to(device)
            
            # Forward pass (freeze text encoder gradients)
            with torch.no_grad():
                text_feats = model.encode_text(tokens)
                text_feats = F.normalize(text_feats, dim=-1)
            
            img_feats = model.encode_image(imgs_tensor)
            img_feats = F.normalize(img_feats, dim=-1)
            
            # Contrastive loss
            logits = img_feats @ text_feats.T * 100.0
            labels = torch.arange(logits.size(0), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        avg_loss = epoch_loss / len(train_dl)
        history['train_loss'].append(avg_loss)
        
        # Validation
        if (epoch + 1) % CONFIG['val_every'] == 0 and len(val_ds) > 0:
            model.eval()
            all_img_feats, all_txt_feats = [], []
            
            with torch.no_grad():
                for imgs, texts in tqdm(val_dl, desc="Validating", leave=False):
                    imgs_tensor = imgs.to(device)
                    tokens = tokenizer(list(texts)).to(device)
                    
                    txt_f = model.encode_text(tokens)
                    img_f = model.encode_image(imgs_tensor)
                    
                    all_img_feats.append(F.normalize(img_f, dim=-1).cpu().numpy())
                    all_txt_feats.append(F.normalize(txt_f, dim=-1).cpu().numpy())
            
            img_feats_np = np.concatenate(all_img_feats, 0)
            txt_feats_np = np.concatenate(all_txt_feats, 0)
            
            recalls = recall_at_k(img_feats_np, txt_feats_np)
            history['val_recall'].append(recalls)
            
            print(f"\n📊 Epoch {epoch+1} | Loss: {avg_loss:.4f} | R@1: {recalls['R@1']:.3f} | R@5: {recalls['R@5']:.3f} | R@10: {recalls['R@10']:.3f}")
        else:
            print(f"\n📊 Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(CONFIG['output_dir'], f'clip_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f"   Checkpoint saved: {ckpt_path}")
    
    # Export ONNX
    print(f"\n{'='*60}")
    print("📦 Exporting ONNX model...")
    onnx_path = os.path.join(CONFIG['output_dir'], 'clip_image_encoder.onnx')
    export_onnx(model, onnx_path, device)
    
    # Build wardrobe index
    if os.path.exists(CONFIG['wardrobe_dir']):
        print(f"\n{'='*60}")
        print("🗂️ Building wardrobe index...")
        build_wardrobe_index(model, preprocess, CONFIG['wardrobe_dir'], CONFIG['output_dir'], device)
    
    # Plot training curves
    print(f"\n{'='*60}")
    print("📈 Saving training curves...")
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history['train_loss'], marker='o')
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)
    
    if history['val_recall']:
        r1 = [r['R@1'] for r in history['val_recall']]
        r5 = [r['R@5'] for r in history['val_recall']]
        r10 = [r['R@10'] for r in history['val_recall']]
        ax[1].plot(r1, marker='o', label='R@1')
        ax[1].plot(r5, marker='s', label='R@5')
        ax[1].plot(r10, marker='^', label='R@10')
        ax[1].set_title('Validation Recall@K')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Recall')
        ax[1].legend()
        ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_curves.png'), dpi=150)
    print(f"   Saved: {os.path.join(CONFIG['output_dir'], 'training_curves.png')}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📁 Output files:")
    for f in os.listdir(CONFIG['output_dir']):
        fpath = os.path.join(CONFIG['output_dir'], f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1e6
            print(f"   ✓ {f} ({size:.1f} MB)")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Copy trained_models/openclip/ to your Flutter app")
    print(f"   2. Update model_config.json with paths")
    print(f"   3. Run: flutter pub get && flutter run")


if __name__ == '__main__':
    main()
