#!/usr/bin/env python
"""
Test the trained OpenCLIP model with sample images
"""
import torch
import torch.nn.functional as F
import open_clip
import numpy as np
from PIL import Image
import os
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load trained model
print('🔄 Loading trained CLIP model...')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.load_state_dict(torch.load('./trained_models/openclip/clip_epoch10.pth', map_location=device))
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print('✅ Model loaded')

# Test with sample DeepFashion2 images
img_dir = './deepfashion2_training/data/deepfashion2/train/train/image'
sample_imgs = glob.glob(os.path.join(img_dir, '*.jpg'))[:5]

print(f'\n📸 Testing with {len(sample_imgs)} sample images...\n')

# Define test queries
queries = [
    'short sleeve top',
    'long sleeve dress',
    'trousers',
    'shorts',
    'skirt'
]

with torch.no_grad():
    # Encode text queries
    tokens = tokenizer(queries).to(device)
    text_feats = model.encode_text(tokens)
    text_feats = F.normalize(text_feats, dim=-1)
    
    # Encode images and find best match
    for img_path in sample_imgs:
        img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        img_feat = model.encode_image(img)
        img_feat = F.normalize(img_feat, dim=-1)
        
        # Compute similarity
        sims = (img_feat @ text_feats.T).squeeze().cpu().numpy()
        best_idx = np.argmax(sims)
        
        print(f'📷 {os.path.basename(img_path)}')
        print(f'   Best match: {queries[best_idx]} (score: {sims[best_idx]:.3f})')
        scores_str = ', '.join([f'{q}: {s:.2f}' for q, s in zip(queries, sims)])
        print(f'   All scores: {scores_str}')
        print()

print('✅ Model inference test complete!')
