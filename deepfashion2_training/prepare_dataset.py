#!/usr/bin/env python3
"""
DeepFashion2 Dataset Preparation Script

This script helps organize the DeepFashion2 dataset into the correct structure
for training the clothing classification model.

The DeepFashion2 dataset contains:
- Annotations in JSON format (with category info)
- Images for each sample

This script will:
1. Read the annotations
2. Organize images by category for ImageDataGenerator
3. Create train/validation splits

Usage:
    python prepare_dataset.py --input-dir data/deepfashion2_raw --output-dir data/deepfashion2
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm

# DeepFashion2 category mapping (1-indexed in original dataset)
CATEGORY_NAMES = {
    1: 'short_sleeve_top',
    2: 'long_sleeve_top',
    3: 'short_sleeve_outwear',
    4: 'long_sleeve_outwear',
    5: 'vest',
    6: 'sling',
    7: 'shorts',
    8: 'trousers',
    9: 'skirt',
    10: 'short_sleeve_dress',
    11: 'long_sleeve_dress',
    12: 'vest_dress',
    13: 'sling_dress'
}

def create_category_folders(output_dir, split='train'):
    """Create category folders for a split"""
    split_dir = Path(output_dir) / split
    for category_name in CATEGORY_NAMES.values():
        category_dir = split_dir / category_name
        category_dir.mkdir(parents=True, exist_ok=True)
    return split_dir

def parse_annotation(annotation_path):
    """Parse a DeepFashion2 annotation file"""
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    items = []
    # DeepFashion2 annotations have items numbered as "item1", "item2", etc.
    for key in data:
        if key.startswith('item'):
            item = data[key]
            if 'category_id' in item:
                items.append({
                    'category_id': item['category_id'],
                    'bounding_box': item.get('bounding_box', None),
                    'style': item.get('style', None),
                })
    return items

def organize_deepfashion2(input_dir, output_dir, validation_split=0.2):
    """
    Organize DeepFashion2 dataset into training structure
    
    Expected input structure:
    input_dir/
        train/
            image/
                000001.jpg
                000002.jpg
                ...
            annos/
                000001.json
                000002.json
                ...
        validation/
            image/
            annos/
    
    Output structure:
    output_dir/
        train/
            short_sleeve_top/
            long_sleeve_top/
            ...
        validation/
            short_sleeve_top/
            ...
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print("Creating output directories...")
    create_category_folders(output_path, 'train')
    create_category_folders(output_path, 'validation')
    
    # Process training set
    for split in ['train', 'validation']:
        print(f"\nProcessing {split} set...")
        
        image_dir = input_path / split / 'image'
        annos_dir = input_path / split / 'annos'
        
        if not image_dir.exists():
            print(f"  [WARNING] {image_dir} not found, skipping...")
            continue
        
        if not annos_dir.exists():
            print(f"  [WARNING] {annos_dir} not found, skipping...")
            continue
        
        # Get all annotation files
        anno_files = list(annos_dir.glob('*.json'))
        print(f"  Found {len(anno_files)} annotation files")
        
        category_counts = defaultdict(int)
        
        for anno_file in tqdm(anno_files, desc=f"  Processing {split}"):
            # Get corresponding image
            image_id = anno_file.stem
            
            # Try different image extensions
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = image_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break
            
            if image_file is None:
                continue
            
            # Parse annotation
            try:
                items = parse_annotation(anno_file)
            except Exception as e:
                print(f"  [WARNING] Failed to parse {anno_file}: {e}")
                continue
            
            # Process each item in the image
            for idx, item in enumerate(items):
                category_id = item['category_id']
                
                if category_id not in CATEGORY_NAMES:
                    continue
                
                category_name = CATEGORY_NAMES[category_id]
                category_counts[category_name] += 1
                
                # Copy image to category folder
                dest_dir = output_path / split / category_name
                dest_file = dest_dir / f"{image_id}_{idx}{image_file.suffix}"
                
                try:
                    shutil.copy2(image_file, dest_file)
                except Exception as e:
                    print(f"  [WARNING] Failed to copy {image_file}: {e}")
        
        # Print summary
        print(f"\n  {split.upper()} Summary:")
        for category, count in sorted(category_counts.items()):
            print(f"    {category}: {count} images")

def organize_simple_structure(input_dir, output_dir, validation_split=0.2):
    """
    Alternative organization for simpler dataset structures
    
    If your dataset is already organized by category:
    input_dir/
        category1/
            image1.jpg
            image2.jpg
        category2/
            ...
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print("Creating output directories...")
    create_category_folders(output_path, 'train')
    create_category_folders(output_path, 'validation')
    
    # Find category folders
    category_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    for category_dir in category_dirs:
        category_name = category_dir.name
        
        # Check if it matches our expected categories
        if category_name not in CATEGORY_NAMES.values():
            print(f"  [WARNING] Unknown category: {category_name}, skipping...")
            continue
        
        # Get all images in category
        images = list(category_dir.glob('*.jpg')) + \
                 list(category_dir.glob('*.jpeg')) + \
                 list(category_dir.glob('*.png'))
        
        if not images:
            continue
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * (1 - validation_split))
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        print(f"  {category_name}: {len(train_images)} train, {len(val_images)} val")
        
        # Copy to output
        for img in train_images:
            dest = output_path / 'train' / category_name / img.name
            shutil.copy2(img, dest)
        
        for img in val_images:
            dest = output_path / 'validation' / category_name / img.name
            shutil.copy2(img, dest)

def verify_dataset(dataset_dir):
    """Verify the dataset structure is correct"""
    dataset_path = Path(dataset_dir)
    
    print("\n=== Dataset Verification ===")
    
    for split in ['train', 'validation']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"  [ERROR] {split} directory not found!")
            continue
        
        print(f"\n{split.upper()}:")
        total = 0
        
        for category_name in sorted(CATEGORY_NAMES.values()):
            category_dir = split_dir / category_name
            if category_dir.exists():
                count = len(list(category_dir.glob('*')))
                total += count
                status = "[OK]" if count > 0 else "[EMPTY]"
                print(f"  {status} {category_name}: {count} images")
            else:
                print(f"  [MISSING] {category_name}")
        
        print(f"  TOTAL: {total} images")

def main():
    parser = argparse.ArgumentParser(
        description='Prepare DeepFashion2 dataset for training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        default='data/deepfashion2_raw',
        help='Input directory containing raw DeepFashion2 data'
    )
    parser.add_argument(
        '--output-dir',
        default='data/deepfashion2',
        help='Output directory for organized dataset'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation (default: 0.2)'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple folder-based organization instead of annotation parsing'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing dataset structure'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.output_dir)
        return
    
    print("=== DeepFashion2 Dataset Preparation ===")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Validation split: {args.validation_split}")
    print()
    
    if args.simple:
        organize_simple_structure(
            args.input_dir,
            args.output_dir,
            args.validation_split
        )
    else:
        organize_deepfashion2(
            args.input_dir,
            args.output_dir,
            args.validation_split
        )
    
    print("\n=== Verifying output ===")
    verify_dataset(args.output_dir)
    
    print("\n=== Done! ===")
    print(f"Dataset is ready at: {args.output_dir}")
    print("You can now run: python train_model.py --epochs 50")

if __name__ == "__main__":
    main()
