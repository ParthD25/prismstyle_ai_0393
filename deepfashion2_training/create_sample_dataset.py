#!/usr/bin/env python3
"""
Sample Dataset Generator for Testing

Creates a small synthetic dataset to test the training pipeline
without needing to download the full 30GB DeepFashion2 dataset.

This generates random colored rectangles with patterns to simulate
clothing items for each category.

Usage:
    python create_sample_dataset.py --output-dir data/deepfashion2 --samples-per-class 100
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from pathlib import Path

# DeepFashion2 categories
CATEGORIES = [
    'short_sleeve_top',
    'long_sleeve_top',
    'short_sleeve_outwear',
    'long_sleeve_outwear',
    'vest',
    'sling',
    'shorts',
    'trousers',
    'skirt',
    'short_sleeve_dress',
    'long_sleeve_dress',
    'vest_dress',
    'sling_dress'
]

# Category-specific color palettes
CATEGORY_COLORS = {
    'short_sleeve_top': [(255, 255, 255), (200, 200, 200), (100, 150, 200), (255, 200, 200)],
    'long_sleeve_top': [(50, 50, 80), (100, 100, 120), (150, 100, 100), (80, 120, 80)],
    'short_sleeve_outwear': [(50, 80, 50), (100, 80, 60), (80, 80, 100)],
    'long_sleeve_outwear': [(30, 30, 30), (60, 60, 60), (100, 70, 50), (50, 50, 100)],
    'vest': [(200, 180, 150), (150, 150, 150), (100, 100, 100)],
    'sling': [(255, 200, 200), (200, 200, 255), (255, 255, 200)],
    'shorts': [(100, 150, 200), (150, 100, 80), (200, 200, 200)],
    'trousers': [(50, 50, 100), (30, 30, 30), (100, 80, 60), (150, 150, 150)],
    'skirt': [(200, 100, 150), (100, 150, 200), (255, 200, 150)],
    'short_sleeve_dress': [(255, 150, 150), (150, 200, 255), (255, 255, 200)],
    'long_sleeve_dress': [(100, 50, 80), (50, 80, 100), (80, 100, 50)],
    'vest_dress': [(200, 150, 100), (150, 150, 200), (200, 200, 150)],
    'sling_dress': [(255, 200, 220), (220, 200, 255), (200, 255, 220)]
}

# Category-specific shapes (aspect ratios and patterns)
CATEGORY_SHAPES = {
    'short_sleeve_top': {'aspect': (1.0, 0.8), 'pattern': 'rectangle'},
    'long_sleeve_top': {'aspect': (1.0, 1.0), 'pattern': 'rectangle'},
    'short_sleeve_outwear': {'aspect': (1.1, 0.9), 'pattern': 'rectangle'},
    'long_sleeve_outwear': {'aspect': (1.2, 1.0), 'pattern': 'rectangle'},
    'vest': {'aspect': (0.8, 0.9), 'pattern': 'vest'},
    'sling': {'aspect': (0.7, 0.8), 'pattern': 'sling'},
    'shorts': {'aspect': (1.2, 0.5), 'pattern': 'shorts'},
    'trousers': {'aspect': (0.6, 1.5), 'pattern': 'trousers'},
    'skirt': {'aspect': (0.8, 0.8), 'pattern': 'skirt'},
    'short_sleeve_dress': {'aspect': (0.7, 1.3), 'pattern': 'dress'},
    'long_sleeve_dress': {'aspect': (0.7, 1.5), 'pattern': 'dress'},
    'vest_dress': {'aspect': (0.7, 1.2), 'pattern': 'dress'},
    'sling_dress': {'aspect': (0.6, 1.3), 'pattern': 'dress'}
}


def generate_pattern(draw, bbox, pattern_type, color):
    """Add pattern to the clothing item"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    if pattern_type == 'stripes':
        stripe_width = random.randint(5, 15)
        for i in range(0, width, stripe_width * 2):
            stripe_color = tuple(max(0, c - 30) for c in color)
            draw.rectangle([x1 + i, y1, x1 + i + stripe_width, y2], fill=stripe_color)
    
    elif pattern_type == 'dots':
        dot_radius = random.randint(3, 8)
        dot_color = tuple(max(0, c - 40) for c in color)
        for _ in range(random.randint(10, 30)):
            cx = random.randint(x1 + dot_radius, x2 - dot_radius)
            cy = random.randint(y1 + dot_radius, y2 - dot_radius)
            draw.ellipse([cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius], fill=dot_color)
    
    elif pattern_type == 'plaid':
        line_color = tuple(max(0, c - 50) for c in color)
        for i in range(x1, x2, 20):
            draw.line([(i, y1), (i, y2)], fill=line_color, width=2)
        for i in range(y1, y2, 20):
            draw.line([(x1, i), (x2, i)], fill=line_color, width=2)


def generate_clothing_image(category, size=224):
    """Generate a synthetic clothing image for a category"""
    # Create background
    bg_color = (240 + random.randint(-20, 20), 240 + random.randint(-20, 20), 240 + random.randint(-20, 20))
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Get category-specific parameters
    colors = CATEGORY_COLORS.get(category, [(150, 150, 150)])
    shape_info = CATEGORY_SHAPES.get(category, {'aspect': (1.0, 1.0), 'pattern': 'rectangle'})
    
    # Select random color
    base_color = random.choice(colors)
    # Add some variation
    color = tuple(min(255, max(0, c + random.randint(-20, 20))) for c in base_color)
    
    # Calculate item dimensions based on aspect ratio
    aspect_w, aspect_h = shape_info['aspect']
    margin = size * 0.15
    
    item_width = int((size - 2 * margin) * aspect_w * random.uniform(0.8, 1.0))
    item_height = int((size - 2 * margin) * aspect_h * random.uniform(0.8, 1.0))
    
    # Center the item
    x1 = (size - item_width) // 2
    y1 = (size - item_height) // 2
    x2 = x1 + item_width
    y2 = y1 + item_height
    
    # Draw base shape
    pattern = shape_info['pattern']
    
    if pattern == 'rectangle':
        draw.rectangle([x1, y1, x2, y2], fill=color)
    elif pattern == 'vest':
        # V-neck vest shape
        points = [(x1, y1 + 20), (size//2, y1 + 40), (x2, y1 + 20), (x2, y2), (x1, y2)]
        draw.polygon(points, fill=color)
    elif pattern == 'sling':
        # Sling top with straps
        draw.rectangle([x1 + 20, y1, x2 - 20, y2], fill=color)
        draw.line([(x1 + 30, y1), (size//2 - 10, y1 - 30)], fill=color, width=5)
        draw.line([(x2 - 30, y1), (size//2 + 10, y1 - 30)], fill=color, width=5)
    elif pattern == 'shorts':
        # Shorts with two legs
        leg_gap = item_width // 4
        draw.rectangle([x1, y1, x2, y1 + item_height//3], fill=color)
        draw.rectangle([x1, y1 + item_height//3, size//2 - leg_gap//2, y2], fill=color)
        draw.rectangle([size//2 + leg_gap//2, y1 + item_height//3, x2, y2], fill=color)
    elif pattern == 'trousers':
        # Full length pants
        leg_gap = item_width // 5
        draw.rectangle([x1, y1, x2, y1 + item_height//4], fill=color)
        draw.rectangle([x1, y1 + item_height//4, size//2 - leg_gap//2, y2], fill=color)
        draw.rectangle([size//2 + leg_gap//2, y1 + item_height//4, x2, y2], fill=color)
    elif pattern == 'skirt':
        # A-line skirt
        points = [(x1 + 20, y1), (x2 - 20, y1), (x2 + 10, y2), (x1 - 10, y2)]
        draw.polygon(points, fill=color)
    elif pattern == 'dress':
        # Dress shape
        # Top part
        draw.rectangle([x1 + 10, y1, x2 - 10, y1 + item_height//3], fill=color)
        # Skirt part (flared)
        points = [(x1 + 10, y1 + item_height//3), (x2 - 10, y1 + item_height//3), 
                  (x2 + 15, y2), (x1 - 15, y2)]
        draw.polygon(points, fill=color)
    
    # Add random pattern (30% chance)
    if random.random() < 0.3:
        pattern_type = random.choice(['stripes', 'dots', 'plaid'])
        generate_pattern(draw, (x1, y1, x2, y2), pattern_type, color)
    
    # Add some noise/texture
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Random rotation
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=bg_color, expand=False)
    
    return img


def create_sample_dataset(output_dir, samples_per_class=100, validation_split=0.2):
    """Create a complete sample dataset"""
    output_path = Path(output_dir)
    
    print("="*60)
    print("  Creating Sample Dataset for Testing")
    print("="*60)
    print(f"  Output directory: {output_path}")
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Validation split: {validation_split}")
    print(f"  Total samples: {samples_per_class * len(CATEGORIES)}")
    print("="*60)
    
    # Create directories
    for split in ['train', 'validation']:
        for category in CATEGORIES:
            dir_path = output_path / split / category
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate images for each category
    for category in CATEGORIES:
        print(f"\nGenerating {category}...")
        
        train_count = int(samples_per_class * (1 - validation_split))
        val_count = samples_per_class - train_count
        
        # Training images
        for i in range(train_count):
            img = generate_clothing_image(category)
            img_path = output_path / 'train' / category / f"{category}_{i:04d}.jpg"
            img.save(img_path, 'JPEG', quality=90)
        
        # Validation images
        for i in range(val_count):
            img = generate_clothing_image(category)
            img_path = output_path / 'validation' / category / f"{category}_{i:04d}.jpg"
            img.save(img_path, 'JPEG', quality=90)
        
        print(f"  [OK] {train_count} train + {val_count} validation images")
    
    print("\n" + "="*60)
    print("  Sample Dataset Created Successfully!")
    print("="*60)
    print(f"\n  You can now test training with:")
    print(f"    python train_model.py --epochs 10 --batch-size 16")
    print(f"\n  Note: This synthetic data is for testing only.")
    print(f"  For production, use the real DeepFashion2 dataset.")


def main():
    parser = argparse.ArgumentParser(description='Create sample dataset for testing')
    parser.add_argument('--output-dir', default='data/deepfashion2', help='Output directory')
    parser.add_argument('--samples-per-class', type=int, default=100, help='Samples per class')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split')
    
    args = parser.parse_args()
    
    create_sample_dataset(
        args.output_dir,
        args.samples_per_class,
        args.validation_split
    )


if __name__ == "__main__":
    main()
