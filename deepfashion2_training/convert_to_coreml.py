#!/usr/bin/env python3
"""
PyTorch to Core ML Conversion Script for PrismStyle AI

Converts the trained ConvNeXt/EfficientNet model to Core ML format
for on-device inference in iOS.

Requirements:
    pip install coremltools torch torchvision onnx onnx-simplify

Usage:
    python convert_to_coreml.py --model-path models/best_model_v4.pth --output clothing_classifier.mlmodel
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# Categories must match training
CATEGORIES = [
    "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear",
    "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"
]

DISPLAY_NAMES = [
    "T-Shirt", "Long Sleeve Shirt", "Short Jacket", "Coat/Jacket",
    "Vest", "Camisole", "Shorts", "Pants", "Skirt",
    "Short Sleeve Dress", "Long Sleeve Dress", "Vest Dress", "Slip Dress"
]


def create_convnext_model(num_classes=13):
    """Recreate ConvNeXt model architecture"""
    model = models.convnext_base(weights=None)
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
    """Recreate EfficientNetB3 model architecture"""
    model = models.efficientnet_b3(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes)
    )
    return model


def load_model(model_path):
    """Load trained model from checkpoint"""
    print(f"[1/5] Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine model type from state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Check if ConvNeXt or EfficientNet
    if any('features.0.0.weight' in k and 'conv' not in k for k in state_dict.keys()):
        print("     Detected: ConvNeXt architecture")
        model = create_convnext_model()
    else:
        print("     Detected: EfficientNet architecture")
        model = create_efficientnet_model()
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    if 'val_acc' in checkpoint:
        print(f"     Model accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model


def convert_via_onnx(model, output_path, image_size=384):
    """Convert PyTorch → ONNX → Core ML"""
    import onnx
    import coremltools as ct
    
    print(f"[2/5] Exporting to ONNX...")
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, image_size, image_size)
    onnx_path = output_path.replace('.mlmodel', '.onnx').replace('.mlpackage', '.onnx')
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=14,
        input_names=['image'],
        output_names=['classLabel_probs'],
        dynamic_axes=None  # Fixed batch size for Core ML
    )
    
    print(f"     ONNX saved: {onnx_path}")
    
    # Simplify ONNX (optional but recommended)
    try:
        from onnxsim import simplify
        onnx_model = onnx.load(onnx_path)
        onnx_model, check = simplify(onnx_model)
        onnx.save(onnx_model, onnx_path)
        print("     ONNX simplified")
    except ImportError:
        print("     (onnx-simplify not installed, skipping)")
    
    print(f"[3/5] Converting ONNX to Core ML...")
    
    # Convert to Core ML
    mlmodel = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_ios_deployment_target='14.0',
    )
    
    return mlmodel, onnx_path


def convert_direct(model, output_path, image_size=384):
    """Direct PyTorch to Core ML conversion (preferred)"""
    import coremltools as ct
    
    print(f"[2/5] Converting PyTorch to Core ML directly...")
    
    # Trace the model
    model.eval()
    example_input = torch.rand(1, 3, image_size, image_size)
    traced_model = torch.jit.trace(model, example_input)
    
    print(f"[3/5] Running Core ML conversion...")
    
    # Convert with image input
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, image_size, image_size),
                scale=1/255.0,
                bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                color_layout=ct.colorlayout.RGB
            )
        ],
        outputs=[
            ct.TensorType(name="classLabel_probs")
        ],
        classifier_config=ct.ClassifierConfig(CATEGORIES),
        minimum_deployment_target=ct.target.iOS14,
        convert_to="mlprogram"  # ML Program format (iOS 15+)
    )
    
    return mlmodel, None


def add_metadata(mlmodel):
    """Add metadata to Core ML model"""
    print(f"[4/5] Adding metadata...")
    
    mlmodel.short_description = "DeepFashion2 Clothing Classifier for PrismStyle AI"
    mlmodel.author = "PrismStyle AI"
    mlmodel.license = "MIT"
    mlmodel.version = "4.0.0"
    
    # Input/output descriptions
    mlmodel.input_description["image"] = f"Clothing item photo (RGB)"
    mlmodel.output_description["classLabel"] = "Predicted clothing category"
    mlmodel.output_description["classLabel_probs"] = "Probability for each category"
    
    return mlmodel


def verify_model(mlmodel, image_size=384):
    """Verify the converted model works"""
    print(f"[5/5] Verifying model...")
    
    try:
        import coremltools as ct
        from PIL import Image
        
        # Create test image
        test_image = Image.new('RGB', (image_size, image_size), color=(128, 128, 128))
        
        # Run prediction
        prediction = mlmodel.predict({'image': test_image})
        
        if 'classLabel' in prediction:
            print(f"     Test prediction: {prediction['classLabel']}")
            probs = prediction.get('classLabel_probs', {})
            top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for cat, prob in top3:
                print(f"       {cat}: {prob*100:.1f}%")
            return True
        else:
            print(f"     Output keys: {prediction.keys()}")
            return True
            
    except Exception as e:
        print(f"     Verification error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to Core ML')
    parser.add_argument('--model-path', required=True, help='Path to .pth model')
    parser.add_argument('--output', default='FashionClassifier.mlpackage', help='Output path')
    parser.add_argument('--image-size', type=int, default=384, help='Input image size')
    parser.add_argument('--use-onnx', action='store_true', help='Convert via ONNX')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  PrismStyle AI - Core ML Conversion")
    print("=" * 60)
    
    # Check coremltools
    try:
        import coremltools as ct
        print(f"[OK] coremltools version: {ct.__version__}")
    except ImportError:
        print("[ERROR] coremltools not installed")
        print("        pip install coremltools")
        sys.exit(1)
    
    # Load model
    model = load_model(args.model_path)
    
    # Convert
    if args.use_onnx:
        mlmodel, onnx_path = convert_via_onnx(model, args.output, args.image_size)
    else:
        try:
            mlmodel, _ = convert_direct(model, args.output, args.image_size)
        except Exception as e:
            print(f"     Direct conversion failed: {e}")
            print("     Falling back to ONNX conversion...")
            mlmodel, _ = convert_via_onnx(model, args.output, args.image_size)
    
    # Add metadata
    mlmodel = add_metadata(mlmodel)
    
    # Save
    mlmodel.save(args.output)
    print(f"\n[OK] Core ML model saved: {args.output}")
    
    # Get size
    if os.path.isdir(args.output):
        size = sum(os.path.getsize(os.path.join(dp, f)) 
                   for dp, _, files in os.walk(args.output) for f in files)
    else:
        size = os.path.getsize(args.output)
    print(f"     Size: {size / 1e6:.1f} MB")
    
    # Verify
    verify_model(mlmodel, args.image_size)
    
    print("\n" + "=" * 60)
    print("  Conversion Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Copy {args.output} to ios/Runner/Resources/")
    print(f"  2. Add to Xcode project")
    print(f"  3. Update CoreMLHandler.swift to load '{os.path.basename(args.output)}'")


if __name__ == "__main__":
    main()
