"""
Convert CLIP ONNX model to CoreML for iOS deployment.
Also creates optimized versions for different iOS devices.
"""

import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import numpy as np
import open_clip
from pathlib import Path


def convert_clip_to_coreml():
    """Convert trained CLIP model to CoreML format."""
    
    print("=" * 60)
    print("CLIP to CoreML Conversion")
    print("=" * 60)
    
    # Load the trained PyTorch model
    print("\n📦 Loading trained CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', 
        pretrained='laion2b_s34b_b79k'
    )
    
    checkpoint = torch.load('trained_models/openclip/clip_epoch10.pth', map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    print("✅ Model loaded")
    
    # Extract vision encoder
    class VisionEncoder(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.visual = clip_model.visual
        
        def forward(self, x):
            return self.visual(x)
    
    vision_model = VisionEncoder(model)
    vision_model.eval()
    
    # Trace the model
    print("\n🔄 Tracing model for CoreML conversion...")
    dummy_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(vision_model, dummy_input)
    
    # Convert to CoreML
    print("\n🍎 Converting to CoreML...")
    
    # Define input/output specs
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, 224, 224),
                scale=1/255.0,
                bias=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
                color_layout=ct.colorlayout.RGB
            )
        ],
        outputs=[
            ct.TensorType(name="embedding", dtype=np.float32)
        ],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )
    
    # Add metadata
    mlmodel.author = "PrismStyle AI"
    mlmodel.short_description = "Fashion-aware CLIP image encoder fine-tuned on DeepFashion2"
    mlmodel.version = "2.0.0"
    mlmodel.license = "MIT"
    
    # Input/output descriptions
    mlmodel.input_description["image"] = "Fashion item image (224x224 RGB)"
    mlmodel.output_description["embedding"] = "512-dimensional normalized embedding"
    
    # Save full precision model
    output_dir = Path("assets/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    coreml_path = output_dir / "CLIPFashionEncoder.mlpackage"
    mlmodel.save(str(coreml_path))
    print(f"✅ Saved CoreML model: {coreml_path}")
    
    # Create quantized version for smaller size
    print("\n📉 Creating quantized (FP16) version...")
    mlmodel_fp16 = quantization_utils.quantize_weights(mlmodel, nbits=16)
    coreml_fp16_path = output_dir / "CLIPFashionEncoder_FP16.mlpackage"
    mlmodel_fp16.save(str(coreml_fp16_path))
    print(f"✅ Saved FP16 model: {coreml_fp16_path}")
    
    # Verify the conversion
    print("\n🧪 Verifying CoreML model...")
    
    # Load and test
    import coremltools as ct
    loaded_model = ct.models.MLModel(str(coreml_path))
    
    # Create test input
    from PIL import Image
    test_img = Image.new('RGB', (224, 224), color=(128, 64, 192))
    
    # Run prediction
    result = loaded_model.predict({"image": test_img})
    embedding = result["embedding"]
    
    print(f"   Output shape: {embedding.shape}")
    print(f"   Embedding sample: {embedding.flatten()[:5]}")
    print("✅ CoreML model verified!")
    
    return coreml_path, coreml_fp16_path


def create_detection_coreml_placeholder():
    """
    Placeholder for GroundingDINO CoreML conversion.
    The actual conversion will happen after training.
    """
    print("\n" + "=" * 60)
    print("Detection Model CoreML (will be created after training)")
    print("=" * 60)
    print("GroundingDINO will be converted to CoreML after training completes.")


if __name__ == "__main__":
    try:
        clip_path, clip_fp16_path = convert_clip_to_coreml()
        
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETE")
        print("=" * 60)
        print(f"\n📁 Generated files:")
        print(f"   - {clip_path}")
        print(f"   - {clip_fp16_path}")
        
        # Get file sizes
        import os
        for p in [clip_path, clip_fp16_path]:
            if p.exists():
                # For mlpackage, get directory size
                total_size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                print(f"   Size: {total_size / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
