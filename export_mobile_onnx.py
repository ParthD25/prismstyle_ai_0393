"""
Export ONNX models with lower opset for mobile compatibility.
Android ONNX Runtime only supports IR version <= 9 (opset <= 13)
"""
import torch
import timm
import onnx
import os

def main():
    print("=" * 60)
    print("Exporting ONNX models for mobile (opset 13)")
    print("=" * 60)
    
    output_dir = "assets/models"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export Fashion Detector (EfficientNet-B0)
    print("\n📦 Exporting Fashion Detector (EfficientNet-B0)...")
    checkpoint_path = "trained_models/fashion_detector_v2/best_model.pth"
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=13)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_path = os.path.join(output_dir, "clothing_classifier.onnx")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,  # Mobile compatible
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify
        onnx_model = onnx.load(onnx_path)
        print(f"   ✅ Exported: {onnx_path}")
        print(f"   Size: {os.path.getsize(onnx_path) / 1e6:.2f} MB")
        print(f"   IR version: {onnx_model.ir_version}")
        print(f"   Opset: {onnx_model.opset_import[0].version}")
    else:
        print(f"   ⚠️ Checkpoint not found: {checkpoint_path}")
    
    # 2. Export CLIP Image Encoder
    print("\n📦 Exporting CLIP Image Encoder...")
    try:
        import open_clip
        
        clip_model, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        
        # Load fine-tuned weights if available
        finetuned_path = "trained_models/openclip/clip_epoch10.pth"
        if os.path.exists(finetuned_path):
            clip_model.load_state_dict(torch.load(finetuned_path, map_location='cpu'))
            print("   Loaded fine-tuned weights")
        
        clip_model.eval()
        
        class CLIPImageEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.visual = model.visual
            def forward(self, x):
                return self.visual(x)
        
        encoder = CLIPImageEncoder(clip_model)
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_path = os.path.join(output_dir, "clip_image_encoder.onnx")
        
        torch.onnx.export(
            encoder,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['embedding'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'embedding': {0: 'batch_size'}
            }
        )
        
        onnx_model = onnx.load(onnx_path)
        print(f"   ✅ Exported: {onnx_path}")
        print(f"   Size: {os.path.getsize(onnx_path) / 1e6:.2f} MB")
        print(f"   IR version: {onnx_model.ir_version}")
        print(f"   Opset: {onnx_model.opset_import[0].version}")
        
    except Exception as e:
        print(f"   ⚠️ CLIP export failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Export complete! Models in:", output_dir)
    print("=" * 60)

if __name__ == "__main__":
    main()
