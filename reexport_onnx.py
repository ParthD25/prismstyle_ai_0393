"""Re-export CLIP ONNX model with correct external data filename."""
import torch
import open_clip

print("Loading OpenCLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

print("Loading trained weights...")
checkpoint = torch.load('trained_models/openclip/clip_epoch10.pth', map_location='cpu')

# The checkpoint is a raw state dict (not wrapped in 'model_state_dict')
model.load_state_dict(checkpoint, strict=True)
model.eval()
print("✅ Weights loaded successfully")

# Export just the vision encoder
class VisionEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
    
    def forward(self, x):
        return self.visual(x)

vision_model = VisionEncoder(model)
dummy_input = torch.randn(1, 3, 224, 224)

print("Exporting to ONNX...")
torch.onnx.export(
    vision_model,
    dummy_input,
    'assets/models/clip_image_encoder.onnx',
    input_names=['image'],
    output_names=['embedding'],
    dynamic_axes={'image': {0: 'batch'}, 'embedding': {0: 'batch'}},
    opset_version=18
)

print("✅ ONNX export complete!")
print("Files created:")
import os
for f in os.listdir('assets/models'):
    if 'clip' in f.lower():
        size = os.path.getsize(f'assets/models/{f}')
        print(f"  - {f}: {size/1024/1024:.1f} MB")
