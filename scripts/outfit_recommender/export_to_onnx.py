import os
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig, SegformerForSemanticSegmentation
import open_clip
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image

def export_fashion_clip(model_id="Marqo/marqo-fashionCLIP", output_path="assets/models/outfit_recommender/fashion_clip.onnx"):
    print(f"Exporting {model_id} via Manual Mapping (Legacy Exporter)...")
    
    # Load open_clip model (source of weights)
    print("Loading open_clip model...")
    model, _, _ = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionCLIP')
    model.eval()
    src_state = model.visual.state_dict()

    # Create transformers CLIPVisionModel (target for export)
    print("Creating transformers CLIPVisionModel...")
    config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch16")
    tgt_model = CLIPVisionModel(config)
    tgt_model.eval()
    
    new_state = {}
    new_state['vision_model.embeddings.patch_embedding.weight'] = src_state['conv1.weight']
    new_state['vision_model.embeddings.class_embedding'] = src_state['class_embedding']
    new_state['vision_model.embeddings.position_embedding.weight'] = src_state['positional_embedding']
    new_state['vision_model.pre_layrnorm.weight'] = src_state['ln_pre.weight']
    new_state['vision_model.pre_layrnorm.bias'] = src_state['ln_pre.bias']
    
    for i in range(config.num_hidden_layers):
        prefix_src = f'transformer.resblocks.{i}.'
        prefix_tgt = f'vision_model.encoder.layers.{i}.'
        new_state[f'{prefix_tgt}layer_norm1.weight'] = src_state[f'{prefix_src}ln_1.weight']
        new_state[f'{prefix_tgt}layer_norm1.bias'] = src_state[f'{prefix_src}ln_1.bias']
        in_proj_weight = src_state[f'{prefix_src}attn.in_proj_weight']
        in_proj_bias = src_state[f'{prefix_src}attn.in_proj_bias']
        dim = config.hidden_size
        new_state[f'{prefix_tgt}self_attn.q_proj.weight'] = in_proj_weight[:dim, :]
        new_state[f'{prefix_tgt}self_attn.k_proj.weight'] = in_proj_weight[dim:2*dim, :]
        new_state[f'{prefix_tgt}self_attn.v_proj.weight'] = in_proj_weight[2*dim:, :]
        new_state[f'{prefix_tgt}self_attn.q_proj.bias'] = in_proj_bias[:dim]
        new_state[f'{prefix_tgt}self_attn.k_proj.bias'] = in_proj_bias[dim:2*dim]
        new_state[f'{prefix_tgt}self_attn.v_proj.bias'] = in_proj_bias[2*dim:]
        new_state[f'{prefix_tgt}self_attn.out_proj.weight'] = src_state[f'{prefix_src}attn.out_proj.weight']
        new_state[f'{prefix_tgt}self_attn.out_proj.bias'] = src_state[f'{prefix_src}attn.out_proj.bias']
        new_state[f'{prefix_tgt}layer_norm2.weight'] = src_state[f'{prefix_src}ln_2.weight']
        new_state[f'{prefix_tgt}layer_norm2.bias'] = src_state[f'{prefix_src}ln_2.bias']
        new_state[f'{prefix_tgt}mlp.fc1.weight'] = src_state[f'{prefix_src}mlp.c_fc.weight']
        new_state[f'{prefix_tgt}mlp.fc1.bias'] = src_state[f'{prefix_src}mlp.c_fc.bias']
        new_state[f'{prefix_tgt}mlp.fc2.weight'] = src_state[f'{prefix_src}mlp.c_proj.weight']
        new_state[f'{prefix_tgt}mlp.fc2.bias'] = src_state[f'{prefix_src}mlp.c_proj.bias']

    new_state['vision_model.post_layernorm.weight'] = src_state['ln_post.weight']
    new_state['vision_model.post_layernorm.bias'] = src_state['ln_post.bias']

    tgt_model.load_state_dict(new_state, strict=False)
    
    class WrappedModel(torch.nn.Module):
        def __init__(self, base, proj):
            super().__init__()
            self.base = base
            self.proj = proj
        def forward(self, x):
            out = self.base(x)
            return torch.matmul(out.pooler_output, self.proj)

    final_model = WrappedModel(tgt_model, src_state['proj'])
    final_model.eval()

    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        final_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"Successfully exported FashionCLIP to {output_path}")

def export_human_parser(model_id="fashn-ai/fashn-human-parser", output_path="assets/models/outfit_recommender/human_parser.onnx"):
    print(f"Exporting {model_id} via Optimum...")
    import subprocess
    output_dir = os.path.dirname(output_path) + "/human_parser_opt"
    cmd = [
        "python", "-m", "optimum.exporters.onnx",
        "--model", model_id,
        "--task", "semantic-segmentation",
        output_dir
    ]
    try:
        subprocess.run(cmd, check=True)
        import shutil
        if os.path.exists(f"{output_dir}/model.onnx"):
            shutil.move(f"{output_dir}/model.onnx", output_path)
            print(f"Successfully exported Human Parser to {output_path}")
    except Exception as e:
        print(f"Optimum export failed: {e}")

if __name__ == "__main__":
    os.makedirs("assets/models/outfit_recommender", exist_ok=True)
    export_fashion_clip()
    export_human_parser()
