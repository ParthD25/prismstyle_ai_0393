import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import json

class FashionEmbedder:
    def __init__(self, model_path="assets/models/outfit_recommender/fashion_clip.onnx"):
        self.model_path = model_path
        if os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path)
        else:
            self.session = None
            print(f"Warning: Model {model_path} not found.")

    def preprocess(self, image_path, size=(224, 224)):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)
        img_data = np.array(img).transpose(2, 0, 1)
        img_data = img_data.astype('float32') / 255.0
        # Normalize with CLIP-specific stats
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1).astype('float32')
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1).astype('float32')
        img_data = (img_data - mean) / std
        return np.expand_dims(img_data, axis=0)

    def embed(self, image_path):
        if not self.session:
            return np.zeros((1, 512)) # Dummy
        
        input_data = self.preprocess(image_path)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_data})
        # Normalize embedding
        embedding = output[0]
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding

def embed_closet(closet_dir="assets/wardrobe_sample", output_file="assets/models/outfit_recommender/embeddings.npy"):
    embedder = FashionEmbedder()
    items = [f for f in os.listdir(closet_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    embeddings = []
    metadata = []

    for i, item in enumerate(items):
        path = os.path.join(closet_dir, item)
        print(f"Embedding {i+1}/{len(items)}: {item}")
        vec = embedder.embed(path)
        embeddings.append(vec[0])
        metadata.append({"item_id": i, "filename": item, "path": path})

    np.save(output_file, np.array(embeddings))
    with open("assets/models/outfit_recommender/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(embeddings)} embeddings to {output_file}")

if __name__ == "__main__":
    embed_closet()
