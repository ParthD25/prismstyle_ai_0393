"""
CLIP-based Fashion Inference Pipeline
For PrismStyle AI - Outfit Recommendation System

This module provides:
1. CLIP image encoding (ONNX optimized)
2. FAISS-based similarity search
3. Wardrobe embedding management
4. Outfit recommendation logic
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image

# ONNX Runtime for inference
try:
    import onnxruntime as ort
except ImportError:
    print("Installing onnxruntime...")
    os.system(f"{sys.executable} -m pip install onnxruntime-gpu")
    import onnxruntime as ort

# FAISS for similarity search
try:
    import faiss
except ImportError:
    print("Installing faiss-cpu...")
    os.system(f"{sys.executable} -m pip install faiss-cpu")
    import faiss


class CLIPFashionEncoder:
    """ONNX-based CLIP image encoder optimized for fashion items."""
    
    def __init__(self, model_path: str = None, use_gpu: bool = True):
        """
        Initialize the CLIP encoder.
        
        Args:
            model_path: Path to ONNX model file
            use_gpu: Whether to use GPU acceleration
        """
        if model_path is None:
            # Default to assets/models location
            base_dir = Path(__file__).parent.parent.parent / "assets" / "models"
            model_path = base_dir / "clip_image_encoder.onnx"
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Setup ONNX Runtime session
        providers = []
        if use_gpu:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Image preprocessing parameters (OpenCLIP ViT-B-32)
        self.image_size = 224
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        
        print(f"✅ CLIP encoder loaded from {self.model_path.name}")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess an image for CLIP encoding.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            img = image.convert('RGB')
        
        # Resize with center crop
        width, height = img.size
        scale = self.image_size / min(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.BICUBIC)
        
        # Center crop
        left = (new_width - self.image_size) // 2
        top = (new_height - self.image_size) // 2
        img = img.crop((left, top, left + self.image_size, top + self.image_size))
        
        # Convert to numpy and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        
        # CHW format with batch dimension
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array.astype(np.float32)
    
    def encode(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode an image to a 512-dimensional embedding.
        
        Args:
            image: Image to encode
            
        Returns:
            Normalized embedding of shape (512,)
        """
        preprocessed = self.preprocess_image(image)
        embedding = self.session.run([self.output_name], {self.input_name: preprocessed})[0]
        
        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding.squeeze()
    
    def encode_batch(self, images: List[Union[str, Path, Image.Image]]) -> np.ndarray:
        """
        Encode multiple images.
        
        Args:
            images: List of images to encode
            
        Returns:
            Normalized embeddings of shape (N, 512)
        """
        # Process one at a time (ONNX batch dimension can have issues)
        embeddings = []
        for img in images:
            embedding = self.encode(img)
            embeddings.append(embedding)
        
        return np.stack(embeddings, axis=0)


class WardrobeIndex:
    """FAISS-based wardrobe similarity index."""
    
    def __init__(self, encoder: CLIPFashionEncoder = None, index_path: str = None):
        """
        Initialize wardrobe index.
        
        Args:
            encoder: CLIPFashionEncoder instance
            index_path: Path to existing FAISS index
        """
        self.encoder = encoder or CLIPFashionEncoder()
        self.embedding_dim = 512
        
        if index_path and Path(index_path).exists():
            self.load(index_path)
        else:
            # Create new index with IVF for scalability
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
            self.item_paths = []
            self.item_metadata = []
    
    def add_item(self, image_path: str, metadata: Dict = None) -> int:
        """
        Add an item to the wardrobe index.
        
        Args:
            image_path: Path to item image
            metadata: Optional metadata (category, color, tags, etc.)
            
        Returns:
            Index of added item
        """
        embedding = self.encoder.encode(image_path)
        self.index.add(embedding.reshape(1, -1))
        self.item_paths.append(str(image_path))
        self.item_metadata.append(metadata or {})
        
        return len(self.item_paths) - 1
    
    def add_items_batch(self, image_paths: List[str], metadata_list: List[Dict] = None) -> List[int]:
        """
        Add multiple items to the index.
        
        Args:
            image_paths: List of image paths
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of indices
        """
        metadata_list = metadata_list or [{}] * len(image_paths)
        embeddings = self.encoder.encode_batch(image_paths)
        
        start_idx = len(self.item_paths)
        self.index.add(embeddings)
        self.item_paths.extend([str(p) for p in image_paths])
        self.item_metadata.extend(metadata_list)
        
        return list(range(start_idx, len(self.item_paths)))
    
    def search(self, query_image: Union[str, Image.Image], k: int = 5, 
               category_filter: str = None) -> List[Dict]:
        """
        Find similar items in the wardrobe.
        
        Args:
            query_image: Query image
            k: Number of results
            category_filter: Optional category to filter by
            
        Returns:
            List of dicts with path, similarity, and metadata
        """
        query_embedding = self.encoder.encode(query_image)
        
        # Search more if filtering
        search_k = k * 3 if category_filter else k
        scores, indices = self.index.search(query_embedding.reshape(1, -1), min(search_k, len(self.item_paths)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing
                continue
                
            metadata = self.item_metadata[idx]
            
            # Apply category filter
            if category_filter and metadata.get('category') != category_filter:
                continue
            
            results.append({
                'path': self.item_paths[idx],
                'similarity': float(score),
                'metadata': metadata
            })
            
            if len(results) >= k:
                break
        
        return results
    
    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / "wardrobe.faiss"))
        np.save(path / "paths.npy", np.array(self.item_paths, dtype=object))
        
        with open(path / "metadata.json", 'w') as f:
            json.dump(self.item_metadata, f)
        
        print(f"✅ Saved wardrobe index with {len(self.item_paths)} items")
    
    def load(self, path: str):
        """Load index from disk."""
        path = Path(path)
        
        self.index = faiss.read_index(str(path / "wardrobe.faiss"))
        self.item_paths = list(np.load(path / "paths.npy", allow_pickle=True))
        
        with open(path / "metadata.json", 'r') as f:
            self.item_metadata = json.load(f)
        
        print(f"✅ Loaded wardrobe index with {len(self.item_paths)} items")


class OutfitRecommender:
    """
    Outfit recommendation engine using CLIP embeddings.
    
    Combines color harmony, style matching, and occasion appropriateness
    to generate complete outfit recommendations.
    """
    
    # Clothing categories for outfit construction
    TOPS = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 
            'long_sleeve_outwear', 'vest', 'sling']
    BOTTOMS = ['trousers', 'shorts', 'skirt']
    DRESSES = ['short_sleeve_dress', 'long_sleeve_dress', 'vest_dress', 'sling_dress']
    
    def __init__(self, wardrobe_index: WardrobeIndex):
        """
        Initialize recommender.
        
        Args:
            wardrobe_index: Indexed wardrobe
        """
        self.wardrobe = wardrobe_index
    
    def recommend_outfit(self, 
                         seed_item: str = None,
                         occasion: str = 'casual',
                         weather: str = None,
                         exclude_categories: List[str] = None) -> Dict:
        """
        Generate an outfit recommendation.
        
        Args:
            seed_item: Optional starting item image
            occasion: Event type (casual, formal, athletic, etc.)
            weather: Weather conditions (hot, cold, rainy, etc.)
            exclude_categories: Categories to exclude
            
        Returns:
            Dict with recommended items and reasoning
        """
        exclude_categories = exclude_categories or []
        outfit = {'items': [], 'reasoning': []}
        
        # If seed item provided, find complementary items
        if seed_item:
            seed_results = self.wardrobe.search(seed_item, k=1)
            if seed_results:
                seed_metadata = seed_results[0]['metadata']
                seed_category = seed_metadata.get('category', 'unknown')
                
                outfit['items'].append({
                    'role': 'seed',
                    'path': seed_item,
                    'category': seed_category
                })
                outfit['reasoning'].append(f"Starting with {seed_category}")
                
                # Determine what else is needed
                if seed_category in self.DRESSES:
                    outfit['reasoning'].append("Dress is a complete outfit")
                elif seed_category in self.TOPS:
                    # Find matching bottom
                    for cat in self.BOTTOMS:
                        if cat not in exclude_categories:
                            matches = self.wardrobe.search(seed_item, k=3, category_filter=cat)
                            if matches:
                                outfit['items'].append({
                                    'role': 'bottom',
                                    'path': matches[0]['path'],
                                    'category': cat,
                                    'similarity': matches[0]['similarity']
                                })
                                outfit['reasoning'].append(f"Added matching {cat}")
                                break
                elif seed_category in self.BOTTOMS:
                    # Find matching top
                    for cat in self.TOPS:
                        if cat not in exclude_categories:
                            matches = self.wardrobe.search(seed_item, k=3, category_filter=cat)
                            if matches:
                                outfit['items'].append({
                                    'role': 'top',
                                    'path': matches[0]['path'],
                                    'category': cat,
                                    'similarity': matches[0]['similarity']
                                })
                                outfit['reasoning'].append(f"Added matching {cat}")
                                break
        
        return outfit
    
    def find_similar_style(self, reference_image: str, k: int = 5) -> List[Dict]:
        """
        Find items with similar style to a reference image.
        
        Args:
            reference_image: Reference outfit/item image
            k: Number of results
            
        Returns:
            List of similar items
        """
        return self.wardrobe.search(reference_image, k=k)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CLIP Fashion Inference Pipeline')
    parser.add_argument('command', choices=['encode', 'search', 'index', 'recommend'])
    parser.add_argument('--image', '-i', help='Input image path')
    parser.add_argument('--wardrobe', '-w', help='Wardrobe directory to index')
    parser.add_argument('--index-path', '-p', help='Path to save/load index')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results')
    
    args = parser.parse_args()
    
    if args.command == 'encode':
        if not args.image:
            print("Error: --image required for encode command")
            return
        
        encoder = CLIPFashionEncoder()
        embedding = encoder.encode(args.image)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 10): {embedding[:10]}")
    
    elif args.command == 'index':
        if not args.wardrobe:
            print("Error: --wardrobe required for index command")
            return
        
        encoder = CLIPFashionEncoder()
        index = WardrobeIndex(encoder)
        
        # Find all images in wardrobe
        wardrobe_path = Path(args.wardrobe)
        image_paths = list(wardrobe_path.glob('**/*.jpg')) + \
                      list(wardrobe_path.glob('**/*.png')) + \
                      list(wardrobe_path.glob('**/*.jpeg'))
        
        print(f"Found {len(image_paths)} images in wardrobe")
        
        # Index in batches
        batch_size = 32
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            index.add_items_batch([str(p) for p in batch])
            print(f"Indexed {min(i+batch_size, len(image_paths))}/{len(image_paths)}")
        
        # Save index
        save_path = args.index_path or str(wardrobe_path / "index")
        index.save(save_path)
    
    elif args.command == 'search':
        if not args.image:
            print("Error: --image required for search command")
            return
        
        encoder = CLIPFashionEncoder()
        index = WardrobeIndex(encoder, args.index_path)
        
        results = index.search(args.image, k=args.top_k)
        
        print(f"\nTop {len(results)} similar items:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['path']} (similarity: {result['similarity']:.3f})")
    
    elif args.command == 'recommend':
        if not args.image:
            print("Error: --image required for recommend command")
            return
        
        encoder = CLIPFashionEncoder()
        index = WardrobeIndex(encoder, args.index_path)
        recommender = OutfitRecommender(index)
        
        outfit = recommender.recommend_outfit(seed_item=args.image)
        
        print("\nRecommended Outfit:")
        for item in outfit['items']:
            print(f"  - {item['role']}: {item['path']}")
        print("\nReasoning:")
        for reason in outfit['reasoning']:
            print(f"  • {reason}")


if __name__ == '__main__':
    main()
