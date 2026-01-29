import os
import numpy as np
import json
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import argparse

class Recommender:
    def __init__(self, embeddings_path="assets/models/outfit_recommender/embeddings.npy", metadata_path="assets/models/outfit_recommender/metadata.json"):
        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            self.closet_embeddings = np.load(embeddings_path)
            with open(metadata_path, 'r') as f:
                self.closet_metadata = json.load(f)
        else:
            self.closet_embeddings = None
            self.closet_metadata = []
            print("Warning: Closet embeddings not found. Please run embed_wardrobe.py first.")

    def get_context_score(self, category, context):
        # score = c*(context_match_rules)
        score = 0.0
        occasion = context.get("occasion", "casual").lower()
        time = context.get("time", "day").lower()
        
        rules = {
            "formal": {"top": 0.8, "shoes": 0.9, "dress": 1.0, "shorts": -0.5},
            "casual": {"top": 1.0, "bottom": 1.0, "shoes": 0.8},
            "business": {"top": 0.9, "bottom": 0.9, "shoes": 0.9}
        }
        
        category_base = category.split('_')[-1] # simple mapping
        score += rules.get(occasion, {}).get(category_base, 0.5)
        
        if time == "night" and "dark" in category.lower():
            score += 0.2
            
        return score

    def recommend(self, query_embedding, context, top_k=3):
        if self.closet_embeddings is None:
            return []

        # Similarity score
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.closet_embeddings)[0]
        
        recommendations = []
        for i, sim in enumerate(similarities):
            item = self.closet_metadata[i]
            # Context re-ranking
            # score = a*(embedding_similarity) + c*(context_match_rules)
            context_score = self.get_context_score(item["filename"], context)
            final_score = 0.7 * sim + 0.3 * context_score
            
            recommendations.append({
                "item_id": item["item_id"],
                "filename": item["filename"],
                "score": float(final_score),
                "similarity": float(sim)
            })

        # Rank
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_k]

    def get_swap_suggestion(self, current_outfit_score, best_recommendation, context):
        # Template-based swap suggestions
        occasion = context.get("occasion", "casual")
        if current_outfit_score < 0.6:
            return f"This outfit looks a bit {occasion} for the occasion. Consider swapping your shoes to '{best_recommendation['filename']}' for a better match."
        return "Your outfit is solid! But adding a different accessory might spice it up."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_img", help="Path to query image")
    parser.add_argument("--occasion", default="casual")
    args = parser.parse_args()
    
    if args.query_img:
        from embed_wardrobe import FashionEmbedder
        embedder = FashionEmbedder()
        query_vec = embedder.embed(args.query_img)
        
        rec = Recommender()
        results = rec.recommend(query_vec[0], {"occasion": args.occasion})
        
        print("\n--- Outfit Recommendations ---")
        for i, res in enumerate(results):
            print(f"{i+1}. {res['filename']} (Score: {res['score']:.4f})")
        
        if results:
            print("\nStyle Tip:", rec.get_swap_suggestion(0.5, results[0], {"occasion": args.occasion}))
    else:
        print("Please provide a query image with --query_img")
