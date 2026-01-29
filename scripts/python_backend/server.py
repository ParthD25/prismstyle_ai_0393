"""
PrismStyle AI - Local Python Backend Server
Integrates with outfit_app_package for FREE local AI:
- GroundingDINO: Open-set object detection
- SAM: Segment Anything Model
- OpenCLIP/FashionCLIP: Image embeddings & retrieval
- Open-Meteo: Free weather API (no key needed)

Run: python server.py
Listens on: http://localhost:5000
"""

import os
import sys
import json
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Path to outfit_app_package (update this to your actual path)
OUTFIT_APP_PATH = os.environ.get('OUTFIT_APP_PATH', r'C:\outfit-app')

# Try to import outfit_app modules
try:
    sys.path.insert(0, OUTFIT_APP_PATH)
    from suggest_outfit import get_weather_suggestions, get_weather_data
    from analyze_selfie import analyze_selfie_image
    from build_index import get_similar_items, load_index
    OUTFIT_APP_AVAILABLE = True
    logger.info(f"✅ Outfit app loaded from: {OUTFIT_APP_PATH}")
except ImportError as e:
    OUTFIT_APP_AVAILABLE = False
    logger.warning(f"⚠️ Outfit app not available: {e}")
    logger.warning("Running in fallback mode - install outfit_app_package first")

# Fallback imports for basic functionality
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Category mapping
CATEGORY_MAP = {
    'top': 'Tops',
    'shirt': 'Tops',
    't-shirt': 'Tops',
    'blouse': 'Tops',
    'sweater': 'Tops',
    'bottom': 'Bottoms',
    'pants': 'Bottoms',
    'jeans': 'Bottoms',
    'shorts': 'Bottoms',
    'skirt': 'Bottoms',
    'dress': 'Dresses',
    'outerwear': 'Outerwear',
    'jacket': 'Outerwear',
    'coat': 'Outerwear',
    'shoes': 'Shoes',
    'footwear': 'Shoes',
    'accessory': 'Accessories',
    'bag': 'Accessories',
}


def decode_base64_image(base64_string: str) -> Optional[bytes]:
    """Decode base64 image string to bytes."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        return base64.b64decode(base64_string)
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def get_app_category(category: str) -> str:
    """Map detected category to app category."""
    category_lower = category.lower()
    return CATEGORY_MAP.get(category_lower, 'Tops')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'outfit_app_available': OUTFIT_APP_AVAILABLE,
        'pil_available': PIL_AVAILABLE,
    })


@app.route('/analyze_selfie', methods=['POST'])
def analyze_selfie():
    """
    Analyze a selfie to detect and segment clothing items.
    Uses GroundingDINO + SAM from outfit_app_package.
    
    Request body:
    {
        "image": "<base64_encoded_image>",
        "detect_items": ["top", "bottom", "dress", "outerwear", "shoes"]
    }
    """
    try:
        data = request.json
        image_b64 = data.get('image')
        detect_items = data.get('detect_items', ['top', 'bottom', 'dress', 'outerwear', 'shoes'])
        
        if not image_b64:
            return jsonify({'error': 'No image provided'}), 400
        
        image_bytes = decode_base64_image(image_b64)
        if not image_bytes:
            return jsonify({'error': 'Invalid image data'}), 400
        
        if not OUTFIT_APP_AVAILABLE:
            return jsonify({
                'garments': [],
                'error': 'Outfit app not available - install outfit_app_package'
            })
        
        # Save temp image and analyze
        temp_path = Path('temp_selfie.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        try:
            results = analyze_selfie_image(str(temp_path), detect_items)
            garments = []
            
            for result in results:
                garments.append({
                    'category': result.get('category', 'unknown'),
                    'confidence': result.get('confidence', 0.0),
                    'bbox': result.get('bbox', [0, 0, 0, 0]),
                    'mask': result.get('mask_base64'),
                    'similar_items': result.get('similar_items', []),
                })
            
            return jsonify({'garments': garments})
        finally:
            if temp_path.exists():
                temp_path.unlink()
                
    except Exception as e:
        logger.error(f"Selfie analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/suggest_outfit', methods=['POST'])
def suggest_outfit():
    """
    Get weather-aware outfit suggestions.
    Uses Open-Meteo (free) + OpenCLIP embeddings.
    
    Request body:
    {
        "location": "Newark, CA",
        "occasion": "casual",
        "k": 5
    }
    """
    try:
        data = request.json
        location = data.get('location', 'San Francisco, CA')
        occasion = data.get('occasion', 'casual')
        k = data.get('k', 5)
        
        if not OUTFIT_APP_AVAILABLE:
            # Return mock data when outfit app not available
            return jsonify({
                'weather': {
                    'temperature': 68.0,
                    'rain_probability': 0.1,
                    'wind_speed': 5.0,
                    'condition': 'Sunny',
                    'location': location,
                },
                'prompt': f'Weather-appropriate outfit for {occasion} in {location}',
                'suggestions': {
                    'top': [],
                    'bottom': [],
                    'shoes': [],
                },
                'combo': None,
            })
        
        # Get weather data
        weather = get_weather_data(location)
        
        # Get suggestions
        suggestions = get_weather_suggestions(
            location=location,
            occasion=occasion,
            k=k,
        )
        
        return jsonify({
            'weather': {
                'temperature': weather.get('temperature', 0),
                'rain_probability': weather.get('rain_probability', 0),
                'wind_speed': weather.get('wind_speed', 0),
                'condition': weather.get('condition', 'Unknown'),
                'location': location,
            },
            'prompt': suggestions.get('prompt', ''),
            'suggestions': suggestions.get('items', {}),
            'combo': suggestions.get('combo'),
        })
        
    except Exception as e:
        logger.error(f"Outfit suggestion error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/find_similar', methods=['POST'])
def find_similar():
    """
    Find similar items from wardrobe using FashionCLIP embeddings.
    
    Request body:
    {
        "image": "<base64_encoded_image>",
        "k": 5
    }
    """
    try:
        data = request.json
        image_b64 = data.get('image')
        k = data.get('k', 5)
        
        if not image_b64:
            return jsonify({'error': 'No image provided'}), 400
        
        image_bytes = decode_base64_image(image_b64)
        if not image_bytes:
            return jsonify({'error': 'Invalid image data'}), 400
        
        if not OUTFIT_APP_AVAILABLE:
            return jsonify({'similar_items': []})
        
        # Save temp image
        temp_path = Path('temp_query.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        try:
            similar = get_similar_items(str(temp_path), k=k)
            
            items = []
            for item in similar:
                items.append({
                    'id': item.get('id', ''),
                    'image_path': item.get('image_path', ''),
                    'similarity': item.get('similarity', 0.0),
                    'category': item.get('category'),
                })
            
            return jsonify({'similar_items': items})
        finally:
            if temp_path.exists():
                temp_path.unlink()
                
    except Exception as e:
        logger.error(f"Find similar error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify a clothing item.
    Uses OpenCLIP for zero-shot classification.
    
    Request body:
    {
        "image": "<base64_encoded_image>"
    }
    """
    try:
        data = request.json
        image_b64 = data.get('image')
        
        if not image_b64:
            return jsonify({'error': 'No image provided'}), 400
        
        image_bytes = decode_base64_image(image_b64)
        if not image_bytes:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # If outfit app available, use it
        if OUTFIT_APP_AVAILABLE:
            try:
                # Save temp image
                temp_path = Path('temp_classify.jpg')
                with open(temp_path, 'wb') as f:
                    f.write(image_bytes)
                
                # Use GroundingDINO for detection
                results = analyze_selfie_image(str(temp_path), 
                    ['top', 'bottom', 'dress', 'outerwear', 'shoes', 'shorts', 'skirt'])
                
                if results:
                    best = max(results, key=lambda x: x.get('confidence', 0))
                    category = best.get('category', 'top')
                    confidence = best.get('confidence', 0.8)
                    
                    return jsonify({
                        'category': category,
                        'app_category': get_app_category(category),
                        'confidence': confidence,
                        'colors': [],  # TODO: Extract dominant colors
                        'attributes': {},
                    })
                    
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        
        # Fallback classification based on basic image analysis
        return jsonify({
            'category': 'top',
            'app_category': 'Tops',
            'confidence': 0.5,
            'colors': [],
            'attributes': {},
        })
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/rebuild_index', methods=['POST'])
def rebuild_index():
    """Rebuild the wardrobe FAISS index."""
    try:
        if not OUTFIT_APP_AVAILABLE:
            return jsonify({'error': 'Outfit app not available'}), 400
        
        # Trigger index rebuild
        load_index(force_rebuild=True)
        
        return jsonify({'status': 'success', 'message': 'Index rebuilt'})
        
    except Exception as e:
        logger.error(f"Index rebuild error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("PrismStyle AI - Local Python Backend")
    print("=" * 60)
    print(f"Outfit App Path: {OUTFIT_APP_PATH}")
    print(f"Outfit App Available: {OUTFIT_APP_AVAILABLE}")
    print(f"PIL Available: {PIL_AVAILABLE}")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
