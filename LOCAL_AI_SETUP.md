# PrismStyle AI - Local Python AI Setup

## Overview
Your app now supports a **FREE local AI backend** that uses:
- **GroundingDINO** - Open-set object detection (detects any clothing item by text prompt)
- **SAM (Segment Anything)** - Precise garment segmentation
- **OpenCLIP / FashionCLIP** - Fashion-aware embeddings for similarity search
- **Open-Meteo** - Free weather API (already integrated, no key needed)

**No paid APIs required!** Everything runs locally on your computer.

---

## Quick Setup

### 1. Extract the outfit_app_package
Extract `outfit_app_package_updated.zip` to `C:\outfit-app\`

### 2. Run the setup script (first time only)
```batch
C:\outfit-app> setup_and_download_models.bat
```
This will:
- Create Python environment with GPU PyTorch
- Download model weights (SAM, GroundingDINO, FashionCLIP)
- Build your wardrobe FAISS index

### 3. Start the Flask backend for Flutter
```batch
C:\Users\pdave\Downloads\prismstyle_ai_0393-main\scripts\python_backend> start_backend.bat
```
This runs on `http://localhost:5000`

### 4. Run the Flutter app
The app will automatically connect to the local Python backend!

---

## AI Model Weights (Priority)

| Model | Weight | Status |
|-------|--------|--------|
| Local Python AI (GroundingDINO + OpenCLIP) | 35% | PRIMARY - runs locally |
| TFLite (custom trained) | 20% | Available after you train |
| ONNX (H100-trained) | 15% | Fallback |
| Apple Vision (iOS) | 15% | iOS only - on-device Apple AI |
| Apple Core ML (iOS) | 10% | iOS only - on-device Apple AI |
| Heuristic | 5% | Always available |

> **Note:** On iOS, Apple's on-device AI (Vision Framework, Core ML, Visual Intelligence) is prioritized for fast, private inference without needing any API keys.

---

## Features

### Weather-Aware Outfit Suggestions
Uses Open-Meteo (free) to fetch weather and suggests appropriate outfits:
```batch
run_suggest_weather.bat "Newark, CA"
```

### Selfie Analysis
Detects and segments clothing from selfies, finds similar items in your wardrobe:
```batch
run_analyze_selfie.bat C:\path\to\selfie.jpg
```

### Wardrobe Similarity Search
Find items similar to any photo using FashionCLIP embeddings.

---

## Files Added

### Flutter Services
- `lib/services/local_python_ai_service.dart` - Connects to Python backend

### Python Backend
- `scripts/python_backend/server.py` - Flask API server
- `scripts/python_backend/start_backend.bat` - Start script
- `scripts/python_backend/requirements.txt` - Python dependencies

---

## Tomorrow: Model Training

When you run the training tomorrow, the TFLite model will be saved to:
```
assets/models/deepfashion2_classifier.tflite
```

This will enable the TFLite classifier (25% weight) for even better accuracy!

---

## Troubleshooting

### "Local Python AI not available"
Make sure the Flask backend is running:
```batch
scripts\python_backend\start_backend.bat
```

### "Outfit app not found"
Update `OUTFIT_APP_PATH` in `start_backend.bat` to your actual path.

### First run slow
The first run downloads model weights (~2GB). After that, everything is cached locally.
