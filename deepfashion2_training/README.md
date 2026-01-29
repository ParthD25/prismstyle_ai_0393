# DeepFashion2 Training Repository

## 🎯 Overview

Complete training pipeline for DeepFashion2 clothing classification model optimized for mobile deployment (TFLite + Core ML).

## 🖥️ System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060/3070/3080/4070/4080 recommended)
- **RAM**: 32GB minimum (64GB recommended)
- **Storage**: 2TB SSD (for dataset + checkpoints)
- **CPU**: Modern multi-core processor

### Software
- **OS**: Ubuntu 20.04/22.04 LTS or Windows 10/11 with WSL2
- **Python**: 3.8-3.11
- **CUDA**: 11.8 or 12.1
- **cuDNN**: Compatible with CUDA version

## 📁 Project Structure

```
deepfashion2_training/
├── config.py                 # Training configuration
├── requirements.txt          # Python dependencies
├── download_dataset.py       # Dataset downloader
├── train_model.py            # Main training script
├── convert_model.py          # Model conversion to mobile formats
├── data/                     # Dataset storage
│   └── deepfashion2/         # Extracted dataset
├── models/                   # Trained models
│   └── checkpoints/          # Model checkpoints
├── converted_models/         # TFLite and Core ML models
├── logs/                     # Training logs and metrics
└── notebooks/                # Jupyter notebooks
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd deepfashion2_training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download and extract DeepFashion2 dataset
python download_dataset.py --base-path .

# Or if you have the dataset already:
# Place it in data/deepfashion2/ with train/validation folders
```

### 3. Train Model

```bash
# Start training
python train_model.py --epochs 50 --batch-size 32

# Monitor training with TensorBoard
tensorboard --logdir logs/
```

### 4. Convert to Mobile Formats

```bash
# Convert trained model to TFLite and Core ML
python convert_model.py --model-path models/final_model_finetuned.h5 --output-dir converted_models
```

### 5. Integrate with Flutter App

```bash
# Copy models to Flutter project
cp converted_models/deepfashion2_classifier.tflite ../assets/models/
cp converted_models/FashionClassifier.mlmodel ../ios/Runner/Resources/
```

## ⚙️ Configuration

Edit `config.py` to customize training:

```python
# Key parameters
NUM_CLASSES = 13           # DeepFashion2 categories
IMAGE_SIZE = 224           # Input size
BATCH_SIZE = 32            # Adjust for your GPU
EPOCHS = 50                # Training epochs
LEARNING_RATE = 0.001      # Starting learning rate

# Hardware settings
GPU_MEMORY_LIMIT = None    # Limit GPU memory (GB) or None
MIXED_PRECISION = True     # Enable for faster training
```

## 📊 Expected Results

### Training Timeline (RTX 4070)
- **Epoch 1-10**:  ~15 mins/epoch (transfer learning)
- **Epoch 11-30**: ~12 mins/epoch (fine-tuning)
- **Total**: ~8-10 hours for 50 epochs

### Model Performance
- **Top-1 Accuracy**: 90-92%
- **Top-5 Accuracy**: 98-99%
- **Inference Speed**: 
  - RTX 4070: ~8ms
  - iPhone 14 Pro: ~15ms
  - Mid-range Android: ~45ms

### Model Sizes
- **Keras (.h5)**: ~35MB
- **TFLite (float16)**: ~18MB
- **TFLite (quantized)**: ~9MB
- **Core ML**: ~35MB

## 🔧 Advanced Usage

### Resume Training
```bash
python train_model.py --model-path models/checkpoints/best_model.h5
```

### Custom Dataset Organization
Structure your data as:
```
data/deepfashion2/
├── train/
│   ├── short_sleeve_top/
│   ├── long_sleeve_top/
│   └── ... (13 categories)
└── validation/
    ├── short_sleeve_top/
    ├── long_sleeve_top/
    └── ... (13 categories)
```

### Distributed Training
Modify `train_model.py` to use `tf.distribute.MirroredStrategy()` for multi-GPU setups.

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 16

# Limit GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true
# or in config.py: GPU_MEMORY_LIMIT = 8
```

### Slow Training
```bash
# Enable mixed precision
config.MIXED_PRECISION = True

# Use faster data loading
# Ensure SSD storage for dataset
```

### Import Errors
```bash
# Reinstall TensorFlow with GPU support
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow[and-cuda]==2.13.0
```

## 📈 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/
# Visit http://localhost:6006
```

### Training Metrics
- Loss curves
- Accuracy progression
- Learning rate schedule
- Confusion matrix
- Classification report

## 📱 Mobile Integration

### Flutter Setup
1. Place TFLite model in `assets/models/`
2. Update `pubspec.yaml`:
```yaml
assets:
  - assets/models/deepfashion2_classifier.tflite
```

### iOS Setup
1. Place Core ML model in `ios/Runner/Resources/`
2. Add to Xcode project
3. Configure build settings

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- DeepFashion2 dataset creators
- TensorFlow/Keras team
- Core ML Tools team
- EfficientNet authors

---

**Happy Training!** 🚀