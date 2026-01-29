# DeepFashion2 Training Configuration
import os

# Base path (works on both Windows and Unix)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset Configuration
DATASET_PATH = os.path.join(BASE_DIR, "data", "deepfashion2")
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "annotations")
IMAGES_PATH = DATASET_PATH  # ImageDataGenerator will use train/validation subfolders

# Model Configuration
NUM_CLASSES = 13  # DeepFashion2 major categories
IMAGE_SIZE = 224  # Input size for EfficientNet
BATCH_SIZE = 32   # Adjust based on GPU memory (16 for 8GB VRAM, 32 for 12GB+)
LEARNING_RATE = 0.001
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# For achieving 95%+ accuracy, use these enhanced settings:
USE_ENHANCED_AUGMENTATION = True
USE_MIXUP = True
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.1

# Categories (DeepFashion2 major categories)
CATEGORIES = [
    'short_sleeve_top',
    'long_sleeve_top', 
    'short_sleeve_outwear',
    'long_sleeve_outwear',
    'vest',
    'sling',
    'shorts',
    'trousers',
    'skirt',
    'short_sleeve_dress',
    'long_sleeve_dress',
    'vest_dress',
    'sling_dress'
]

# Training Configuration
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.h5")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Hardware Configuration
GPU_MEMORY_LIMIT = None  # Set to integer GB if needed (e.g., 8)
MIXED_PRECISION = True   # Use mixed precision training for speed

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "converted_models"), exist_ok=True)
