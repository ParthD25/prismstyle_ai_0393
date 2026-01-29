#!/usr/bin/env python3
"""
DeepFashion2 Dataset Downloader and Extractor

Downloads and prepares the DeepFashion2 dataset for training.
Password: 2019Deepfashion2
"""

import os
import zipfile
import requests
from tqdm import tqdm
import argparse
from pathlib import Path

# DeepFashion2 download URLs (you'll need to get these from the official source)
DEEPFASHION2_URLS = [
    "https://drive.google.com/uc?id=1KFW0XQT8i9Xmt7TVOcSL3x7x6vKC71sY",  # Part 1
    "https://drive.google.com/uc?id=1KFW0XQT8i9Xmt7TVOcSL3x7x6vKC71sY",  # Part 2
    # Add actual URLs here
]

def download_file(url, filename, chunk_size=1024):
    """Download file with progress bar"""
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def setup_directories(base_path):
    """Create required directories"""
    dirs = [
        base_path,
        f"{base_path}/data",
        f"{base_path}/data/deepfashion2",
        f"{base_path}/models",
        f"{base_path}/models/checkpoints",
        f"{base_path}/logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description='Download DeepFashion2 dataset')
    parser.add_argument('--base-path', default='.', help='Base path for project')
    parser.add_argument('--skip-download', action='store_true', help='Skip download if files exist')
    
    args = parser.parse_args()
    
    base_path = args.base_path
    setup_directories(base_path)
    
    # Check if dataset already exists
    dataset_path = f"{base_path}/data/deepfashion2/train"
    if os.path.exists(dataset_path) and not args.skip_download:
        print("Dataset already exists. Use --skip-download to skip.")
        return
    
    # Download dataset parts
    for i, url in enumerate(DEEPFASHION2_URLS):
        zip_filename = f"{base_path}/data/deepfashion2_part_{i+1}.zip"
        
        if not os.path.exists(zip_filename) or not args.skip_download:
            download_file(url, zip_filename)
        
        # Extract
        extract_zip(zip_filename, f"{base_path}/data/deepfashion2")
        
        # Remove zip to save space
        os.remove(zip_filename)
        print(f"Removed {zip_filename}")
    
    print("Dataset setup complete!")

if __name__ == "__main__":
    main()