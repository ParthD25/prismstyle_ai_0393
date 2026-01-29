import os
from PIL import Image
import json

def verify_closet(closet_dir="outfit_pipeline/closet"):
    print(f"Verifying closet in {closet_dir}...")
    if not os.path.exists(closet_dir):
        print(f"Error: {closet_dir} does not exist.")
        return

    items = os.listdir(closet_dir)
    valid_count = 0
    errors = []

    for item in items:
        item_path = os.path.join(closet_dir, item)
        if item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            try:
                with Image.open(item_path) as img:
                    img.verify()
                valid_count += 1
            except Exception as e:
                errors.append(f"Invalid image {item}: {e}")

    print(f"Found {len(items)} items.")
    print(f"Valid images: {valid_count}")
    if errors:
        print("\nErrors found:")
        for err in errors:
            print(f"  - {err}")

if __name__ == "__main__":
    verify_closet()
