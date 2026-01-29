import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import json

class OutfitParser:
    def __init__(self, model_path="assets/models/outfit_recommender/human_parser.onnx"):
        self.model_path = model_path
        if os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path)
        else:
            self.session = None
            print(f"Warning: Model {model_path} not found. Skipping segmentation.")

    def preprocess(self, image_path, size=(512, 512)):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)
        img_data = np.array(img).transpose(2, 0, 1)
        img_data = img_data.astype('float32') / 255.0
        return np.expand_dims(img_data, axis=0)

    def parse(self, image_path):
        if not self.session:
            return None
        
        input_data = self.preprocess(image_path)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_data})
        mask = np.argmax(output[0][0], axis=0)
        return mask

    def crop_segments(self, image_path, output_dir="assets/wardrobe_sample/output"):
        mask = self.parse(image_path)
        if mask is None:
            return
        
        img = cv2.imread(image_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Classes for fashn-human-parser (Segformer mapping)
        classes = {
            "top": [3], "bottom": [5, 6], "dress": [4], 
            "shoes": [15], "bag": [8], "hat": [9],
            "accessory": [7, 10, 11, 17] # belt, scarf, glasses, jewelry
        }
        
        results = []
        for name, ids in classes.items():
            combined_mask = np.isin(mask, ids).astype('uint8')
            if np.sum(combined_mask) == 0:
                continue
                
            # Resize mask to original image size
            combined_mask = cv2.resize(combined_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Find bounding box
            coords = cv2.findNonZero(combined_mask)
            x, y, w, h = cv2.boundingRect(coords)
            
            crop = img[y:y+h, x:x+w]
            save_path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(save_path, crop)
            results.append({"category": name, "path": save_path})
            
        return results

if __name__ == "__main__":
    parser = OutfitParser()
    # parser.crop_segments("path/to/test.jpg")
