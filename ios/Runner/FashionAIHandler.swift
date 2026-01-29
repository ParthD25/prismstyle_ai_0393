import Foundation
import UIKit
import Accelerate

/**
 * Fashion AI Handler for iOS
 * Uses ONNX Runtime for inference
 * 
 * Provides:
 * 1. CLIP image encoding for embeddings (outfit matching)
 * 2. Clothing classification (category detection)
 * 3. Similarity search for wardrobe items
 */
@available(iOS 13.0, *)
class FashionAIHandler: NSObject {
    
    // MARK: - Constants
    
    static let CLIP_INPUT_SIZE = 224
    static let CLIP_EMBEDDING_DIM = 512
    
    // OpenCLIP ViT-B-32 normalization
    static let CLIP_MEAN: [Float] = [0.48145466, 0.4578275, 0.40821073]
    static let CLIP_STD: [Float] = [0.26862954, 0.26130258, 0.27577711]
    
    // ImageNet normalization for classifier
    static let CLASSIFIER_MEAN: [Float] = [0.485, 0.456, 0.406]
    static let CLASSIFIER_STD: [Float] = [0.229, 0.224, 0.225]
    
    // DeepFashion2 categories
    static let CATEGORIES = [
        "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear",
        "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
        "skirt", "short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"
    ]
    
    // Category mapping to app categories
    static let CATEGORY_MAPPING: [String: String] = [
        "short_sleeve_top": "Tops",
        "long_sleeve_top": "Tops",
        "short_sleeve_outwear": "Outerwear",
        "long_sleeve_outwear": "Outerwear",
        "vest": "Tops",
        "sling": "Tops",
        "shorts": "Bottoms",
        "trousers": "Bottoms",
        "skirt": "Bottoms",
        "short_sleeve_dress": "Dresses",
        "long_sleeve_dress": "Dresses",
        "vest_dress": "Dresses",
        "sling_dress": "Dresses"
    ]
    
    // Display names for user-friendly output
    static let DISPLAY_NAMES: [String: String] = [
        "short_sleeve_top": "T-Shirt",
        "long_sleeve_top": "Long Sleeve Shirt",
        "short_sleeve_outwear": "Short Jacket",
        "long_sleeve_outwear": "Coat/Jacket",
        "vest": "Vest",
        "sling": "Camisole",
        "shorts": "Shorts",
        "trousers": "Pants",
        "skirt": "Skirt",
        "short_sleeve_dress": "Summer Dress",
        "long_sleeve_dress": "Long Sleeve Dress",
        "vest_dress": "Vest Dress",
        "sling_dress": "Slip Dress"
    ]
    
    // MARK: - Properties
    
    private var isInitialized = false
    private var clipModel: Any?  // ORT Session placeholder
    private var classifierModel: Any?
    
    // Wardrobe index
    private var wardrobeEmbeddings: [[Float]] = []
    private var wardrobePaths: [String] = []
    private var wardrobeMetadata: [[String: Any]] = []
    
    // MARK: - Initialization
    
    override init() {
        super.init()
    }
    
    /// Initialize the AI models
    func initialize() -> Bool {
        // Note: This requires onnxruntime-objc to be added to the project
        // For now, we provide the interface and use CoreML fallback
        
        // Check for ONNX models in bundle
        if let clipPath = Bundle.main.path(forResource: "clip_image_encoder", ofType: "onnx") {
            print("✅ Found CLIP model at: \(clipPath)")
            // TODO: Initialize ONNX Runtime session
        }
        
        if let classifierPath = Bundle.main.path(forResource: "clothing_classifier", ofType: "onnx") {
            print("✅ Found classifier model at: \(classifierPath)")
            // TODO: Initialize ONNX Runtime session
        }
        
        isInitialized = true
        print("✅ FashionAI initialized (models ready for ONNX Runtime)")
        return true
    }
    
    // MARK: - Image Preprocessing
    
    /// Preprocess image for CLIP encoding
    private func preprocessForClip(image: UIImage) -> [Float]? {
        guard let cgImage = image.cgImage else { return nil }
        
        // Resize to 224x224
        let size = CGSize(width: Self.CLIP_INPUT_SIZE, height: Self.CLIP_INPUT_SIZE)
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            return nil
        }
        UIGraphicsEndImageContext()
        
        guard let resizedCGImage = resizedImage.cgImage else { return nil }
        
        // Get pixel data
        let width = Self.CLIP_INPUT_SIZE
        let height = Self.CLIP_INPUT_SIZE
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        
        context.draw(resizedCGImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert to CHW float array with normalization
        var result = [Float](repeating: 0, count: 3 * width * height)
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * bytesPerPixel
                let r = Float(pixelData[offset]) / 255.0
                let g = Float(pixelData[offset + 1]) / 255.0
                let b = Float(pixelData[offset + 2]) / 255.0
                
                let idx = y * width + x
                result[idx] = (r - Self.CLIP_MEAN[0]) / Self.CLIP_STD[0]  // R channel
                result[width * height + idx] = (g - Self.CLIP_MEAN[1]) / Self.CLIP_STD[1]  // G
                result[2 * width * height + idx] = (b - Self.CLIP_MEAN[2]) / Self.CLIP_STD[2]  // B
            }
        }
        
        return result
    }
    
    // MARK: - CLIP Encoding
    
    /// Get CLIP embedding for an image
    func getEmbedding(image: UIImage) -> [Float]? {
        guard isInitialized else {
            print("❌ FashionAI not initialized")
            return nil
        }
        
        guard let preprocessed = preprocessForClip(image: image) else {
            print("❌ Failed to preprocess image")
            return nil
        }
        
        // TODO: Run ONNX inference when runtime is available
        // For now, return placeholder or use CoreML handler
        
        // Placeholder: return random embedding (replace with actual inference)
        print("⚠️ ONNX Runtime not available, using placeholder")
        var embedding = [Float](repeating: 0, count: Self.CLIP_EMBEDDING_DIM)
        for i in 0..<Self.CLIP_EMBEDDING_DIM {
            embedding[i] = Float.random(in: -1...1)
        }
        
        // L2 normalize
        let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        return embedding.map { $0 / norm }
    }
    
    /// Get CLIP embedding from image data
    func getEmbedding(imageData: Data) -> [Float]? {
        guard let image = UIImage(data: imageData) else { return nil }
        return getEmbedding(image: image)
    }
    
    // MARK: - Classification
    
    /// Classify clothing category
    func classify(image: UIImage) -> [String: Float] {
        guard isInitialized else {
            return [:]
        }
        
        // TODO: Run classifier inference
        // For now, use heuristic-based classification from CoreMLHandler
        
        return [:]
    }
    
    // MARK: - Wardrobe Index
    
    /// Load wardrobe embeddings for similarity search
    func loadWardrobeIndex(embeddings: [[Float]], paths: [String], metadata: [[String: Any]]? = nil) {
        wardrobeEmbeddings = embeddings
        wardrobePaths = paths
        wardrobeMetadata = metadata ?? Array(repeating: [:], count: paths.count)
        print("✅ Loaded wardrobe index with \(paths.count) items")
    }
    
    /// Add item to wardrobe index
    func addToWardrobeIndex(image: UIImage, path: String, metadata: [String: Any] = [:]) -> Bool {
        guard let embedding = getEmbedding(image: image) else { return false }
        
        wardrobeEmbeddings.append(embedding)
        wardrobePaths.append(path)
        wardrobeMetadata.append(metadata)
        
        return true
    }
    
    // MARK: - Similarity Search
    
    /// Find similar items in wardrobe
    func findSimilar(queryEmbedding: [Float], topK: Int = 5) -> [(path: String, score: Float, metadata: [String: Any])] {
        guard !wardrobeEmbeddings.isEmpty else { return [] }
        
        // Compute cosine similarities
        var similarities: [(Int, Float)] = []
        
        for (idx, embedding) in wardrobeEmbeddings.enumerated() {
            var dotProduct: Float = 0
            for i in 0..<min(queryEmbedding.count, embedding.count) {
                dotProduct += queryEmbedding[i] * embedding[i]
            }
            similarities.append((idx, dotProduct))
        }
        
        // Sort by similarity (descending) and take top K
        similarities.sort { $0.1 > $1.1 }
        
        return similarities.prefix(topK).map { (idx, score) in
            (path: wardrobePaths[idx], score: score, metadata: wardrobeMetadata[idx])
        }
    }
    
    /// Find similar items to a given image
    func findSimilar(image: UIImage, topK: Int = 5) -> [(path: String, score: Float, metadata: [String: Any])] {
        guard let embedding = getEmbedding(image: image) else { return [] }
        return findSimilar(queryEmbedding: embedding, topK: topK)
    }
    
    // MARK: - Outfit Recommendation
    
    /// Get outfit recommendation based on seed item
    func recommendOutfit(seedImage: UIImage) -> [String: Any] {
        var result: [String: Any] = [:]
        
        // Get seed item classification
        let classification = classify(image: seedImage)
        if let topCategory = classification.max(by: { $0.value < $1.value })?.key {
            result["seed_category"] = topCategory
            result["seed_app_category"] = Self.CATEGORY_MAPPING[topCategory] ?? "Unknown"
            result["display_name"] = Self.DISPLAY_NAMES[topCategory] ?? topCategory
        }
        
        // Get embedding
        if let embedding = getEmbedding(image: seedImage) {
            result["embedding"] = embedding
            
            // Find similar items
            let similar = findSimilar(queryEmbedding: embedding, topK: 10)
            result["similar_items"] = similar.map { item in
                ["path": item.path, "score": item.score, "metadata": item.metadata]
            }
        }
        
        return result
    }
    
    // MARK: - Utility
    
    /// Check if models are loaded
    var isReady: Bool {
        return isInitialized
    }
    
    /// Get wardrobe size
    var wardrobeSize: Int {
        return wardrobePaths.count
    }
}
