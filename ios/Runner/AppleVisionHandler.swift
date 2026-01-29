import Foundation
import Vision
import UIKit

/// Apple Vision Framework handler for clothing classification
/// Uses Vision's built-in image classification (FREE)
/// Enhanced with DeepFashion2 category mapping for better accuracy
@available(iOS 13.0, *)
class AppleVisionHandler: NSObject {
    
    private var isInitialized = false
    
    // DeepFashion2 category mapping (matches trained model)
    private let deepFashion2Categories: [String: [String]] = [
        "short_sleeve_top": ["t-shirt", "tee", "polo", "tank top", "crop top", "jersey"],
        "long_sleeve_top": ["shirt", "blouse", "sweater", "pullover", "long sleeve", "henley", "thermal"],
        "short_sleeve_outwear": ["short jacket", "bolero", "shrug"],
        "long_sleeve_outwear": ["jacket", "coat", "blazer", "hoodie", "cardigan", "parka", "windbreaker", "overcoat"],
        "vest": ["vest", "waistcoat", "gilet", "sleeveless jacket"],
        "sling": ["camisole", "spaghetti", "strap top", "halter", "tube top"],
        "shorts": ["shorts", "short pants", "bermuda", "hot pants", "athletic shorts"],
        "trousers": ["pants", "jeans", "trousers", "slacks", "chinos", "khakis", "leggings", "joggers"],
        "skirt": ["skirt", "mini skirt", "maxi skirt", "pencil skirt", "a-line skirt"],
        "short_sleeve_dress": ["summer dress", "casual dress", "shift dress", "sundress"],
        "long_sleeve_dress": ["formal dress", "winter dress", "maxi dress", "wrap dress"],
        "vest_dress": ["pinafore", "jumper dress", "overall dress"],
        "sling_dress": ["slip dress", "cocktail dress", "evening dress", "party dress"]
    ]
    
    // User-friendly display names
    private let displayNames: [String: String] = [
        "short_sleeve_top": "T-Shirt / Short Sleeve Top",
        "long_sleeve_top": "Shirt / Long Sleeve Top",
        "short_sleeve_outwear": "Short Sleeve Jacket",
        "long_sleeve_outwear": "Jacket / Coat",
        "vest": "Vest",
        "sling": "Sling / Camisole",
        "shorts": "Shorts",
        "trousers": "Pants / Trousers",
        "skirt": "Skirt",
        "short_sleeve_dress": "Short Sleeve Dress",
        "long_sleeve_dress": "Long Sleeve Dress",
        "vest_dress": "Vest Dress",
        "sling_dress": "Sling / Slip Dress"
    ]
    
    // Category groups for higher-level classification
    private let categoryGroups: [String: [String]] = [
        "Tops": ["short_sleeve_top", "long_sleeve_top", "vest", "sling"],
        "Bottoms": ["shorts", "trousers", "skirt"],
        "Dresses": ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"],
        "Outerwear": ["short_sleeve_outwear", "long_sleeve_outwear"]
    ]
    
    /// Initialize Apple Vision Framework
    func initialize() -> Bool {
        isInitialized = true
        print("✅ Apple Vision Framework initialized with DeepFashion2 mapping")
        return true
    }
    
    /// Classify clothing image using Apple Vision Framework
    /// - Parameter imageData: Image bytes
    /// - Returns: Dictionary of category predictions with confidence scores
    func classifyImage(imageData: Data) -> [String: Double] {
        guard isInitialized else {
            print("❌ Apple Vision not initialized")
            return [:]
        }
        
        guard let image = UIImage(data: imageData),
              let cgImage = image.cgImage else {
            print("❌ Failed to convert image data")
            return [:]
        }
        
        var predictions: [String: Double] = [:]
        let semaphore = DispatchSemaphore(value: 0)
        
        // Create Vision request
        let request = VNClassifyImageRequest { [weak self] request, error in
            defer { semaphore.signal() }
            
            if let error = error {
                print("❌ Vision classification error: \(error.localizedDescription)")
                return
            }
            
            guard let observations = request.results as? [VNClassificationObservation] else {
                print("❌ No classification results")
                return
            }
            
            // Map Vision classifications to DeepFashion2 categories
            predictions = self?.mapVisionToDeepFashion2(observations: observations) ?? [:]
        }
        
        // Configure request for best quality
        request.imageCropAndScaleOption = .scaleFit
        if #available(iOS 15.0, *) {
            request.revision = VNClassifyImageRequestRevision2
        }
        
        // Perform request
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        do {
            try handler.perform([request])
            _ = semaphore.wait(timeout: .now() + 5.0)
        } catch {
            print("❌ Failed to perform Vision request: \(error.localizedDescription)")
        }
        
        return predictions
    }
    
    /// Map Apple Vision classifications to DeepFashion2 categories
    private func mapVisionToDeepFashion2(observations: [VNClassificationObservation]) -> [String: Double] {
        var clothingPredictions: [String: Double] = [:]
        
        // Process top 20 Vision predictions for better coverage
        for observation in observations.prefix(20) {
            let identifier = observation.identifier.lowercased()
            let confidence = Double(observation.confidence)
            
            // Skip low confidence predictions
            guard confidence > 0.01 else { continue }
            
            // Match to DeepFashion2 categories
            for (category, keywords) in deepFashion2Categories {
                if keywords.contains(where: { identifier.contains($0) }) {
                    let currentScore = clothingPredictions[category] ?? 0.0
                    clothingPredictions[category] = max(currentScore, confidence)
                }
            }
            
            // Also check for generic clothing terms and map to likely categories
            if identifier.contains("clothing") || identifier.contains("garment") {
                // Boost all categories slightly
                for category in deepFashion2Categories.keys {
                    let boost = confidence * 0.1
                    clothingPredictions[category] = (clothingPredictions[category] ?? 0.0) + boost
                }
            }
            
            // Check for fabric patterns that indicate clothing
            if identifier.contains("fabric") || identifier.contains("textile") || 
               identifier.contains("denim") || identifier.contains("cotton") {
                // These often indicate pants or tops
                clothingPredictions["trousers"] = max(clothingPredictions["trousers"] ?? 0.0, confidence * 0.3)
                clothingPredictions["short_sleeve_top"] = max(clothingPredictions["short_sleeve_top"] ?? 0.0, confidence * 0.3)
            }
        }
        
        // Normalize predictions
        let maxScore = clothingPredictions.values.max() ?? 1.0
        if maxScore > 0 {
            for (category, score) in clothingPredictions {
                clothingPredictions[category] = score / maxScore
            }
        }
        
        // Add group-level predictions for ensemble integration
        var groupPredictions: [String: Double] = [:]
        for (group, categories) in categoryGroups {
            let groupScore = categories.compactMap { clothingPredictions[$0] }.max() ?? 0.0
            groupPredictions[group] = groupScore
        }
        
        // Merge detailed and group predictions
        let mergedPredictions = clothingPredictions.merging(groupPredictions) { old, new in max(old, new) }
        
        print("📊 Apple Vision DeepFashion2 predictions: \(mergedPredictions)")
        return mergedPredictions
    }
    
    /// Get user-friendly display name for a category
    func getDisplayName(for category: String) -> String {
        return displayNames[category] ?? category.replacingOccurrences(of: "_", with: " ").capitalized
    }
    
    /// Get the group a category belongs to
    func getGroup(for category: String) -> String? {
        for (group, categories) in categoryGroups {
            if categories.contains(category) {
                return group
            }
        }
        return nil
    }
}
