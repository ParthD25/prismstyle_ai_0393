import Foundation
import CoreML
import Vision
import UIKit

/// Apple Core ML handler for clothing classification
/// Uses custom trained DeepFashion2 Core ML model for fashion classification
/// Falls back to heuristic analysis if model is not available
@available(iOS 13.0, *)
class CoreMLHandler: NSObject {
    
    private var isInitialized = false
    private var visionModel: VNCoreMLModel?
    private var hasCustomModel = false
    private var modelInputSize: Int = 384  // Default, will be updated from model
    
    /// DeepFashion2 category names (must match training config)
    private let deepFashion2Categories = [
        "short_sleeve_top",
        "long_sleeve_top",
        "short_sleeve_outwear",
        "long_sleeve_outwear",
        "vest",
        "sling",
        "shorts",
        "trousers",
        "skirt",
        "short_sleeve_dress",
        "long_sleeve_dress",
        "vest_dress",
        "sling_dress"
    ]
    
    /// User-friendly display names
    private let displayNames: [String: String] = [
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
    
    /// Map DeepFashion2 categories to app categories
    private let categoryMapping: [String: String] = [
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
    
    /// Initialize Core ML with fashion classification model
    func initialize() -> Bool {
        isInitialized = true
        
        // Try to load custom Core ML model - check multiple possible names/extensions
        let modelNames = ["FashionClassifier", "DeepFashion2Classifier", "ClothingClassifier"]
        let extensions = ["mlmodelc", "mlpackage", "mlmodel"]
        
        for modelName in modelNames {
            for ext in extensions {
                if let modelURL = Bundle.main.url(forResource: modelName, withExtension: ext) {
                    do {
                        let config = MLModelConfiguration()
                        config.computeUnits = .all // Use Neural Engine if available
                        
                        let mlModel = try MLModel(contentsOf: modelURL, configuration: config)
                        visionModel = try VNCoreMLModel(for: mlModel)
                        hasCustomModel = true
                        print("✅ Core ML custom model loaded: \(modelName).\(ext)")
                        print("✅ Core ML Framework initialized with trained model")
                        return true
                    } catch {
                        print("⚠️ Failed to load \(modelName).\(ext): \(error.localizedDescription)")
                    }
                }
            }
        }
        
        print("⚠️ No custom Core ML model found, using heuristic fallback")
        print("✅ Core ML Framework initialized (heuristic mode)")
        return true
    }
    
    /// Check if custom trained model is loaded
    func hasTrainedModel() -> Bool {
        return hasCustomModel
    }
    
    /// Classify clothing image using Core ML
    /// - Parameter imageData: Image bytes
    /// - Returns: Dictionary of category predictions with confidence scores
    func classifyImage(imageData: Data) -> [String: Double] {
        guard isInitialized else {
            print("❌ Core ML not initialized")
            return [:]
        }
        
        guard let image = UIImage(data: imageData),
              let cgImage = image.cgImage else {
            print("❌ Failed to convert image data")
            return [:]
        }
        
        // If custom model is loaded, use it
        if let visionModel = visionModel {
            return classifyWithCustomModel(cgImage: cgImage, model: visionModel)
        }
        
        // Fallback: Use heuristic-based analysis
        return classifyWithHeuristics(image: image)
    }
    
    /// Classify using custom Core ML model
    private func classifyWithCustomModel(cgImage: CGImage, model: VNCoreMLModel) -> [String: Double] {
        var predictions: [String: Double] = [:]
        var appCategoryScores: [String: Double] = [:]
        let semaphore = DispatchSemaphore(value: 0)
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            defer { semaphore.signal() }
            
            if let error = error {
                print("❌ Core ML classification error: \(error.localizedDescription)")
                return
            }
            
            guard let observations = request.results as? [VNClassificationObservation] else {
                print("❌ No Core ML results")
                return
            }
            
            // Process all observations
            for observation in observations {
                let identifier = observation.identifier
                let confidence = Double(observation.confidence)
                
                // Store raw prediction
                predictions[identifier] = confidence
                
                // Map to app category and aggregate
                if let appCategory = self?.categoryMapping[identifier] {
                    appCategoryScores[appCategory] = (appCategoryScores[appCategory] ?? 0) + confidence
                } else {
                    // If identifier is already an app category
                    appCategoryScores[identifier] = confidence
                }
            }
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
            _ = semaphore.wait(timeout: .now() + 10.0)
        } catch {
            print("❌ Failed to perform Core ML request: \(error.localizedDescription)")
        }
        
        // Merge detailed and app categories
        for (category, score) in appCategoryScores {
            predictions[category] = score
        }
        
        // Log top predictions
        let sortedPredictions = predictions.sorted { $0.value > $1.value }
        if !sortedPredictions.isEmpty {
            print("📊 Core ML top predictions:")
            for (idx, pred) in sortedPredictions.prefix(5).enumerated() {
                print("   \(idx + 1). \(pred.key): \(String(format: "%.2f%%", pred.value * 100))")
            }
        }
        
        return predictions
    }
    
    /// Fallback classification using heuristics (when no model is available)
    private func classifyWithHeuristics(image: UIImage) -> [String: Double] {
        var predictions: [String: Double] = [:]
        
        // Analyze image properties
        let aspectRatio = image.size.width / image.size.height
        let colorAnalysis = analyzeColors(from: image)
        let brightness = colorAnalysis.brightness
        let dominantColor = colorAnalysis.dominantColor
        
        // Category determination based on heuristics
        if aspectRatio > 1.5 {
            // Wide image - likely shoes or accessories
            predictions["Shoes"] = 0.55
            predictions["Accessories"] = 0.35
            predictions["Tops"] = 0.10
        } else if aspectRatio < 0.6 {
            // Tall image - likely dresses or full-body
            predictions["Dresses"] = 0.60
            predictions["Outerwear"] = 0.25
            predictions["Bottoms"] = 0.15
        } else if aspectRatio < 0.8 {
            // Somewhat tall - could be dress or trousers
            if brightness < 0.4 {
                predictions["Bottoms"] = 0.50
                predictions["Dresses"] = 0.30
                predictions["Outerwear"] = 0.20
            } else {
                predictions["Dresses"] = 0.50
                predictions["Tops"] = 0.30
                predictions["Bottoms"] = 0.20
            }
        } else {
            // Square-ish - could be tops or bottoms
            if brightness > 0.6 {
                predictions["Tops"] = 0.55
                predictions["Accessories"] = 0.25
                predictions["Dresses"] = 0.20
            } else {
                predictions["Bottoms"] = 0.45
                predictions["Tops"] = 0.30
                predictions["Outerwear"] = 0.25
            }
        }
        
        // Color-based refinement
        switch dominantColor {
        case "blue", "dark_blue":
            predictions["Bottoms"] = (predictions["Bottoms"] ?? 0) + 0.15
        case "black":
            predictions["Bottoms"] = (predictions["Bottoms"] ?? 0) + 0.10
            predictions["Outerwear"] = (predictions["Outerwear"] ?? 0) + 0.10
        case "white", "cream":
            predictions["Tops"] = (predictions["Tops"] ?? 0) + 0.15
        case "red", "pink":
            predictions["Dresses"] = (predictions["Dresses"] ?? 0) + 0.10
        default:
            break
        }
        
        // Normalize predictions
        let total = predictions.values.reduce(0, +)
        if total > 0 {
            for (key, value) in predictions {
                predictions[key] = value / total
            }
        }
        
        print("📊 Core ML heuristic predictions: \(predictions)")
        return predictions
    }
    
    /// Analyze colors from image
    private func analyzeColors(from image: UIImage) -> (dominantColor: String, brightness: Double) {
        guard let cgImage = image.cgImage else { 
            return ("unknown", 0.5) 
        }
        
        let width = min(cgImage.width, 50)
        let height = min(cgImage.height, 50)
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return ("unknown", 0.5)
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        var totalR: Double = 0, totalG: Double = 0, totalB: Double = 0
        var totalBrightness: Double = 0
        let pixelCount = width * height
        
        for i in stride(from: 0, to: pixelCount * bytesPerPixel, by: bytesPerPixel) {
            let r = Double(pixelData[i])
            let g = Double(pixelData[i + 1])
            let b = Double(pixelData[i + 2])
            
            totalR += r
            totalG += g
            totalB += b
            totalBrightness += (r + g + b) / 3.0
        }
        
        let avgR = totalR / Double(pixelCount)
        let avgG = totalG / Double(pixelCount)
        let avgB = totalB / Double(pixelCount)
        let brightness = totalBrightness / Double(pixelCount) / 255.0
        
        // Determine dominant color
        var dominantColor: String
        if brightness > 0.85 {
            dominantColor = "white"
        } else if brightness < 0.15 {
            dominantColor = "black"
        } else if avgR > avgG + 30 && avgR > avgB + 30 {
            dominantColor = avgR > 180 ? "red" : "dark_red"
        } else if avgG > avgR + 30 && avgG > avgB + 30 {
            dominantColor = avgG > 180 ? "green" : "dark_green"
        } else if avgB > avgR + 30 && avgB > avgG + 30 {
            dominantColor = avgB > 180 ? "blue" : "dark_blue"
        } else if abs(avgR - avgG) < 20 && abs(avgG - avgB) < 20 {
            dominantColor = brightness > 0.5 ? "gray" : "dark_gray"
        } else if avgR > 200 && avgG > 180 && avgB < 150 {
            dominantColor = "yellow"
        } else if avgR > 200 && avgG < 150 && avgB > 150 {
            dominantColor = "pink"
        } else {
            dominantColor = "neutral"
        }
        
        return (dominantColor, brightness)
    }
}
