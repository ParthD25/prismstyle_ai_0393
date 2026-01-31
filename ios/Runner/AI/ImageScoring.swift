import Foundation
import UIKit
import Vision
import Accelerate
import CoreImage

/// Image Quality Scoring Service for PrismStyle AI
/// Analyzes wardrobe photos for quality metrics that impact AI accuracy
/// and user experience.
///
/// Key Features:
/// - Laplacian variance sharpness detection
/// - Vision framework foreground masking
/// - Lighting quality assessment
/// - Composition analysis
/// - Overall quality score for AI pipeline decisions
///
/// Usage:
/// let score = ImageScoring.shared.analyzeImage(image)
@available(iOS 15.0, *)
public final class ImageScoring {
    
    // MARK: - Singleton
    
    public static let shared = ImageScoring()
    private init() {
        ciContext = CIContext(options: [.useSoftwareRenderer: false])
    }
    
    // MARK: - Properties
    
    private let ciContext: CIContext
    
    // MARK: - Types
    
    /// Comprehensive image quality result
    public struct QualityResult {
        public let overallScore: Double        // 0-1, composite score
        public let sharpnessScore: Double      // 0-1, Laplacian variance
        public let lightingScore: Double       // 0-1, exposure quality
        public let compositionScore: Double    // 0-1, subject centering
        public let foregroundScore: Double     // 0-1, foreground isolation
        public let contrastScore: Double       // 0-1, dynamic range
        public let noiseScore: Double          // 0-1, noise level (higher = less noise)
        
        public let hasForegroundSubject: Bool  // Did Vision detect a subject?
        public let foregroundMask: CGImage?    // Optional foreground mask
        public let qualityLevel: QualityLevel  // Categorical assessment
        public let suggestions: [String]       // Improvement suggestions
        
        public init(
            overallScore: Double,
            sharpnessScore: Double,
            lightingScore: Double,
            compositionScore: Double,
            foregroundScore: Double,
            contrastScore: Double,
            noiseScore: Double,
            hasForegroundSubject: Bool,
            foregroundMask: CGImage?,
            qualityLevel: QualityLevel,
            suggestions: [String]
        ) {
            self.overallScore = overallScore
            self.sharpnessScore = sharpnessScore
            self.lightingScore = lightingScore
            self.compositionScore = compositionScore
            self.foregroundScore = foregroundScore
            self.contrastScore = contrastScore
            self.noiseScore = noiseScore
            self.hasForegroundSubject = hasForegroundSubject
            self.foregroundMask = foregroundMask
            self.qualityLevel = qualityLevel
            self.suggestions = suggestions
        }
    }
    
    /// Quality level categories
    public enum QualityLevel: String {
        case excellent = "Excellent"  // 0.85+
        case good = "Good"            // 0.70-0.85
        case acceptable = "Acceptable" // 0.50-0.70
        case poor = "Poor"            // 0.30-0.50
        case unusable = "Unusable"    // <0.30
        
        public static func from(score: Double) -> QualityLevel {
            switch score {
            case 0.85...: return .excellent
            case 0.70..<0.85: return .good
            case 0.50..<0.70: return .acceptable
            case 0.30..<0.50: return .poor
            default: return .unusable
            }
        }
    }
    
    // MARK: - Main Analysis API
    
    /// Analyze image quality comprehensively
    public func analyzeImage(_ image: UIImage, includeForegroundMask: Bool = true) -> QualityResult {
        guard let cgImage = image.cgImage else {
            return emptyResult(reason: "Invalid image")
        }
        
        // Convert to grayscale for analysis
        let grayImage = convertToGrayscale(cgImage)
        
        // Calculate individual metrics
        let sharpness = calculateSharpness(grayImage ?? cgImage)
        let lighting = calculateLightingQuality(cgImage)
        let contrast = calculateContrast(grayImage ?? cgImage)
        let noise = calculateNoiseLevel(grayImage ?? cgImage)
        
        // Get foreground analysis
        var foregroundScore: Double = 0.5
        var hasForeground = false
        var foregroundMask: CGImage?
        
        if includeForegroundMask {
            let fgResult = analyzeForeground(cgImage)
            foregroundScore = fgResult.score
            hasForeground = fgResult.hasSubject
            foregroundMask = fgResult.mask
        }
        
        // Calculate composition (requires foreground info)
        let composition = calculateComposition(cgImage, hasForeground: hasForeground)
        
        // Weighted overall score
        let overallScore = 
            sharpness * 0.30 +          // Sharpness most important for AI
            lighting * 0.20 +
            foregroundScore * 0.20 +
            contrast * 0.15 +
            noise * 0.10 +
            composition * 0.05
        
        let qualityLevel = QualityLevel.from(score: overallScore)
        let suggestions = generateSuggestions(
            sharpness: sharpness,
            lighting: lighting,
            foreground: foregroundScore,
            contrast: contrast,
            noise: noise
        )
        
        return QualityResult(
            overallScore: overallScore,
            sharpnessScore: sharpness,
            lightingScore: lighting,
            compositionScore: composition,
            foregroundScore: foregroundScore,
            contrastScore: contrast,
            noiseScore: noise,
            hasForegroundSubject: hasForeground,
            foregroundMask: foregroundMask,
            qualityLevel: qualityLevel,
            suggestions: suggestions
        )
    }
    
    /// Quick sharpness check (faster, less comprehensive)
    public func quickSharpnessCheck(_ image: UIImage) -> Double {
        guard let cgImage = image.cgImage else { return 0 }
        
        // Resize for speed
        let resized = resizeImage(cgImage, maxSize: 256)
        let gray = convertToGrayscale(resized ?? cgImage)
        
        return calculateSharpness(gray ?? resized ?? cgImage)
    }
    
    /// Check if image meets minimum quality threshold
    public func meetsQualityThreshold(_ image: UIImage, threshold: Double = 0.5) -> Bool {
        let result = analyzeImage(image, includeForegroundMask: false)
        return result.overallScore >= threshold
    }
    
    // MARK: - Sharpness (Laplacian Variance)
    
    /// Calculate sharpness using Laplacian variance
    /// Higher variance = sharper image
    private func calculateSharpness(_ cgImage: CGImage) -> Double {
        let width = cgImage.width
        let height = cgImage.height
        
        guard width > 10 && height > 10 else { return 0 }
        
        // Get pixel data
        guard let pixelData = getGrayscalePixels(cgImage) else { return 0.5 }
        
        // Laplacian kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
        var laplacianSum: Double = 0
        var laplacianSumSq: Double = 0
        var count = 0
        
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let idx = y * width + x
                
                let center = Double(pixelData[idx])
                let top = Double(pixelData[idx - width])
                let bottom = Double(pixelData[idx + width])
                let left = Double(pixelData[idx - 1])
                let right = Double(pixelData[idx + 1])
                
                let laplacian = top + bottom + left + right - 4 * center
                laplacianSum += laplacian
                laplacianSumSq += laplacian * laplacian
                count += 1
            }
        }
        
        guard count > 0 else { return 0.5 }
        
        let mean = laplacianSum / Double(count)
        let variance = (laplacianSumSq / Double(count)) - (mean * mean)
        
        // Normalize variance to 0-1 score
        // Typical sharp images have variance > 500, blurry < 100
        let normalizedVariance = min(1.0, variance / 1000.0)
        
        return sqrt(normalizedVariance)  // Square root for better distribution
    }
    
    // MARK: - Lighting Quality
    
    private func calculateLightingQuality(_ cgImage: CGImage) -> Double {
        guard let pixelData = getRGBPixels(cgImage) else { return 0.5 }
        
        let pixelCount = cgImage.width * cgImage.height
        var luminanceSum: Double = 0
        var luminanceValues: [Double] = []
        luminanceValues.reserveCapacity(min(pixelCount, 10000))
        
        let stride = max(1, pixelCount / 10000)  // Sample max 10000 pixels
        
        for i in Swift.stride(from: 0, to: pixelCount * 4, by: 4 * stride) {
            let r = Double(pixelData[i])
            let g = Double(pixelData[i + 1])
            let b = Double(pixelData[i + 2])
            
            // Luminance formula
            let luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            luminanceSum += luminance
            luminanceValues.append(luminance)
        }
        
        guard !luminanceValues.isEmpty else { return 0.5 }
        
        let avgLuminance = luminanceSum / Double(luminanceValues.count)
        
        // Check for under/over exposure
        // Ideal average luminance is around 0.45-0.55
        let exposureScore: Double
        if avgLuminance < 0.15 {
            exposureScore = avgLuminance / 0.15 * 0.5  // Very dark
        } else if avgLuminance > 0.85 {
            exposureScore = (1.0 - avgLuminance) / 0.15 * 0.5  // Very bright
        } else if avgLuminance >= 0.35 && avgLuminance <= 0.65 {
            exposureScore = 1.0  // Good range
        } else {
            exposureScore = 0.75  // Acceptable range
        }
        
        // Check for clipping (too many pixels at extremes)
        let darkClipped = luminanceValues.filter { $0 < 0.05 }.count
        let brightClipped = luminanceValues.filter { $0 > 0.95 }.count
        let clipRatio = Double(darkClipped + brightClipped) / Double(luminanceValues.count)
        let clipPenalty = min(0.3, clipRatio)
        
        return max(0, exposureScore - clipPenalty)
    }
    
    // MARK: - Contrast
    
    private func calculateContrast(_ cgImage: CGImage) -> Double {
        guard let pixelData = getGrayscalePixels(cgImage) else { return 0.5 }
        
        var minVal: UInt8 = 255
        var maxVal: UInt8 = 0
        
        let stride = max(1, pixelData.count / 10000)
        
        for i in Swift.stride(from: 0, to: pixelData.count, by: stride) {
            minVal = min(minVal, pixelData[i])
            maxVal = max(maxVal, pixelData[i])
        }
        
        let range = Int(maxVal) - Int(minVal)
        
        // Ideal range is 200-255 (good dynamic range)
        if range >= 200 {
            return 1.0
        } else if range >= 150 {
            return 0.8
        } else if range >= 100 {
            return 0.6
        } else if range >= 50 {
            return 0.4
        }
        
        return 0.2
    }
    
    // MARK: - Noise Level
    
    private func calculateNoiseLevel(_ cgImage: CGImage) -> Double {
        // Estimate noise using local variance in flat regions
        guard let pixelData = getGrayscalePixels(cgImage) else { return 0.5 }
        
        let width = cgImage.width
        let height = cgImage.height
        
        var localVariances: [Double] = []
        let blockSize = 8
        
        // Sample blocks
        for by in stride(from: blockSize, to: height - blockSize, by: blockSize * 4) {
            for bx in stride(from: blockSize, to: width - blockSize, by: blockSize * 4) {
                var sum: Double = 0
                var sumSq: Double = 0
                var count = 0
                
                for y in 0..<blockSize {
                    for x in 0..<blockSize {
                        let idx = (by + y) * width + (bx + x)
                        let val = Double(pixelData[idx])
                        sum += val
                        sumSq += val * val
                        count += 1
                    }
                }
                
                if count > 0 {
                    let mean = sum / Double(count)
                    let variance = (sumSq / Double(count)) - (mean * mean)
                    localVariances.append(variance)
                }
            }
        }
        
        guard !localVariances.isEmpty else { return 0.5 }
        
        // Find minimum variance regions (flat areas)
        localVariances.sort()
        let lowVariances = Array(localVariances.prefix(localVariances.count / 4))
        let avgLowVariance = lowVariances.reduce(0, +) / Double(max(1, lowVariances.count))
        
        // Lower variance in flat regions = less noise
        // Typical noise variance is 100-500 in noisy images, <50 in clean images
        let noiseScore = max(0, 1.0 - (avgLowVariance / 200.0))
        
        return min(1.0, noiseScore)
    }
    
    // MARK: - Foreground Analysis (Vision Framework)
    
    private func analyzeForeground(_ cgImage: CGImage) -> (score: Double, hasSubject: Bool, mask: CGImage?) {
        var score: Double = 0.5
        var hasSubject = false
        var maskImage: CGImage?
        
        let semaphore = DispatchSemaphore(value: 0)
        
        // Use Vision's person/subject segmentation
        if #available(iOS 17.0, *) {
            let request = VNGenerateForegroundInstanceMaskRequest { request, error in
                defer { semaphore.signal() }
                
                if let error = error {
                    print("⚠️ Foreground mask error: \(error)")
                    return
                }
                
                guard let observation = request.results?.first as? VNInstanceMaskObservation else {
                    return
                }
                
                hasSubject = true
                
                // Get mask coverage
                do {
                    let allInstances = observation.allInstances
                    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
                    let mask = try observation.generateScaledMaskForImage(forInstances: allInstances, from: handler)
                    
                    // Convert CVPixelBuffer to CGImage
                    let ciImage = CIImage(cvPixelBuffer: mask)
                    let context = CIContext()
                    maskImage = context.createCGImage(ciImage, from: ciImage.extent)
                    
                    // Calculate foreground coverage
                    if let maskCG = maskImage {
                        let coverage = self.calculateMaskCoverage(maskCG)
                        // Good coverage is 10-80% (clothing item centered, not too small or filling frame)
                        if coverage >= 0.1 && coverage <= 0.8 {
                            score = 0.5 + 0.5 * (1.0 - abs(coverage - 0.4) / 0.4)
                        } else if coverage < 0.1 {
                            score = 0.3  // Too small
                        } else {
                            score = 0.6  // Too large but okay
                        }
                    }
                } catch {
                    print("⚠️ Mask generation error: \(error)")
                }
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
                _ = semaphore.wait(timeout: .now() + 3.0)
            } catch {
                print("⚠️ Vision request failed: \(error)")
            }
        } else {
            // Fallback for iOS 15-16: Use saliency
            let request = VNGenerateAttentionBasedSaliencyImageRequest { request, error in
                defer { semaphore.signal() }
                
                if let result = request.results?.first as? VNSaliencyImageObservation {
                    hasSubject = true
                    
                    // Analyze saliency distribution
                    if let salientObjects = result.salientObjects, !salientObjects.isEmpty {
                        let totalArea = salientObjects.reduce(0.0) { $0 + Double($1.boundingBox.width * $1.boundingBox.height) }
                        score = min(1.0, 0.5 + totalArea)
                    }
                }
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
                _ = semaphore.wait(timeout: .now() + 2.0)
            } catch {
                print("⚠️ Saliency request failed: \(error)")
            }
        }
        
        return (score, hasSubject, maskImage)
    }
    
    private func calculateMaskCoverage(_ mask: CGImage) -> Double {
        guard let pixelData = getGrayscalePixels(mask) else { return 0 }
        
        let foregroundPixels = pixelData.filter { $0 > 128 }.count
        return Double(foregroundPixels) / Double(max(1, pixelData.count))
    }
    
    // MARK: - Composition
    
    private func calculateComposition(_ cgImage: CGImage, hasForeground: Bool) -> Double {
        // Basic composition check: is the subject reasonably centered?
        if !hasForeground {
            return 0.5  // Can't assess without foreground
        }
        
        // Use saliency to find subject location
        var centerScore: Double = 0.5
        
        let semaphore = DispatchSemaphore(value: 0)
        
        let request = VNGenerateAttentionBasedSaliencyImageRequest { request, error in
            defer { semaphore.signal() }
            
            guard let result = request.results?.first as? VNSaliencyImageObservation,
                  let salientObjects = result.salientObjects,
                  let mainObject = salientObjects.first else {
                return
            }
            
            let bbox = mainObject.boundingBox
            let centerX = bbox.midX
            let centerY = bbox.midY
            
            // How far from center (0.5, 0.5)?
            let distFromCenter = sqrt(pow(centerX - 0.5, 2) + pow(centerY - 0.5, 2))
            
            // Rule of thirds bonus: near 1/3 or 2/3 lines is also good
            let thirdX = min(abs(centerX - 0.33), abs(centerX - 0.67))
            let thirdY = min(abs(centerY - 0.33), abs(centerY - 0.67))
            let thirdBonus = (thirdX < 0.1 || thirdY < 0.1) ? 0.1 : 0
            
            centerScore = max(0, 1.0 - distFromCenter) + thirdBonus
        }
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
        _ = semaphore.wait(timeout: .now() + 1.0)
        
        return min(1.0, centerScore)
    }
    
    // MARK: - Helpers
    
    private func emptyResult(reason: String) -> QualityResult {
        return QualityResult(
            overallScore: 0,
            sharpnessScore: 0,
            lightingScore: 0,
            compositionScore: 0,
            foregroundScore: 0,
            contrastScore: 0,
            noiseScore: 0,
            hasForegroundSubject: false,
            foregroundMask: nil,
            qualityLevel: .unusable,
            suggestions: [reason]
        )
    }
    
    private func generateSuggestions(
        sharpness: Double,
        lighting: Double,
        foreground: Double,
        contrast: Double,
        noise: Double
    ) -> [String] {
        var suggestions: [String] = []
        
        if sharpness < 0.5 {
            suggestions.append("Image appears blurry - try holding your device steadier or tap to focus")
        }
        
        if lighting < 0.5 {
            if lighting < 0.3 {
                suggestions.append("Image is too dark - try better lighting or moving to a brighter area")
            } else {
                suggestions.append("Image may be overexposed - try reducing brightness or moving away from direct light")
            }
        }
        
        if foreground < 0.4 {
            suggestions.append("Clothing item not clearly visible - try positioning the item more centrally")
        }
        
        if contrast < 0.5 {
            suggestions.append("Low contrast - try placing item against a contrasting background")
        }
        
        if noise < 0.4 {
            suggestions.append("Image appears noisy - try using better lighting to reduce grain")
        }
        
        if suggestions.isEmpty {
            suggestions.append("Image quality looks good!")
        }
        
        return suggestions
    }
    
    private func convertToGrayscale(_ cgImage: CGImage) -> CGImage? {
        let width = cgImage.width
        let height = cgImage.height
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }
    
    private func resizeImage(_ cgImage: CGImage, maxSize: Int) -> CGImage? {
        let width = cgImage.width
        let height = cgImage.height
        
        let scale = Double(maxSize) / Double(max(width, height))
        if scale >= 1.0 { return cgImage }
        
        let newWidth = Int(Double(width) * scale)
        let newHeight = Int(Double(height) * scale)
        
        guard let colorSpace = cgImage.colorSpace,
              let context = CGContext(
                data: nil,
                width: newWidth,
                height: newHeight,
                bitsPerComponent: cgImage.bitsPerComponent,
                bytesPerRow: 0,
                space: colorSpace,
                bitmapInfo: cgImage.bitmapInfo.rawValue
              ) else { return nil }
        
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))
        
        return context.makeImage()
    }
    
    private func getGrayscalePixels(_ cgImage: CGImage) -> [UInt8]? {
        let width = cgImage.width
        let height = cgImage.height
        
        var pixelData = [UInt8](repeating: 0, count: width * height)
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return pixelData
    }
    
    private func getRGBPixels(_ cgImage: CGImage) -> [UInt8]? {
        let width = cgImage.width
        let height = cgImage.height
        
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return pixelData
    }
}
