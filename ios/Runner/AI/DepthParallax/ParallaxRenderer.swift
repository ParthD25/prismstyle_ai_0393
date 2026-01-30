import Foundation
import UIKit
import CoreImage
import CoreImage.CIFilterBuiltins
import Accelerate

/// Parallax Renderer for PrismStyle AI
/// Creates immersive 2.5D depth-based parallax effects for wardrobe items
/// using device motion and depth map data.
///
/// Features:
/// - Real-time parallax from device tilt
/// - Depth-based layer separation
/// - Smooth interpolation for fluid motion
/// - Fallback pseudo-depth for non-LiDAR devices
///
/// Usage:
/// let renderer = ParallaxRenderer()
/// let parallaxImage = renderer.render(image, depth: depthMap, tilt: (x, y))
@available(iOS 15.0, *)
public final class ParallaxRenderer {
    
    // MARK: - Types
    
    /// Parallax configuration
    public struct Configuration {
        /// Maximum parallax shift in points (each direction)
        public var maxShift: CGFloat = 15.0
        
        /// Depth influence factor (how much depth affects parallax)
        public var depthInfluence: CGFloat = 1.0
        
        /// Smoothing factor for motion (0-1, higher = smoother but laggier)
        public var smoothingFactor: CGFloat = 0.3
        
        /// Enable edge softening to hide artifacts
        public var softEdges: Bool = true
        
        /// Edge softening width in pixels
        public var edgeSoftWidth: CGFloat = 20.0
        
        /// Number of depth layers for pseudo-3D effect
        public var depthLayers: Int = 3
        
        /// Enable subtle shadow casting
        public var enableShadows: Bool = true
        
        /// Shadow opacity
        public var shadowOpacity: CGFloat = 0.3
        
        public init() {}
    }
    
    /// Rendered parallax result
    public struct RenderResult {
        public let image: UIImage
        public let layers: [UIImage]?  // Individual depth layers if requested
        public let appliedShift: CGPoint
    }
    
    // MARK: - Properties
    
    public var configuration = Configuration()
    
    private let ciContext: CIContext
    
    // Smoothed tilt values (for fluid motion)
    private var smoothedTiltX: CGFloat = 0
    private var smoothedTiltY: CGFloat = 0
    
    // Cached filters for performance
    private var displacementFilter: CIFilter?
    private var blurFilter: CIFilter?
    
    // MARK: - Initialization
    
    public init() {
        // Use GPU-accelerated context
        ciContext = CIContext(options: [
            .useSoftwareRenderer: false,
            .workingColorSpace: CGColorSpaceCreateDeviceRGB()
        ])
        
        setupFilters()
    }
    
    private func setupFilters() {
        displacementFilter = CIFilter(name: "CIDisplacementDistortion")
        blurFilter = CIFilter(name: "CIGaussianBlur")
    }
    
    // MARK: - Rendering API
    
    /// Render parallax effect on image using depth map
    /// - Parameters:
    ///   - image: Source image
    ///   - depthMap: Depth buffer (optional, will generate pseudo-depth if nil)
    ///   - tiltX: Device tilt on X axis (-1 to 1)
    ///   - tiltY: Device tilt on Y axis (-1 to 1)
    /// - Returns: Rendered image with parallax effect
    public func render(
        image: UIImage,
        depthMap: CVPixelBuffer? = nil,
        tiltX: CGFloat,
        tiltY: CGFloat
    ) -> RenderResult {
        guard let cgImage = image.cgImage else {
            return RenderResult(image: image, layers: nil, appliedShift: .zero)
        }
        
        // Smooth the tilt values
        let smoothing = configuration.smoothingFactor
        smoothedTiltX = smoothedTiltX * smoothing + tiltX * (1 - smoothing)
        smoothedTiltY = smoothedTiltY * smoothing + tiltY * (1 - smoothing)
        
        // Calculate shift based on smoothed tilt
        let shiftX = smoothedTiltX * configuration.maxShift
        let shiftY = smoothedTiltY * configuration.maxShift
        
        let ciImage = CIImage(cgImage: cgImage)
        
        // If we have real depth data, use displacement mapping
        if let depth = depthMap {
            let rendered = renderWithDepth(ciImage, depthBuffer: depth, shiftX: shiftX, shiftY: shiftY)
            if let output = createUIImage(from: rendered) {
                return RenderResult(image: output, layers: nil, appliedShift: CGPoint(x: shiftX, y: shiftY))
            }
        }
        
        // Fallback: Use layered pseudo-depth rendering
        let (rendered, layers) = renderWithPseudoDepth(
            ciImage,
            shiftX: shiftX,
            shiftY: shiftY
        )
        
        if let output = createUIImage(from: rendered) {
            let layerImages = layers.compactMap { createUIImage(from: $0) }
            return RenderResult(
                image: output,
                layers: layerImages.isEmpty ? nil : layerImages,
                appliedShift: CGPoint(x: shiftX, y: shiftY)
            )
        }
        
        return RenderResult(image: image, layers: nil, appliedShift: .zero)
    }
    
    /// Render parallax for a wardrobe card (optimized for cards)
    public func renderCardParallax(
        foregroundImage: UIImage,
        backgroundBlur: CGFloat = 10.0,
        tiltX: CGFloat,
        tiltY: CGFloat
    ) -> UIImage {
        guard let cgImage = foregroundImage.cgImage else { return foregroundImage }
        
        let ciImage = CIImage(cgImage: cgImage)
        
        // Smooth tilt
        let smoothing = configuration.smoothingFactor
        smoothedTiltX = smoothedTiltX * smoothing + tiltX * (1 - smoothing)
        smoothedTiltY = smoothedTiltY * smoothing + tiltY * (1 - smoothing)
        
        // Create 3-layer effect:
        // 1. Blurred background (moves opposite to tilt)
        // 2. Mid-ground (moves slightly)
        // 3. Foreground subject (moves with tilt)
        
        let maxShift = configuration.maxShift
        
        // Background layer (opposite direction, larger shift)
        let bgShiftX = -smoothedTiltX * maxShift * 1.2
        let bgShiftY = -smoothedTiltY * maxShift * 1.2
        
        var backgroundLayer = ciImage
            .applyingFilter("CIAffineTransform", parameters: [
                "inputTransform": NSValue(cgAffineTransform: CGAffineTransform(translationX: bgShiftX, y: bgShiftY))
            ])
        
        if backgroundBlur > 0 {
            backgroundLayer = backgroundLayer
                .applyingFilter("CIGaussianBlur", parameters: ["inputRadius": backgroundBlur])
        }
        
        // Foreground layer (same direction as tilt)
        let fgShiftX = smoothedTiltX * maxShift * 0.8
        let fgShiftY = smoothedTiltY * maxShift * 0.8
        
        let foregroundLayer = ciImage
            .applyingFilter("CIAffineTransform", parameters: [
                "inputTransform": NSValue(cgAffineTransform: CGAffineTransform(translationX: fgShiftX, y: fgShiftY))
            ])
        
        // Composite: use center-weighted mask
        let centerMask = createCenterGradientMask(size: ciImage.extent.size)
        
        // Use screen blend for subtle depth illusion
        var composited = backgroundLayer
        if let mask = centerMask {
            composited = foregroundLayer.applyingFilter("CIBlendWithMask", parameters: [
                "inputBackgroundImage": backgroundLayer,
                "inputMaskImage": mask
            ])
        }
        
        // Add subtle shadow for foreground lift
        if configuration.enableShadows {
            composited = addDropShadow(to: composited, offsetX: fgShiftX * 0.3, offsetY: fgShiftY * 0.3)
        }
        
        // Crop to original bounds
        let cropped = composited.cropped(to: ciImage.extent)
        
        return createUIImage(from: cropped) ?? foregroundImage
    }
    
    // MARK: - Depth-Based Rendering
    
    private func renderWithDepth(
        _ image: CIImage,
        depthBuffer: CVPixelBuffer,
        shiftX: CGFloat,
        shiftY: CGFloat
    ) -> CIImage {
        let depthImage = CIImage(cvPixelBuffer: depthBuffer)
        
        // Scale depth to image size
        let scaleX = image.extent.width / depthImage.extent.width
        let scaleY = image.extent.height / depthImage.extent.height
        
        let scaledDepth = depthImage
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        // Use displacement filter for depth-aware parallax
        guard let displacement = displacementFilter else { return image }
        
        // Create displacement map from tilt and depth
        let displacementScale = configuration.depthInfluence * configuration.maxShift
        
        displacement.setValue(image, forKey: kCIInputImageKey)
        displacement.setValue(scaledDepth, forKey: "inputDisplacementImage")
        displacement.setValue(displacementScale, forKey: "inputScale")
        
        guard let output = displacement.outputImage else { return image }
        
        // Apply additional shift based on tilt
        let shifted = output.transformed(by: CGAffineTransform(
            translationX: shiftX * 0.5,
            y: shiftY * 0.5
        ))
        
        return shifted.cropped(to: image.extent)
    }
    
    // MARK: - Pseudo-Depth Rendering
    
    private func renderWithPseudoDepth(
        _ image: CIImage,
        shiftX: CGFloat,
        shiftY: CGFloat
    ) -> (CIImage, [CIImage]) {
        var layers: [CIImage] = []
        let numLayers = configuration.depthLayers
        
        // Create layers with different shifts (simulating depth)
        for i in 0..<numLayers {
            let layerDepth = CGFloat(i) / CGFloat(numLayers - 1)  // 0 to 1
            
            // Far layers move opposite, near layers move with tilt
            let depthFactor = (layerDepth - 0.5) * 2  // -1 to 1
            let layerShiftX = shiftX * depthFactor
            let layerShiftY = shiftY * depthFactor
            
            let shifted = image.transformed(by: CGAffineTransform(
                translationX: layerShiftX,
                y: layerShiftY
            ))
            
            layers.append(shifted)
        }
        
        // Simple composite: use the middle layer as primary
        // with subtle influence from other layers
        guard layers.count >= 2 else {
            return (image, layers)
        }
        
        // Blend layers with depth-based opacity
        var composited = layers[0]
        for i in 1..<layers.count {
            let opacity = CGFloat(i) / CGFloat(layers.count)
            composited = layers[i].applyingFilter("CISourceOverCompositing", parameters: [
                "inputBackgroundImage": composited.applyingFilter("CIColorMatrix", parameters: [
                    "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1 - opacity * 0.3)
                ])
            ])
        }
        
        return (composited.cropped(to: image.extent), layers)
    }
    
    // MARK: - Helper Methods
    
    private func createCenterGradientMask(size: CGSize) -> CIImage? {
        let gradient = CIFilter(name: "CIRadialGradient")
        gradient?.setValue(CIVector(x: size.width / 2, y: size.height / 2), forKey: "inputCenter")
        gradient?.setValue(min(size.width, size.height) * 0.3, forKey: "inputRadius0")
        gradient?.setValue(min(size.width, size.height) * 0.7, forKey: "inputRadius1")
        gradient?.setValue(CIColor.white, forKey: "inputColor0")
        gradient?.setValue(CIColor.black, forKey: "inputColor1")
        
        return gradient?.outputImage?.cropped(to: CGRect(origin: .zero, size: size))
    }
    
    private func addDropShadow(to image: CIImage, offsetX: CGFloat, offsetY: CGFloat) -> CIImage {
        // Create shadow from image
        let shadow = image
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: configuration.shadowOpacity)
            ])
            .applyingFilter("CIGaussianBlur", parameters: ["inputRadius": 5.0])
            .transformed(by: CGAffineTransform(translationX: offsetX, y: -offsetY))
        
        return image.applyingFilter("CISourceOverCompositing", parameters: [
            "inputBackgroundImage": shadow
        ])
    }
    
    private func createUIImage(from ciImage: CIImage) -> UIImage? {
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        return UIImage(cgImage: cgImage)
    }
    
    // MARK: - Reset
    
    /// Reset smoothed values (call when view appears)
    public func resetSmoothing() {
        smoothedTiltX = 0
        smoothedTiltY = 0
    }
}

// MARK: - UIKit Integration

@available(iOS 15.0, *)
public extension ParallaxRenderer {
    
    /// Create a parallax-enabled image view
    class ParallaxImageView: UIImageView {
        
        private let renderer = ParallaxRenderer()
        private var originalImage: UIImage?
        private var depthMap: CVPixelBuffer?
        
        /// Set the image with optional depth map
        public func setImage(_ image: UIImage, depth: CVPixelBuffer? = nil) {
            self.originalImage = image
            self.depthMap = depth
            self.image = image
        }
        
        /// Update parallax based on device tilt
        public func updateParallax(tiltX: CGFloat, tiltY: CGFloat) {
            guard let original = originalImage else { return }
            
            let result = renderer.render(
                image: original,
                depthMap: depthMap,
                tiltX: tiltX,
                tiltY: tiltY
            )
            
            self.image = result.image
        }
        
        /// Reset to original image
        public func resetParallax() {
            renderer.resetSmoothing()
            if let original = originalImage {
                self.image = original
            }
        }
    }
}
