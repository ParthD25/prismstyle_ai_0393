import Foundation
import AVFoundation
import UIKit
import CoreImage

/// Depth Capture Service for PrismStyle AI
/// Captures depth data from iPhone 12+ Pro models with LiDAR scanner
/// for creating immersive 2.5D parallax effects on wardrobe items.
///
/// Requirements:
/// - iPhone 12 Pro or later (LiDAR-enabled devices)
/// - iOS 15.0+
///
/// Features:
/// - Real-time depth capture
/// - Depth map generation for static images
/// - Photo depth data extraction
/// - TrueDepth camera fallback for non-LiDAR devices
@available(iOS 15.0, *)
public final class DepthCaptureService: NSObject {
    
    // MARK: - Singleton
    
    public static let shared = DepthCaptureService()
    
    // MARK: - Types
    
    /// Depth capture result
    public struct DepthResult {
        public let depthMap: CVPixelBuffer?      // Depth data
        public let depthImage: CGImage?          // Visualized depth
        public let colorImage: CGImage?          // Corresponding color image
        public let minDepth: Float               // Nearest depth value
        public let maxDepth: Float               // Farthest depth value
        public let hasValidDepth: Bool           // Whether depth data is usable
        
        public static let empty = DepthResult(
            depthMap: nil,
            depthImage: nil,
            colorImage: nil,
            minDepth: 0,
            maxDepth: 0,
            hasValidDepth: false
        )
    }
    
    /// Depth capture capabilities
    public struct DeviceCapabilities {
        public let hasLiDAR: Bool
        public let hasTrueDepth: Bool
        public let supportedResolutions: [CGSize]
        public let maxFrameRate: Double
    }
    
    // MARK: - Properties
    
    private var captureSession: AVCaptureSession?
    private var depthDataOutput: AVCaptureDepthDataOutput?
    private var videoDataOutput: AVCaptureVideoDataOutput?
    private var photoOutput: AVCapturePhotoOutput?
    
    private var currentDevice: AVCaptureDevice?
    private var isConfigured = false
    
    private let sessionQueue = DispatchQueue(label: "com.prismstyle.depth.session")
    private let processingQueue = DispatchQueue(label: "com.prismstyle.depth.processing")
    
    // Depth data callback
    public var onDepthDataReceived: ((DepthResult) -> Void)?
    
    // Latest captured depth
    private(set) var latestDepthResult: DepthResult = .empty
    
    // MARK: - Initialization
    
    private override init() {
        super.init()
    }
    
    // MARK: - Capability Checking
    
    /// Check if device supports depth capture
    public var isDepthCaptureSupported: Bool {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInLiDARDepthCamera, .builtInTrueDepthCamera],
            mediaType: .video,
            position: .back
        )
        return !discoverySession.devices.isEmpty
    }
    
    /// Check if device has LiDAR scanner
    public var hasLiDAR: Bool {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInLiDARDepthCamera],
            mediaType: .video,
            position: .back
        )
        return !discoverySession.devices.isEmpty
    }
    
    /// Get device capabilities
    public func getCapabilities() -> DeviceCapabilities {
        let lidarSession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInLiDARDepthCamera],
            mediaType: .video,
            position: .back
        )
        
        let trueDepthSession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInTrueDepthCamera],
            mediaType: .video,
            position: .front
        )
        
        var resolutions: [CGSize] = []
        if let device = lidarSession.devices.first ?? trueDepthSession.devices.first {
            for format in device.formats {
                if format.supportedDepthDataFormats.isEmpty { continue }
                let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                let size = CGSize(width: Int(dimensions.width), height: Int(dimensions.height))
                if !resolutions.contains(size) {
                    resolutions.append(size)
                }
            }
        }
        
        return DeviceCapabilities(
            hasLiDAR: !lidarSession.devices.isEmpty,
            hasTrueDepth: !trueDepthSession.devices.isEmpty,
            supportedResolutions: resolutions,
            maxFrameRate: 30.0
        )
    }
    
    // MARK: - Session Configuration
    
    /// Configure capture session for depth capture
    public func configureSession(completion: @escaping (Bool, Error?) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self = self else {
                completion(false, nil)
                return
            }
            
            do {
                try self.setupCaptureSession()
                self.isConfigured = true
                DispatchQueue.main.async {
                    completion(true, nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completion(false, error)
                }
            }
        }
    }
    
    private func setupCaptureSession() throws {
        let session = AVCaptureSession()
        session.beginConfiguration()
        
        // Find depth-capable camera
        guard let device = findDepthCamera() else {
            throw DepthCaptureError.noDepthCamera
        }
        
        currentDevice = device
        
        // Configure device for depth
        try device.lockForConfiguration()
        
        // Find format with depth support
        if let format = findBestDepthFormat(for: device) {
            device.activeFormat = format
            
            // Configure depth format
            if let depthFormat = format.supportedDepthDataFormats.first(where: {
                CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32
            }) ?? format.supportedDepthDataFormats.first {
                device.activeDepthDataFormat = depthFormat
            }
        }
        
        device.unlockForConfiguration()
        
        // Add video input
        let input = try AVCaptureDeviceInput(device: device)
        if session.canAddInput(input) {
            session.addInput(input)
        }
        
        // Add depth output
        let depthOutput = AVCaptureDepthDataOutput()
        depthOutput.isFilteringEnabled = true  // Smooth depth data
        if session.canAddOutput(depthOutput) {
            session.addOutput(depthOutput)
            depthOutput.setDelegate(self, callbackQueue: processingQueue)
        }
        depthDataOutput = depthOutput
        
        // Add video output for color frames
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
            videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        }
        videoDataOutput = videoOutput
        
        // Add photo output for captures
        let photoOut = AVCapturePhotoOutput()
        photoOut.isDepthDataDeliveryEnabled = photoOut.isDepthDataDeliverySupported
        if session.canAddOutput(photoOut) {
            session.addOutput(photoOut)
        }
        photoOutput = photoOut
        
        session.commitConfiguration()
        captureSession = session
    }
    
    private func findDepthCamera() -> AVCaptureDevice? {
        // Prefer LiDAR camera
        let lidarSession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInLiDARDepthCamera],
            mediaType: .video,
            position: .back
        )
        
        if let device = lidarSession.devices.first {
            return device
        }
        
        // Fallback to TrueDepth
        let trueDepthSession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInTrueDepthCamera],
            mediaType: .video,
            position: .front
        )
        
        return trueDepthSession.devices.first
    }
    
    private func findBestDepthFormat(for device: AVCaptureDevice) -> AVCaptureDevice.Format? {
        // Find format with depth support and good resolution
        let depthFormats = device.formats.filter { !$0.supportedDepthDataFormats.isEmpty }
        
        // Sort by resolution (prefer higher but not too high for performance)
        return depthFormats.max { format1, format2 in
            let dim1 = CMVideoFormatDescriptionGetDimensions(format1.formatDescription)
            let dim2 = CMVideoFormatDescriptionGetDimensions(format2.formatDescription)
            
            let res1 = Int(dim1.width) * Int(dim1.height)
            let res2 = Int(dim2.width) * Int(dim2.height)
            
            // Cap at 4K for performance
            let capped1 = min(res1, 3840 * 2160)
            let capped2 = min(res2, 3840 * 2160)
            
            return capped1 < capped2
        }
    }
    
    // MARK: - Session Control
    
    /// Start depth capture session
    public func startCapture() {
        sessionQueue.async { [weak self] in
            guard let session = self?.captureSession, !session.isRunning else { return }
            session.startRunning()
            print("✅ Depth capture started")
        }
    }
    
    /// Stop depth capture session
    public func stopCapture() {
        sessionQueue.async { [weak self] in
            guard let session = self?.captureSession, session.isRunning else { return }
            session.stopRunning()
            print("✅ Depth capture stopped")
        }
    }
    
    // MARK: - Static Image Depth Extraction
    
    /// Extract depth data from a photo (if available)
    /// Works with photos taken in Portrait mode or with depth-enabled cameras
    public func extractDepthFromPhoto(_ image: UIImage) -> DepthResult {
        guard let cgImage = image.cgImage else {
            return .empty
        }
        
        // Try to get depth from image source (works with HEIF/HEIC files)
        guard let imageSource = CGImageSourceCreateWithData(
            image.jpegData(compressionQuality: 1.0)! as CFData,
            nil
        ) else {
            return .empty
        }
        
        // Check for auxiliary depth image
        let depthProperties = CGImageSourceCopyAuxiliaryDataInfoAtIndex(
            imageSource,
            0,
            kCGImageAuxiliaryDataTypeDisparity
        ) as? [String: Any]
        
        if let depthData = depthProperties?[kCGImageAuxiliaryDataInfoData as String] as? Data,
           let depthDescription = depthProperties?[kCGImageAuxiliaryDataInfoDataDescription as String] {
            
            // Create depth pixel buffer from data
            if let depthBuffer = createDepthBuffer(from: depthData, description: depthDescription as! [String: Any]) {
                let (minDepth, maxDepth) = getDepthRange(depthBuffer)
                let depthImage = visualizeDepth(depthBuffer)
                
                return DepthResult(
                    depthMap: depthBuffer,
                    depthImage: depthImage,
                    colorImage: cgImage,
                    minDepth: minDepth,
                    maxDepth: maxDepth,
                    hasValidDepth: true
                )
            }
        }
        
        return DepthResult(
            depthMap: nil,
            depthImage: nil,
            colorImage: cgImage,
            minDepth: 0,
            maxDepth: 0,
            hasValidDepth: false
        )
    }
    
    /// Generate synthetic depth map using Vision framework
    /// Fallback for images without embedded depth data
    @available(iOS 17.0, *)
    public func generateDepthMap(for image: UIImage, completion: @escaping (DepthResult) -> Void) {
        guard let cgImage = image.cgImage else {
            completion(.empty)
            return
        }
        
        processingQueue.async {
            // Use Vision's person/object segmentation as depth proxy
            var result: DepthResult = .empty
            let semaphore = DispatchSemaphore(value: 0)
            
            // For iOS 17+, we can use depth estimation models
            // For now, use saliency as depth proxy
            let request = VNGenerateAttentionBasedSaliencyImageRequest { request, error in
                defer { semaphore.signal() }
                
                guard let observation = request.results?.first as? VNSaliencyImageObservation,
                      let pixelBuffer = observation.pixelBuffer else {
                    return
                }
                
                // Use saliency as proxy depth (more salient = closer)
                let depthImage = self.visualizeDepth(pixelBuffer)
                let (minD, maxD) = self.getDepthRange(pixelBuffer)
                
                result = DepthResult(
                    depthMap: pixelBuffer,
                    depthImage: depthImage,
                    colorImage: cgImage,
                    minDepth: minD,
                    maxDepth: maxD,
                    hasValidDepth: true
                )
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try? handler.perform([request])
            _ = semaphore.wait(timeout: .now() + 3.0)
            
            DispatchQueue.main.async {
                completion(result)
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func createDepthBuffer(from data: Data, description: [String: Any]) -> CVPixelBuffer? {
        guard let width = description[kCVPixelBufferWidthKey as String] as? Int,
              let height = description[kCVPixelBufferHeightKey as String] as? Int else {
            return nil
        }
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_DepthFloat32,
            nil,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        if let baseAddress = CVPixelBufferGetBaseAddress(buffer) {
            data.copyBytes(to: baseAddress.assumingMemoryBound(to: UInt8.self), count: data.count)
        }
        
        return buffer
    }
    
    private func getDepthRange(_ depthBuffer: CVPixelBuffer) -> (Float, Float) {
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthBuffer) else {
            return (0, 1)
        }
        
        var minDepth: Float = .greatestFiniteMagnitude
        var maxDepth: Float = -.greatestFiniteMagnitude
        
        let floatBuffer = baseAddress.bindMemory(to: Float.self, capacity: width * height)
        
        for y in 0..<height {
            for x in 0..<width {
                let index = y * (bytesPerRow / MemoryLayout<Float>.size) + x
                let depth = floatBuffer[index]
                
                if depth.isFinite && depth > 0 {
                    minDepth = min(minDepth, depth)
                    maxDepth = max(maxDepth, depth)
                }
            }
        }
        
        if minDepth == .greatestFiniteMagnitude {
            return (0, 1)
        }
        
        return (minDepth, maxDepth)
    }
    
    private func visualizeDepth(_ depthBuffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: depthBuffer)
        
        // Normalize and colorize
        let coloredDepth = ciImage
            .applyingFilter("CIColorMap", parameters: [
                "inputGradientImage": createDepthGradient()
            ])
        
        let context = CIContext()
        return context.createCGImage(coloredDepth, from: coloredDepth.extent)
    }
    
    private func createDepthGradient() -> CIImage {
        // Create blue (far) to red (near) gradient
        let gradientFilter = CIFilter(name: "CILinearGradient")!
        gradientFilter.setValue(CIVector(x: 0, y: 0), forKey: "inputPoint0")
        gradientFilter.setValue(CIVector(x: 256, y: 0), forKey: "inputPoint1")
        gradientFilter.setValue(CIColor.blue, forKey: "inputColor0")
        gradientFilter.setValue(CIColor.red, forKey: "inputColor1")
        
        return gradientFilter.outputImage!.cropped(to: CGRect(x: 0, y: 0, width: 256, height: 1))
    }
    
    // MARK: - Errors
    
    public enum DepthCaptureError: Error {
        case noDepthCamera
        case configurationFailed
        case notAuthorized
    }
}

// MARK: - AVCaptureDepthDataOutputDelegate

@available(iOS 15.0, *)
extension DepthCaptureService: AVCaptureDepthDataOutputDelegate {
    
    public func depthDataOutput(
        _ output: AVCaptureDepthDataOutput,
        didOutput depthData: AVDepthData,
        timestamp: CMTime,
        connection: AVCaptureConnection
    ) {
        // Convert to float32 if needed
        let convertedDepth: AVDepthData
        if depthData.depthDataType != kCVPixelFormatType_DepthFloat32 {
            convertedDepth = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        } else {
            convertedDepth = depthData
        }
        
        let depthBuffer = convertedDepth.depthDataMap
        let (minD, maxD) = getDepthRange(depthBuffer)
        let depthImage = visualizeDepth(depthBuffer)
        
        let result = DepthResult(
            depthMap: depthBuffer,
            depthImage: depthImage,
            colorImage: nil,
            minDepth: minD,
            maxDepth: maxD,
            hasValidDepth: true
        )
        
        latestDepthResult = result
        
        DispatchQueue.main.async { [weak self] in
            self?.onDepthDataReceived?(result)
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

@available(iOS 15.0, *)
extension DepthCaptureService: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Update color image in latest result
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        let context = CIContext()
        
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            var updatedResult = latestDepthResult
            updatedResult = DepthResult(
                depthMap: updatedResult.depthMap,
                depthImage: updatedResult.depthImage,
                colorImage: cgImage,
                minDepth: updatedResult.minDepth,
                maxDepth: updatedResult.maxDepth,
                hasValidDepth: updatedResult.hasValidDepth
            )
            latestDepthResult = updatedResult
        }
    }
}

// MARK: - Vision Import
import Vision
