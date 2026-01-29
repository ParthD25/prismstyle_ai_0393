import UIKit
import Flutter
import VisionKit

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
  
  // Apple Vision and Core ML handlers
  private var appleVisionHandler: AppleVisionHandler?
  private var coreMLHandler: CoreMLHandler?
  private var visualIntelligenceHandler: VisualIntelligenceHandler?
  private var fashionAIHandler: FashionAIHandler?
  
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)
    
    // Initialize Apple frameworks
    if #available(iOS 13.0, *) {
      appleVisionHandler = AppleVisionHandler()
      coreMLHandler = CoreMLHandler()
      fashionAIHandler = FashionAIHandler()
    }
    
    // Initialize Visual Intelligence for iPhone 16+ (iOS 18.2+)
    if #available(iOS 18.2, *) {
      visualIntelligenceHandler = VisualIntelligenceHandler.shared
    }
    
    // Setup MethodChannels for Apple frameworks
    setupMethodChannels()
    
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
  
  /// Setup MethodChannels for Apple Vision and Core ML
  private func setupMethodChannels() {
    guard let controller = window?.rootViewController as? FlutterViewController else {
      return
    }
    
    // Apple Vision Framework channel
    let appleVisionChannel = FlutterMethodChannel(
      name: "com.prismstyle_ai/apple_vision",
      binaryMessenger: controller.binaryMessenger
    )
    
    appleVisionChannel.setMethodCallHandler { [weak self] (call, result) in
      guard #available(iOS 13.0, *) else {
        result(FlutterError(code: "UNAVAILABLE", message: "iOS 13+ required", details: nil))
        return
      }
      
      switch call.method {
      case "initialize":
        let success = self?.appleVisionHandler?.initialize() ?? false
        result(success)
        
      case "classifyImage":
        guard let args = call.arguments as? [String: Any],
              let imageData = args["imageBytes"] as? FlutterStandardTypedData else {
          result(FlutterError(code: "INVALID_ARGS", message: "Image bytes required", details: nil))
          return
        }
        
        let predictions = self?.appleVisionHandler?.classifyImage(imageData: imageData.data) ?? [:]
        result(predictions)
        
      default:
        result(FlutterMethodNotImplemented)
      }
    }
    
    // Core ML channel
    let coreMLChannel = FlutterMethodChannel(
      name: "com.prismstyle_ai/coreml",
      binaryMessenger: controller.binaryMessenger
    )
    
    coreMLChannel.setMethodCallHandler { [weak self] (call, result) in
      guard #available(iOS 13.0, *) else {
        result(FlutterError(code: "UNAVAILABLE", message: "iOS 13+ required", details: nil))
        return
      }
      
      switch call.method {
      case "initialize":
        let success = self?.coreMLHandler?.initialize() ?? false
        result(success)
        
      case "classifyImage":
        guard let args = call.arguments as? [String: Any],
              let imageData = args["imageBytes"] as? FlutterStandardTypedData else {
          result(FlutterError(code: "INVALID_ARGS", message: "Image bytes required", details: nil))
          return
        }
        
        let predictions = self?.coreMLHandler?.classifyImage(imageData: imageData.data) ?? [:]
        result(predictions)
        
      case "hasTrainedModel":
        let hasTrained = self?.coreMLHandler?.hasTrainedModel() ?? false
        result(hasTrained)
        
      case "getModelStatus":
        let status: [String: Any] = [
          "hasTrainedModel": self?.coreMLHandler?.hasTrainedModel() ?? false,
          "isInitialized": true,
          "platform": "iOS"
        ]
        result(status)
        
      default:
        result(FlutterMethodNotImplemented)
      }
    }
    
    // Visual Intelligence channel (iPhone 16+, iOS 18.2+)
    if #available(iOS 18.2, *) {
      let visualIntelligenceChannel = FlutterMethodChannel(
        name: "com.prismstyle_ai/visual_intelligence",
        binaryMessenger: controller.binaryMessenger
      )
      
      visualIntelligenceChannel.setMethodCallHandler { [weak self] (call, result) in
        self?.visualIntelligenceHandler?.handleMethodCall(call, result: result)
      }
      
      print("✅ Visual Intelligence MethodChannel registered (iPhone 16+)")
    }
    
    // CLIP Encoder channel for fashion embeddings
    if #available(iOS 13.0, *) {
      let clipChannel = FlutterMethodChannel(
        name: "com.prismstyle_ai/clip_encoder",
        binaryMessenger: controller.binaryMessenger
      )
      
      clipChannel.setMethodCallHandler { [weak self] (call, result) in
        switch call.method {
        case "initialize":
          let success = self?.fashionAIHandler?.initialize() ?? false
          result(success)
          
        case "getEmbedding":
          guard let args = call.arguments as? [String: Any],
                let imageData = args["imageData"] as? FlutterStandardTypedData else {
            result(FlutterError(code: "INVALID_ARGS", message: "imageData required", details: nil))
            return
          }
          
          let embedding = self?.fashionAIHandler?.getEmbedding(imageData: imageData.data)
          result(embedding)
          
        case "findSimilar":
          guard let args = call.arguments as? [String: Any],
                let imageData = args["imageData"] as? FlutterStandardTypedData else {
            result(FlutterError(code: "INVALID_ARGS", message: "imageData required", details: nil))
            return
          }
          
          let topK = args["topK"] as? Int ?? 5
          guard let image = UIImage(data: imageData.data) else {
            result(FlutterError(code: "INVALID_IMAGE", message: "Could not decode image", details: nil))
            return
          }
          
          let similar = self?.fashionAIHandler?.findSimilar(image: image, topK: topK) ?? []
          let response = similar.map { item in
            ["path": item.path, "score": item.score, "metadata": item.metadata] as [String: Any]
          }
          result(response)
          
        case "addToIndex":
          guard let args = call.arguments as? [String: Any],
                let imageData = args["imageData"] as? FlutterStandardTypedData,
                let path = args["path"] as? String else {
            result(FlutterError(code: "INVALID_ARGS", message: "imageData and path required", details: nil))
            return
          }
          
          guard let image = UIImage(data: imageData.data) else {
            result(FlutterError(code: "INVALID_IMAGE", message: "Could not decode image", details: nil))
            return
          }
          
          let metadata = args["metadata"] as? [String: Any] ?? [:]
          let success = self?.fashionAIHandler?.addToWardrobeIndex(image: image, path: path, metadata: metadata) ?? false
          result(success)
          
        default:
          result(FlutterMethodNotImplemented)
        }
      }
      
      print("✅ CLIP Encoder MethodChannel registered")
    }
    
    print("✅ Apple Vision and Core ML MethodChannels registered")
  }
}
