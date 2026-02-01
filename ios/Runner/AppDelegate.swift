import UIKit
import Flutter
import VisionKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  
  // Apple Vision and Core ML handlers (legacy, kept for compatibility)
  private var appleVisionHandler: AppleVisionHandler?
  private var coreMLHandler: CoreMLHandler?
  private var visualIntelligenceHandler: Any?
  private var fashionAIHandler: FashionAIHandler?
  
  // New Unified AI Module (iOS 15+) - use computed property to avoid @available issue
  private var _aiContainer: Any?
  
  @available(iOS 15.0, *)
  private var aiContainer: AIContainer {
    if _aiContainer == nil {
      _aiContainer = AIContainer.shared
    }
    return _aiContainer as! AIContainer
  }
  
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)
    
    // Initialize legacy Apple frameworks (iOS 13+)
    if #available(iOS 13.0, *) {
      appleVisionHandler = AppleVisionHandler()
      coreMLHandler = CoreMLHandler()
      fashionAIHandler = FashionAIHandler()
    }
    
    // Initialize Unified AI Module (iOS 15+)
    if #available(iOS 15.0, *) {
      _ = aiContainer  // Force lazy initialization
      print("✅ Unified AI Module initialized")
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
        (self?.visualIntelligenceHandler as? VisualIntelligenceHandler)?.handleMethodCall(call, result: result)
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
    
    // New Unified AI Module channel (iOS 15+)
    if #available(iOS 15.0, *) {
      let aiChannel = FlutterMethodChannel(
        name: "com.prismstyle_ai/ai_module",
        binaryMessenger: controller.binaryMessenger
      )
      
      aiChannel.setMethodCallHandler { [weak self] (call, result) in
        guard #available(iOS 15.0, *) else {
          result(FlutterError(code: "UNAVAILABLE", message: "iOS 15+ required", details: nil))
          return
        }
        
        self?.handleAIModuleCall(call, result: result)
      }
      
      print("✅ Unified AI Module MethodChannel registered")
    }
    
    print("✅ Apple Vision and Core ML MethodChannels registered")
  }
  
  // MARK: - Unified AI Module Method Handler
  
  @available(iOS 15.0, *)
  private func handleAIModuleCall(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    
    // MARK: Outfit Scoring
    case "scoreOutfit":
      guard let args = call.arguments as? [String: Any],
            let itemsData = args["items"] as? [[String: Any]] else {
        result(FlutterError(code: "INVALID_ARGS", message: "items required", details: nil))
        return
      }
      
      let occasionStr = args["occasion"] as? String ?? "casual"
      let occasion = CompatibilityEngine.Occasion(rawValue: occasionStr) ?? .casual
      
      // Convert items from Flutter format
      let items = itemsData.compactMap { itemDict -> CompatibilityEngine.WardrobeItem? in
        guard let id = itemDict["id"] as? String,
              let category = itemDict["category"] as? String,
              let subcategory = itemDict["subcategory"] as? String else {
          return nil
        }
        
        // Parse color
        var dominantColor = ColorTheory.LabColor(L: 50, a: 0, b: 0)
        if let colorHex = itemDict["dominantColor"] as? String {
          dominantColor = ColorTheory.shared.hexToLab(colorHex) ?? dominantColor
        }
        
        return CompatibilityEngine.WardrobeItem(
          id: id,
          category: category,
          subcategory: subcategory,
          dominantColor: dominantColor,
          formalityScore: itemDict["formality"] as? Double ?? 0.5
        )
      }
      
      let score = aiContainer.scoreOutfit(items: items, occasion: occasion)
      
      let response: [String: Any] = [
        "score": score.score,
        "rawScore": score.rawScore,
        "colorHarmony": score.components.colorHarmony,
        "occasionMatch": score.components.occasionMatch,
        "styleCoherence": score.components.styleCoherence,
        "formalityMatch": score.components.formalityMatch,
        "suggestions": score.suggestions,
        "explanation": score.explanation?.summary ?? ""
      ]
      result(response)
      
    // MARK: Color Analysis
    case "analyzeColors":
      guard let args = call.arguments as? [String: Any],
            let colorHexes = args["colors"] as? [String] else {
        result(FlutterError(code: "INVALID_ARGS", message: "colors required", details: nil))
        return
      }
      
      let colors = colorHexes.compactMap { UIColor(hex: $0) }
      let harmony = aiContainer.analyzeColorHarmony(colors)
      
      let response: [String: Any] = [
        "overallScore": harmony.overallScore,
        "harmonyType": harmony.harmonyType.rawValue,
        "temperatureCompatible": harmony.temperatureCompatible,
        "seasonallyCoherent": harmony.seasonallyCoherent,
        "followsRules": harmony.followsRules
      ]
      result(response)
      
    // MARK: Image Quality
    case "analyzeImageQuality":
      guard let args = call.arguments as? [String: Any],
            let imageData = args["imageData"] as? FlutterStandardTypedData else {
        result(FlutterError(code: "INVALID_ARGS", message: "imageData required", details: nil))
        return
      }
      
      guard let image = UIImage(data: imageData.data) else {
        result(FlutterError(code: "INVALID_IMAGE", message: "Could not decode image", details: nil))
        return
      }
      
      let quality = aiContainer.analyzeImageQuality(image)
      
      let response: [String: Any] = [
        "overallScore": quality.overallScore,
        "sharpnessScore": quality.sharpnessScore,
        "lightingScore": quality.lightingScore,
        "compositionScore": quality.compositionScore,
        "qualityLevel": quality.qualityLevel.rawValue,
        "suggestions": quality.suggestions,
        "hasForeground": quality.hasForegroundSubject
      ]
      result(response)
      
    // MARK: Learning
    case "recordInteraction":
      guard let args = call.arguments as? [String: Any],
            let typeStr = args["type"] as? String,
            let itemIds = args["itemIds"] as? [String] else {
        result(FlutterError(code: "INVALID_ARGS", message: "type and itemIds required", details: nil))
        return
      }
      
      let type = UserLearning.InteractionType(rawValue: typeStr) ?? .viewed
      let metadata = args["metadata"] as? [[String: Any]] ?? []
      let occasion = args["occasion"] as? String
      
      aiContainer.recordInteraction(type: type, itemIds: itemIds, itemMetadata: metadata, occasion: occasion)
      result(true)
      
    // MARK: Learning Status
    case "getLearningStatus":
      let status = aiContainer.learning.getLearningStatus()
      result(status)
      
    // MARK: Reset Learning
    case "resetLearning":
      aiContainer.resetLearning()
      result(true)
      
    // MARK: Diagnostics
    case "getDiagnostics":
      let diagnostics = aiContainer.getDiagnostics()
      result(diagnostics)
      
    // MARK: Feature Flags
    case "setFeatureFlag":
      guard let args = call.arguments as? [String: Any],
            let feature = args["feature"] as? String,
            let enabled = args["enabled"] as? Bool else {
        result(FlutterError(code: "INVALID_ARGS", message: "feature and enabled required", details: nil))
        return
      }
      
      aiContainer.setFeatureFlag(feature, enabled: enabled)
      result(true)
      
    // MARK: Quick Pair Check
    case "quickPairScore":
      guard let args = call.arguments as? [String: Any],
            let item1Data = args["item1"] as? [String: Any],
            let item2Data = args["item2"] as? [String: Any] else {
        result(FlutterError(code: "INVALID_ARGS", message: "item1 and item2 required", details: nil))
        return
      }
      
      func parseItem(_ data: [String: Any]) -> CompatibilityEngine.WardrobeItem? {
        guard let id = data["id"] as? String,
              let category = data["category"] as? String,
              let subcategory = data["subcategory"] as? String else {
          return nil
        }
        
        var color = ColorTheory.LabColor(L: 50, a: 0, b: 0)
        if let hex = data["dominantColor"] as? String {
          color = ColorTheory.shared.hexToLab(hex) ?? color
        }
        
        return CompatibilityEngine.WardrobeItem(
          id: id, category: category, subcategory: subcategory,
          dominantColor: color, formalityScore: data["formality"] as? Double ?? 0.5
        )
      }
      
      if let item1 = parseItem(item1Data), let item2 = parseItem(item2Data) {
        let score = aiContainer.quickPairScore(item1, item2)
        result(score)
      } else {
        result(FlutterError(code: "INVALID_ITEMS", message: "Could not parse items", details: nil))
      }
      
    default:
      result(FlutterMethodNotImplemented)
    }
  }
}

// MARK: - UIColor Extension for Hex Support

extension UIColor {
  convenience init?(hex: String) {
    var hexSanitized = hex.trimmingCharacters(in: .whitespacesAndNewlines)
    hexSanitized = hexSanitized.replacingOccurrences(of: "#", with: "")
    
    guard hexSanitized.count == 6,
          let rgb = UInt32(hexSanitized, radix: 16) else {
      return nil
    }
    
    let r = CGFloat((rgb >> 16) & 0xFF) / 255.0
    let g = CGFloat((rgb >> 8) & 0xFF) / 255.0
    let b = CGFloat(rgb & 0xFF) / 255.0
    
    self.init(red: r, green: g, blue: b, alpha: 1.0)
  }
}
