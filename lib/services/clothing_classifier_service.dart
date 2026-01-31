import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import 'ensemble_ai_service.dart';
import 'context_aware_outfit_service.dart';

/// Top-level function for pattern analysis (runs in isolate)
String _decodeAndAnalyzePattern(Uint8List imageBytes) {
  final image = img.decodeImage(imageBytes);
  if (image == null) return 'Solid';

  // Resize for faster processing
  final resized = img.copyResize(image, width: 64, height: 64);
  
  // Calculate color variance across the image
  final colorVariance = _calculateColorVariance(resized);
  final edgeScore = _calculateEdgeScore(resized);
  final repetitionScore = _calculateRepetitionScore(resized);
  
  // Classify based on analysis scores
  if (colorVariance < 15 && edgeScore < 0.1) {
    return 'Solid';
  } else if (_detectStripes(resized)) {
    return 'Striped';
  } else if (_detectPlaid(resized)) {
    return 'Plaid';
  } else if (repetitionScore > 0.6 && colorVariance > 40) {
    return 'Floral';
  } else if (edgeScore > 0.4 && repetitionScore > 0.5) {
    return 'Geometric';
  } else if (colorVariance > 50 && edgeScore > 0.3) {
    return 'Printed';
  } else if (colorVariance > 25) {
    return 'Patterned';
  }
  
  return 'Solid';
}

/// Calculate color variance across the image
double _calculateColorVariance(img.Image image) {
  double sumR = 0, sumG = 0, sumB = 0;
  int count = 0;
  
  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      final pixel = image.getPixel(x, y);
      sumR += pixel.r;
      sumG += pixel.g;
      sumB += pixel.b;
      count++;
    }
  }
  
  final avgR = sumR / count;
  final avgG = sumG / count;
  final avgB = sumB / count;
  
  double variance = 0;
  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      final pixel = image.getPixel(x, y);
      variance += pow(pixel.r - avgR, 2) + pow(pixel.g - avgG, 2) + pow(pixel.b - avgB, 2);
    }
  }
  
  return sqrt(variance / count);
}

/// Calculate edge score using gradient analysis
double _calculateEdgeScore(img.Image image) {
  double edgeStrength = 0;
  int count = 0;
  
  for (int y = 1; y < image.height - 1; y++) {
    for (int x = 1; x < image.width - 1; x++) {
      final current = image.getPixel(x, y);
      final right = image.getPixel(x + 1, y);
      final bottom = image.getPixel(x, y + 1);
      
      // Horizontal gradient
      final gx = (right.r - current.r).abs() + 
                 (right.g - current.g).abs() + 
                 (right.b - current.b).abs();
      
      // Vertical gradient
      final gy = (bottom.r - current.r).abs() + 
                 (bottom.g - current.g).abs() + 
                 (bottom.b - current.b).abs();
      
      edgeStrength += sqrt(gx * gx + gy * gy);
      count++;
    }
  }
  
  return (edgeStrength / count) / 255;
}

/// Calculate repetition score
double _calculateRepetitionScore(img.Image image) {
  // Analyze rows for repetitive patterns
  double rowSimilarity = 0;
  int comparisons = 0;
  
  for (int y = 0; y < image.height ~/ 2; y++) {
    double rowDiff = 0;
    for (int x = 0; x < image.width; x++) {
      final p1 = image.getPixel(x, y);
      final p2 = image.getPixel(x, y + image.height ~/ 4);
      rowDiff += (p1.r - p2.r).abs() + (p1.g - p2.g).abs() + (p1.b - p2.b).abs();
    }
    rowSimilarity += 1 - (rowDiff / (image.width * 765));
    comparisons++;
  }
  
  return comparisons > 0 ? rowSimilarity / comparisons : 0;
}

/// Detect horizontal or vertical stripes
bool _detectStripes(img.Image image) {
  // Check horizontal stripes
  int horizontalTransitions = 0;
  int verticalTransitions = 0;
  
  // Check middle column for horizontal stripes
  final midX = image.width ~/ 2;
  for (int y = 1; y < image.height; y++) {
    final prev = image.getPixel(midX, y - 1);
    final curr = image.getPixel(midX, y);
    final diff = (prev.r - curr.r).abs() + (prev.g - curr.g).abs() + (prev.b - curr.b).abs();
    if (diff > 100) horizontalTransitions++;
  }
  
  // Check middle row for vertical stripes
  final midY = image.height ~/ 2;
  for (int x = 1; x < image.width; x++) {
    final prev = image.getPixel(x - 1, midY);
    final curr = image.getPixel(x, midY);
    final diff = (prev.r - curr.r).abs() + (prev.g - curr.g).abs() + (prev.b - curr.b).abs();
    if (diff > 100) verticalTransitions++;
  }
  
  // Stripes have multiple regular transitions
  return horizontalTransitions >= 3 || verticalTransitions >= 3;
}

/// Detect plaid/checkered pattern
bool _detectPlaid(img.Image image) {
  // Plaid has both horizontal AND vertical transitions
  int horizontalTransitions = 0;
  int verticalTransitions = 0;
  
  final midX = image.width ~/ 2;
  for (int y = 1; y < image.height; y++) {
    final prev = image.getPixel(midX, y - 1);
    final curr = image.getPixel(midX, y);
    final diff = (prev.r - curr.r).abs() + (prev.g - curr.g).abs() + (prev.b - curr.b).abs();
    if (diff > 80) horizontalTransitions++;
  }
  
  final midY = image.height ~/ 2;
  for (int x = 1; x < image.width; x++) {
    final prev = image.getPixel(x - 1, midY);
    final curr = image.getPixel(x, midY);
    final diff = (prev.r - curr.r).abs() + (prev.g - curr.g).abs() + (prev.b - curr.b).abs();
    if (diff > 80) verticalTransitions++;
  }
  
  // Plaid has significant transitions in both directions
  return horizontalTransitions >= 2 && verticalTransitions >= 2;
}

/// Top-level function for texture analysis (runs in isolate)
String _analyzeTexture(Uint8List imageBytes) {
  final image = img.decodeImage(imageBytes);
  if (image == null) return 'Cotton';

  final resized = img.copyResize(image, width: 64, height: 64);
  
  // Analyze texture properties
  final smoothness = _calculateSmoothness(resized);
  final brightness = _calculateBrightness(resized);
  final colorSaturation = _calculateSaturation(resized);
  
  // Classify material based on texture properties
  if (smoothness > 0.8 && brightness > 0.7) {
    return 'Silk';
  } else if (smoothness > 0.7 && colorSaturation > 0.5) {
    return 'Polyester';
  } else if (smoothness < 0.3 && brightness < 0.4) {
    return 'Denim';
  } else if (smoothness < 0.4 && colorSaturation < 0.3) {
    return 'Wool';
  } else if (smoothness > 0.5 && _hasSheen(resized)) {
    return 'Satin';
  } else if (smoothness < 0.5 && brightness > 0.5) {
    return 'Linen';
  } else if (brightness < 0.3) {
    return 'Leather';
  }
  
  return 'Cotton';
}

/// Calculate smoothness (inverse of texture variance)
double _calculateSmoothness(img.Image image) {
  double variance = 0;
  int count = 0;
  
  for (int y = 1; y < image.height - 1; y++) {
    for (int x = 1; x < image.width - 1; x++) {
      final current = image.getPixel(x, y);
      final neighbors = [
        image.getPixel(x - 1, y),
        image.getPixel(x + 1, y),
        image.getPixel(x, y - 1),
        image.getPixel(x, y + 1),
      ];
      
      double localVariance = 0;
      for (final neighbor in neighbors) {
        localVariance += (current.r - neighbor.r).abs() +
                        (current.g - neighbor.g).abs() +
                        (current.b - neighbor.b).abs();
      }
      variance += localVariance / 4;
      count++;
    }
  }
  
  // Invert: high variance = low smoothness
  final avgVariance = variance / count / 255;
  return 1.0 - avgVariance.clamp(0.0, 1.0);
}

/// Calculate average brightness
double _calculateBrightness(img.Image image) {
  double sum = 0;
  int count = 0;
  
  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      final pixel = image.getPixel(x, y);
      sum += (pixel.r + pixel.g + pixel.b) / 3;
      count++;
    }
  }
  
  return sum / count / 255;
}

/// Calculate color saturation
double _calculateSaturation(img.Image image) {
  double totalSaturation = 0;
  int count = 0;
  
  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      final pixel = image.getPixel(x, y);
      final r = pixel.r.toDouble();
      final g = pixel.g.toDouble();
      final b = pixel.b.toDouble();
      
      final maxVal = max(max(r, g), b);
      final minVal = min(min(r, g), b);
      
      if (maxVal > 0) {
        totalSaturation += (maxVal - minVal) / maxVal;
      }
      count++;
    }
  }
  
  return totalSaturation / count;
}

/// Check for sheen/gloss (indicates silk/satin)
bool _hasSheen(img.Image image) {
  int brightSpots = 0;
  final threshold = 230;
  
  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      final pixel = image.getPixel(x, y);
      if (pixel.r > threshold && pixel.g > threshold && pixel.b > threshold) {
        brightSpots++;
      }
    }
  }
  
  // Sheen materials have scattered bright spots
  final percentage = brightSpots / (image.width * image.height);
  return percentage > 0.05 && percentage < 0.25;
}

/// Clothing Classifier Service using Ensemble AI + Context Awareness
/// Wrapper around EnsembleAIService and ContextAwareOutfitService
/// Provides backward compatibility with weather/occasion context
class ClothingClassifierService {
  static ClothingClassifierService? _instance;
  static ClothingClassifierService get instance =>
      _instance ??= ClothingClassifierService._();

  ClothingClassifierService._();

  final EnsembleAIService _ensembleAI = EnsembleAIService.instance;
  final ContextAwareOutfitService _contextService =
      ContextAwareOutfitService.instance;

  // Category mapping for app compatibility
  static const Map<String, String> categoryMapping = {
    'Tops': 'Tops',
    'Bottoms': 'Bottoms',
    'Dresses': 'Dresses',
    'Outerwear': 'Tops',
    'Shoes': 'Shoes',
    'Accessories': 'Accessories',
    'Bags': 'Accessories',
    'Jewelry': 'Accessories',
    'Swimwear': 'Dresses',
    'Undergarments': 'Tops',
  };

  /// Initialize the ensemble AI service
  Future<void> initialize() async {
    await _ensembleAI.initialize();
    await _contextService.initialize();
    debugPrint(
      'ClothingClassifierService initialized with ensemble AI + context awareness',
    );
  }

  /// Check if service is ready
  bool get isModelLoaded => _ensembleAI.isReady;

  /// Classify clothing with context (weather, occasion, time)
  Future<ClothingClassificationResult> classifyWithContext({
    required Uint8List imageBytes,
    String? occasion,
    String? location,
  }) async {
    try {
      final contextResult = await _contextService.classifyClothingWithContext(
        imageBytes: imageBytes,
        occasion: occasion,
        location: location,
      );

      final appCategory = categoryMapping[contextResult.category] ?? 'Tops';
      final tags = _getTagsForCategory(appCategory);

      return ClothingClassificationResult(
        category: contextResult.category,
        appCategory: appCategory,
        confidence: contextResult.confidence,
        suggestedTags: tags,
        colorAnalysis: {},
        allPredictions: {},
        contextualSuitability: contextResult.suitability,
        contextualSuggestions: contextResult.suggestions,
        weatherContext: contextResult.weatherContext,
      );
    } catch (e) {
      debugPrint('Context-aware classification failed: $e');
      return ClothingClassificationResult.unknown();
    }
  }

  /// Classify clothing using ensemble approach (backward compatible)
  Future<ClothingClassificationResult> classifyImageBytes(
    Uint8List imageBytes,
  ) async {
    try {
      final ensembleResult = await _ensembleAI.classifyClothing(imageBytes);

      // Convert to compatible result format
      final appCategory =
          categoryMapping[ensembleResult.primaryCategory] ?? 'Tops';
      final tags = _getTagsForCategory(appCategory);

      // Extract color information from heuristic analysis if available
      final heuristicResult = ensembleResult.individualResults['heuristic'];
      final colorInfo = heuristicResult?.metadata ?? {};

      return ClothingClassificationResult(
        category: ensembleResult.primaryCategory,
        appCategory: appCategory,
        confidence: ensembleResult.confidence,
        suggestedTags: tags,
        colorAnalysis: colorInfo,
        allPredictions: {
          for (final pred in ensembleResult.predictions)
            pred.category: pred.confidence,
        },
      );
    } catch (e) {
      debugPrint('Ensemble classification failed: $e');
      return ClothingClassificationResult.unknown();
    }
  }

  /// Classify image from file path
  Future<ClothingClassificationResult> classifyImage(String imagePath) async {
    try {
      final file = File(imagePath);
      if (!await file.exists()) {
        throw Exception('Image file not found: $imagePath');
      }

      final bytes = await file.readAsBytes();
      return classifyImageBytes(bytes);
    } catch (e) {
      debugPrint('Error classifying image: $e');
      return ClothingClassificationResult.unknown();
    }
  }

  /// Get style tags for category
  List<String> _getTagsForCategory(String category) {
    switch (category) {
      case 'Tops':
        return ['casual', 'versatile', 'everyday'];
      case 'Bottoms':
        return ['classic', 'versatile', 'everyday'];
      case 'Dresses':
        return ['feminine', 'elegant', 'versatile'];
      case 'Shoes':
        return ['comfortable', 'stylish', 'versatile'];
      case 'Accessories':
        return ['statement', 'functional', 'stylish'];
      default:
        return ['versatile'];
    }
  }

  /// Get pattern detection using image analysis
  /// Analyzes edge patterns, color variance, and repetition
  Future<String> detectPattern(Uint8List imageBytes) async {
    try {
      final result = await _analyzePatternFromBytes(imageBytes);
      return result;
    } catch (e) {
      debugPrint('Pattern detection error: $e');
      return 'Solid';
    }
  }

  /// Analyze image bytes for pattern detection
  Future<String> _analyzePatternFromBytes(Uint8List imageBytes) async {
    // Use image package to decode and analyze
    final img = await compute(_decodeAndAnalyzePattern, imageBytes);
    return img;
  }

  /// Get material estimation based on texture analysis
  Future<String> estimateMaterial(Uint8List imageBytes) async {
    try {
      final result = await _analyzeMaterialFromBytes(imageBytes);
      return result;
    } catch (e) {
      debugPrint('Material estimation error: $e');
      return 'Cotton';
    }
  }

  /// Analyze material from image bytes
  Future<String> _analyzeMaterialFromBytes(Uint8List imageBytes) async {
    final material = await compute(_analyzeTexture, imageBytes);
    return material;
  }

  /// Get model status
  Map<String, bool> getModelStatus() => _ensembleAI.getModelStatus();

  /// Get ensemble explanation
  Future<String> getClassificationExplanation(Uint8List imageBytes) async {
    final result = await _ensembleAI.classifyClothing(imageBytes);
    return result.getExplanation();
  }

  /// Dispose resources
  void dispose() {
    _ensembleAI.dispose();
  }
}

/// Classification result model (backward compatible)
class ClothingClassificationResult {
  final String category; // Detailed category from ensemble
  final String appCategory; // App category (Tops, Bottoms, etc.)
  final double confidence;
  final List<String> suggestedTags;
  final Map<String, dynamic> colorAnalysis;
  final Map<String, double> allPredictions;
  final String? pattern;
  final String? material;
  final dynamic contextualSuitability; // SuitabilityScore from context service
  final List<String>? contextualSuggestions;
  final dynamic weatherContext; // WeatherContext from context service

  ClothingClassificationResult({
    required this.category,
    required this.appCategory,
    required this.confidence,
    required this.suggestedTags,
    required this.colorAnalysis,
    required this.allPredictions,
    this.pattern,
    this.material,
    this.contextualSuitability,
    this.contextualSuggestions,
    this.weatherContext,
  });

  /// Create an unknown/failed classification result
  factory ClothingClassificationResult.unknown() {
    return ClothingClassificationResult(
      category: 'Unknown',
      appCategory: 'Tops',
      confidence: 0.0,
      suggestedTags: ['uncategorized'],
      colorAnalysis: {},
      allPredictions: {},
    );
  }

  /// Get the primary color from analysis
  String get primaryColor => colorAnalysis['dominantColor'] ?? 'Unknown';

  /// Check if classification is confident (above threshold)
  bool get isConfident => confidence >= 0.7;

  Map<String, dynamic> toJson() {
    return {
      'category': category,
      'appCategory': appCategory,
      'confidence': confidence,
      'suggestedTags': suggestedTags,
      'primaryColor': primaryColor,
      'pattern': pattern ?? 'Solid',
      'material': material ?? 'Unknown',
    };
  }
}
