import 'dart:io';
import 'package:flutter/foundation.dart';

import 'ensemble_ai_service.dart';
import 'context_aware_outfit_service.dart';

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

  /// Get pattern detection (using ensemble analysis)
  Future<String> detectPattern(Uint8List imageBytes) async {
    // TODO: Implement actual pattern detection using ensemble approach
    // For now, return placeholder
    return 'Solid';
  }

  /// Get material estimation (using ensemble analysis)
  Future<String> estimateMaterial(Uint8List imageBytes) async {
    // TODO: Implement actual material estimation using ensemble approach
    // For now, return placeholder
    return 'Cotton';
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
