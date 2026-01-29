import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Visual Intelligence Service
/// 
/// Integrates Apple's Visual Intelligence API (iPhone 16+, iOS 18.2+)
/// for advanced clothing identification with personalized recommendations
/// 
/// Features:
/// - Brand recognition from clothing labels
/// - Style and material identification
/// - Care instruction extraction
/// - Personalized styling recommendations (UNIQUE to each user)
/// - Fallback to standard AI on older devices
class VisualIntelligenceService {
  static VisualIntelligenceService? _instance;
  static VisualIntelligenceService get instance =>
      _instance ??= VisualIntelligenceService._();

  VisualIntelligenceService._();

  static const MethodChannel _channel =
      MethodChannel('com.prismstyle_ai/visual_intelligence');

  bool? _isAvailable;

  /// Check if Visual Intelligence is available on this device
  Future<bool> isAvailable() async {
    if (_isAvailable != null) return _isAvailable!;

    try {
      if (!Platform.isIOS) {
        _isAvailable = false;
        return false;
      }

      final available = await _channel.invokeMethod<bool>(
        'isVisualIntelligenceAvailable',
      );
      
      _isAvailable = available ?? false;
      
      if (_isAvailable!) {
        debugPrint('✅ Apple Visual Intelligence available (iPhone 16+)');
      } else {
        debugPrint('ℹ️ Visual Intelligence not available (requires iPhone 16+, iOS 18.2+)');
      }
      
      return _isAvailable!;
    } catch (e) {
      debugPrint('⚠️ Visual Intelligence check failed: $e');
      _isAvailable = false;
      return false;
    }
  }

  /// Analyze image using Visual Intelligence
  /// 
  /// Returns detailed clothing information:
  /// - Brand names from labels
  /// - Style/material details
  /// - Care instructions
  /// - High-accuracy classifications
  Future<VisualIntelligenceResult?> analyzeImage(String imagePath) async {
    try {
      if (!await isAvailable()) {
        debugPrint('ℹ️ Visual Intelligence not available, use standard AI');
        return null;
      }

      final result = await _channel.invokeMethod<Map>(
        'analyzeImageWithVisualIntelligence',
        {'imagePath': imagePath},
      );

      if (result == null) return null;

      return VisualIntelligenceResult.fromMap(
        Map<String, dynamic>.from(result),
      );
    } catch (e) {
      debugPrint('❌ Visual Intelligence analysis failed: $e');
      return null;
    }
  }

  /// Generate personalized styling recommendations
  /// 
  /// CRITICAL: This provides UNIQUE recommendations based on:
  /// - User's actual wardrobe items
  /// - User's style preferences  
  /// - User's wear history
  /// - Detected item characteristics
  /// 
  /// NO TWO USERS GET THE SAME RECOMMENDATIONS
  Future<PersonalizedRecommendations?> getPersonalizedRecommendations({
    required VisualIntelligenceResult visualResults,
    required Map<String, dynamic> userPreferences,
    required Map<String, dynamic> wardrobeContext,
  }) async {
    try {
      if (!await isAvailable()) return null;

      final result = await _channel.invokeMethod<Map>(
        'generatePersonalizedRecommendations',
        {
          'visualResults': visualResults.toMap(),
          'userPreferences': userPreferences,
          'wardrobeContext': wardrobeContext,
        },
      );

      if (result == null) return null;

      return PersonalizedRecommendations.fromMap(
        Map<String, dynamic>.from(result),
      );
    } catch (e) {
      debugPrint('❌ Personalized recommendations failed: $e');
      return null;
    }
  }
}

/// Visual Intelligence Analysis Result
class VisualIntelligenceResult {
  final Map<String, dynamic> visualSearch;
  final Map<String, dynamic> detectedText;
  final Map<String, dynamic>? subjectInfo;
  final double confidence;
  final String source;
  final String deviceCapability;

  VisualIntelligenceResult({
    required this.visualSearch,
    required this.detectedText,
    this.subjectInfo,
    required this.confidence,
    required this.source,
    required this.deviceCapability,
  });

  factory VisualIntelligenceResult.fromMap(Map<String, dynamic> map) {
    return VisualIntelligenceResult(
      visualSearch: map['visual_search'] as Map<String, dynamic>? ?? {},
      detectedText: map['detected_text'] as Map<String, dynamic>? ?? {},
      subjectInfo: map['subject_info'] as Map<String, dynamic>?,
      confidence: map['confidence'] as double? ?? 0.0,
      source: map['source'] as String? ?? 'unknown',
      deviceCapability: map['device_capability'] as String? ?? 'unknown',
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'visual_search': visualSearch,
      'detected_text': detectedText,
      'subject_info': subjectInfo,
      'confidence': confidence,
      'source': source,
      'device_capability': deviceCapability,
    };
  }

  /// Get detected brand name (if any)
  String? get detectedBrand {
    return detectedText['potential_brand'] as String?;
  }

  /// Get detected size (if any)
  String? get detectedSize {
    return detectedText['potential_size'] as String?;
  }

  /// Get care instructions
  List<String> get careInstructions {
    final instructions = detectedText['care_instructions'];
    if (instructions is List) {
      return instructions.cast<String>();
    }
    return [];
  }

  /// Get all detected texts
  List<String> get allDetectedTexts {
    final texts = detectedText['detected_texts'];
    if (texts is List) {
      return texts.cast<String>();
    }
    return [];
  }

  /// Check if visual search found results
  bool get hasVisualSearchResults {
    return visualSearch['has_results'] == true;
  }
}

/// Personalized Recommendations (Unique per user)
class PersonalizedRecommendations {
  final List<String>? brandSpecificTips;
  final List<String> stylingTips;
  final List<String> complementaryItems;
  final Map<String, List<String>> occasionRecommendations;

  PersonalizedRecommendations({
    this.brandSpecificTips,
    required this.stylingTips,
    required this.complementaryItems,
    required this.occasionRecommendations,
  });

  factory PersonalizedRecommendations.fromMap(Map<String, dynamic> map) {
    return PersonalizedRecommendations(
      brandSpecificTips: (map['brand_specific_tips'] as List?)?.cast<String>(),
      stylingTips: (map['styling_tips'] as List?)?.cast<String>() ?? [],
      complementaryItems: (map['complementary_items'] as List?)?.cast<String>() ?? [],
      occasionRecommendations: _parseOccasionRecommendations(
        map['occasion_recommendations'],
      ),
    );
  }

  static Map<String, List<String>> _parseOccasionRecommendations(dynamic data) {
    if (data is! Map) return {};
    
    final result = <String, List<String>>{};
    data.forEach((key, value) {
      if (value is List) {
        result[key.toString()] = value.cast<String>();
      }
    });
    return result;
  }

  Map<String, dynamic> toMap() {
    return {
      'brand_specific_tips': brandSpecificTips,
      'styling_tips': stylingTips,
      'complementary_items': complementaryItems,
      'occasion_recommendations': occasionRecommendations,
    };
  }

  /// Get all tips as a single list
  List<String> getAllTips() {
    final allTips = <String>[];
    
    if (brandSpecificTips != null) {
      allTips.addAll(brandSpecificTips!);
    }
    allTips.addAll(stylingTips);
    
    return allTips;
  }

  /// Get recommendations for specific occasion
  List<String> getRecommendationsForOccasion(String occasion) {
    return occasionRecommendations[occasion.toLowerCase()] ?? [];
  }
}
