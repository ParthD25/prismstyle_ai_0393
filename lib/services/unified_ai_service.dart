import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Unified AI Module Service for PrismStyle AI
/// Communicates with the iOS native AI module for advanced features:
/// - CIELAB ΔE 2000 color harmony
/// - Multi-factor outfit scoring
/// - EWMA user preference learning
/// - Image quality assessment
///
/// This service provides consistent AI functionality across platforms
/// with iOS getting enhanced native performance.
class UnifiedAIService {
  static UnifiedAIService? _instance;
  static UnifiedAIService get instance => _instance ??= UnifiedAIService._();

  UnifiedAIService._();

  bool _isInitialized = false;
  bool _isAvailable = false;

  /// Method channel for iOS native AI module
  static const MethodChannel _aiChannel = MethodChannel(
    'com.prismstyle_ai/ai_module',
  );

  /// Check if running on iOS 15+
  bool get isIOSWithAIModule => !kIsWeb && Platform.isIOS && _isAvailable;

  /// Initialize the service
  Future<void> initialize() async {
    if (_isInitialized) return;

    // Check if iOS AI module is available
    if (!kIsWeb && Platform.isIOS) {
      try {
        final diagnostics = await _aiChannel.invokeMethod('getDiagnostics');
        _isAvailable = diagnostics != null;
        debugPrint('✅ iOS Unified AI Module available');
      } catch (e) {
        _isAvailable = false;
        debugPrint('⚠️ iOS Unified AI Module not available: $e');
      }
    }

    _isInitialized = true;
  }

  // MARK: - Outfit Scoring

  /// Score an outfit using the unified AI module
  /// Returns comprehensive scoring with explanations
  Future<OutfitScore> scoreOutfit({
    required List<WardrobeItemData> items,
    String occasion = 'casual',
  }) async {
    if (!isIOSWithAIModule) {
      // Return fallback score for non-iOS platforms
      return OutfitScore.fallback(items.length);
    }

    try {
      final itemsData = items
          .map((item) => {
                'id': item.id,
                'category': item.category,
                'subcategory': item.subcategory,
                'dominantColor': item.dominantColorHex,
                'formality': item.formalityScore,
              })
          .toList();

      final result = await _aiChannel.invokeMethod<Map>('scoreOutfit', {
        'items': itemsData,
        'occasion': occasion,
      });

      if (result == null) return OutfitScore.fallback(items.length);

      return OutfitScore(
        score: (result['score'] as num?)?.toDouble() ?? 0.5,
        rawScore: (result['rawScore'] as num?)?.toDouble() ?? 0.5,
        colorHarmony: (result['colorHarmony'] as num?)?.toDouble() ?? 0.5,
        occasionMatch: (result['occasionMatch'] as num?)?.toDouble() ?? 0.5,
        styleCoherence: (result['styleCoherence'] as num?)?.toDouble() ?? 0.5,
        formalityMatch: (result['formalityMatch'] as num?)?.toDouble() ?? 0.5,
        suggestions: (result['suggestions'] as List<dynamic>?)
                ?.map((e) => e.toString())
                .toList() ??
            [],
        explanation: result['explanation'] as String? ?? '',
      );
    } catch (e) {
      debugPrint('Error scoring outfit: $e');
      return OutfitScore.fallback(items.length);
    }
  }

  /// Quick check if two items go well together
  Future<double> quickPairScore(
    WardrobeItemData item1,
    WardrobeItemData item2,
  ) async {
    if (!isIOSWithAIModule) return 0.7;

    try {
      final result = await _aiChannel.invokeMethod<double>('quickPairScore', {
        'item1': {
          'id': item1.id,
          'category': item1.category,
          'subcategory': item1.subcategory,
          'dominantColor': item1.dominantColorHex,
          'formality': item1.formalityScore,
        },
        'item2': {
          'id': item2.id,
          'category': item2.category,
          'subcategory': item2.subcategory,
          'dominantColor': item2.dominantColorHex,
          'formality': item2.formalityScore,
        },
      });

      return result ?? 0.7;
    } catch (e) {
      debugPrint('Error in quick pair score: $e');
      return 0.7;
    }
  }

  // MARK: - Color Analysis

  /// Analyze color harmony for a set of colors
  Future<ColorHarmonyResult> analyzeColors(List<String> colorHexes) async {
    if (!isIOSWithAIModule || colorHexes.length < 2) {
      return ColorHarmonyResult.neutral();
    }

    try {
      final result = await _aiChannel.invokeMethod<Map>('analyzeColors', {
        'colors': colorHexes,
      });

      if (result == null) return ColorHarmonyResult.neutral();

      return ColorHarmonyResult(
        overallScore: (result['overallScore'] as num?)?.toDouble() ?? 0.5,
        harmonyType: result['harmonyType'] as String? ?? 'neutral',
        temperatureCompatible: result['temperatureCompatible'] as bool? ?? true,
        seasonallyCoherent: result['seasonallyCoherent'] as bool? ?? true,
        followsRules: result['followsRules'] as bool? ?? true,
      );
    } catch (e) {
      debugPrint('Error analyzing colors: $e');
      return ColorHarmonyResult.neutral();
    }
  }

  // MARK: - Image Quality

  /// Analyze image quality before adding to wardrobe
  Future<ImageQualityResult> analyzeImageQuality(Uint8List imageBytes) async {
    if (!isIOSWithAIModule) {
      return ImageQualityResult.unknown();
    }

    try {
      final result = await _aiChannel.invokeMethod<Map>('analyzeImageQuality', {
        'imageData': imageBytes,
      });

      if (result == null) return ImageQualityResult.unknown();

      return ImageQualityResult(
        overallScore: (result['overallScore'] as num?)?.toDouble() ?? 0.5,
        sharpnessScore: (result['sharpnessScore'] as num?)?.toDouble() ?? 0.5,
        lightingScore: (result['lightingScore'] as num?)?.toDouble() ?? 0.5,
        compositionScore:
            (result['compositionScore'] as num?)?.toDouble() ?? 0.5,
        qualityLevel: result['qualityLevel'] as String? ?? 'Unknown',
        suggestions: (result['suggestions'] as List<dynamic>?)
                ?.map((e) => e.toString())
                .toList() ??
            [],
        hasForeground: result['hasForeground'] as bool? ?? false,
      );
    } catch (e) {
      debugPrint('Error analyzing image quality: $e');
      return ImageQualityResult.unknown();
    }
  }

  // MARK: - Learning

  /// Record user interaction for preference learning
  Future<void> recordInteraction({
    required InteractionType type,
    required List<String> itemIds,
    List<Map<String, dynamic>>? metadata,
    String? occasion,
  }) async {
    if (!isIOSWithAIModule) return;

    try {
      await _aiChannel.invokeMethod('recordInteraction', {
        'type': type.value,
        'itemIds': itemIds,
        'metadata': metadata ?? [],
        'occasion': occasion,
      });
    } catch (e) {
      debugPrint('Error recording interaction: $e');
    }
  }

  /// Get learning status
  Future<Map<String, dynamic>> getLearningStatus() async {
    if (!isIOSWithAIModule) {
      return {'hasPersonalization': false, 'totalInteractions': 0};
    }

    try {
      final result =
          await _aiChannel.invokeMethod<Map<dynamic, dynamic>>('getLearningStatus');
      return result?.cast<String, dynamic>() ??
          {'hasPersonalization': false, 'totalInteractions': 0};
    } catch (e) {
      debugPrint('Error getting learning status: $e');
      return {'hasPersonalization': false, 'totalInteractions': 0};
    }
  }

  /// Reset all learned preferences
  Future<void> resetLearning() async {
    if (!isIOSWithAIModule) return;

    try {
      await _aiChannel.invokeMethod('resetLearning');
      debugPrint('✅ Learning data reset');
    } catch (e) {
      debugPrint('Error resetting learning: $e');
    }
  }

  // MARK: - Feature Flags

  /// Set a feature flag
  Future<void> setFeatureFlag(String feature, bool enabled) async {
    if (!isIOSWithAIModule) return;

    try {
      await _aiChannel.invokeMethod('setFeatureFlag', {
        'feature': feature,
        'enabled': enabled,
      });
    } catch (e) {
      debugPrint('Error setting feature flag: $e');
    }
  }

  // MARK: - Diagnostics

  /// Get diagnostic information
  Future<Map<String, dynamic>> getDiagnostics() async {
    if (!isIOSWithAIModule) {
      return {'isAvailable': false, 'platform': Platform.operatingSystem};
    }

    try {
      final result =
          await _aiChannel.invokeMethod<Map<dynamic, dynamic>>('getDiagnostics');
      return result?.cast<String, dynamic>() ?? {'isAvailable': false};
    } catch (e) {
      debugPrint('Error getting diagnostics: $e');
      return {'isAvailable': false, 'error': e.toString()};
    }
  }
}

// MARK: - Data Types

/// Wardrobe item data for AI scoring
class WardrobeItemData {
  final String id;
  final String category;
  final String subcategory;
  final String dominantColorHex;
  final double formalityScore;

  WardrobeItemData({
    required this.id,
    required this.category,
    required this.subcategory,
    required this.dominantColorHex,
    this.formalityScore = 0.5,
  });
}

/// Outfit scoring result
class OutfitScore {
  final double score;
  final double rawScore;
  final double colorHarmony;
  final double occasionMatch;
  final double styleCoherence;
  final double formalityMatch;
  final List<String> suggestions;
  final String explanation;

  OutfitScore({
    required this.score,
    required this.rawScore,
    required this.colorHarmony,
    required this.occasionMatch,
    required this.styleCoherence,
    required this.formalityMatch,
    required this.suggestions,
    required this.explanation,
  });

  factory OutfitScore.fallback(int itemCount) {
    return OutfitScore(
      score: itemCount >= 2 ? 0.7 : 0.5,
      rawScore: 0.7,
      colorHarmony: 0.7,
      occasionMatch: 0.7,
      styleCoherence: 0.7,
      formalityMatch: 0.7,
      suggestions: [],
      explanation:
          itemCount >= 2 ? 'Outfit scored with fallback method' : 'Add more items',
    );
  }
}

/// Color harmony analysis result
class ColorHarmonyResult {
  final double overallScore;
  final String harmonyType;
  final bool temperatureCompatible;
  final bool seasonallyCoherent;
  final bool followsRules;

  ColorHarmonyResult({
    required this.overallScore,
    required this.harmonyType,
    required this.temperatureCompatible,
    required this.seasonallyCoherent,
    required this.followsRules,
  });

  factory ColorHarmonyResult.neutral() {
    return ColorHarmonyResult(
      overallScore: 0.5,
      harmonyType: 'neutral',
      temperatureCompatible: true,
      seasonallyCoherent: true,
      followsRules: true,
    );
  }
}

/// Image quality analysis result
class ImageQualityResult {
  final double overallScore;
  final double sharpnessScore;
  final double lightingScore;
  final double compositionScore;
  final String qualityLevel;
  final List<String> suggestions;
  final bool hasForeground;

  ImageQualityResult({
    required this.overallScore,
    required this.sharpnessScore,
    required this.lightingScore,
    required this.compositionScore,
    required this.qualityLevel,
    required this.suggestions,
    required this.hasForeground,
  });

  factory ImageQualityResult.unknown() {
    return ImageQualityResult(
      overallScore: 0.5,
      sharpnessScore: 0.5,
      lightingScore: 0.5,
      compositionScore: 0.5,
      qualityLevel: 'Unknown',
      suggestions: ['AI quality analysis not available on this platform'],
      hasForeground: false,
    );
  }

  bool get isAcceptable => overallScore >= 0.5;
  bool get isGood => overallScore >= 0.7;
  bool get isExcellent => overallScore >= 0.85;
}

/// Interaction types for learning
enum InteractionType {
  selected('selected'),
  dismissed('dismissed'),
  worn('worn'),
  favorited('favorited'),
  unfavorited('unfavorited'),
  rated('rated'),
  viewed('viewed');

  final String value;
  const InteractionType(this.value);
}
