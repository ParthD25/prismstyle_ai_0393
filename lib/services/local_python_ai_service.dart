import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:dio/dio.dart';

/// Local Python AI Backend Service for PrismStyle AI
/// Connects to the local outfit_app_package Python backend
/// Uses: GroundingDINO + SAM + OpenCLIP/FashionCLIP - ALL FREE, NO API KEYS
///
/// This service communicates with a local Flask/FastAPI server that runs
/// the Python AI models for garment detection, segmentation, and retrieval
class LocalPythonAIService {
  static LocalPythonAIService? _instance;
  static LocalPythonAIService get instance =>
      _instance ??= LocalPythonAIService._();

  LocalPythonAIService._();

  final Dio _dio = Dio();
  bool _isServerRunning = false;
  String _serverUrl = 'http://localhost:5000';

  // Path to the Python backend
  String? _pythonBackendPath;

  /// Initialize the service and check if Python backend is running
  Future<void> initialize({String? backendPath}) async {
    _pythonBackendPath = backendPath;

    // Configure Dio
    _dio.options.connectTimeout = const Duration(seconds: 10);
    _dio.options.receiveTimeout = const Duration(seconds: 60);
    _dio.options.sendTimeout = const Duration(seconds: 60);

    // Check if server is running
    await _checkServerHealth();
  }

  /// Check if the Python backend server is running
  Future<bool> _checkServerHealth() async {
    try {
      final response = await _dio.get('$_serverUrl/health');
      _isServerRunning = response.statusCode == 200;
      debugPrint(
        _isServerRunning
            ? '✅ Local Python AI backend connected'
            : '⚠️ Local Python AI backend not responding',
      );
      return _isServerRunning;
    } catch (e) {
      debugPrint('⚠️ Local Python AI backend not running: $e');
      _isServerRunning = false;
      return false;
    }
  }

  /// Check if service is ready
  bool get isReady => _isServerRunning;

  /// Get server URL
  String get serverUrl => _serverUrl;

  /// Set custom server URL
  void setServerUrl(String url) {
    _serverUrl = url;
  }

  /// Analyze a selfie to detect and segment clothing items
  /// Uses GroundingDINO for detection + SAM for segmentation
  Future<SelfieAnalysisResult> analyzeSelfie(Uint8List imageBytes) async {
    if (!_isServerRunning) {
      await _checkServerHealth();
      if (!_isServerRunning) {
        throw Exception('Local Python AI backend not running');
      }
    }

    try {
      // Convert image to base64 for API call
      final base64Image = base64Encode(imageBytes);

      final response = await _dio.post(
        '$_serverUrl/analyze_selfie',
        data: {
          'image': base64Image,
          'detect_items': ['top', 'bottom', 'dress', 'outerwear', 'shoes'],
        },
      );

      if (response.statusCode == 200) {
        final data = response.data as Map<String, dynamic>;
        return SelfieAnalysisResult.fromJson(data);
      } else {
        throw Exception('Analysis failed: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Selfie analysis error: $e');
      rethrow;
    }
  }

  /// Get weather-aware outfit suggestions
  /// Uses Open-Meteo (free) for weather + OpenCLIP for retrieval
  Future<WeatherOutfitSuggestion> getWeatherSuggestions({
    required String location,
    String occasion = 'casual',
    int topK = 5,
  }) async {
    if (!_isServerRunning) {
      await _checkServerHealth();
      if (!_isServerRunning) {
        throw Exception('Local Python AI backend not running');
      }
    }

    try {
      final response = await _dio.post(
        '$_serverUrl/suggest_outfit',
        data: {'location': location, 'occasion': occasion, 'k': topK},
      );

      if (response.statusCode == 200) {
        final data = response.data as Map<String, dynamic>;
        return WeatherOutfitSuggestion.fromJson(data);
      } else {
        throw Exception('Suggestion failed: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Weather suggestion error: $e');
      rethrow;
    }
  }

  /// Find similar items from wardrobe using FashionCLIP embeddings
  Future<List<SimilarItem>> findSimilarItems({
    required Uint8List imageBytes,
    int topK = 5,
  }) async {
    if (!_isServerRunning) {
      await _checkServerHealth();
      if (!_isServerRunning) {
        throw Exception('Local Python AI backend not running');
      }
    }

    try {
      final base64Image = base64Encode(imageBytes);

      final response = await _dio.post(
        '$_serverUrl/find_similar',
        data: {'image': base64Image, 'k': topK},
      );

      if (response.statusCode == 200) {
        final data = response.data as Map<String, dynamic>;
        final items = (data['similar_items'] as List)
            .map((item) => SimilarItem.fromJson(item))
            .toList();
        return items;
      } else {
        throw Exception('Similar search failed: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Find similar error: $e');
      rethrow;
    }
  }

  /// Build/refresh the wardrobe index (call after adding new photos)
  Future<void> rebuildIndex() async {
    if (!_isServerRunning) {
      throw Exception('Local Python AI backend not running');
    }

    try {
      final response = await _dio.post('$_serverUrl/rebuild_index');
      if (response.statusCode != 200) {
        throw Exception('Index rebuild failed: ${response.statusCode}');
      }
      debugPrint('Wardrobe index rebuilt successfully');
    } catch (e) {
      debugPrint('Index rebuild error: $e');
      rethrow;
    }
  }

  /// Classify a clothing item category
  Future<LocalAIClassificationResult> classifyClothing(
    Uint8List imageBytes,
  ) async {
    if (!_isServerRunning) {
      await _checkServerHealth();
      if (!_isServerRunning) {
        // Return fallback result if server not running
        return LocalAIClassificationResult(
          category: 'Unknown',
          appCategory: 'Tops',
          confidence: 0.0,
          colors: [],
          attributes: {},
          isLocalAI: false,
        );
      }
    }

    try {
      final base64Image = base64Encode(imageBytes);

      final response = await _dio.post(
        '$_serverUrl/classify',
        data: {'image': base64Image},
      );

      if (response.statusCode == 200) {
        final data = response.data as Map<String, dynamic>;
        return LocalAIClassificationResult.fromJson(data);
      } else {
        throw Exception('Classification failed: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Classification error: $e');
      return LocalAIClassificationResult(
        category: 'Unknown',
        appCategory: 'Tops',
        confidence: 0.0,
        colors: [],
        attributes: {},
        isLocalAI: false,
      );
    }
  }

  void dispose() {
    _dio.close();
  }
}

/// Result from selfie analysis (detection + segmentation)
class SelfieAnalysisResult {
  final List<DetectedGarment> garments;
  final String? errorMessage;

  SelfieAnalysisResult({required this.garments, this.errorMessage});

  factory SelfieAnalysisResult.fromJson(Map<String, dynamic> json) {
    return SelfieAnalysisResult(
      garments: (json['garments'] as List? ?? [])
          .map((g) => DetectedGarment.fromJson(g))
          .toList(),
      errorMessage: json['error'] as String?,
    );
  }

  bool get hasError => errorMessage != null;
}

/// A garment detected and segmented from a selfie
class DetectedGarment {
  final String category; // 'top', 'bottom', 'dress', 'outerwear', 'shoes'
  final double confidence;
  final List<int> boundingBox; // [x1, y1, x2, y2]
  final String? maskBase64; // Segmentation mask
  final List<SimilarItem> similarFromWardrobe;

  DetectedGarment({
    required this.category,
    required this.confidence,
    required this.boundingBox,
    this.maskBase64,
    this.similarFromWardrobe = const [],
  });

  factory DetectedGarment.fromJson(Map<String, dynamic> json) {
    return DetectedGarment(
      category: json['category'] as String? ?? 'unknown',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      boundingBox: (json['bbox'] as List? ?? [0, 0, 0, 0]).cast<int>(),
      maskBase64: json['mask'] as String?,
      similarFromWardrobe: (json['similar_items'] as List? ?? [])
          .map((item) => SimilarItem.fromJson(item))
          .toList(),
    );
  }
}

/// Weather-aware outfit suggestion
class WeatherOutfitSuggestion {
  final WeatherInfo weather;
  final String prompt; // Generated prompt based on weather
  final Map<String, List<SuggestedItem>> suggestions; // By category
  final OutfitCombo? recommendedCombo;

  WeatherOutfitSuggestion({
    required this.weather,
    required this.prompt,
    required this.suggestions,
    this.recommendedCombo,
  });

  factory WeatherOutfitSuggestion.fromJson(Map<String, dynamic> json) {
    final suggestionsMap = <String, List<SuggestedItem>>{};
    final suggestionsData = json['suggestions'] as Map<String, dynamic>? ?? {};

    suggestionsData.forEach((category, items) {
      suggestionsMap[category] = (items as List)
          .map((item) => SuggestedItem.fromJson(item))
          .toList();
    });

    return WeatherOutfitSuggestion(
      weather: WeatherInfo.fromJson(json['weather'] ?? {}),
      prompt: json['prompt'] as String? ?? '',
      suggestions: suggestionsMap,
      recommendedCombo: json['combo'] != null
          ? OutfitCombo.fromJson(json['combo'])
          : null,
    );
  }
}

/// Weather information from Open-Meteo
class WeatherInfo {
  final double temperature;
  final double rainProbability;
  final double windSpeed;
  final String condition;
  final String location;

  WeatherInfo({
    required this.temperature,
    required this.rainProbability,
    required this.windSpeed,
    required this.condition,
    required this.location,
  });

  factory WeatherInfo.fromJson(Map<String, dynamic> json) {
    return WeatherInfo(
      temperature: (json['temperature'] as num?)?.toDouble() ?? 0.0,
      rainProbability: (json['rain_probability'] as num?)?.toDouble() ?? 0.0,
      windSpeed: (json['wind_speed'] as num?)?.toDouble() ?? 0.0,
      condition: json['condition'] as String? ?? 'Unknown',
      location: json['location'] as String? ?? 'Unknown',
    );
  }
}

/// A suggested item from wardrobe
class SuggestedItem {
  final String id;
  final String imagePath;
  final double score;
  final String? category;

  SuggestedItem({
    required this.id,
    required this.imagePath,
    required this.score,
    this.category,
  });

  factory SuggestedItem.fromJson(Map<String, dynamic> json) {
    return SuggestedItem(
      id: json['id'] as String? ?? '',
      imagePath: json['image_path'] as String? ?? '',
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
      category: json['category'] as String?,
    );
  }
}

/// A complete outfit combination
class OutfitCombo {
  final SuggestedItem? top;
  final SuggestedItem? bottom;
  final SuggestedItem? outerwear;
  final SuggestedItem? shoes;
  final double totalScore;

  OutfitCombo({
    this.top,
    this.bottom,
    this.outerwear,
    this.shoes,
    this.totalScore = 0.0,
  });

  factory OutfitCombo.fromJson(Map<String, dynamic> json) {
    return OutfitCombo(
      top: json['top'] != null ? SuggestedItem.fromJson(json['top']) : null,
      bottom: json['bottom'] != null
          ? SuggestedItem.fromJson(json['bottom'])
          : null,
      outerwear: json['outerwear'] != null
          ? SuggestedItem.fromJson(json['outerwear'])
          : null,
      shoes: json['shoes'] != null
          ? SuggestedItem.fromJson(json['shoes'])
          : null,
      totalScore: (json['total_score'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

/// Similar item from wardrobe (found via FashionCLIP embeddings)
class SimilarItem {
  final String id;
  final String imagePath;
  final double similarity; // 0.0 - 1.0
  final String? category;

  SimilarItem({
    required this.id,
    required this.imagePath,
    required this.similarity,
    this.category,
  });

  factory SimilarItem.fromJson(Map<String, dynamic> json) {
    return SimilarItem(
      id: json['id'] as String? ?? '',
      imagePath: json['image_path'] as String? ?? '',
      similarity: (json['similarity'] as num?)?.toDouble() ?? 0.0,
      category: json['category'] as String?,
    );
  }
}

/// Classification result from local AI
class LocalAIClassificationResult {
  final String category;
  final String appCategory;
  final double confidence;
  final List<String> colors;
  final Map<String, dynamic> attributes;
  final bool isLocalAI;

  LocalAIClassificationResult({
    required this.category,
    required this.appCategory,
    required this.confidence,
    required this.colors,
    required this.attributes,
    required this.isLocalAI,
  });

  factory LocalAIClassificationResult.fromJson(Map<String, dynamic> json) {
    return LocalAIClassificationResult(
      category: json['category'] as String? ?? 'Unknown',
      appCategory: json['app_category'] as String? ?? 'Tops',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      colors: (json['colors'] as List? ?? []).cast<String>(),
      attributes: json['attributes'] as Map<String, dynamic>? ?? {},
      isLocalAI: true,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'category': category,
      'app_category': appCategory,
      'confidence': confidence,
      'colors': colors,
      'attributes': attributes,
      'is_local_ai': isLocalAI,
    };
  }
}
