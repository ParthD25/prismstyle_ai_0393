import 'package:flutter/foundation.dart';
import './ensemble_ai_service.dart';
import './weather_service.dart';
import './location_service.dart';

/// Context-Aware Recommendation Service
/// Combines clothing classification with weather, occasion, and time-of-day context
/// to provide intelligent outfit recommendations
class ContextAwareRecommendationService {
  static ContextAwareRecommendationService? _instance;
  static ContextAwareRecommendationService get instance =>
      _instance ??= ContextAwareRecommendationService._();

  ContextAwareRecommendationService._();

  final EnsembleAIService _ensembleAI = EnsembleAIService.instance;
  final WeatherService _weatherService = WeatherService.instance;
  final LocationService _locationService = LocationService.instance;

  /// Classify clothing with full context awareness
  Future<ContextualClassificationResult> classifyWithContext({
    required Uint8List imageBytes,
    String? occasion,
    String? timeOfDay,
    String? location,
    Map<String, dynamic>? weatherData,
  }) async {
    // Get base classification from ensemble AI
    final classification = await _ensembleAI.classifyClothing(imageBytes);

    // Get weather data if not provided
    final weather = weatherData ?? await _getWeatherData(location);

    // Calculate contextual suitability scores
    final suitability = _calculateSuitability(
      classification: classification,
      weather: weather,
      occasion: occasion,
      timeOfDay: timeOfDay,
    );

    return ContextualClassificationResult(
      classification: classification,
      weather: weather,
      occasion: occasion,
      timeOfDay: timeOfDay,
      suitability: suitability,
      recommendations: _generateRecommendations(
        classification: classification,
        weather: weather,
        occasion: occasion,
        timeOfDay: timeOfDay,
        suitability: suitability,
      ),
    );
  }

  /// Get current weather data
  Future<Map<String, dynamic>> _getWeatherData(String? location) async {
    try {
      // Try to get location-based weather
      final position = await _locationService.getCurrentPosition();
      if (position != null) {
        final weather = await _weatherService.getWeatherForLocation(
          latitude: position.latitude,
          longitude: position.longitude,
        );
        return weather.toJson();
      }

      // Return default weather if unable to fetch
      return {
        'temperature': 20,
        'condition': 'Clear',
        'humidity': 50,
        'windSpeed': 10,
        'icon': 'sunny',
      };
    } catch (e) {
      debugPrint('Failed to get weather data: $e');
      return {
        'temperature': 20,
        'condition': 'Unknown',
        'humidity': 50,
        'windSpeed': 10,
        'icon': 'partly_sunny',
      };
    }
  }

  /// Calculate suitability scores for different contexts
  Map<String, SuitabilityScore> _calculateSuitability({
    required EnsembleClassificationResult classification,
    required Map<String, dynamic> weather,
    String? occasion,
    String? timeOfDay,
  }) {
    final category = classification.primaryCategory;
    final temp = weather['temperature'] as num? ?? 20;
    final condition = weather['condition'] as String? ?? 'Clear';

    return {
      'weather': _calculateWeatherSuitability(category, temp, condition),
      'occasion': _calculateOccasionSuitability(category, occasion),
      'timeOfDay': _calculateTimeOfDaySuitability(category, timeOfDay),
      'overall': _calculateOverallSuitability(
        category,
        temp,
        condition,
        occasion,
        timeOfDay,
      ),
    };
  }

  /// Calculate weather-based suitability
  SuitabilityScore _calculateWeatherSuitability(
    String category,
    num temperature,
    String condition,
  ) {
    double score = 0.5;
    List<String> reasons = [];

    // Temperature-based scoring
    if (category.contains('short_sleeve')) {
      if (temperature >= 20) {
        score += 0.3;
        reasons.add('Perfect for warm weather');
      } else if (temperature < 15) {
        score -= 0.3;
        reasons.add('Too cold for short sleeves');
      }
    }

    if (category.contains('long_sleeve')) {
      if (temperature <= 20) {
        score += 0.3;
        reasons.add('Great for cooler weather');
      } else if (temperature > 28) {
        score -= 0.2;
        reasons.add('Might be too warm');
      }
    }

    if (category.contains('outwear')) {
      if (temperature < 18) {
        score += 0.4;
        reasons.add('Essential for cold weather');
      } else if (temperature > 25) {
        score -= 0.3;
        reasons.add('Too warm for outerwear');
      }
    }

    if (category == 'shorts') {
      if (temperature >= 22) {
        score += 0.3;
        reasons.add('Perfect shorts weather');
      } else if (temperature < 18) {
        score -= 0.3;
        reasons.add('Too cold for shorts');
      }
    }

    // Weather condition scoring
    if (condition.toLowerCase().contains('rain')) {
      if (category.contains('outwear')) {
        score += 0.2;
        reasons.add('Good protection from rain');
      }
    }

    // Clamp score between 0 and 1
    score = score.clamp(0.0, 1.0);

    return SuitabilityScore(
      score: score,
      level: _getSuitabilityLevel(score),
      reasons: reasons.isEmpty ? ['Suitable for current weather'] : reasons,
    );
  }

  /// Calculate occasion-based suitability
  SuitabilityScore _calculateOccasionSuitability(
    String category,
    String? occasion,
  ) {
    if (occasion == null) {
      return SuitabilityScore(
        score: 0.7,
        level: 'Good',
        reasons: ['Versatile piece'],
      );
    }

    double score = 0.5;
    List<String> reasons = [];

    switch (occasion.toLowerCase()) {
      case 'formal':
      case 'business':
      case 'professional':
        if (category == 'long_sleeve_top' || category == 'trousers') {
          score = 0.9;
          reasons.add('Perfect for formal occasions');
        } else if (category.contains('dress')) {
          score = 0.85;
          reasons.add('Elegant choice for formal events');
        } else if (category.contains('outwear') &&
            !category.contains('short_sleeve')) {
          score = 0.8;
          reasons.add('Professional outerwear');
        } else if (category == 'shorts' || category == 'sling') {
          score = 0.2;
          reasons.add('Too casual for formal settings');
        }
        break;

      case 'casual':
      case 'everyday':
      case 'relaxed':
        if (category.contains('short_sleeve') || category == 'shorts') {
          score = 0.9;
          reasons.add('Great for casual wear');
        } else if (category == 'trousers' || category == 'long_sleeve_top') {
          score = 0.75;
          reasons.add('Comfortable casual option');
        }
        break;

      case 'party':
      case 'evening':
      case 'night out':
        if (category.contains('dress')) {
          score = 0.95;
          reasons.add('Perfect party attire');
        } else if (category == 'sling' || category == 'vest') {
          score = 0.8;
          reasons.add('Stylish evening choice');
        }
        break;

      case 'sport':
      case 'gym':
      case 'workout':
      case 'athletic':
        if (category == 'shorts' || category == 'short_sleeve_top') {
          score = 0.9;
          reasons.add('Ideal for physical activity');
        } else if (category.contains('dress') || category.contains('outwear')) {
          score = 0.3;
          reasons.add('Not suitable for sports');
        }
        break;

      default:
        score = 0.7;
        reasons.add('Suitable for most occasions');
    }

    return SuitabilityScore(
      score: score,
      level: _getSuitabilityLevel(score),
      reasons: reasons,
    );
  }

  /// Calculate time-of-day suitability
  SuitabilityScore _calculateTimeOfDaySuitability(
    String category,
    String? timeOfDay,
  ) {
    if (timeOfDay == null) {
      return SuitabilityScore(
        score: 0.7,
        level: 'Good',
        reasons: ['All-day suitable'],
      );
    }

    double score = 0.7;
    List<String> reasons = [];

    switch (timeOfDay.toLowerCase()) {
      case 'morning':
      case 'breakfast':
        if (category == 'long_sleeve_top' || category == 'trousers') {
          score = 0.85;
          reasons.add('Professional morning attire');
        }
        break;

      case 'afternoon':
      case 'lunch':
      case 'daytime':
        if (category.contains('short_sleeve') || category == 'shorts') {
          score = 0.9;
          reasons.add('Perfect for afternoon activities');
        }
        break;

      case 'evening':
      case 'dinner':
      case 'night':
        if (category.contains('dress')) {
          score = 0.9;
          reasons.add('Elegant evening choice');
        } else if (category.contains('long_sleeve')) {
          score = 0.85;
          reasons.add('Great for evening wear');
        } else if (category == 'shorts') {
          score = 0.5;
          reasons.add('Might be too casual for evening');
        }
        break;

      default:
        score = 0.7;
        reasons.add('Suitable for most times');
    }

    return SuitabilityScore(
      score: score,
      level: _getSuitabilityLevel(score),
      reasons: reasons,
    );
  }

  /// Calculate overall suitability
  SuitabilityScore _calculateOverallSuitability(
    String category,
    num temperature,
    String condition,
    String? occasion,
    String? timeOfDay,
  ) {
    final weatherScore = _calculateWeatherSuitability(
      category,
      temperature,
      condition,
    );
    final occasionScore = _calculateOccasionSuitability(category, occasion);
    final timeScore = _calculateTimeOfDaySuitability(category, timeOfDay);

    // Weighted average
    final overallScore =
        (weatherScore.score * 0.4) +
        (occasionScore.score * 0.35) +
        (timeScore.score * 0.25);

    final allReasons = [
      ...weatherScore.reasons,
      ...occasionScore.reasons,
      ...timeScore.reasons,
    ];

    return SuitabilityScore(
      score: overallScore,
      level: _getSuitabilityLevel(overallScore),
      reasons: allReasons.take(3).toList(),
    );
  }

  /// Get suitability level label
  String _getSuitabilityLevel(double score) {
    if (score >= 0.8) return 'Excellent';
    if (score >= 0.6) return 'Good';
    if (score >= 0.4) return 'Fair';
    return 'Poor';
  }

  /// Generate contextual recommendations
  List<String> _generateRecommendations({
    required EnsembleClassificationResult classification,
    required Map<String, dynamic> weather,
    String? occasion,
    String? timeOfDay,
    required Map<String, SuitabilityScore> suitability,
  }) {
    final recommendations = <String>[];
    final category = classification.primaryCategory;
    final temp = weather['temperature'] as num? ?? 20;
    final overallScore = suitability['overall']?.score ?? 0.5;

    // Add context-specific recommendations
    if (overallScore < 0.5) {
      recommendations.add(
        'Consider choosing a different item for these conditions',
      );
    }

    // Weather-based recommendations
    if (temp < 15 &&
        !category.contains('long_sleeve') &&
        !category.contains('outwear')) {
      recommendations.add('Pair with a jacket or sweater for warmth');
    }

    if (temp > 28 && category.contains('long_sleeve')) {
      recommendations.add(
        'Choose breathable fabrics or consider short sleeves',
      );
    }

    // Occasion-based recommendations
    if (occasion?.toLowerCase() == 'formal' && category == 'shorts') {
      recommendations.add('Switch to trousers or a dress for formal settings');
    }

    if (occasion?.toLowerCase() == 'sport' && category.contains('dress')) {
      recommendations.add(
        'Choose athletic wear for better comfort during activities',
      );
    }

    // Styling recommendations
    if (category.contains('top')) {
      recommendations.add(
        'Pair with ${temp < 20 ? "trousers" : "shorts or a skirt"}',
      );
    }

    if (category.contains('dress')) {
      recommendations.add(
        'Complete the look with appropriate footwear and accessories',
      );
    }

    // Add at least one positive recommendation
    if (recommendations.isEmpty || overallScore >= 0.7) {
      recommendations.insert(0, 'Great choice for the current conditions!');
    }

    return recommendations.take(3).toList();
  }
}

/// Contextual classification result with weather and occasion data
class ContextualClassificationResult {
  final EnsembleClassificationResult classification;
  final Map<String, dynamic> weather;
  final String? occasion;
  final String? timeOfDay;
  final Map<String, SuitabilityScore> suitability;
  final List<String> recommendations;

  ContextualClassificationResult({
    required this.classification,
    required this.weather,
    this.occasion,
    this.timeOfDay,
    required this.suitability,
    required this.recommendations,
  });

  /// Get overall suitability score (0-1)
  double get overallScore => suitability['overall']?.score ?? 0.5;

  /// Check if item is suitable for context
  bool get isSuitable => overallScore >= 0.6;

  Map<String, dynamic> toJson() {
    return {
      'category': classification.primaryCategory,
      'confidence': classification.confidence,
      'weather': weather,
      'occasion': occasion,
      'timeOfDay': timeOfDay,
      'overallScore': overallScore,
      'isSuitable': isSuitable,
      'suitability': suitability.map(
        (key, value) => MapEntry(key, value.toJson()),
      ),
      'recommendations': recommendations,
    };
  }
}

/// Suitability score with level and reasons
class SuitabilityScore {
  final double score; // 0.0 to 1.0
  final String level; // Excellent, Good, Fair, Poor
  final List<String> reasons;

  SuitabilityScore({
    required this.score,
    required this.level,
    required this.reasons,
  });

  Map<String, dynamic> toJson() {
    return {'score': score, 'level': level, 'reasons': reasons};
  }
}
