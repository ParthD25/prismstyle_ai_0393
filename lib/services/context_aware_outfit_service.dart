import 'package:flutter/foundation.dart';
import './ensemble_ai_service.dart';
import './weather_service.dart';
import './location_service.dart';

/// Context-Aware Outfit Recommendation Service
/// Integrates weather, occasion, season, and user preferences
/// for intelligent outfit recommendations
class ContextAwareOutfitService {
  static ContextAwareOutfitService? _instance;
  static ContextAwareOutfitService get instance =>
      _instance ??= ContextAwareOutfitService._();

  ContextAwareOutfitService._();

  final EnsembleAIService _aiService = EnsembleAIService.instance;
  final WeatherService _weatherService = WeatherService.instance;
  final LocationService _locationService = LocationService.instance;

  /// Occasion types
  static const List<String> occasions = [
    'Casual',
    'Formal',
    'Business',
    'Athletic',
    'Party',
    'Beach',
    'Date Night',
    'Wedding',
    'Interview',
    'Outdoor',
  ];

  /// Initialize all services
  Future<void> initialize() async {
    await _aiService.initialize();
    await _locationService.initialize();
    // Weather service initializes on demand
  }

  /// Get smart outfit recommendations based on context
  Future<OutfitRecommendation> getSmartRecommendation({
    required String occasion,
    String? location,
    DateTime? dateTime,
    Map<String, dynamic>? userPreferences,
  }) async {
    try {
      // Get weather data
      final weatherData = await _getWeatherContext(location);

      // Get temporal context (season, time of day)
      final temporalContext = _getTemporalContext(dateTime ?? DateTime.now());

      // Build complete context
      final context = OutfitContext(
        occasion: occasion,
        weather: weatherData,
        temporal: temporalContext,
        userPreferences: userPreferences ?? {},
      );

      // Generate recommendations based on context
      final recommendations = await _generateContextualRecommendations(context);

      return recommendations;
    } catch (e) {
      debugPrint('Error generating smart recommendation: $e');
      return OutfitRecommendation.empty();
    }
  }

  /// Classify clothing with context awareness
  Future<ContextualClothingResult> classifyClothingWithContext({
    required Uint8List imageBytes,
    String? occasion,
    String? location,
  }) async {
    // Get base AI classification
    final aiResult = await _aiService.classifyClothing(imageBytes);

    // Get weather context if location provided
    WeatherContext? weatherContext;
    if (location != null) {
      weatherContext = await _getWeatherContext(location);
    }

    // Analyze suitability for occasion and weather
    final suitability = _analyzeSuitability(
      category: aiResult.primaryCategory,
      occasion: occasion,
      weather: weatherContext,
    );

    return ContextualClothingResult(
      category: aiResult.primaryCategory,
      confidence: aiResult.confidence,
      suitability: suitability,
      weatherContext: weatherContext,
      suggestions: _generateSuggestions(
        category: aiResult.primaryCategory,
        occasion: occasion,
        weather: weatherContext,
      ),
    );
  }

  /// Get weather context for location
  Future<WeatherContext?> _getWeatherContext(String? location) async {
    if (location == null) return null;

    try {
      final weatherData = await _weatherService.getWeather();
      if (weatherData == null) return null;

      return WeatherContext(
        temperature: weatherData['temperature'] ?? 20.0,
        condition: weatherData['condition'] ?? 'Clear',
        humidity: weatherData['humidity'] ?? 50.0,
        windSpeed: weatherData['windSpeed'] ?? 0.0,
        isRaining: (weatherData['condition'] ?? '').toLowerCase().contains(
          'rain',
        ),
        isSnowing: (weatherData['condition'] ?? '').toLowerCase().contains(
          'snow',
        ),
      );
    } catch (e) {
      debugPrint('Failed to get weather context: $e');
      return null;
    }
  }

  /// Get temporal context (season, time of day)
  TemporalContext _getTemporalContext(DateTime dateTime) {
    final hour = dateTime.hour;
    final month = dateTime.month;

    // Determine time of day
    String timeOfDay;
    if (hour >= 5 && hour < 12) {
      timeOfDay = 'Morning';
    } else if (hour >= 12 && hour < 17) {
      timeOfDay = 'Afternoon';
    } else if (hour >= 17 && hour < 21) {
      timeOfDay = 'Evening';
    } else {
      timeOfDay = 'Night';
    }

    // Determine season (Northern Hemisphere)
    String season;
    if (month >= 3 && month <= 5) {
      season = 'Spring';
    } else if (month >= 6 && month <= 8) {
      season = 'Summer';
    } else if (month >= 9 && month <= 11) {
      season = 'Fall';
    } else {
      season = 'Winter';
    }

    return TemporalContext(
      season: season,
      timeOfDay: timeOfDay,
      month: month,
      hour: hour,
    );
  }

  /// Generate contextual recommendations
  Future<OutfitRecommendation> _generateContextualRecommendations(
    OutfitContext context,
  ) async {
    // Determine recommended categories based on weather
    final topCategories = _getRecommendedTopCategories(context);
    final bottomCategories = _getRecommendedBottomCategories(context);
    final outerwearCategories = _getRecommendedOuterwearCategories(context);

    // Generate styling tips
    final tips = _generateStylingTips(context);

    return OutfitRecommendation(
      recommendedTops: topCategories,
      recommendedBottoms: bottomCategories,
      recommendedOuterwear: outerwearCategories,
      stylingTips: tips,
      context: context,
    );
  }

  /// Get recommended top categories based on context
  List<String> _getRecommendedTopCategories(OutfitContext context) {
    final recommendations = <String>[];
    final temp = context.weather?.temperature ?? 20.0;
    final occasion = context.occasion.toLowerCase();

    // Temperature-based recommendations
    if (temp > 25) {
      recommendations.addAll(['short_sleeve_top', 'sling', 'vest']);
    } else if (temp > 15) {
      recommendations.addAll(['short_sleeve_top', 'long_sleeve_top']);
    } else {
      recommendations.addAll(['long_sleeve_top']);
    }

    // Occasion-based filtering
    if (occasion.contains('formal') ||
        occasion.contains('business') ||
        occasion.contains('interview')) {
      recommendations.removeWhere((cat) => cat == 'sling' || cat == 'vest');
    } else if (occasion.contains('athletic') || occasion.contains('beach')) {
      recommendations.removeWhere((cat) => cat == 'long_sleeve_top');
    }

    return recommendations;
  }

  /// Get recommended bottom categories based on context
  List<String> _getRecommendedBottomCategories(OutfitContext context) {
    final recommendations = <String>[];
    final temp = context.weather?.temperature ?? 20.0;
    final occasion = context.occasion.toLowerCase();

    // Temperature-based recommendations
    if (temp > 25) {
      recommendations.addAll(['shorts', 'skirt']);
    } else if (temp > 15) {
      recommendations.addAll(['trousers', 'skirt']);
    } else {
      recommendations.addAll(['trousers']);
    }

    // Occasion-based filtering
    if (occasion.contains('formal') ||
        occasion.contains('business') ||
        occasion.contains('interview')) {
      recommendations.removeWhere((cat) => cat == 'shorts');
      recommendations.add('trousers');
    } else if (occasion.contains('athletic')) {
      recommendations.add('shorts');
    }

    return recommendations;
  }

  /// Get recommended outerwear based on context
  List<String> _getRecommendedOuterwearCategories(OutfitContext context) {
    final recommendations = <String>[];
    final temp = context.weather?.temperature ?? 20.0;
    final isRaining = context.weather?.isRaining ?? false;
    final isSnowing = context.weather?.isSnowing ?? false;

    if (temp < 15 || isRaining || isSnowing) {
      recommendations.addAll(['long_sleeve_outwear']);
    }

    if (temp > 10 && temp < 20) {
      recommendations.addAll(['short_sleeve_outwear']);
    }

    return recommendations;
  }

  /// Analyze clothing suitability for occasion and weather
  SuitabilityScore _analyzeSuitability({
    required String category,
    String? occasion,
    WeatherContext? weather,
  }) {
    double score = 1.0;
    final reasons = <String>[];

    // Weather suitability
    if (weather != null) {
      final temp = weather.temperature;

      // Check temperature appropriateness
      if (category.contains('short_sleeve') && temp < 15) {
        score -= 0.3;
        reasons.add(
          'Too cold for short sleeves (${temp.toStringAsFixed(0)}°C)',
        );
      } else if (category.contains('long_sleeve') && temp > 28) {
        score -= 0.2;
        reasons.add('Might be too warm (${temp.toStringAsFixed(0)}°C)');
      }

      // Check for rain
      if (weather.isRaining && !category.contains('outwear')) {
        score -= 0.2;
        reasons.add('Consider adding a jacket (rainy weather)');
      }
    }

    // Occasion suitability
    if (occasion != null) {
      final occ = occasion.toLowerCase();

      if ((occ.contains('formal') ||
              occ.contains('business') ||
              occ.contains('interview')) &&
          (category.contains('sling') || category.contains('shorts'))) {
        score -= 0.4;
        reasons.add('Too casual for $occasion');
      }

      if (occ.contains('athletic') && category.contains('dress')) {
        score -= 0.5;
        reasons.add('Not suitable for athletic activities');
      }
    }

    return SuitabilityScore(score: score.clamp(0.0, 1.0), reasons: reasons);
  }

  /// Generate contextual suggestions
  List<String> _generateSuggestions({
    required String category,
    String? occasion,
    WeatherContext? weather,
  }) {
    final suggestions = <String>[];

    // Weather-based suggestions
    if (weather != null) {
      if (weather.temperature < 10) {
        suggestions.add('Layer up with warm clothing');
        suggestions.add('Consider a heavy jacket or coat');
      } else if (weather.temperature > 30) {
        suggestions.add('Wear light, breathable fabrics');
        suggestions.add('Stay hydrated and wear light colors');
      }

      if (weather.isRaining) {
        suggestions.add('Bring an umbrella or wear waterproof jacket');
      }

      if (weather.humidity > 70) {
        suggestions.add('Choose moisture-wicking fabrics');
      }
    }

    // Occasion-based suggestions
    if (occasion != null) {
      final occ = occasion.toLowerCase();

      if (occ.contains('formal') || occ.contains('business')) {
        suggestions.add('Pair with formal shoes and accessories');
        suggestions.add('Keep colors neutral or professional');
      } else if (occ.contains('date')) {
        suggestions.add('Add a statement piece or accessory');
        suggestions.add('Choose flattering colors and fits');
      } else if (occ.contains('athletic')) {
        suggestions.add('Wear comfortable athletic shoes');
        suggestions.add('Choose breathable, stretchy fabrics');
      }
    }

    return suggestions;
  }

  /// Generate styling tips based on context
  List<String> _generateStylingTips(OutfitContext context) {
    final tips = <String>[];
    final temp = context.weather?.temperature ?? 20.0;
    final season = context.temporal.season;

    // Season-specific tips
    switch (season) {
      case 'Spring':
        tips.add('Layer pieces for changing temperatures');
        tips.add('Incorporate pastel colors');
        break;
      case 'Summer':
        tips.add('Choose light, breathable fabrics');
        tips.add('Opt for bright colors and patterns');
        break;
      case 'Fall':
        tips.add('Try earth tones and warm colors');
        tips.add('Layer with cardigans or light jackets');
        break;
      case 'Winter':
        tips.add('Layer for warmth without bulk');
        tips.add('Add cozy textures like knits');
        break;
    }

    // Occasion-specific tips
    final occasion = context.occasion.toLowerCase();
    if (occasion.contains('formal')) {
      tips.add('Keep accessories minimal and elegant');
    } else if (occasion.contains('casual')) {
      tips.add('Mix and match for personal style');
    }

    return tips;
  }
}

/// Weather context data
class WeatherContext {
  final double temperature;
  final String condition;
  final double humidity;
  final double windSpeed;
  final bool isRaining;
  final bool isSnowing;

  WeatherContext({
    required this.temperature,
    required this.condition,
    required this.humidity,
    required this.windSpeed,
    required this.isRaining,
    required this.isSnowing,
  });

  Map<String, dynamic> toJson() => {
    'temperature': temperature,
    'condition': condition,
    'humidity': humidity,
    'windSpeed': windSpeed,
    'isRaining': isRaining,
    'isSnowing': isSnowing,
  };
}

/// Temporal context (time and season)
class TemporalContext {
  final String season;
  final String timeOfDay;
  final int month;
  final int hour;

  TemporalContext({
    required this.season,
    required this.timeOfDay,
    required this.month,
    required this.hour,
  });

  Map<String, dynamic> toJson() => {
    'season': season,
    'timeOfDay': timeOfDay,
    'month': month,
    'hour': hour,
  };
}

/// Complete outfit context
class OutfitContext {
  final String occasion;
  final WeatherContext? weather;
  final TemporalContext temporal;
  final Map<String, dynamic> userPreferences;

  OutfitContext({
    required this.occasion,
    this.weather,
    required this.temporal,
    required this.userPreferences,
  });
}

/// Outfit recommendation result
class OutfitRecommendation {
  final List<String> recommendedTops;
  final List<String> recommendedBottoms;
  final List<String> recommendedOuterwear;
  final List<String> stylingTips;
  final OutfitContext context;

  OutfitRecommendation({
    required this.recommendedTops,
    required this.recommendedBottoms,
    required this.recommendedOuterwear,
    required this.stylingTips,
    required this.context,
  });

  factory OutfitRecommendation.empty() => OutfitRecommendation(
    recommendedTops: [],
    recommendedBottoms: [],
    recommendedOuterwear: [],
    stylingTips: [],
    context: OutfitContext(
      occasion: 'Casual',
      temporal: TemporalContext(
        season: 'Spring',
        timeOfDay: 'Afternoon',
        month: 1,
        hour: 12,
      ),
      userPreferences: {},
    ),
  );
}

/// Contextual clothing classification result
class ContextualClothingResult {
  final String category;
  final double confidence;
  final SuitabilityScore suitability;
  final WeatherContext? weatherContext;
  final List<String> suggestions;

  ContextualClothingResult({
    required this.category,
    required this.confidence,
    required this.suitability,
    this.weatherContext,
    required this.suggestions,
  });
}

/// Suitability score for clothing item
class SuitabilityScore {
  final double score;
  final List<String> reasons;

  SuitabilityScore({required this.score, required this.reasons});

  bool get isHighlySuitable => score >= 0.8;
  bool get isSuitable => score >= 0.6;
  bool get isModeratelySuitable => score >= 0.4;
  bool get isNotSuitable => score < 0.4;

  String get label {
    if (isHighlySuitable) return 'Highly Suitable';
    if (isSuitable) return 'Suitable';
    if (isModeratelySuitable) return 'Moderately Suitable';
    return 'Not Suitable';
  }
}
