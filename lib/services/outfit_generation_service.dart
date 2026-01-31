import 'package:flutter/foundation.dart';
import '../models/wardrobe_item.dart';
import './context_aware_outfit_service.dart';
import './wardrobe_service.dart';
import './personalized_learning_service.dart';

/// Outfit Generation Service
///
/// Generates personalized outfit combinations using:
/// - Color harmony algorithms (complementary, analogous, triadic)
/// - Style matching rules
/// - Season/occasion appropriateness
/// - User's wear history and preferences
/// - ML-based compatibility scoring
class OutfitGenerationService {
  static OutfitGenerationService? _instance;
  static OutfitGenerationService get instance =>
      _instance ??= OutfitGenerationService._();

  OutfitGenerationService._();

  final WardrobeService _wardrobeService = WardrobeService.instance;
  final ContextAwareOutfitService _contextService =
      ContextAwareOutfitService.instance;
  final PersonalizedLearningService _learningService =
      PersonalizedLearningService.instance;

  /// Generate personalized outfit recommendations
  ///
  /// Uses user's actual wardrobe items and preferences for unique combinations
  Future<List<OutfitCombination>> generateOutfits({
    required String userId,
    String? occasion,
    String? location,
    int maxCombinations = 10,
    Map<String, dynamic>? userPreferences,
  }) async {
    try {
      // 1. Fetch user's wardrobe
      final items = await _wardrobeService.getAllItems();
      if (items.isEmpty) {
        debugPrint('⚠️ No wardrobe items found for user');
        return [];
      }

      // 2. Get context recommendations
      final contextRec = await _contextService.getSmartRecommendation(
        occasion: occasion ?? 'Casual',
        location: location,
        dateTime: DateTime.now(),
        userPreferences: userPreferences,
      );

      // 3. Filter items by context (season, weather, occasion)
      final filteredItems = _filterItemsByContext(items, contextRec);

      // 4. Group items by category
      final grouped = _groupByCategory(filteredItems);

      // 5. Generate all valid combinations
      final combinations = _generateCombinations(
        grouped,
        contextRec,
        userPreferences ?? {},
      );

      // 6. Score combinations using multiple algorithms
      final scored = combinations.map((combo) {
        return _scoreOutfitCombination(combo, userPreferences ?? {});
      }).toList();

      // 7. Sort by score and return top results
      scored.sort((a, b) => b.totalScore.compareTo(a.totalScore));

      // 8. Apply diversity filter (avoid too similar outfits)
      final diverse = _applyDiversityFilter(scored, maxCombinations);

      debugPrint(
        '✅ Generated ${diverse.length} personalized outfit combinations',
      );
      return diverse.take(maxCombinations).toList();
    } catch (e) {
      debugPrint('❌ Error generating outfits: $e');
      return [];
    }
  }

  /// Filter items based on context (weather, season, occasion)
  List<WardrobeItem> _filterItemsByContext(
    List<WardrobeItem> items,
    OutfitRecommendation context,
  ) {
    return items.where((item) {
      // Check if item is appropriate for season
      if (item.season.isNotEmpty) {
        final currentSeason = context.context.temporal.season;
        if (!item.season.contains(currentSeason)) {
          return false;
        }
      }

      // Check if item matches occasion
      if (item.occasion.isNotEmpty) {
        final targetOccasion = context.context.occasion;
        if (!item.occasion.any(
          (occ) => occ.toLowerCase().contains(targetOccasion.toLowerCase()),
        )) {
          return false;
        }
      }

      return true;
    }).toList();
  }

  /// Group items by category for combination generation
  Map<String, List<WardrobeItem>> _groupByCategory(List<WardrobeItem> items) {
    final grouped = <String, List<WardrobeItem>>{};

    for (final item in items) {
      grouped.putIfAbsent(item.category, () => []).add(item);
    }

    return grouped;
  }

  /// Generate all valid outfit combinations
  List<OutfitCombination> _generateCombinations(
    Map<String, List<WardrobeItem>> grouped,
    OutfitRecommendation context,
    Map<String, dynamic> userPrefs,
  ) {
    final combinations = <OutfitCombination>[];

    final tops = grouped['Tops'] ?? [];
    final bottoms = grouped['Bottoms'] ?? [];
    final dresses = grouped['Dresses'] ?? [];
    final shoes = grouped['Shoes'] ?? [];
    final outerwear = grouped['Outerwear'] ?? [];
    // ignore: unused_local_variable - Reserved for accessory combinations
    final accessories = grouped['Accessories'] ?? [];

    // Combination 1: Top + Bottom + Shoes (+ optional Outerwear/Accessories)
    for (final top in tops) {
      for (final bottom in bottoms) {
        for (final shoe in shoes) {
          combinations.add(
            OutfitCombination(items: [top, bottom, shoe], context: context),
          );

          // Add with outerwear
          if (outerwear.isNotEmpty) {
            for (final outer in outerwear) {
              combinations.add(
                OutfitCombination(
                  items: [top, bottom, shoe, outer],
                  context: context,
                ),
              );
            }
          }
        }
      }
    }

    // Combination 2: Dress + Shoes (+ optional Outerwear/Accessories)
    for (final dress in dresses) {
      for (final shoe in shoes) {
        combinations.add(
          OutfitCombination(items: [dress, shoe], context: context),
        );

        if (outerwear.isNotEmpty) {
          for (final outer in outerwear) {
            combinations.add(
              OutfitCombination(items: [dress, shoe, outer], context: context),
            );
          }
        }
      }
    }

    debugPrint('Generated ${combinations.length} raw combinations');
    return combinations;
  }

  /// Score outfit combination using multiple algorithms
  OutfitCombination _scoreOutfitCombination(
    OutfitCombination combo,
    Map<String, dynamic> userPrefs,
  ) {
    double colorHarmonyScore = _calculateColorHarmony(combo.items);
    double styleConsistencyScore = _calculateStyleConsistency(combo.items);
    double seasonalFitScore = _calculateSeasonalFit(combo.items, combo.context);
    double personalPreferenceScore = _calculatePersonalPreference(
      combo.items,
      userPrefs,
    );
    double wearHistoryScore = _calculateWearHistoryScore(combo.items);

    // Weighted scoring (user preferences matter most for personalization)
    final totalScore =
        (colorHarmonyScore * 0.25) +
        (styleConsistencyScore * 0.20) +
        (seasonalFitScore * 0.15) +
        (personalPreferenceScore * 0.30) + // Highest weight for personalization
        (wearHistoryScore * 0.10);

    return combo.copyWith(
      colorHarmonyScore: colorHarmonyScore,
      styleConsistencyScore: styleConsistencyScore,
      seasonalFitScore: seasonalFitScore,
      personalPreferenceScore: personalPreferenceScore,
      wearHistoryScore: wearHistoryScore,
      totalScore: totalScore,
    );
  }

  /// Calculate color harmony using color theory algorithms
  ///
  /// Algorithms: Complementary, Analogous, Triadic, Monochromatic
  double _calculateColorHarmony(List<WardrobeItem> items) {
    final colors = items
        .where((item) => item.color != null && item.color!.isNotEmpty)
        .map((item) => item.color!)
        .toList();

    if (colors.length < 2) return 0.8; // Neutral score for single color

    // Convert color names to HSV for analysis
    final hsvColors = colors.map(_colorNameToHSV).toList();

    double score = 0.0;
    int comparisons = 0;

    // Check all color pairs
    for (int i = 0; i < hsvColors.length; i++) {
      for (int j = i + 1; j < hsvColors.length; j++) {
        final color1 = hsvColors[i];
        final color2 = hsvColors[j];

        // Calculate hue difference
        final hueDiff = (color1['h']! - color2['h']!).abs();
        final normalizedDiff = hueDiff > 180 ? 360 - hueDiff : hueDiff;

        // Check color harmony rules
        double pairScore = 0.0;

        // Complementary (180° ± 30°)
        if (normalizedDiff >= 150 && normalizedDiff <= 210) {
          pairScore = 0.95;
        }
        // Analogous (30° ± 15°)
        else if (normalizedDiff <= 45) {
          pairScore = 0.90;
        }
        // Triadic (120° ± 20°)
        else if (normalizedDiff >= 100 && normalizedDiff <= 140) {
          pairScore = 0.85;
        }
        // Split-complementary (150° ± 20°)
        else if (normalizedDiff >= 130 && normalizedDiff <= 170) {
          pairScore = 0.80;
        }
        // Neutral compatibility (gray, white, black)
        else if (color1['s']! < 0.2 || color2['s']! < 0.2) {
          pairScore = 0.75; // Neutrals go with everything
        }
        // Poor match
        else {
          pairScore = 0.50;
        }

        score += pairScore;
        comparisons++;
      }
    }

    return comparisons > 0 ? score / comparisons : 0.7;
  }

  /// Calculate style consistency (formal, casual, athletic, etc.)
  double _calculateStyleConsistency(List<WardrobeItem> items) {
    final styles = items
        .where((item) => item.style != null && item.style!.isNotEmpty)
        .map((item) => item.style!)
        .toList();

    if (styles.isEmpty) return 0.7;

    // Count style occurrences
    final styleCounts = <String, int>{};
    for (final style in styles) {
      styleCounts[style] = (styleCounts[style] ?? 0) + 1;
    }

    // Check if styles are compatible
    final hasConsistency = _checkStyleCompatibility(styleCounts.keys.toList());

    return hasConsistency ? 0.9 : 0.6;
  }

  /// Check if styles are compatible
  bool _checkStyleCompatibility(List<String> styles) {
    // Define incompatible style pairs
    final incompatible = [
      ['Formal', 'Athletic'],
      ['Formal', 'Casual'],
      ['Business', 'Athletic'],
    ];

    for (final pair in incompatible) {
      if (styles.contains(pair[0]) && styles.contains(pair[1])) {
        return false;
      }
    }

    return true;
  }

  /// Calculate seasonal fitness
  double _calculateSeasonalFit(
    List<WardrobeItem> items,
    OutfitRecommendation context,
  ) {
    final currentSeason = context.context.temporal.season;
    int matchingItems = 0;

    for (final item in items) {
      if (item.season.isEmpty || item.season.contains(currentSeason)) {
        matchingItems++;
      }
    }

    return items.isNotEmpty ? matchingItems / items.length : 0.5;
  }

  /// Calculate personal preference score based on user's history
  ///
  /// THIS IS KEY FOR PERSONALIZATION - uses ML-learned preferences from feedback
  double _calculatePersonalPreference(
    List<WardrobeItem> items,
    Map<String, dynamic> userPrefs,
  ) {
    double score = 0.5; // Start neutral
    int components = 0;

    // 1. Use ML-learned preferences from feedback service
    final mlScore = _learningService.calculateOutfitScore(
      items,
      occasion: userPrefs['occasion'] as String?,
    );
    score += mlScore;
    components++;

    // 2. Check favorite items (user-marked)
    final favCount = items.where((item) => item.isFavorite).length;
    if (favCount > 0) {
      score += (favCount * 0.15).clamp(0.0, 0.3);
      components++;
    }

    // 3. Use learned color preferences
    double colorScore = 0;
    for (final item in items) {
      if (item.color != null) {
        colorScore += _learningService.getColorScore(item.color);
      }
    }
    if (items.isNotEmpty) {
      score += colorScore / items.length;
      components++;
    }

    // 4. Use learned style preferences
    double styleScore = 0;
    for (final item in items) {
      if (item.style != null) {
        styleScore += _learningService.getStyleScore(item.style);
      }
    }
    final itemsWithStyle = items.where((i) => i.style != null).length;
    if (itemsWithStyle > 0) {
      score += styleScore / itemsWithStyle;
      components++;
    }

    // 5. Use learned item pair compatibility
    double pairScore = 0;
    int pairs = 0;
    for (int i = 0; i < items.length; i++) {
      for (int j = i + 1; j < items.length; j++) {
        pairScore += _learningService.getItemPairScore(items[i].id, items[j].id);
        pairs++;
      }
    }
    if (pairs > 0) {
      score += pairScore / pairs;
      components++;
    }

    // 6. Fallback: use explicitly passed preferences
    final favoriteColors =
        (userPrefs['favorite_colors'] as List?)?.cast<String>() ?? [];
    if (favoriteColors.isNotEmpty) {
      for (final item in items) {
        if (item.color != null && favoriteColors.contains(item.color)) {
          score += 0.05;
        }
      }
    }

    final preferredStyles =
        (userPrefs['preferred_styles'] as List?)?.cast<String>() ?? [];
    if (preferredStyles.isNotEmpty) {
      for (final item in items) {
        if (item.style != null && preferredStyles.contains(item.style)) {
          score += 0.05;
        }
      }
    }

    return components > 0 ? (score / components).clamp(0.0, 1.0) : 0.5;
  }

  /// Calculate wear history score (promote underutilized items)
  double _calculateWearHistoryScore(List<WardrobeItem> items) {
    if (items.isEmpty) return 0.5;

    // Calculate average wear count
    final totalWears = items.fold<int>(0, (sum, item) => sum + item.wearCount);
    final avgWears = totalWears / items.length;

    // Lower wear count = higher score (encourage variety)
    if (avgWears < 2) return 0.9;
    if (avgWears < 5) return 0.8;
    if (avgWears < 10) return 0.7;
    return 0.6;
  }

  /// Apply diversity filter to avoid similar outfits
  List<OutfitCombination> _applyDiversityFilter(
    List<OutfitCombination> combinations,
    int maxResults,
  ) {
    final diverse = <OutfitCombination>[];
    final usedItemSets = <Set<String>>[];

    for (final combo in combinations) {
      final itemSet = combo.items.map((item) => item.id).toSet();

      // Check if this combination is too similar to existing ones
      bool isSimilar = false;
      for (final usedSet in usedItemSets) {
        final overlap = itemSet.intersection(usedSet).length;
        // If more than 60% overlap, consider it similar
        if (overlap / itemSet.length > 0.6) {
          isSimilar = true;
          break;
        }
      }

      if (!isSimilar) {
        diverse.add(combo);
        usedItemSets.add(itemSet);
      }

      if (diverse.length >= maxResults) break;
    }

    return diverse;
  }

  /// Convert color name to HSV (simplified)
  Map<String, double> _colorNameToHSV(String colorName) {
    // Simplified color mapping (in production, use a proper color library)
    final colorMap = {
      // Primary colors
      'Red': {'h': 0.0, 's': 1.0, 'v': 1.0},
      'Orange': {'h': 30.0, 's': 1.0, 'v': 1.0},
      'Yellow': {'h': 60.0, 's': 1.0, 'v': 1.0},
      'Green': {'h': 120.0, 's': 1.0, 'v': 1.0},
      'Blue': {'h': 240.0, 's': 1.0, 'v': 1.0},
      'Purple': {'h': 280.0, 's': 1.0, 'v': 1.0},
      'Pink': {'h': 330.0, 's': 0.7, 'v': 1.0},

      // Neutrals
      'White': {'h': 0.0, 's': 0.0, 'v': 1.0},
      'Black': {'h': 0.0, 's': 0.0, 'v': 0.0},
      'Gray': {'h': 0.0, 's': 0.0, 'v': 0.5},
      'Brown': {'h': 30.0, 's': 0.6, 'v': 0.5},
      'Beige': {'h': 40.0, 's': 0.3, 'v': 0.8},

      // Variations
      'Navy': {'h': 240.0, 's': 0.8, 'v': 0.5},
      'Dark Blue': {'h': 240.0, 's': 0.9, 'v': 0.6},
      'Light Blue': {'h': 200.0, 's': 0.5, 'v': 0.9},
    };

    return colorMap[colorName] ?? {'h': 0.0, 's': 0.5, 'v': 0.7};
  }

  /// Record user feedback on an outfit (thumbs up/down)
  /// This trains the ML personalization system
  Future<void> recordOutfitFeedback({
    required List<WardrobeItem> items,
    required bool isPositive,
    String? occasion,
    Map<String, dynamic>? context,
  }) async {
    await _learningService.recordFeedback(
      items: items,
      isPositive: isPositive,
      occasion: occasion,
      context: context,
    );
  }

  /// Get personalization summary for debugging
  Map<String, dynamic> getPersonalizationSummary() {
    return _learningService.getPersonalizationSummary();
  }
}

/// Outfit Combination with scoring
class OutfitCombination {
  final List<WardrobeItem> items;
  final OutfitRecommendation context;
  final double colorHarmonyScore;
  final double styleConsistencyScore;
  final double seasonalFitScore;
  final double personalPreferenceScore;
  final double wearHistoryScore;
  final double totalScore;

  OutfitCombination({
    required this.items,
    required this.context,
    this.colorHarmonyScore = 0.0,
    this.styleConsistencyScore = 0.0,
    this.seasonalFitScore = 0.0,
    this.personalPreferenceScore = 0.0,
    this.wearHistoryScore = 0.0,
    this.totalScore = 0.0,
  });

  OutfitCombination copyWith({
    List<WardrobeItem>? items,
    OutfitRecommendation? context,
    double? colorHarmonyScore,
    double? styleConsistencyScore,
    double? seasonalFitScore,
    double? personalPreferenceScore,
    double? wearHistoryScore,
    double? totalScore,
  }) {
    return OutfitCombination(
      items: items ?? this.items,
      context: context ?? this.context,
      colorHarmonyScore: colorHarmonyScore ?? this.colorHarmonyScore,
      styleConsistencyScore:
          styleConsistencyScore ?? this.styleConsistencyScore,
      seasonalFitScore: seasonalFitScore ?? this.seasonalFitScore,
      personalPreferenceScore:
          personalPreferenceScore ?? this.personalPreferenceScore,
      wearHistoryScore: wearHistoryScore ?? this.wearHistoryScore,
      totalScore: totalScore ?? this.totalScore,
    );
  }

  /// Get personalized styling tips
  List<String> getPersonalizedTips() {
    final tips = <String>[];

    // Color harmony tips
    if (colorHarmonyScore > 0.85) {
      tips.add(
        '🎨 Perfect color harmony! This combination follows complementary color theory.',
      );
    } else if (colorHarmonyScore < 0.6) {
      tips.add('💡 Try swapping one item for better color balance.');
    }

    // Seasonal tips
    if (seasonalFitScore > 0.9) {
      tips.add('🌤️ Ideal for ${context.context.temporal.season} weather!');
    }

    // Personal preference
    if (personalPreferenceScore > 0.8) {
      tips.add('⭐ This matches your style preferences perfectly!');
    }

    // Wear history
    if (wearHistoryScore > 0.85) {
      tips.add('👔 Great mix of underutilized items - let them shine!');
    }

    return tips;
  }
}
