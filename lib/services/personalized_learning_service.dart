import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/wardrobe_item.dart';

/// Personalized Learning Service - On-Device User Preference Learning
/// 
/// Learns from user feedback (thumbs up/down) to improve outfit recommendations.
/// Uses EWMA (Exponentially Weighted Moving Average) for preference learning.
/// All data stored locally - NO cloud costs, NO subscriptions required.
class PersonalizedLearningService {
  static PersonalizedLearningService? _instance;
  static PersonalizedLearningService get instance =>
      _instance ??= PersonalizedLearningService._();

  PersonalizedLearningService._();

  static const String _feedbackKey = 'outfit_feedback_history';
  static const String _combinationKey = 'item_combination_scores';
  static const String _colorPrefKey = 'color_preferences';
  static const String _stylePrefKey = 'style_preferences';
  static const String _occasionPrefKey = 'occasion_preferences';
  
  // EWMA decay factor (0.1 = learns slowly, 0.3 = learns faster)
  static const double _learningRate = 0.2;
  
  SharedPreferences? _prefs;
  
  // In-memory caches
  Map<String, double> _colorPreferences = {};
  Map<String, double> _stylePreferences = {};
  Map<String, double> _occasionPreferences = {};
  Map<String, double> _itemPairScores = {}; // "itemId1_itemId2" -> score
  final Map<String, double> _categoryPairScores = {}; // "category1_category2" -> score
  List<OutfitFeedback> _feedbackHistory = [];
  
  bool _isInitialized = false;

  /// Initialize the service and load saved preferences
  Future<void> initialize() async {
    if (_isInitialized) return;
    
    _prefs = await SharedPreferences.getInstance();
    await _loadPreferences();
    _isInitialized = true;
    debugPrint('✅ PersonalizedLearningService initialized');
  }

  /// Load all saved preferences from local storage
  Future<void> _loadPreferences() async {
    try {
      // Load color preferences
      final colorJson = _prefs?.getString(_colorPrefKey);
      if (colorJson != null) {
        _colorPreferences = Map<String, double>.from(json.decode(colorJson));
      }
      
      // Load style preferences
      final styleJson = _prefs?.getString(_stylePrefKey);
      if (styleJson != null) {
        _stylePreferences = Map<String, double>.from(json.decode(styleJson));
      }
      
      // Load occasion preferences
      final occasionJson = _prefs?.getString(_occasionPrefKey);
      if (occasionJson != null) {
        _occasionPreferences = Map<String, double>.from(json.decode(occasionJson));
      }
      
      // Load item pair scores
      final combinationJson = _prefs?.getString(_combinationKey);
      if (combinationJson != null) {
        _itemPairScores = Map<String, double>.from(json.decode(combinationJson));
      }
      
      // Load feedback history
      final feedbackJson = _prefs?.getString(_feedbackKey);
      if (feedbackJson != null) {
        final List<dynamic> feedbackList = json.decode(feedbackJson);
        _feedbackHistory = feedbackList
            .map((e) => OutfitFeedback.fromJson(e))
            .toList();
      }
      
      debugPrint('Loaded ${_feedbackHistory.length} feedback entries');
    } catch (e) {
      debugPrint('Error loading preferences: $e');
    }
  }

  /// Save all preferences to local storage
  Future<void> _savePreferences() async {
    try {
      await _prefs?.setString(_colorPrefKey, json.encode(_colorPreferences));
      await _prefs?.setString(_stylePrefKey, json.encode(_stylePreferences));
      await _prefs?.setString(_occasionPrefKey, json.encode(_occasionPreferences));
      await _prefs?.setString(_combinationKey, json.encode(_itemPairScores));
      
      // Save last 1000 feedback entries to prevent unbounded growth
      final recentFeedback = _feedbackHistory.length > 1000
          ? _feedbackHistory.sublist(_feedbackHistory.length - 1000)
          : _feedbackHistory;
      await _prefs?.setString(
        _feedbackKey,
        json.encode(recentFeedback.map((e) => e.toJson()).toList()),
      );
    } catch (e) {
      debugPrint('Error saving preferences: $e');
    }
  }

  /// Record user feedback on an outfit combination
  /// 
  /// [items] - The wardrobe items in the outfit
  /// [isPositive] - true for thumbs up, false for thumbs down
  /// [occasion] - The occasion the outfit was generated for
  /// [context] - Additional context (weather, time of day, etc.)
  Future<void> recordFeedback({
    required List<WardrobeItem> items,
    required bool isPositive,
    String? occasion,
    Map<String, dynamic>? context,
  }) async {
    if (!_isInitialized) await initialize();
    
    final feedback = OutfitFeedback(
      itemIds: items.map((e) => e.id).toList(),
      isPositive: isPositive,
      occasion: occasion,
      timestamp: DateTime.now(),
      context: context ?? {},
    );
    
    _feedbackHistory.add(feedback);
    
    // Update preference scores using EWMA
    final delta = isPositive ? 1.0 : -0.5; // Positive has stronger effect
    
    // Learn color preferences
    for (final item in items) {
      if (item.color != null) {
        _updatePreference(_colorPreferences, item.color!, delta);
      }
      if (item.secondaryColor != null) {
        _updatePreference(_colorPreferences, item.secondaryColor!, delta * 0.5);
      }
    }
    
    // Learn style preferences
    for (final item in items) {
      if (item.style != null) {
        _updatePreference(_stylePreferences, item.style!, delta);
      }
    }
    
    // Learn occasion preferences
    if (occasion != null) {
      _updatePreference(_occasionPreferences, occasion, delta);
    }
    
    // Learn item pair compatibility
    for (int i = 0; i < items.length; i++) {
      for (int j = i + 1; j < items.length; j++) {
        final pairKey = _getPairKey(items[i].id, items[j].id);
        _updatePreference(_itemPairScores, pairKey, delta);
        
        // Also learn category pairs
        final categoryKey = _getPairKey(items[i].category, items[j].category);
        _updatePreference(_categoryPairScores, categoryKey, delta * 0.5);
      }
    }
    
    await _savePreferences();
    
    debugPrint('📝 Recorded ${isPositive ? "positive" : "negative"} feedback');
    debugPrint('   Colors: ${items.map((e) => e.color).join(", ")}');
    debugPrint('   Styles: ${items.map((e) => e.style).join(", ")}');
  }

  /// Update a preference score using EWMA
  void _updatePreference(Map<String, double> prefs, String key, double delta) {
    final currentScore = prefs[key] ?? 0.5; // Start neutral
    final newScore = currentScore + _learningRate * (delta - (currentScore - 0.5));
    prefs[key] = newScore.clamp(0.0, 1.0);
  }

  /// Get a consistent key for an item pair (order independent)
  String _getPairKey(String id1, String id2) {
    return id1.compareTo(id2) < 0 ? '${id1}_$id2' : '${id2}_$id1';
  }

  /// Get personalized score for a color
  double getColorScore(String? color) {
    if (color == null) return 0.5;
    return _colorPreferences[color] ?? 0.5;
  }

  /// Get personalized score for a style
  double getStyleScore(String? style) {
    if (style == null) return 0.5;
    return _stylePreferences[style] ?? 0.5;
  }

  /// Get personalized score for an occasion
  double getOccasionScore(String? occasion) {
    if (occasion == null) return 0.5;
    return _occasionPreferences[occasion] ?? 0.5;
  }

  /// Get compatibility score for two items based on learned preferences
  double getItemPairScore(String id1, String id2) {
    final key = _getPairKey(id1, id2);
    return _itemPairScores[key] ?? 0.5;
  }

  /// Get compatibility score for two categories
  double getCategoryPairScore(String cat1, String cat2) {
    final key = _getPairKey(cat1, cat2);
    return _categoryPairScores[key] ?? 0.5;
  }

  /// Calculate personalized score for an entire outfit combination
  /// 
  /// Returns a score from 0.0 to 1.0 based on learned preferences
  double calculateOutfitScore(List<WardrobeItem> items, {String? occasion}) {
    if (items.isEmpty) return 0.5;
    
    double totalScore = 0.0;
    int components = 0;
    
    // Color preference scores
    for (final item in items) {
      if (item.color != null) {
        totalScore += getColorScore(item.color);
        components++;
      }
    }
    
    // Style preference scores
    for (final item in items) {
      if (item.style != null) {
        totalScore += getStyleScore(item.style);
        components++;
      }
    }
    
    // Item pair compatibility scores
    for (int i = 0; i < items.length; i++) {
      for (int j = i + 1; j < items.length; j++) {
        totalScore += getItemPairScore(items[i].id, items[j].id);
        components++;
      }
    }
    
    // Occasion preference
    if (occasion != null) {
      totalScore += getOccasionScore(occasion);
      components++;
    }
    
    // Favorite items bonus
    final favoriteCount = items.where((i) => i.isFavorite).length;
    if (favoriteCount > 0) {
      totalScore += favoriteCount * 0.1;
      components++;
    }
    
    return components > 0 ? (totalScore / components).clamp(0.0, 1.0) : 0.5;
  }

  /// Get user's top preferred colors (learned from feedback)
  List<String> getPreferredColors({int limit = 5}) {
    final sorted = _colorPreferences.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return sorted
        .where((e) => e.value > 0.5) // Only preferences above neutral
        .take(limit)
        .map((e) => e.key)
        .toList();
  }

  /// Get user's top preferred styles
  List<String> getPreferredStyles({int limit = 5}) {
    final sorted = _stylePreferences.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return sorted
        .where((e) => e.value > 0.5)
        .take(limit)
        .map((e) => e.key)
        .toList();
  }

  /// Get personalization summary for debugging/display
  Map<String, dynamic> getPersonalizationSummary() {
    return {
      'total_feedback': _feedbackHistory.length,
      'positive_feedback': _feedbackHistory.where((f) => f.isPositive).length,
      'negative_feedback': _feedbackHistory.where((f) => !f.isPositive).length,
      'learned_colors': _colorPreferences.length,
      'learned_styles': _stylePreferences.length,
      'learned_pairs': _itemPairScores.length,
      'preferred_colors': getPreferredColors(),
      'preferred_styles': getPreferredStyles(),
    };
  }

  /// Export preferences as user profile (for debugging)
  Map<String, dynamic> exportUserProfile() {
    return {
      'color_preferences': _colorPreferences,
      'style_preferences': _stylePreferences,
      'occasion_preferences': _occasionPreferences,
      'feedback_count': _feedbackHistory.length,
      'last_updated': DateTime.now().toIso8601String(),
    };
  }

  /// Clear all learned preferences (reset)
  Future<void> resetPreferences() async {
    _colorPreferences.clear();
    _stylePreferences.clear();
    _occasionPreferences.clear();
    _itemPairScores.clear();
    _categoryPairScores.clear();
    _feedbackHistory.clear();
    await _savePreferences();
    debugPrint('🗑️ All preferences reset');
  }
}

/// Outfit feedback record
class OutfitFeedback {
  final List<String> itemIds;
  final bool isPositive;
  final String? occasion;
  final DateTime timestamp;
  final Map<String, dynamic> context;

  OutfitFeedback({
    required this.itemIds,
    required this.isPositive,
    this.occasion,
    required this.timestamp,
    this.context = const {},
  });

  factory OutfitFeedback.fromJson(Map<String, dynamic> json) {
    return OutfitFeedback(
      itemIds: List<String>.from(json['item_ids'] ?? []),
      isPositive: json['is_positive'] ?? true,
      occasion: json['occasion'],
      timestamp: DateTime.parse(json['timestamp']),
      context: json['context'] ?? {},
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'item_ids': itemIds,
      'is_positive': isPositive,
      'occasion': occasion,
      'timestamp': timestamp.toIso8601String(),
      'context': context,
    };
  }
}
