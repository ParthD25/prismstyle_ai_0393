/// Outfit Model
/// Represents a complete outfit combination stored in Supabase
class Outfit {
  final String id;
  final String userId;
  final String? name;
  final String? occasion;
  final String? weatherCondition;
  final String? temperatureRange;
  final Map<String, dynamic> outfitData;
  final List<String> itemIds;
  final double? compatibilityScore;
  final String? styleNotes;
  final bool isSaved;
  final bool isShared;
  final int timesWorn;
  final DateTime? lastWornAt;
  final DateTime createdAt;
  final DateTime updatedAt;

  Outfit({
    required this.id,
    required this.userId,
    this.name,
    this.occasion,
    this.weatherCondition,
    this.temperatureRange,
    this.outfitData = const {},
    this.itemIds = const [],
    this.compatibilityScore,
    this.styleNotes,
    this.isSaved = false,
    this.isShared = false,
    this.timesWorn = 0,
    this.lastWornAt,
    required this.createdAt,
    required this.updatedAt,
  });

  /// Create from Supabase JSON
  factory Outfit.fromJson(Map<String, dynamic> json) {
    return Outfit(
      id: json['id'] as String,
      userId: json['user_id'] as String,
      name: json['name'] as String?,
      occasion: json['occasion'] as String?,
      weatherCondition: json['weather_condition'] as String?,
      temperatureRange: json['temperature_range'] as String?,
      outfitData: json['outfit_data'] as Map<String, dynamic>? ?? {},
      itemIds: _parseStringList(json['item_ids']),
      compatibilityScore: _parseDouble(json['compatibility_score']),
      styleNotes: json['style_notes'] as String?,
      isSaved: json['is_saved'] as bool? ?? false,
      isShared: json['is_shared'] as bool? ?? false,
      timesWorn: json['times_worn'] as int? ?? 0,
      lastWornAt: _parseDateTime(json['last_worn_at']),
      createdAt: DateTime.parse(json['created_at'] as String),
      updatedAt: DateTime.parse(json['updated_at'] as String),
    );
  }

  /// Convert to Supabase JSON
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'user_id': userId,
      'name': name,
      'occasion': occasion,
      'weather_condition': weatherCondition,
      'temperature_range': temperatureRange,
      'outfit_data': outfitData,
      'item_ids': itemIds,
      'compatibility_score': compatibilityScore,
      'style_notes': styleNotes,
      'is_saved': isSaved,
      'is_shared': isShared,
      'times_worn': timesWorn,
      'last_worn_at': lastWornAt?.toIso8601String(),
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
    };
  }

  /// Create for insert
  Map<String, dynamic> toInsertJson() {
    return {
      'user_id': userId,
      'name': name,
      'occasion': occasion,
      'weather_condition': weatherCondition,
      'temperature_range': temperatureRange,
      'outfit_data': outfitData,
      'item_ids': itemIds,
      'compatibility_score': compatibilityScore,
      'style_notes': styleNotes,
      'is_saved': isSaved,
      'is_shared': isShared,
    };
  }

  /// Copy with modifications
  Outfit copyWith({
    String? id,
    String? userId,
    String? name,
    String? occasion,
    String? weatherCondition,
    String? temperatureRange,
    Map<String, dynamic>? outfitData,
    List<String>? itemIds,
    double? compatibilityScore,
    String? styleNotes,
    bool? isSaved,
    bool? isShared,
    int? timesWorn,
    DateTime? lastWornAt,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return Outfit(
      id: id ?? this.id,
      userId: userId ?? this.userId,
      name: name ?? this.name,
      occasion: occasion ?? this.occasion,
      weatherCondition: weatherCondition ?? this.weatherCondition,
      temperatureRange: temperatureRange ?? this.temperatureRange,
      outfitData: outfitData ?? this.outfitData,
      itemIds: itemIds ?? this.itemIds,
      compatibilityScore: compatibilityScore ?? this.compatibilityScore,
      styleNotes: styleNotes ?? this.styleNotes,
      isSaved: isSaved ?? this.isSaved,
      isShared: isShared ?? this.isShared,
      timesWorn: timesWorn ?? this.timesWorn,
      lastWornAt: lastWornAt ?? this.lastWornAt,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  // Helper parsers
  static List<String> _parseStringList(dynamic value) {
    if (value == null) return [];
    if (value is List) return value.map((e) => e.toString()).toList();
    return [];
  }

  static double? _parseDouble(dynamic value) {
    if (value == null) return null;
    if (value is double) return value;
    if (value is int) return value.toDouble();
    if (value is String) return double.tryParse(value);
    return null;
  }

  static DateTime? _parseDateTime(dynamic value) {
    if (value == null) return null;
    if (value is DateTime) return value;
    if (value is String) return DateTime.tryParse(value);
    return null;
  }
}

/// Outfit Feedback Model
class OutfitFeedback {
  final String id;
  final String outfitId;
  final String fromUserId;
  final int? rating;
  final String? comment;
  final String? reaction;
  final DateTime createdAt;

  OutfitFeedback({
    required this.id,
    required this.outfitId,
    required this.fromUserId,
    this.rating,
    this.comment,
    this.reaction,
    required this.createdAt,
  });

  factory OutfitFeedback.fromJson(Map<String, dynamic> json) {
    return OutfitFeedback(
      id: json['id'] as String,
      outfitId: json['outfit_id'] as String,
      fromUserId: json['from_user_id'] as String,
      rating: json['rating'] as int?,
      comment: json['comment'] as String?,
      reaction: json['reaction'] as String?,
      createdAt: DateTime.parse(json['created_at'] as String),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'outfit_id': outfitId,
      'from_user_id': fromUserId,
      'rating': rating,
      'comment': comment,
      'reaction': reaction,
      'created_at': createdAt.toIso8601String(),
    };
  }
}

/// Reactions available for outfits
class OutfitReactions {
  static const String love = 'love';
  static const String like = 'like';
  static const String fire = 'fire';
  static const String cool = 'cool';
  static const String meh = 'meh';

  static const List<String> all = [love, like, fire, cool, meh];
  
  static String getEmoji(String reaction) {
    switch (reaction) {
      case love:
        return '❤️';
      case like:
        return '👍';
      case fire:
        return '🔥';
      case cool:
        return '😎';
      case meh:
        return '😐';
      default:
        return '👍';
    }
  }
}
