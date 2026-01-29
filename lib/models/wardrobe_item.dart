/// Wardrobe Item Model
/// Represents a clothing item stored in Supabase
class WardrobeItem {
  final String id;
  final String userId;
  final String? name;
  final String? photoUrl;
  final String category;
  final String? subcategory;
  final String? color;
  final String? secondaryColor;
  final String pattern;
  final String? material;
  final String? style;
  final List<String> season;
  final List<String> occasion;
  final String? brand;
  final String? size;
  final double? aiConfidence;
  final Map<String, dynamic> aiPredictions;
  final Map<String, dynamic> colorAnalysis;
  final List<String> tags;
  final bool isFavorite;
  final int wearCount;
  final DateTime? lastWornAt;
  final Map<String, dynamic> metadata;
  final DateTime createdAt;
  final DateTime updatedAt;

  WardrobeItem({
    required this.id,
    required this.userId,
    this.name,
    this.photoUrl,
    required this.category,
    this.subcategory,
    this.color,
    this.secondaryColor,
    this.pattern = 'Solid',
    this.material,
    this.style,
    this.season = const [],
    this.occasion = const [],
    this.brand,
    this.size,
    this.aiConfidence,
    this.aiPredictions = const {},
    this.colorAnalysis = const {},
    this.tags = const [],
    this.isFavorite = false,
    this.wearCount = 0,
    this.lastWornAt,
    this.metadata = const {},
    required this.createdAt,
    required this.updatedAt,
  });

  /// Create from Supabase JSON
  factory WardrobeItem.fromJson(Map<String, dynamic> json) {
    return WardrobeItem(
      id: json['id'] as String,
      userId: json['user_id'] as String,
      name: json['name'] as String?,
      photoUrl: json['photo_url'] as String?,
      category: json['category'] as String,
      subcategory: json['subcategory'] as String?,
      color: json['color'] as String?,
      secondaryColor: json['secondary_color'] as String?,
      pattern: json['pattern'] as String? ?? 'Solid',
      material: json['material'] as String?,
      style: json['style'] as String?,
      season: _parseStringList(json['season']),
      occasion: _parseStringList(json['occasion']),
      brand: json['brand'] as String?,
      size: json['size'] as String?,
      aiConfidence: _parseDouble(json['ai_confidence']),
      aiPredictions: json['ai_predictions'] as Map<String, dynamic>? ?? {},
      colorAnalysis: json['color_analysis'] as Map<String, dynamic>? ?? {},
      tags: _parseStringList(json['tags']),
      isFavorite: json['is_favorite'] as bool? ?? false,
      wearCount: json['wear_count'] as int? ?? 0,
      lastWornAt: _parseDateTime(json['last_worn_at']),
      metadata: json['metadata'] as Map<String, dynamic>? ?? {},
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
      'photo_url': photoUrl,
      'category': category,
      'subcategory': subcategory,
      'color': color,
      'secondary_color': secondaryColor,
      'pattern': pattern,
      'material': material,
      'style': style,
      'season': season,
      'occasion': occasion,
      'brand': brand,
      'size': size,
      'ai_confidence': aiConfidence,
      'ai_predictions': aiPredictions,
      'color_analysis': colorAnalysis,
      'tags': tags,
      'is_favorite': isFavorite,
      'wear_count': wearCount,
      'last_worn_at': lastWornAt?.toIso8601String(),
      'metadata': metadata,
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
    };
  }

  /// Create for insert (without id, timestamps)
  Map<String, dynamic> toInsertJson() {
    return {
      'user_id': userId,
      'name': name,
      'photo_url': photoUrl,
      'category': category,
      'subcategory': subcategory,
      'color': color,
      'secondary_color': secondaryColor,
      'pattern': pattern,
      'material': material,
      'style': style,
      'season': season,
      'occasion': occasion,
      'brand': brand,
      'size': size,
      'ai_confidence': aiConfidence,
      'ai_predictions': aiPredictions,
      'color_analysis': colorAnalysis,
      'tags': tags,
      'is_favorite': isFavorite,
      'metadata': metadata,
    };
  }

  /// Copy with modifications
  WardrobeItem copyWith({
    String? id,
    String? userId,
    String? name,
    String? photoUrl,
    String? category,
    String? subcategory,
    String? color,
    String? secondaryColor,
    String? pattern,
    String? material,
    String? style,
    List<String>? season,
    List<String>? occasion,
    String? brand,
    String? size,
    double? aiConfidence,
    Map<String, dynamic>? aiPredictions,
    Map<String, dynamic>? colorAnalysis,
    List<String>? tags,
    bool? isFavorite,
    int? wearCount,
    DateTime? lastWornAt,
    Map<String, dynamic>? metadata,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return WardrobeItem(
      id: id ?? this.id,
      userId: userId ?? this.userId,
      name: name ?? this.name,
      photoUrl: photoUrl ?? this.photoUrl,
      category: category ?? this.category,
      subcategory: subcategory ?? this.subcategory,
      color: color ?? this.color,
      secondaryColor: secondaryColor ?? this.secondaryColor,
      pattern: pattern ?? this.pattern,
      material: material ?? this.material,
      style: style ?? this.style,
      season: season ?? this.season,
      occasion: occasion ?? this.occasion,
      brand: brand ?? this.brand,
      size: size ?? this.size,
      aiConfidence: aiConfidence ?? this.aiConfidence,
      aiPredictions: aiPredictions ?? this.aiPredictions,
      colorAnalysis: colorAnalysis ?? this.colorAnalysis,
      tags: tags ?? this.tags,
      isFavorite: isFavorite ?? this.isFavorite,
      wearCount: wearCount ?? this.wearCount,
      lastWornAt: lastWornAt ?? this.lastWornAt,
      metadata: metadata ?? this.metadata,
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

/// Categories supported by the app
class WardrobeCategories {
  static const String tops = 'Tops';
  static const String bottoms = 'Bottoms';
  static const String dresses = 'Dresses';
  static const String shoes = 'Shoes';
  static const String accessories = 'Accessories';
  static const String outerwear = 'Outerwear';
  static const String activewear = 'Activewear';
  static const String formal = 'Formal';

  static const List<String> all = [
    tops,
    bottoms,
    dresses,
    shoes,
    accessories,
    outerwear,
    activewear,
    formal,
  ];
}
