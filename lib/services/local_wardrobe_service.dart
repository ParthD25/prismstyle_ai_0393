import 'dart:io';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import '../models/wardrobe_item.dart';
import './clothing_classifier_service.dart';

/// Local Wardrobe Service
/// Manages wardrobe items with local SQLite storage - NO AUTHENTICATION REQUIRED
///
/// Features:
/// - Local SQLite database for offline storage
/// - Image storage in app documents directory
/// - AI-powered classification integration
/// - No cloud dependencies or auth requirements
class LocalWardrobeService {
  static LocalWardrobeService? _instance;
  static LocalWardrobeService get instance =>
      _instance ??= LocalWardrobeService._();

  LocalWardrobeService._();

  Database? _database;
  final ClothingClassifierService _classifier =
      ClothingClassifierService.instance;
  bool _isInitialized = false;

  static const String _tableName = 'wardrobe_items';
  static const String _dbName = 'prismstyle_wardrobe.db';
  static const String _localUserId = 'local_user';

  // In-memory cache for fast access
  final Map<String, WardrobeItem> _cache = {};

  /// Initialize the service and database
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Initialize classifier
      await _classifier.initialize();

      // Initialize database
      await _initDatabase();

      // Load items into cache
      await _loadAllItemsToCache();

      _isInitialized = true;
      debugPrint(
        '✅ LocalWardrobeService initialized with ${_cache.length} items',
      );
    } catch (e) {
      debugPrint('⚠️ LocalWardrobeService initialization error: $e');
    }
  }

  /// Initialize SQLite database
  Future<void> _initDatabase() async {
    final databasesPath = await getDatabasesPath();
    final dbPath = path.join(databasesPath, _dbName);

    _database = await openDatabase(
      dbPath,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE $_tableName (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT,
            photo_url TEXT,
            category TEXT NOT NULL,
            subcategory TEXT,
            color TEXT,
            secondary_color TEXT,
            pattern TEXT DEFAULT 'Solid',
            material TEXT,
            style TEXT,
            season TEXT,
            occasion TEXT,
            brand TEXT,
            size TEXT,
            ai_confidence REAL,
            ai_predictions TEXT,
            color_analysis TEXT,
            tags TEXT,
            is_favorite INTEGER DEFAULT 0,
            wear_count INTEGER DEFAULT 0,
            last_worn_at TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
          )
        ''');
        debugPrint('Database table created');
      },
    );
  }

  /// Load all items from database to cache
  Future<void> _loadAllItemsToCache() async {
    if (_database == null) return;

    final List<Map<String, dynamic>> maps = await _database!.query(_tableName);
    _cache.clear();

    for (final map in maps) {
      final item = _mapToWardrobeItem(map);
      _cache[item.id] = item;
    }
  }

  /// Convert database map to WardrobeItem
  WardrobeItem _mapToWardrobeItem(Map<String, dynamic> map) {
    return WardrobeItem(
      id: map['id'] as String,
      userId: map['user_id'] as String,
      name: map['name'] as String?,
      photoUrl: map['photo_url'] as String?,
      category: map['category'] as String,
      subcategory: map['subcategory'] as String?,
      color: map['color'] as String?,
      secondaryColor: map['secondary_color'] as String?,
      pattern: map['pattern'] as String? ?? 'Solid',
      material: map['material'] as String?,
      style: map['style'] as String?,
      season: _parseJsonList(map['season']),
      occasion: _parseJsonList(map['occasion']),
      brand: map['brand'] as String?,
      size: map['size'] as String?,
      aiConfidence: map['ai_confidence'] as double?,
      aiPredictions: _parseJsonMap(map['ai_predictions']),
      colorAnalysis: _parseJsonMap(map['color_analysis']),
      tags: _parseJsonList(map['tags']),
      isFavorite: (map['is_favorite'] as int?) == 1,
      wearCount: map['wear_count'] as int? ?? 0,
      lastWornAt: map['last_worn_at'] != null
          ? DateTime.tryParse(map['last_worn_at'] as String)
          : null,
      metadata: _parseJsonMap(map['metadata']),
      createdAt: DateTime.parse(map['created_at'] as String),
      updatedAt: DateTime.parse(map['updated_at'] as String),
    );
  }

  List<String> _parseJsonList(dynamic value) {
    if (value == null) return [];
    if (value is String) {
      try {
        final list = json.decode(value);
        if (list is List) return list.cast<String>();
      } catch (_) {}
    }
    return [];
  }

  Map<String, dynamic> _parseJsonMap(dynamic value) {
    if (value == null) return {};
    if (value is String) {
      try {
        final map = json.decode(value);
        if (map is Map) return map.cast<String, dynamic>();
      } catch (_) {}
    }
    return {};
  }

  /// Add item with AI classification (NO AUTH REQUIRED)
  Future<WardrobeItem?> addItem({
    required File imageFile,
    String? name,
    String? occasion,
    String? location,
    Map<String, dynamic>? additionalData,
  }) async {
    try {
      // 1. Read image bytes for classification
      final imageBytes = await imageFile.readAsBytes();

      // 2. Run AI classification
      debugPrint('🔄 Classifying clothing item...');
      final classification = await _classifier.classifyWithContext(
        imageBytes: imageBytes,
        occasion: occasion,
        location: location,
      );

      // 3. Save image to local storage
      debugPrint('📤 Saving image locally...');
      final localPath = await _saveImageLocally(imageBytes);

      // 4. Generate unique ID
      final id = 'item_${DateTime.now().millisecondsSinceEpoch}';
      final now = DateTime.now();

      // 5. Create item with all required fields
      final item = WardrobeItem(
        id: id,
        userId: _localUserId,
        name: name ?? _generateItemName(classification),
        photoUrl: localPath,
        category: classification.appCategory,
        subcategory: classification.category,
        color: classification.primaryColor,
        pattern: classification.pattern ?? 'Solid',
        material: classification.material,
        season: [],
        occasion: occasion != null ? [occasion] : [],
        aiConfidence: classification.confidence,
        aiPredictions: classification.allPredictions,
        colorAnalysis: classification.colorAnalysis,
        tags: classification.suggestedTags,
        metadata: additionalData ?? {},
        createdAt: now,
        updatedAt: now,
      );

      // 6. Insert into database
      await _database?.insert(_tableName, {
        'id': item.id,
        'user_id': item.userId,
        'name': item.name,
        'photo_url': item.photoUrl,
        'category': item.category,
        'subcategory': item.subcategory,
        'color': item.color,
        'secondary_color': item.secondaryColor,
        'pattern': item.pattern,
        'material': item.material,
        'style': item.style,
        'season': json.encode(item.season),
        'occasion': json.encode(item.occasion),
        'brand': item.brand,
        'size': item.size,
        'ai_confidence': item.aiConfidence,
        'ai_predictions': json.encode(item.aiPredictions),
        'color_analysis': json.encode(item.colorAnalysis),
        'tags': json.encode(item.tags),
        'is_favorite': item.isFavorite ? 1 : 0,
        'wear_count': item.wearCount,
        'last_worn_at': item.lastWornAt?.toIso8601String(),
        'metadata': json.encode(item.metadata),
        'created_at': item.createdAt.toIso8601String(),
        'updated_at': item.updatedAt.toIso8601String(),
      });

      // 7. Update cache
      _cache[item.id] = item;

      debugPrint('✅ Item added successfully: ${item.name}');
      return item;
    } catch (e) {
      debugPrint('❌ Error adding item: $e');
      return null;
    }
  }

  /// Save image to local storage and return path
  Future<String> _saveImageLocally(Uint8List imageBytes) async {
    final appDir = await getApplicationDocumentsDirectory();
    final imagesDir = Directory(path.join(appDir.path, 'wardrobe_images'));

    if (!await imagesDir.exists()) {
      await imagesDir.create(recursive: true);
    }

    final fileName = 'img_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final filePath = path.join(imagesDir.path, fileName);

    final file = File(filePath);
    await file.writeAsBytes(imageBytes);

    return filePath;
  }

  /// Generate item name from classification
  String _generateItemName(dynamic classification) {
    final category = classification.category as String? ?? 'Item';
    final color = classification.primaryColor as String? ?? '';

    // Convert category to readable name
    final readableName = category
        .replaceAll('_', ' ')
        .split(' ')
        .map(
          (w) => w.isNotEmpty ? '${w[0].toUpperCase()}${w.substring(1)}' : '',
        )
        .join(' ');

    return color.isNotEmpty ? '$color $readableName' : readableName;
  }

  /// Get all items (from cache)
  Future<List<WardrobeItem>> getAllItems() async {
    if (!_isInitialized) await initialize();
    return _cache.values.toList()
      ..sort((a, b) => b.createdAt.compareTo(a.createdAt));
  }

  /// Get items by category
  Future<List<WardrobeItem>> getItemsByCategory(String category) async {
    if (!_isInitialized) await initialize();
    return _cache.values
        .where(
          (item) => item.category == category || item.subcategory == category,
        )
        .toList();
  }

  /// Get item by ID
  Future<WardrobeItem?> getItemById(String id) async {
    return _cache[id];
  }

  /// Update item
  Future<bool> updateItem(WardrobeItem item) async {
    try {
      final updatedItem = item.copyWith(updatedAt: DateTime.now());

      await _database?.update(
        _tableName,
        {
          'name': updatedItem.name,
          'category': updatedItem.category,
          'subcategory': updatedItem.subcategory,
          'color': updatedItem.color,
          'pattern': updatedItem.pattern,
          'material': updatedItem.material,
          'brand': updatedItem.brand,
          'size': updatedItem.size,
          'season': json.encode(updatedItem.season),
          'occasion': json.encode(updatedItem.occasion),
          'tags': json.encode(updatedItem.tags),
          'is_favorite': updatedItem.isFavorite ? 1 : 0,
          'wear_count': updatedItem.wearCount,
          'last_worn_at': updatedItem.lastWornAt?.toIso8601String(),
          'updated_at': updatedItem.updatedAt.toIso8601String(),
        },
        where: 'id = ?',
        whereArgs: [item.id],
      );

      _cache[item.id] = updatedItem;
      return true;
    } catch (e) {
      debugPrint('Error updating item: $e');
      return false;
    }
  }

  /// Delete item
  Future<bool> deleteItem(String id) async {
    try {
      // Delete image file
      final item = _cache[id];
      if (item != null && item.photoUrl != null) {
        final file = File(item.photoUrl!);
        if (await file.exists()) {
          await file.delete();
        }
      }

      // Delete from database
      await _database?.delete(_tableName, where: 'id = ?', whereArgs: [id]);

      // Remove from cache
      _cache.remove(id);
      return true;
    } catch (e) {
      debugPrint('Error deleting item: $e');
      return false;
    }
  }

  /// Toggle favorite status
  Future<bool> toggleFavorite(String id) async {
    final item = _cache[id];
    if (item == null) return false;

    final updatedItem = item.copyWith(isFavorite: !item.isFavorite);
    return updateItem(updatedItem);
  }

  /// Record item worn
  Future<bool> recordWear(String id) async {
    final item = _cache[id];
    if (item == null) return false;

    final updatedItem = item.copyWith(
      wearCount: item.wearCount + 1,
      lastWornAt: DateTime.now(),
    );
    return updateItem(updatedItem);
  }

  /// Get wardrobe statistics
  Future<Map<String, dynamic>> getStatistics() async {
    if (!_isInitialized) await initialize();

    final items = _cache.values.toList();

    // Category counts
    final categoryCounts = <String, int>{};
    for (final item in items) {
      categoryCounts[item.category] = (categoryCounts[item.category] ?? 0) + 1;
    }

    // Color distribution
    final colorCounts = <String, int>{};
    for (final item in items) {
      final color = item.color ?? 'Unknown';
      colorCounts[color] = (colorCounts[color] ?? 0) + 1;
    }

    // Most worn items
    final sortedByWear = items
      ..sort((a, b) => b.wearCount.compareTo(a.wearCount));
    final mostWorn = sortedByWear.take(5).toList();

    // Favorite items
    final favorites = items.where((i) => i.isFavorite).toList();

    return {
      'totalItems': items.length,
      'categoryCounts': categoryCounts,
      'colorDistribution': colorCounts,
      'mostWornItems': mostWorn.map((i) => i.toJson()).toList(),
      'favoriteCount': favorites.length,
      'averageWearCount': items.isEmpty
          ? 0
          : items.map((i) => i.wearCount).reduce((a, b) => a + b) /
                items.length,
    };
  }

  /// Search items
  Future<List<WardrobeItem>> searchItems(String query) async {
    if (!_isInitialized) await initialize();

    final lowerQuery = query.toLowerCase();
    return _cache.values.where((item) {
      final name = item.name ?? '';
      return name.toLowerCase().contains(lowerQuery) ||
          item.category.toLowerCase().contains(lowerQuery) ||
          (item.color?.toLowerCase().contains(lowerQuery) ?? false) ||
          item.tags.any((t) => t.toLowerCase().contains(lowerQuery));
    }).toList();
  }

  /// Get items for outfit generation
  Future<Map<String, List<WardrobeItem>>> getItemsForOutfitGeneration() async {
    if (!_isInitialized) await initialize();

    final result = <String, List<WardrobeItem>>{
      'Tops': [],
      'Bottoms': [],
      'Dresses': [],
      'Outerwear': [],
      'Accessories': [],
      'Shoes': [],
    };

    for (final item in _cache.values) {
      if (result.containsKey(item.category)) {
        result[item.category]!.add(item);
      }
    }

    return result;
  }

  /// Check if service is ready
  bool get isReady => _isInitialized;

  /// Get item count
  int get itemCount => _cache.length;

  /// Dispose resources
  Future<void> dispose() async {
    await _database?.close();
    _database = null;
    _cache.clear();
    _isInitialized = false;
  }
}
