import 'dart:io';
import 'package:flutter/foundation.dart';
import '../models/wardrobe_item.dart';
import './clothing_classifier_service.dart';
import './local_wardrobe_service.dart';

/// Wardrobe Service
/// Manages wardrobe items with LOCAL storage - NO AUTHENTICATION REQUIRED
///
/// This is a wrapper around LocalWardrobeService for backward compatibility.
/// All data is stored locally on device with SQLite database.
///
/// Features:
/// - CRUD operations stored locally
/// - Image storage in app documents directory
/// - AI-powered classification integration
/// - No cloud dependencies or auth requirements
class WardrobeService {
  static WardrobeService? _instance;
  static WardrobeService get instance => _instance ??= WardrobeService._();

  WardrobeService._();

  // Use local storage service instead of Supabase
  final LocalWardrobeService _localService = LocalWardrobeService.instance;
  final ClothingClassifierService _classifier =
      ClothingClassifierService.instance;

  // Cache for fast access
  final Map<String, WardrobeItem> _cache = {};
  bool _isInitialized = false;

  /// Initialize the service
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      await _classifier.initialize();
      await _localService.initialize();

      // Load items to local cache
      final items = await _localService.getAllItems();
      for (final item in items) {
        _cache[item.id] = item;
      }

      _isInitialized = true;
      debugPrint(
        '✅ WardrobeService initialized with ${_cache.length} items (LOCAL MODE)',
      );
    } catch (e) {
      debugPrint('⚠️ WardrobeService initialization error: $e');
    }
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
      final item = await _localService.addItem(
        imageFile: imageFile,
        name: name,
        occasion: occasion,
        location: location,
        additionalData: additionalData,
      );

      if (item != null) {
        _cache[item.id] = item;
      }

      return item;
    } catch (e) {
      debugPrint('❌ Error adding item: $e');
      return null;
    }
  }

  /// Add item from bytes (NO AUTH REQUIRED)
  Future<WardrobeItem?> addItemFromBytes({
    required Uint8List imageBytes,
    String? name,
    String? occasion,
    String? location,
    Map<String, dynamic>? additionalData,
  }) async {
    try {
      // Create temporary file
      final tempDir = Directory.systemTemp;
      final tempFile = File(
        '${tempDir.path}/temp_${DateTime.now().millisecondsSinceEpoch}.jpg',
      );
      await tempFile.writeAsBytes(imageBytes);

      final item = await addItem(
        imageFile: tempFile,
        name: name,
        occasion: occasion,
        location: location,
        additionalData: additionalData,
      );

      // Clean up temp file
      if (await tempFile.exists()) {
        await tempFile.delete();
      }

      return item;
    } catch (e) {
      debugPrint('❌ Error adding item from bytes: $e');
      return null;
    }
  }

  /// Get all items (NO AUTH REQUIRED)
  Future<List<WardrobeItem>> getAllItems() async {
    if (!_isInitialized) await initialize();

    try {
      final items = await _localService.getAllItems();

      // Update cache
      _cache.clear();
      for (final item in items) {
        _cache[item.id] = item;
      }

      return items;
    } catch (e) {
      debugPrint('Error fetching items: $e');
      return _cache.values.toList();
    }
  }

  /// Get items by category
  Future<List<WardrobeItem>> getItemsByCategory(String category) async {
    if (!_isInitialized) await initialize();
    return _localService.getItemsByCategory(category);
  }

  /// Get favorite items
  Future<List<WardrobeItem>> getFavoriteItems() async {
    if (!_isInitialized) await initialize();
    return _cache.values.where((item) => item.isFavorite).toList();
  }

  /// Search items by query
  Future<List<WardrobeItem>> searchItems(String query) async {
    if (!_isInitialized) await initialize();
    return _localService.searchItems(query);
  }

  /// Update item
  Future<WardrobeItem?> updateItem(
    String itemId,
    Map<String, dynamic> updates,
  ) async {
    try {
      final item = _cache[itemId];
      if (item == null) return null;

      // Create updated item
      final updatedItem = item.copyWith(
        name: updates['name'] as String? ?? item.name,
        category: updates['category'] as String? ?? item.category,
        color: updates['color'] as String? ?? item.color,
        isFavorite: updates['is_favorite'] as bool? ?? item.isFavorite,
        updatedAt: DateTime.now(),
      );

      final success = await _localService.updateItem(updatedItem);
      if (success) {
        _cache[itemId] = updatedItem;
        return updatedItem;
      }
      return null;
    } catch (e) {
      debugPrint('Error updating item: $e');
      return null;
    }
  }

  /// Toggle favorite status
  Future<bool> toggleFavorite(String itemId) async {
    final success = await _localService.toggleFavorite(itemId);
    if (success) {
      final item = _cache[itemId];
      if (item != null) {
        _cache[itemId] = item.copyWith(isFavorite: !item.isFavorite);
      }
    }
    return success;
  }

  /// Record item worn
  Future<void> incrementWearCount(String itemId) async {
    await _localService.recordWear(itemId);
    final item = _cache[itemId];
    if (item != null) {
      _cache[itemId] = item.copyWith(
        wearCount: item.wearCount + 1,
        lastWornAt: DateTime.now(),
      );
    }
  }

  /// Get item by ID
  Future<WardrobeItem?> getItemById(String itemId) async {
    if (_cache.containsKey(itemId)) {
      return _cache[itemId];
    }
    return _localService.getItemById(itemId);
  }

  /// Delete item
  Future<bool> deleteItem(String itemId) async {
    final success = await _localService.deleteItem(itemId);
    if (success) {
      _cache.remove(itemId);
    }
    return success;
  }

  /// Subscribe to real-time updates (returns stream of cached items since local-only)
  Stream<List<WardrobeItem>> subscribeToWardrobe() {
    // For local storage, return a stream with current items
    return Stream.value(_cache.values.toList());
  }

  /// Get statistics
  Future<Map<String, dynamic>> getStatistics() async {
    return _localService.getStatistics();
  }

  /// Get items organized by category for outfit generation
  Future<Map<String, List<WardrobeItem>>> getItemsForOutfitGeneration() async {
    return _localService.getItemsForOutfitGeneration();
  }

  /// Check if service is ready
  bool get isReady => _isInitialized && _localService.isReady;

  /// Get item count
  int get itemCount => _cache.length;

  /// Clear cache
  void clearCache() {
    _cache.clear();
  }

  /// Dispose resources
  void dispose() {
    clearCache();
  }
}
