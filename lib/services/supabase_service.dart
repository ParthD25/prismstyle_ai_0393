import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

/// Supabase Service for PrismStyle AI
/// Handles database operations, real-time subscriptions, and authentication
/// 
/// Database Schema:
/// - users: User profiles with style preferences
/// - clothing_items: Wardrobe items with AI-detected attributes
/// - outfits: Generated and saved outfits
/// - friend_relationships: Social connections
/// - outfit_feedback: Community feedback on outfits
class SupabaseService {
  static SupabaseService? _instance;
  static SupabaseService get instance => _instance ??= SupabaseService._();

  SupabaseService._();

  static String supabaseUrl = '';
  static String supabaseAnonKey = '';

  // Real-time subscription streams
  StreamSubscription? _wardrobeSubscription;
  StreamSubscription? _outfitsSubscription;
  StreamSubscription? _feedbackSubscription;

  // Stream controllers for real-time updates
  final _wardrobeStreamController = StreamController<List<ClothingItem>>.broadcast();
  final _outfitsStreamController = StreamController<List<Outfit>>.broadcast();
  final _feedbackStreamController = StreamController<List<OutfitFeedback>>.broadcast();

  // Public streams
  Stream<List<ClothingItem>> get wardrobeStream => _wardrobeStreamController.stream;
  Stream<List<Outfit>> get outfitsStream => _outfitsStreamController.stream;
  Stream<List<OutfitFeedback>> get feedbackStream => _feedbackStreamController.stream;

  // Initialize Supabase - call this in main()
  static Future<void> initialize() async {
    // Load credentials from assets/env.json
    try {
      final envString = await rootBundle.loadString('assets/env.json');
      final env = json.decode(envString) as Map<String, dynamic>;
      
      supabaseUrl = env['SUPABASE_URL'] as String? ?? '';
      supabaseAnonKey = env['SUPABASE_ANON_KEY'] as String? ?? '';
      
      if (supabaseUrl.isEmpty || supabaseAnonKey.isEmpty) {
        throw Exception(
          'SUPABASE_URL and SUPABASE_ANON_KEY must be defined in env.json',
        );
      }
      
      await Supabase.initialize(url: supabaseUrl, anonKey: supabaseAnonKey);
      debugPrint('Supabase initialized: ${supabaseUrl.substring(0, 20)}...');
      
    } catch (e) {
      debugPrint('Failed to load Supabase credentials: $e');
      throw Exception('Supabase initialization failed: $e');
    }
  }

  // Get Supabase client
  SupabaseClient get client => Supabase.instance.client;

  // Get current user ID
  String? get currentUserId => client.auth.currentUser?.id;

  // ============== USER OPERATIONS ==============

  /// Get user profile
  Future<UserProfile?> getUserProfile(String userId) async {
    try {
      final response = await client
          .from('users')
          .select()
          .eq('id', userId)
          .single();
      return UserProfile.fromJson(response);
    } catch (e) {
      debugPrint('Error getting user profile: $e');
      return null;
    }
  }

  /// Create or update user profile
  Future<bool> upsertUserProfile(UserProfile profile) async {
    try {
      await client.from('users').upsert(profile.toJson());
      return true;
    } catch (e) {
      debugPrint('Error upserting user profile: $e');
      return false;
    }
  }

  /// Update style preferences
  Future<bool> updateStylePreferences(
    String userId,
    Map<String, dynamic> preferences,
  ) async {
    try {
      await client
          .from('users')
          .update({'style_preferences': preferences})
          .eq('id', userId);
      return true;
    } catch (e) {
      debugPrint('Error updating style preferences: $e');
      return false;
    }
  }

  // ============== CLOTHING ITEMS OPERATIONS ==============

  /// Get all clothing items for a user
  Future<List<ClothingItem>> getClothingItems(String userId) async {
    try {
      final response = await client
          .from('clothing_items')
          .select()
          .eq('user_id', userId)
          .order('created_at', ascending: false);
      return (response as List)
          .map((item) => ClothingItem.fromJson(item))
          .toList();
    } catch (e) {
      debugPrint('Error getting clothing items: $e');
      return [];
    }
  }

  /// Get clothing items by category
  Future<List<ClothingItem>> getClothingItemsByCategory(
    String userId,
    String category,
  ) async {
    try {
      final response = await client
          .from('clothing_items')
          .select()
          .eq('user_id', userId)
          .eq('category', category)
          .order('created_at', ascending: false);
      return (response as List)
          .map((item) => ClothingItem.fromJson(item))
          .toList();
    } catch (e) {
      debugPrint('Error getting clothing items by category: $e');
      return [];
    }
  }

  /// Add a new clothing item
  Future<ClothingItem?> addClothingItem(ClothingItem item) async {
    try {
      final response = await client
          .from('clothing_items')
          .insert(item.toJson())
          .select()
          .single();
      return ClothingItem.fromJson(response);
    } catch (e) {
      debugPrint('Error adding clothing item: $e');
      return null;
    }
  }

  /// Update a clothing item
  Future<bool> updateClothingItem(ClothingItem item) async {
    try {
      if (item.id == null) return false;
      await client
          .from('clothing_items')
          .update(item.toJson())
          .eq('id', item.id!);
      return true;
    } catch (e) {
      debugPrint('Error updating clothing item: $e');
      return false;
    }
  }

  /// Delete a clothing item
  Future<bool> deleteClothingItem(String itemId) async {
    try {
      await client.from('clothing_items').delete().eq('id', itemId);
      return true;
    } catch (e) {
      debugPrint('Error deleting clothing item: $e');
      return false;
    }
  }

  /// Upload clothing image to Supabase Storage
  Future<String?> uploadClothingImage(
    String userId,
    String fileName,
    List<int> imageBytes,
  ) async {
    try {
      final path = '$userId/$fileName';
      await client.storage
          .from('clothing-images')
          .uploadBinary(path, imageBytes as dynamic);
      final publicUrl = client.storage
          .from('clothing-images')
          .getPublicUrl(path);
      return publicUrl;
    } catch (e) {
      debugPrint('Error uploading clothing image: $e');
      return null;
    }
  }

  // ============== OUTFIT OPERATIONS ==============

  /// Get all outfits for a user
  Future<List<Outfit>> getOutfits(String userId) async {
    try {
      final response = await client
          .from('outfits')
          .select()
          .eq('user_id', userId)
          .order('created_at', ascending: false);
      return (response as List)
          .map((item) => Outfit.fromJson(item))
          .toList();
    } catch (e) {
      debugPrint('Error getting outfits: $e');
      return [];
    }
  }

  /// Get saved outfits only
  Future<List<Outfit>> getSavedOutfits(String userId) async {
    try {
      final response = await client
          .from('outfits')
          .select()
          .eq('user_id', userId)
          .eq('is_saved', true)
          .order('created_at', ascending: false);
      return (response as List)
          .map((item) => Outfit.fromJson(item))
          .toList();
    } catch (e) {
      debugPrint('Error getting saved outfits: $e');
      return [];
    }
  }

  /// Save an outfit
  Future<Outfit?> saveOutfit(Outfit outfit) async {
    try {
      final response = await client
          .from('outfits')
          .insert(outfit.toJson())
          .select()
          .single();
      return Outfit.fromJson(response);
    } catch (e) {
      debugPrint('Error saving outfit: $e');
      return null;
    }
  }

  /// Update outfit saved status
  Future<bool> toggleOutfitSaved(String outfitId, bool isSaved) async {
    try {
      await client
          .from('outfits')
          .update({'is_saved': isSaved})
          .eq('id', outfitId);
      return true;
    } catch (e) {
      debugPrint('Error toggling outfit saved status: $e');
      return false;
    }
  }

  /// Delete an outfit
  Future<bool> deleteOutfit(String outfitId) async {
    try {
      await client.from('outfits').delete().eq('id', outfitId);
      return true;
    } catch (e) {
      debugPrint('Error deleting outfit: $e');
      return false;
    }
  }

  // ============== FRIEND RELATIONSHIPS ==============

  /// Get user's friends
  Future<List<FriendRelationship>> getFriends(String userId) async {
    try {
      final response = await client
          .from('friend_relationships')
          .select()
          .or('user_id.eq.$userId,friend_id.eq.$userId')
          .eq('status', 'accepted');
      return (response as List)
          .map((item) => FriendRelationship.fromJson(item))
          .toList();
    } catch (e) {
      debugPrint('Error getting friends: $e');
      return [];
    }
  }

  /// Get pending friend requests
  Future<List<FriendRelationship>> getPendingRequests(String userId) async {
    try {
      final response = await client
          .from('friend_relationships')
          .select()
          .eq('friend_id', userId)
          .eq('status', 'pending');
      return (response as List)
          .map((item) => FriendRelationship.fromJson(item))
          .toList();
    } catch (e) {
      debugPrint('Error getting pending requests: $e');
      return [];
    }
  }

  /// Send friend request
  Future<bool> sendFriendRequest(String userId, String friendId) async {
    try {
      await client.from('friend_relationships').insert({
        'user_id': userId,
        'friend_id': friendId,
        'status': 'pending',
      });
      return true;
    } catch (e) {
      debugPrint('Error sending friend request: $e');
      return false;
    }
  }

  /// Accept friend request
  Future<bool> acceptFriendRequest(String requestId) async {
    try {
      await client
          .from('friend_relationships')
          .update({'status': 'accepted'})
          .eq('id', requestId);
      return true;
    } catch (e) {
      debugPrint('Error accepting friend request: $e');
      return false;
    }
  }

  // ============== OUTFIT FEEDBACK ==============

  /// Get feedback for an outfit
  Future<List<OutfitFeedback>> getOutfitFeedback(String outfitId) async {
    try {
      final response = await client
          .from('outfit_feedback')
          .select()
          .eq('outfit_id', outfitId)
          .order('created_at', ascending: false);
      return (response as List)
          .map((item) => OutfitFeedback.fromJson(item))
          .toList();
    } catch (e) {
      debugPrint('Error getting outfit feedback: $e');
      return [];
    }
  }

  /// Add feedback to an outfit
  Future<OutfitFeedback?> addOutfitFeedback(OutfitFeedback feedback) async {
    try {
      final response = await client
          .from('outfit_feedback')
          .insert(feedback.toJson())
          .select()
          .single();
      return OutfitFeedback.fromJson(response);
    } catch (e) {
      debugPrint('Error adding outfit feedback: $e');
      return null;
    }
  }

  // ============== REAL-TIME SUBSCRIPTIONS ==============

  /// Subscribe to wardrobe changes
  void subscribeToWardrobe(String userId) {
    _wardrobeSubscription?.cancel();
    _wardrobeSubscription = client
        .from('clothing_items')
        .stream(primaryKey: ['id'])
        .eq('user_id', userId)
        .listen((data) {
          final items = data.map((item) => ClothingItem.fromJson(item)).toList();
          _wardrobeStreamController.add(items);
        });
  }

  /// Subscribe to outfit changes
  void subscribeToOutfits(String userId) {
    _outfitsSubscription?.cancel();
    _outfitsSubscription = client
        .from('outfits')
        .stream(primaryKey: ['id'])
        .eq('user_id', userId)
        .listen((data) {
          final outfits = data.map((item) => Outfit.fromJson(item)).toList();
          _outfitsStreamController.add(outfits);
        });
  }

  /// Subscribe to feedback on user's outfits
  void subscribeToFeedback(String outfitId) {
    _feedbackSubscription?.cancel();
    _feedbackSubscription = client
        .from('outfit_feedback')
        .stream(primaryKey: ['id'])
        .eq('outfit_id', outfitId)
        .listen((data) {
          final feedback = data.map((item) => OutfitFeedback.fromJson(item)).toList();
          _feedbackStreamController.add(feedback);
        });
  }

  /// Unsubscribe from all real-time updates
  void unsubscribeAll() {
    _wardrobeSubscription?.cancel();
    _outfitsSubscription?.cancel();
    _feedbackSubscription?.cancel();
  }

  /// Dispose resources
  void dispose() {
    unsubscribeAll();
    _wardrobeStreamController.close();
    _outfitsStreamController.close();
    _feedbackStreamController.close();
  }
}

// ============== DATA MODELS ==============

/// User profile model
class UserProfile {
  final String id;
  final String email;
  final String? gender;
  final Map<String, dynamic>? stylePreferences;
  final DateTime? createdAt;

  UserProfile({
    required this.id,
    required this.email,
    this.gender,
    this.stylePreferences,
    this.createdAt,
  });

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      id: json['id'] as String,
      email: json['email'] as String,
      gender: json['gender'] as String?,
      stylePreferences: json['style_preferences'] as Map<String, dynamic>?,
      createdAt: json['created_at'] != null
          ? DateTime.parse(json['created_at'])
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      'gender': gender,
      'style_preferences': stylePreferences,
    };
  }
}

/// Clothing item model
class ClothingItem {
  final String? id;
  final String userId;
  final String? photoUrl;
  final String category;
  final String? color;
  final String? pattern;
  final String? style;
  final double? aiConfidence;
  final DateTime? createdAt;
  final Map<String, dynamic>? metadata;

  ClothingItem({
    this.id,
    required this.userId,
    this.photoUrl,
    required this.category,
    this.color,
    this.pattern,
    this.style,
    this.aiConfidence,
    this.createdAt,
    this.metadata,
  });

  factory ClothingItem.fromJson(Map<String, dynamic> json) {
    return ClothingItem(
      id: json['id'] as String?,
      userId: json['user_id'] as String,
      photoUrl: json['photo_url'] as String?,
      category: json['category'] as String,
      color: json['color'] as String?,
      pattern: json['pattern'] as String?,
      style: json['style'] as String?,
      aiConfidence: (json['ai_confidence'] as num?)?.toDouble(),
      createdAt: json['created_at'] != null
          ? DateTime.parse(json['created_at'])
          : null,
      metadata: json['metadata'] as Map<String, dynamic>?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      if (id != null) 'id': id,
      'user_id': userId,
      'photo_url': photoUrl,
      'category': category,
      'color': color,
      'pattern': pattern,
      'style': style,
      'ai_confidence': aiConfidence,
      if (metadata != null) 'metadata': metadata,
    };
  }
}

/// Outfit model
class Outfit {
  final String? id;
  final String userId;
  final String? occasion;
  final String? weatherCondition;
  final Map<String, dynamic>? outfitData;
  final bool isSaved;
  final DateTime? createdAt;

  Outfit({
    this.id,
    required this.userId,
    this.occasion,
    this.weatherCondition,
    this.outfitData,
    this.isSaved = false,
    this.createdAt,
  });

  factory Outfit.fromJson(Map<String, dynamic> json) {
    return Outfit(
      id: json['id'] as String?,
      userId: json['user_id'] as String,
      occasion: json['occasion'] as String?,
      weatherCondition: json['weather_condition'] as String?,
      outfitData: json['outfit_data'] as Map<String, dynamic>?,
      isSaved: json['is_saved'] as bool? ?? false,
      createdAt: json['created_at'] != null
          ? DateTime.parse(json['created_at'])
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      if (id != null) 'id': id,
      'user_id': userId,
      'occasion': occasion,
      'weather_condition': weatherCondition,
      'outfit_data': outfitData,
      'is_saved': isSaved,
    };
  }
}

/// Friend relationship model
class FriendRelationship {
  final String? id;
  final String userId;
  final String friendId;
  final String status;
  final DateTime? createdAt;

  FriendRelationship({
    this.id,
    required this.userId,
    required this.friendId,
    required this.status,
    this.createdAt,
  });

  factory FriendRelationship.fromJson(Map<String, dynamic> json) {
    return FriendRelationship(
      id: json['id'] as String?,
      userId: json['user_id'] as String,
      friendId: json['friend_id'] as String,
      status: json['status'] as String,
      createdAt: json['created_at'] != null
          ? DateTime.parse(json['created_at'])
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      if (id != null) 'id': id,
      'user_id': userId,
      'friend_id': friendId,
      'status': status,
    };
  }
}

/// Outfit feedback model
class OutfitFeedback {
  final String? id;
  final String outfitId;
  final String fromUserId;
  final int? rating;
  final String? comment;
  final DateTime? createdAt;

  OutfitFeedback({
    this.id,
    required this.outfitId,
    required this.fromUserId,
    this.rating,
    this.comment,
    this.createdAt,
  });

  factory OutfitFeedback.fromJson(Map<String, dynamic> json) {
    return OutfitFeedback(
      id: json['id'] as String?,
      outfitId: json['outfit_id'] as String,
      fromUserId: json['from_user_id'] as String,
      rating: json['rating'] as int?,
      comment: json['comment'] as String?,
      createdAt: json['created_at'] != null
          ? DateTime.parse(json['created_at'])
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      if (id != null) 'id': id,
      'outfit_id': outfitId,
      'from_user_id': fromUserId,
      'rating': rating,
      'comment': comment,
    };
  }
}
