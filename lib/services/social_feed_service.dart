import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/outfit.dart';

/// Social Feed Service
///
/// Manages outfit sharing and social interactions with:
/// - Infinite scroll pagination
/// - Real-time updates
/// - Image optimization
/// - Engagement tracking (reactions, comments)
class SocialFeedService {
  static SocialFeedService? _instance;
  static SocialFeedService get instance => _instance ??= SocialFeedService._();

  SocialFeedService._();

  final SupabaseClient _supabase = Supabase.instance.client;

  static const String _outfitsTable = 'outfits';
  static const String _feedbackTable = 'outfit_feedback';
  static const int _pageSize = 20; // Items per page

  // Cache for performance
  final Map<String, Outfit> _outfitCache = {};
  final Map<String, List<OutfitFeedback>> _feedbackCache = {};

  /// Get paginated feed with infinite scroll support
  ///
  /// [offset]: Starting position (0 for first page, 20 for second, etc.)
  /// [filter]: Optional filter ('friends', 'popular', 'recent')
  Future<List<Outfit>> getFeed({
    int offset = 0,
    String filter = 'recent',
  }) async {
    try {
      final baseQuery = _supabase
          .from(_outfitsTable)
          .select()
          .eq('is_shared', true);

      // Apply filter and pagination
      late final PostgrestList response;
      switch (filter) {
        case 'friends':
          // TODO: Add friends filter when friend system is implemented
          response = await baseQuery
              .order('created_at', ascending: false)
              .range(offset, offset + _pageSize - 1);
          break;
        case 'popular':
          // Order by times_worn (popularity indicator)
          response = await baseQuery
              .order('times_worn', ascending: false)
              .range(offset, offset + _pageSize - 1);
          break;
        case 'recent':
        default:
          response = await baseQuery
              .order('created_at', ascending: false)
              .range(offset, offset + _pageSize - 1);
      }

      final outfits = (response as List)
          .map((json) => Outfit.fromJson(json))
          .toList();

      // Cache results
      for (final outfit in outfits) {
        _outfitCache[outfit.id] = outfit;
      }

      debugPrint('✅ Loaded ${outfits.length} outfits (offset: $offset)');
      return outfits;
    } catch (e) {
      debugPrint('❌ Error loading feed: $e');
      return [];
    }
  }

  /// Subscribe to real-time feed updates
  /// Returns a stream that emits when new shared outfits are posted
  Stream<List<Outfit>> subscribeToFeedUpdates() {
    return _supabase
        .from(_outfitsTable)
        .stream(primaryKey: ['id'])
        .eq('is_shared', true)
        .order('created_at', ascending: false)
        .map((data) => data.map((json) => Outfit.fromJson(json)).toList());
  }

  /// Share an outfit to the social feed
  Future<bool> shareOutfit(String outfitId) async {
    try {
      await _supabase
          .from(_outfitsTable)
          .update({'is_shared': true})
          .eq('id', outfitId);

      // Update cache
      if (_outfitCache.containsKey(outfitId)) {
        _outfitCache[outfitId] = _outfitCache[outfitId]!.copyWith(
          isShared: true,
        );
      }

      debugPrint('✅ Outfit shared to feed: $outfitId');
      return true;
    } catch (e) {
      debugPrint('❌ Error sharing outfit: $e');
      return false;
    }
  }

  /// Unshare an outfit from the social feed
  Future<bool> unshareOutfit(String outfitId) async {
    try {
      await _supabase
          .from(_outfitsTable)
          .update({'is_shared': false})
          .eq('id', outfitId);

      // Update cache
      if (_outfitCache.containsKey(outfitId)) {
        _outfitCache[outfitId] = _outfitCache[outfitId]!.copyWith(
          isShared: false,
        );
      }

      return true;
    } catch (e) {
      debugPrint('❌ Error unsharing outfit: $e');
      return false;
    }
  }

  /// Add reaction to an outfit
  Future<bool> addReaction({
    required String outfitId,
    required String reaction,
    int? rating,
    String? comment,
  }) async {
    try {
      final userId = _supabase.auth.currentUser?.id;
      if (userId == null) {
        throw Exception('User not authenticated');
      }

      await _supabase.from(_feedbackTable).upsert({
        'outfit_id': outfitId,
        'from_user_id': userId,
        'reaction': reaction,
        'rating': rating,
        'comment': comment,
      });

      // Clear cache to force refresh
      _feedbackCache.remove(outfitId);

      debugPrint('✅ Reaction added: $reaction');
      return true;
    } catch (e) {
      debugPrint('❌ Error adding reaction: $e');
      return false;
    }
  }

  /// Get feedback for an outfit
  Future<List<OutfitFeedback>> getFeedback(String outfitId) async {
    // Check cache first
    if (_feedbackCache.containsKey(outfitId)) {
      return _feedbackCache[outfitId]!;
    }

    try {
      final response = await _supabase
          .from(_feedbackTable)
          .select()
          .eq('outfit_id', outfitId)
          .order('created_at', ascending: false);

      final feedback = (response as List)
          .map((json) => OutfitFeedback.fromJson(json))
          .toList();

      // Cache results
      _feedbackCache[outfitId] = feedback;

      return feedback;
    } catch (e) {
      debugPrint('❌ Error loading feedback: $e');
      return [];
    }
  }

  /// Get aggregated feedback stats
  Future<FeedbackStats> getFeedbackStats(String outfitId) async {
    try {
      final feedback = await getFeedback(outfitId);

      final reactionCounts = <String, int>{};
      double totalRating = 0.0;
      int ratingCount = 0;

      for (final fb in feedback) {
        if (fb.reaction != null) {
          reactionCounts[fb.reaction!] =
              (reactionCounts[fb.reaction!] ?? 0) + 1;
        }
        if (fb.rating != null) {
          totalRating += fb.rating!;
          ratingCount++;
        }
      }

      return FeedbackStats(
        totalReactions: feedback.length,
        reactionCounts: reactionCounts,
        averageRating: ratingCount > 0 ? totalRating / ratingCount : 0.0,
        commentCount: feedback.where((f) => f.comment != null).length,
      );
    } catch (e) {
      debugPrint('❌ Error loading feedback stats: $e');
      return FeedbackStats(
        totalReactions: 0,
        reactionCounts: {},
        averageRating: 0.0,
        commentCount: 0,
      );
    }
  }

  /// Get user's own shared outfits
  Future<List<Outfit>> getUserSharedOutfits() async {
    try {
      final userId = _supabase.auth.currentUser?.id;
      if (userId == null) return [];

      final response = await _supabase
          .from(_outfitsTable)
          .select()
          .eq('user_id', userId)
          .eq('is_shared', true)
          .order('created_at', ascending: false);

      return (response as List).map((json) => Outfit.fromJson(json)).toList();
    } catch (e) {
      debugPrint('❌ Error loading user shared outfits: $e');
      return [];
    }
  }

  /// Search outfits in feed
  Future<List<Outfit>> searchFeed(String query) async {
    try {
      final response = await _supabase
          .from(_outfitsTable)
          .select()
          .eq('is_shared', true)
          .or('name.ilike.%$query%,occasion.ilike.%$query%')
          .order('created_at', ascending: false)
          .limit(50);

      return (response as List).map((json) => Outfit.fromJson(json)).toList();
    } catch (e) {
      debugPrint('❌ Error searching feed: $e');
      return [];
    }
  }

  /// Clear cache
  void clearCache() {
    _outfitCache.clear();
    _feedbackCache.clear();
  }
}

/// Feedback statistics
class FeedbackStats {
  final int totalReactions;
  final Map<String, int> reactionCounts;
  final double averageRating;
  final int commentCount;

  FeedbackStats({
    required this.totalReactions,
    required this.reactionCounts,
    required this.averageRating,
    required this.commentCount,
  });

  /// Get most popular reaction
  String? get mostPopularReaction {
    if (reactionCounts.isEmpty) return null;

    return reactionCounts.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
  }

  /// Get reaction count for specific reaction
  int getReactionCount(String reaction) {
    return reactionCounts[reaction] ?? 0;
  }
}
