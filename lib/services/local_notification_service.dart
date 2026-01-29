import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

/// Local Notification Service for PrismStyle AI (No Firebase)
/// Uses Flutter Local Notifications + Supabase Real-time for all notifications
/// 
/// Features:
/// - Outfit feedback notifications (via Supabase real-time)
/// - Friend request notifications
/// - Weather-based outfit reminders
/// - Daily style tips
class LocalNotificationService {
  static LocalNotificationService? _instance;
  static LocalNotificationService get instance =>
      _instance ??= LocalNotificationService._();

  LocalNotificationService._();

  final FlutterLocalNotificationsPlugin _localNotifications =
      FlutterLocalNotificationsPlugin();

  // Notification channels
  static const String _channelIdOutfits = 'outfit_recommendations';
  static const String _channelIdWeather = 'weather_alerts';
  static const String _channelIdSocial = 'social_interactions';
  static const String _channelIdTips = 'style_tips';

  // Stream controller for notification taps
  final _notificationStreamController =
      StreamController<Map<String, dynamic>>.broadcast();
  Stream<Map<String, dynamic>> get notificationStream =>
      _notificationStreamController.stream;

  // Supabase real-time subscriptions
  RealtimeChannel? _feedbackChannel;
  RealtimeChannel? _friendRequestChannel;

  /// Initialize notification service
  Future<void> initialize() async {
    try {
      // Request notification permission
      await _requestPermission();

      // Initialize local notifications
      await _initializeLocalNotifications();

      // Set up Supabase real-time listeners
      _setupRealtimeListeners();

      debugPrint('LocalNotificationService initialized successfully');
    } catch (e) {
      debugPrint('Error initializing LocalNotificationService: $e');
    }
  }

  /// Request notification permission
  Future<bool> _requestPermission() async {
    final status = await Permission.notification.request();
    return status.isGranted;
  }

  /// Initialize local notifications plugin
  Future<void> _initializeLocalNotifications() async {
    // Android initialization
    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');

    // iOS initialization
    const iosSettings = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );

    const initSettings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _localNotifications.initialize(
      initSettings,
      onDidReceiveNotificationResponse: _onNotificationTapped,
    );

    // Create notification channels for Android
    if (defaultTargetPlatform == TargetPlatform.android) {
      await _createNotificationChannels();
    }
  }

  /// Create Android notification channels
  Future<void> _createNotificationChannels() async {
    final androidPlugin =
        _localNotifications.resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>();

    if (androidPlugin != null) {
      // Outfit recommendations channel
      await androidPlugin.createNotificationChannel(
        const AndroidNotificationChannel(
          _channelIdOutfits,
          'Outfit Recommendations',
          description: 'Daily outfit suggestions based on weather and occasions',
          importance: Importance.high,
        ),
      );

      // Weather alerts channel
      await androidPlugin.createNotificationChannel(
        const AndroidNotificationChannel(
          _channelIdWeather,
          'Weather Alerts',
          description: 'Weather changes that affect your outfit choices',
          importance: Importance.high,
        ),
      );

      // Social interactions channel
      await androidPlugin.createNotificationChannel(
        const AndroidNotificationChannel(
          _channelIdSocial,
          'Social Interactions',
          description: 'Comments and reactions on your outfits',
          importance: Importance.defaultImportance,
        ),
      );

      // Style tips channel
      await androidPlugin.createNotificationChannel(
        const AndroidNotificationChannel(
          _channelIdTips,
          'Style Tips',
          description: 'Daily fashion tips and trends',
          importance: Importance.low,
        ),
      );
    }
  }

  /// Set up Supabase real-time listeners for notifications
  void _setupRealtimeListeners() {
    final supabase = Supabase.instance.client;
    final userId = supabase.auth.currentUser?.id;

    if (userId == null) {
      debugPrint('User not authenticated, skipping real-time setup');
      return;
    }

    // Listen for outfit feedback
    _feedbackChannel = supabase
        .channel('outfit_feedback_notifications')
        .onPostgresChanges(
          event: PostgresChangeEvent.insert,
          schema: 'public',
          table: 'outfit_feedback',
          callback: (payload) {
            _handleFeedbackNotification(payload.newRecord);
          },
        )
        .subscribe();

    // Listen for friend requests
    _friendRequestChannel = supabase
        .channel('friend_request_notifications')
        .onPostgresChanges(
          event: PostgresChangeEvent.insert,
          schema: 'public',
          table: 'friend_relationships',
          filter: PostgresChangeFilter(
            type: PostgresChangeFilterType.eq,
            column: 'friend_id',
            value: userId,
          ),
          callback: (payload) {
            _handleFriendRequestNotification(payload.newRecord);
          },
        )
        .subscribe();

    debugPrint('Real-time listeners set up successfully');
  }

  /// Handle outfit feedback notification
  void _handleFeedbackNotification(Map<String, dynamic> feedback) {
    showSocialNotification(
      title: 'New Outfit Feedback',
      body: 'Someone commented on your outfit!',
      data: {'type': 'feedback', 'feedback_id': feedback['id']},
    );
  }

  /// Handle friend request notification
  void _handleFriendRequestNotification(Map<String, dynamic> request) {
    showSocialNotification(
      title: 'New Friend Request',
      body: 'You have a new friend request',
      data: {'type': 'friend_request', 'request_id': request['id']},
    );
  }

  /// Show local notification
  Future<void> _showLocalNotification({
    required String title,
    required String body,
    String? payload,
    String type = 'general',
  }) async {
    final channelId = _getChannelForType(type);

    final androidDetails = AndroidNotificationDetails(
      channelId,
      _getChannelName(channelId),
      importance: Importance.high,
      priority: Priority.high,
      icon: '@mipmap/ic_launcher',
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    final details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _localNotifications.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      title,
      body,
      details,
      payload: payload,
    );
  }

  /// Handle notification tap
  void _onNotificationTapped(NotificationResponse response) {
    if (response.payload != null) {
      try {
        final data = jsonDecode(response.payload!) as Map<String, dynamic>;
        _notificationStreamController.add(data);
      } catch (e) {
        debugPrint('Error parsing notification payload: $e');
      }
    }
  }

  /// Get channel ID for notification type
  String _getChannelForType(String type) {
    switch (type) {
      case 'outfit':
        return _channelIdOutfits;
      case 'weather':
        return _channelIdWeather;
      case 'social':
        return _channelIdSocial;
      case 'tips':
        return _channelIdTips;
      default:
        return _channelIdOutfits;
    }
  }

  /// Get channel name
  String _getChannelName(String channelId) {
    switch (channelId) {
      case _channelIdOutfits:
        return 'Outfit Recommendations';
      case _channelIdWeather:
        return 'Weather Alerts';
      case _channelIdSocial:
        return 'Social Interactions';
      case _channelIdTips:
        return 'Style Tips';
      default:
        return 'General';
    }
  }

  // ============== PUBLIC NOTIFICATION METHODS ==============

  /// Show outfit recommendation notification
  Future<void> showOutfitRecommendation({
    required String title,
    required String body,
    Map<String, dynamic>? data,
  }) async {
    await _showLocalNotification(
      title: title,
      body: body,
      payload: data != null ? jsonEncode(data) : null,
      type: 'outfit',
    );
  }

  /// Show weather alert notification
  Future<void> showWeatherAlert({
    required String title,
    required String body,
    Map<String, dynamic>? data,
  }) async {
    await _showLocalNotification(
      title: title,
      body: body,
      payload: data != null ? jsonEncode(data) : null,
      type: 'weather',
    );
  }

  /// Show social interaction notification
  Future<void> showSocialNotification({
    required String title,
    required String body,
    Map<String, dynamic>? data,
  }) async {
    await _showLocalNotification(
      title: title,
      body: body,
      payload: data != null ? jsonEncode(data) : null,
      type: 'social',
    );
  }

  /// Show style tip notification
  Future<void> showStyleTip({
    required String title,
    required String body,
    Map<String, dynamic>? data,
  }) async {
    await _showLocalNotification(
      title: title,
      body: body,
      payload: data != null ? jsonEncode(data) : null,
      type: 'tips',
    );
  }

  /// Schedule a notification
  Future<void> scheduleNotification({
    required int id,
    required String title,
    required String body,
    required DateTime scheduledTime,
    String? payload,
    String type = 'outfit',
  }) async {
    final channelId = _getChannelForType(type);

    final androidDetails = AndroidNotificationDetails(
      channelId,
      _getChannelName(channelId),
      importance: Importance.high,
      priority: Priority.high,
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    final details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _localNotifications.zonedSchedule(
      id,
      title,
      body,
      scheduledTime.toUtc().toLocal() as dynamic,
      details,
      androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
      payload: payload,
    );
  }

  /// Cancel a scheduled notification
  Future<void> cancelNotification(int id) async {
    await _localNotifications.cancel(id);
  }

  /// Cancel all notifications
  Future<void> cancelAllNotifications() async {
    await _localNotifications.cancelAll();
  }

  /// Check if notifications are enabled
  Future<bool> areNotificationsEnabled() async {
    final status = await Permission.notification.status;
    return status.isGranted;
  }

  /// Dispose resources
  void dispose() {
    _notificationStreamController.close();
    _feedbackChannel?.unsubscribe();
    _friendRequestChannel?.unsubscribe();
  }
}
