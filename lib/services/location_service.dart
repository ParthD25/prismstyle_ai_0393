import 'package:flutter/foundation.dart';
import 'package:geolocator/geolocator.dart';
import 'package:geocoding/geocoding.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Location service for PrismStyle AI
/// Provides GPS coordinates, city name conversion, and location caching
/// Source: geolocator package - https://pub.dev/packages/geolocator
class LocationService {
  static LocationService? _instance;
  static LocationService get instance => _instance ??= LocationService._();
  
  LocationService._();
  
  // Cache keys
  static const String _latitudeKey = 'cached_latitude';
  static const String _longitudeKey = 'cached_longitude';
  static const String _cityKey = 'cached_city';
  static const String _countryKey = 'cached_country';
  static const String _lastUpdateKey = 'location_last_update';
  
  // Cache duration: 30 minutes
  static const Duration _cacheDuration = Duration(minutes: 30);
  
  // ignore: unused_field - Reserved for position caching
  Position? _cachedPosition;
  LocationData? _cachedLocationData;
  bool _isInitialized = false;
  
  /// Initialize the location service
  Future<void> initialize() async {
    if (_isInitialized) return;
    await checkLocationPermission();
    _isInitialized = true;
  }
  
  /// Check if location services are enabled and permissions granted
  Future<bool> checkLocationPermission() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      debugPrint('Location services are disabled');
      return false;
    }
    
    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        debugPrint('Location permissions are denied');
        return false;
      }
    }
    
    if (permission == LocationPermission.deniedForever) {
      debugPrint('Location permissions are permanently denied');
      return false;
    }
    
    return true;
  }
  
  /// Request location permission
  Future<PermissionStatus> requestLocationPermission() async {
    final status = await Permission.locationWhenInUse.request();
    return status;
  }
  
  /// Get current position with high accuracy
  Future<Position?> getCurrentPosition() async {
    try {
      if (!await checkLocationPermission()) {
        return _getCachedPosition();
      }
      
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
        timeLimit: const Duration(seconds: 15),
      );
      
      _cachedPosition = position;
      await _cachePosition(position);
      
      return position;
    } catch (e) {
      debugPrint('Error getting current position: $e');
      return _getCachedPosition();
    }
  }
  
  /// Get location data with city name and country
  Future<LocationData?> getLocationData() async {
    try {
      // Check cache first
      if (_cachedLocationData != null && !await _isCacheExpired()) {
        return _cachedLocationData;
      }
      
      final position = await getCurrentPosition();
      if (position == null) {
        return _getCachedLocationData();
      }
      
      // Convert coordinates to address (reverse geocoding)
      final placemarks = await placemarkFromCoordinates(
        position.latitude,
        position.longitude,
      );
      
      if (placemarks.isNotEmpty) {
        final placemark = placemarks.first;
        final locationData = LocationData(
          latitude: position.latitude,
          longitude: position.longitude,
          city: placemark.locality ?? placemark.subAdministrativeArea ?? 'Unknown',
          country: placemark.country ?? 'Unknown',
          state: placemark.administrativeArea ?? '',
          postalCode: placemark.postalCode ?? '',
        );
        
        _cachedLocationData = locationData;
        await _cacheLocationData(locationData);
        
        return locationData;
      }
      
      return LocationData(
        latitude: position.latitude,
        longitude: position.longitude,
        city: 'Unknown',
        country: 'Unknown',
      );
    } catch (e) {
      debugPrint('Error getting location data: $e');
      return _getCachedLocationData();
    }
  }
  
  /// Get coordinates from city name (forward geocoding)
  Future<LocationData?> getLocationFromCity(String cityName) async {
    try {
      final locations = await locationFromAddress(cityName);
      if (locations.isNotEmpty) {
        final location = locations.first;
        return LocationData(
          latitude: location.latitude,
          longitude: location.longitude,
          city: cityName,
          country: '',
        );
      }
      return null;
    } catch (e) {
      debugPrint('Error geocoding city name: $e');
      return null;
    }
  }
  
  /// Calculate distance between two points in kilometers
  double calculateDistance({
    required double startLat,
    required double startLng,
    required double endLat,
    required double endLng,
  }) {
    return Geolocator.distanceBetween(
      startLat,
      startLng,
      endLat,
      endLng,
    ) / 1000; // Convert meters to kilometers
  }
  
  /// Listen to location updates
  Stream<Position> getLocationStream({
    LocationAccuracy accuracy = LocationAccuracy.high,
    int distanceFilter = 100, // meters
  }) {
    return Geolocator.getPositionStream(
      locationSettings: LocationSettings(
        accuracy: accuracy,
        distanceFilter: distanceFilter,
      ),
    );
  }
  
  // Private caching methods
  
  Future<void> _cachePosition(Position position) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble(_latitudeKey, position.latitude);
    await prefs.setDouble(_longitudeKey, position.longitude);
    await prefs.setInt(_lastUpdateKey, DateTime.now().millisecondsSinceEpoch);
  }
  
  Future<void> _cacheLocationData(LocationData data) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble(_latitudeKey, data.latitude);
    await prefs.setDouble(_longitudeKey, data.longitude);
    await prefs.setString(_cityKey, data.city);
    await prefs.setString(_countryKey, data.country);
    await prefs.setInt(_lastUpdateKey, DateTime.now().millisecondsSinceEpoch);
  }
  
  Future<Position?> _getCachedPosition() async {
    final prefs = await SharedPreferences.getInstance();
    final lat = prefs.getDouble(_latitudeKey);
    final lng = prefs.getDouble(_longitudeKey);
    
    if (lat != null && lng != null) {
      return Position(
        latitude: lat,
        longitude: lng,
        timestamp: DateTime.now(),
        accuracy: 0,
        altitude: 0,
        altitudeAccuracy: 0,
        heading: 0,
        headingAccuracy: 0,
        speed: 0,
        speedAccuracy: 0,
      );
    }
    return null;
  }
  
  Future<LocationData?> _getCachedLocationData() async {
    final prefs = await SharedPreferences.getInstance();
    final lat = prefs.getDouble(_latitudeKey);
    final lng = prefs.getDouble(_longitudeKey);
    final city = prefs.getString(_cityKey);
    final country = prefs.getString(_countryKey);
    
    if (lat != null && lng != null) {
      return LocationData(
        latitude: lat,
        longitude: lng,
        city: city ?? 'Unknown',
        country: country ?? 'Unknown',
      );
    }
    return null;
  }
  
  Future<bool> _isCacheExpired() async {
    final prefs = await SharedPreferences.getInstance();
    final lastUpdate = prefs.getInt(_lastUpdateKey);
    
    if (lastUpdate == null) return true;
    
    final lastUpdateTime = DateTime.fromMillisecondsSinceEpoch(lastUpdate);
    return DateTime.now().difference(lastUpdateTime) > _cacheDuration;
  }
  
  /// Clear cached location data
  Future<void> clearCache() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_latitudeKey);
    await prefs.remove(_longitudeKey);
    await prefs.remove(_cityKey);
    await prefs.remove(_countryKey);
    await prefs.remove(_lastUpdateKey);
    _cachedPosition = null;
    _cachedLocationData = null;
  }
  
  /// Default location (San Francisco) for when location is unavailable
  LocationData get defaultLocation => LocationData(
    latitude: 37.7749,
    longitude: -122.4194,
    city: 'San Francisco',
    country: 'United States',
    state: 'California',
  );
}

/// Location data model with city name and coordinates
class LocationData {
  final double latitude;
  final double longitude;
  final String city;
  final String country;
  final String state;
  final String postalCode;
  
  LocationData({
    required this.latitude,
    required this.longitude,
    required this.city,
    required this.country,
    this.state = '',
    this.postalCode = '',
  });
  
  String get displayName {
    if (state.isNotEmpty) {
      return '$city, $state';
    }
    return '$city, $country';
  }
  
  Map<String, dynamic> toJson() {
    return {
      'latitude': latitude,
      'longitude': longitude,
      'city': city,
      'country': country,
      'state': state,
      'postalCode': postalCode,
    };
  }
  
  factory LocationData.fromJson(Map<String, dynamic> json) {
    return LocationData(
      latitude: json['latitude'] as double,
      longitude: json['longitude'] as double,
      city: json['city'] as String,
      country: json['country'] as String,
      state: json['state'] as String? ?? '',
      postalCode: json['postalCode'] as String? ?? '',
    );
  }
}
