import 'package:dio/dio.dart';

/// Weather service using Open-Meteo API
/// Free weather API with accurate real-time data
class WeatherService {
  static WeatherService? _instance;
  static WeatherService get instance => _instance ??= WeatherService._();
  
  static const String _baseUrl = 'https://api.open-meteo.com/v1';
  final Dio _dio;

  WeatherService._()
    : _dio = Dio(
        BaseOptions(
          baseUrl: _baseUrl,
          connectTimeout: const Duration(seconds: 10),
          receiveTimeout: const Duration(seconds: 10),
        ),
      );

  /// Get weather for default location (San Francisco)
  Future<Map<String, dynamic>?> getWeather() async {
    try {
      final data = await getSanFranciscoWeather();
      return data.toJson();
    } catch (e) {
      return null;
    }
  }

  /// Fetch weather data for a specific location
  /// Returns temperature, condition, humidity, and wind speed
  Future<WeatherData> getWeatherForLocation({
    required double latitude,
    required double longitude,
  }) async {
    try {
      final response = await _dio.get(
        '/forecast',
        queryParameters: {
          'latitude': latitude,
          'longitude': longitude,
          'current':
              'temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code',
          'temperature_unit': 'fahrenheit',
          'wind_speed_unit': 'mph',
        },
      );

      final current = response.data['current'];
      final weatherCode = current['weather_code'] as int;

      return WeatherData(
        temperature: (current['temperature_2m'] as num).toDouble(),
        condition: _weatherCodeToCondition(weatherCode),
        humidity: current['relative_humidity_2m'] as int,
        windSpeed: (current['wind_speed_10m'] as num).toDouble(),
      );
    } on DioException catch (e) {
      throw WeatherException(
        'Failed to fetch weather data: ${e.message ?? 'Unknown error'}',
      );
    }
  }

  /// Fetch weather for San Francisco (default location)
  Future<WeatherData> getSanFranciscoWeather() async {
    return getWeatherForLocation(latitude: 37.7749, longitude: -122.4194);
  }

  /// Convert WMO weather codes to human-readable conditions
  String _weatherCodeToCondition(int code) {
    // WMO Weather interpretation codes
    if (code == 0) return 'Sunny';
    if (code <= 3) return 'Partly Cloudy';
    if (code <= 48) return 'Cloudy';
    if (code <= 67) return 'Rainy';
    if (code <= 77) return 'Snowy';
    if (code <= 99) return 'Stormy';
    return 'Unknown';
  }
}

/// Weather data model
class WeatherData {
  final double temperature;
  final String condition;
  final int humidity;
  final double windSpeed;

  WeatherData({
    required this.temperature,
    required this.condition,
    required this.humidity,
    required this.windSpeed,
  });

  Map<String, dynamic> toJson() {
    return {
      'temperature': temperature.round(),
      'condition': condition,
      'humidity': humidity,
      'windSpeed': windSpeed.round(),
    };
  }
}

/// Weather exception
class WeatherException implements Exception {
  final String message;

  WeatherException(this.message);

  @override
  String toString() => 'WeatherException: $message';
}
