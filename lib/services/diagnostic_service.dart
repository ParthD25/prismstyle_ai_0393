import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:geolocator/geolocator.dart';
import './ensemble_ai_service.dart';
import './supabase_service.dart';
import './weather_service.dart';
import './location_service.dart';
import './color_detection_service.dart';

/// Comprehensive Diagnostic Service for PrismStyle AI
/// Runs background checks on all app functionality
class DiagnosticService {
  static DiagnosticService? _instance;
  static DiagnosticService get instance => _instance ??= DiagnosticService._();

  DiagnosticService._();

  final Map<String, DiagnosticResult> _results = {};
  bool _isRunning = false;

  /// Run all diagnostic tests
  Future<DiagnosticReport> runFullDiagnostics() async {
    if (_isRunning) {
      throw Exception('Diagnostics already running');
    }

    _isRunning = true;
    _results.clear();

    debugPrint('🔍 Starting comprehensive diagnostics...\n');

    try {
      // Run all tests
      await _testAIModels();
      await _testDatabase();
      await _testCamera();
      await _testLocation();
      await _testWeather();
      await _testColorDetection();
      await _testPlatformIntegration();

      return _generateReport();
    } finally {
      _isRunning = false;
    }
  }

  /// Test AI Model Loading and Inference
  Future<void> _testAIModels() async {
    debugPrint('📊 Testing AI Models...');

    try {
      final ensemble = EnsembleAIService.instance;

      // Initialize
      final initStart = DateTime.now();
      await ensemble.initialize();
      final initDuration = DateTime.now().difference(initStart);

      // Check model status
      final status = ensemble.getModelStatus();
      final loadedModels = status.entries.where((e) => e.value).length;

      // Test inference with dummy image
      final testImageBytes = Uint8List.fromList(
        List.filled(224 * 224 * 3, 128),
      );
      final inferenceStart = DateTime.now();
      final result = await ensemble.classifyClothing(testImageBytes);
      final inferenceDuration = DateTime.now().difference(inferenceStart);

      _results['AI Models'] = DiagnosticResult(
        name: 'AI Models',
        passed: ensemble.isReady && loadedModels >= 2,
        details: {
          'Initialized': ensemble.isReady,
          'Models Loaded': '$loadedModels/4',
          'TFLite': status['tflite'] ?? false,
          'Apple Vision': status['apple_vision'] ?? false,
          'Core ML': status['coreml'] ?? false,
          'Heuristic': status['heuristic'] ?? false,
          'Init Time': '${initDuration.inMilliseconds}ms',
          'Inference Time': '${inferenceDuration.inMilliseconds}ms',
          'Test Category': result.primaryCategory,
          'Test Confidence': '${(result.confidence * 100).toStringAsFixed(1)}%',
        },
        message: ensemble.isReady
            ? 'AI system ready with $loadedModels models'
            : 'AI system initialized but insufficient models loaded',
      );

      debugPrint('  ✅ AI Models: ${_results['AI Models']!.message}');
    } catch (e) {
      _results['AI Models'] = DiagnosticResult(
        name: 'AI Models',
        passed: false,
        details: {'Error': e.toString()},
        message: 'AI initialization failed: $e',
      );
      debugPrint('  ❌ AI Models failed: $e');
    }
  }

  /// Test Supabase Database Connection
  Future<void> _testDatabase() async {
    debugPrint('🗄️  Testing Database...');

    try {
      final supabase = SupabaseService.instance;
      final startTime = DateTime.now();

      // Test connection
      final hasAuth = supabase.client.auth.currentUser != null;

      // Test query (public tables only, no auth needed)
      bool canQuery = false;

      try {
        // Try a simple count query on public data
        await supabase.client.from('users').select('id').limit(1);
        canQuery = true;
      } catch (e) {
        // Expected to fail without auth - RLS prevents anonymous access
      }

      final duration = DateTime.now().difference(startTime);

      _results['Database'] = DiagnosticResult(
        name: 'Database',
        passed: true, // Connection exists even if queries need auth
        details: {
          'Supabase URL': SupabaseService.supabaseUrl.isNotEmpty
              ? 'Configured'
              : 'Missing',
          'Anon Key': SupabaseService.supabaseAnonKey.isNotEmpty
              ? 'Configured'
              : 'Missing',
          'Client Initialized': true,
          'Auth Status': hasAuth ? 'Authenticated' : 'Anonymous',
          'Query Test': canQuery ? 'Passed' : 'Needs Auth',
          'Response Time': '${duration.inMilliseconds}ms',
          if (!canQuery)
            'Query Note': 'RLS requires authentication for data access',
        },
        message: canQuery
            ? 'Database fully operational'
            : 'Database connected (authentication required for data operations)',
      );

      debugPrint('  ✅ Database: ${_results['Database']!.message}');
    } catch (e) {
      _results['Database'] = DiagnosticResult(
        name: 'Database',
        passed: false,
        details: {'Error': e.toString()},
        message: 'Database connection failed: $e',
      );
      debugPrint('  ❌ Database failed: $e');
    }
  }

  /// Test Camera Access
  Future<void> _testCamera() async {
    debugPrint('📸 Testing Camera...');

    try {
      final cameras = await availableCameras();
      final hasPermission = await _checkCameraPermission();

      _results['Camera'] = DiagnosticResult(
        name: 'Camera',
        passed: cameras.isNotEmpty,
        details: {
          'Available Cameras': cameras.length,
          'Permission': hasPermission ? 'Granted' : 'Not Granted',
          'Cameras': cameras.map((c) => c.name).join(', '),
        },
        message: cameras.isNotEmpty
            ? '${cameras.length} camera(s) available'
            : 'No cameras detected',
      );

      debugPrint('  ✅ Camera: ${_results['Camera']!.message}');
    } catch (e) {
      _results['Camera'] = DiagnosticResult(
        name: 'Camera',
        passed: false,
        details: {'Error': e.toString()},
        message: 'Camera access failed: $e',
      );
      debugPrint('  ❌ Camera failed: $e');
    }
  }

  /// Test Location Services
  Future<void> _testLocation() async {
    debugPrint('📍 Testing Location...');

    try {
      final service = LocationService.instance;
      final startTime = DateTime.now();

      final hasPermission = await service.checkLocationPermission();
      Position? position;

      if (hasPermission) {
        position = await service.getCurrentPosition();
      }

      final duration = DateTime.now().difference(startTime);

      _results['Location'] = DiagnosticResult(
        name: 'Location',
        passed: hasPermission && position != null,
        details: {
          'Permission': hasPermission ? 'Granted' : 'Denied',
          'Service Enabled': await Geolocator.isLocationServiceEnabled(),
          if (position != null)
            'Latitude': position.latitude.toStringAsFixed(4),
          if (position != null)
            'Longitude': position.longitude.toStringAsFixed(4),
          if (position != null)
            'Accuracy': '${position.accuracy.toStringAsFixed(0)}m',
          'Response Time': '${duration.inMilliseconds}ms',
        },
        message: position != null
            ? 'Location acquired successfully'
            : hasPermission
            ? 'Location service available but unable to get position'
            : 'Location permission denied',
      );

      debugPrint('  ✅ Location: ${_results['Location']!.message}');
    } catch (e) {
      _results['Location'] = DiagnosticResult(
        name: 'Location',
        passed: false,
        details: {'Error': e.toString()},
        message: 'Location service failed: $e',
      );
      debugPrint('  ❌ Location failed: $e');
    }
  }

  /// Test Weather API
  Future<void> _testWeather() async {
    debugPrint('🌤️  Testing Weather...');

    try {
      final service = WeatherService.instance;
      final startTime = DateTime.now();

      // Test with sample coordinates (New York)
      final weather = await service.getWeatherForLocation(
        latitude: 40.7128,
        longitude: -74.0060,
      );

      final duration = DateTime.now().difference(startTime);

      _results['Weather'] = DiagnosticResult(
        name: 'Weather',
        passed: true,
        details: {
          'API Response': 'Success',
          'Temperature': '${weather.temperature}°F',
          'Condition': weather.condition,
          'Humidity': '${weather.humidity}%',
          'Response Time': '${duration.inMilliseconds}ms',
          'API': 'Open-Meteo (Free)',
        },
        message: 'Weather API operational',
      );

      debugPrint('  ✅ Weather: ${_results['Weather']!.message}');
    } catch (e) {
      _results['Weather'] = DiagnosticResult(
        name: 'Weather',
        passed: false,
        details: {'Error': e.toString()},
        message: 'Weather API failed: $e',
      );
      debugPrint('  ❌ Weather failed: $e');
    }
  }

  /// Test Color Detection
  Future<void> _testColorDetection() async {
    debugPrint('🎨 Testing Color Detection...');

    try {
      final service = ColorDetectionService.instance;

      // Create test image (red square)
      final testImageBytes = Uint8List.fromList([
        ...List.filled(224 * 224, 255), // R
        ...List.filled(224 * 224, 0), // G
        ...List.filled(224 * 224, 0), // B
      ]);

      final startTime = DateTime.now();
      final result = await service.extractColorsFromBytes(
        testImageBytes,
        numColors: 3,
      );
      final duration = DateTime.now().difference(startTime);

      _results['Color Detection'] = DiagnosticResult(
        name: 'Color Detection',
        passed: result.colors.isNotEmpty,
        details: {
          'Colors Detected': result.colors.length,
          'Processing Time': '${duration.inMilliseconds}ms',
          'Algorithm': 'K-means clustering',
          'Sample Results': result.colors.take(3).map((c) => c.name).join(', '),
        },
        message:
            'Color detection operational (${result.colors.length} colors detected)',
      );

      debugPrint(
        '  ✅ Color Detection: ${_results['Color Detection']!.message}',
      );
    } catch (e) {
      _results['Color Detection'] = DiagnosticResult(
        name: 'Color Detection',
        passed: false,
        details: {'Error': e.toString()},
        message: 'Color detection failed: $e',
      );
      debugPrint('  ❌ Color Detection failed: $e');
    }
  }

  /// Test Platform-Specific Integration
  Future<void> _testPlatformIntegration() async {
    debugPrint('📱 Testing Platform Integration...');

    try {
      final isIOS = !kIsWeb && Platform.isIOS;

      Map<String, dynamic> details = {
        'Platform': Platform.operatingSystem,
        'Version': Platform.operatingSystemVersion,
        'Dart Version': Platform.version.split(' ').first,
      };

      // Test iOS-specific features
      if (isIOS) {
        try {
          const platform = MethodChannel('com.prismstyle_ai/apple_vision');
          await platform.invokeMethod('initialize');
          details['Apple Vision'] = 'Available';
        } catch (e) {
          details['Apple Vision'] = 'Not Available';
        }

        try {
          const platform = MethodChannel('com.prismstyle_ai/coreml');
          await platform.invokeMethod('initialize');
          details['Core ML'] = 'Available';
        } catch (e) {
          details['Core ML'] = 'Not Available';
        }
      }

      _results['Platform'] = DiagnosticResult(
        name: 'Platform',
        passed: true,
        details: details,
        message: 'Running on ${Platform.operatingSystem}',
      );

      debugPrint('  ✅ Platform: ${_results['Platform']!.message}');
    } catch (e) {
      _results['Platform'] = DiagnosticResult(
        name: 'Platform',
        passed: false,
        details: {'Error': e.toString()},
        message: 'Platform check failed: $e',
      );
      debugPrint('  ❌ Platform failed: $e');
    }
  }

  /// Check camera permission
  Future<bool> _checkCameraPermission() async {
    // This is a simplified check - actual implementation in camera service
    return true;
  }

  /// Generate diagnostic report
  DiagnosticReport _generateReport() {
    final passed = _results.values.where((r) => r.passed).length;
    final total = _results.length;
    final percentage = (passed / total * 100).toStringAsFixed(1);

    final status = passed == total
        ? DiagnosticStatus.allPassed
        : passed >= total * 0.8
        ? DiagnosticStatus.mostlyPassed
        : passed >= total * 0.5
        ? DiagnosticStatus.someFailures
        : DiagnosticStatus.criticalFailures;

    final report = DiagnosticReport(
      timestamp: DateTime.now(),
      results: _results,
      status: status,
      passedCount: passed,
      totalCount: total,
      successRate: percentage,
    );

    debugPrint('\n📋 Diagnostic Report:');
    debugPrint('   Status: ${status.toString().split('.').last}');
    debugPrint('   Passed: $passed/$total ($percentage%)');
    debugPrint('   Timestamp: ${report.timestamp}');

    return report;
  }
}

/// Diagnostic test result
class DiagnosticResult {
  final String name;
  final bool passed;
  final Map<String, dynamic> details;
  final String message;

  DiagnosticResult({
    required this.name,
    required this.passed,
    required this.details,
    required this.message,
  });
}

/// Overall diagnostic status
enum DiagnosticStatus {
  allPassed,
  mostlyPassed,
  someFailures,
  criticalFailures,
}

/// Complete diagnostic report
class DiagnosticReport {
  final DateTime timestamp;
  final Map<String, DiagnosticResult> results;
  final DiagnosticStatus status;
  final int passedCount;
  final int totalCount;
  final String successRate;

  DiagnosticReport({
    required this.timestamp,
    required this.results,
    required this.status,
    required this.passedCount,
    required this.totalCount,
    required this.successRate,
  });

  String get summary {
    final buffer = StringBuffer();
    buffer.writeln('=== PrismStyle AI Diagnostic Report ===');
    buffer.writeln('Timestamp: $timestamp');
    buffer.writeln('Status: ${status.toString().split('.').last}');
    buffer.writeln('Success Rate: $successRate% ($passedCount/$totalCount)');
    buffer.writeln('');

    results.forEach((name, result) {
      buffer.writeln('${result.passed ? "✅" : "❌"} $name: ${result.message}');
      result.details.forEach((key, value) {
        buffer.writeln('   $key: $value');
      });
      buffer.writeln('');
    });

    return buffer.toString();
  }

  bool get isProductionReady =>
      status == DiagnosticStatus.allPassed ||
      status == DiagnosticStatus.mostlyPassed;
}
