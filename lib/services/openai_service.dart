import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

/// OpenAI Service for AI-powered outfit recommendations and analysis
/// Singleton pattern for managing OpenAI API interactions
class OpenAIService {
  static final OpenAIService _instance = OpenAIService._internal();
  late final Dio _dio;
  static const String apiKey = String.fromEnvironment('OPENAI_API_KEY');

  factory OpenAIService() {
    return _instance;
  }

  OpenAIService._internal() {
    _initializeService();
  }

  void _initializeService() {
    final headers = <String, dynamic>{
      'Content-Type': 'application/json',
    };
    if (apiKey.isNotEmpty) {
      headers['Authorization'] = 'Bearer $apiKey';
    } else {
      debugPrint('⚠️ OPENAI_API_KEY not provided; OpenAI features disabled.');
    }

    _dio = Dio(
      BaseOptions(
        baseUrl: 'https://api.openai.com/v1',
        headers: headers,
      ),
    );
  }

  Dio get dio => _dio;
}
