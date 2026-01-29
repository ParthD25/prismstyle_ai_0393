import 'package:dio/dio.dart';
import 'dart:convert';

/// OpenAI Client for fashion recommendations and outfit analysis
/// Uses GPT-5 models for optimal performance
class OpenAIClient {
  final Dio dio;

  OpenAIClient(this.dio);

  /// Analyze outfit and provide fashion recommendations
  /// Uses GPT-5 with minimal reasoning for fast responses
  Future<OutfitAnalysis> analyzeOutfit({
    required String imageUrl,
    required String context,
    String occasion = 'casual',
    String timeOfDay = 'all_day',
    String weather = 'moderate',
  }) async {
    try {
      final List<Map<String, dynamic>> content = [
        {
          'type': 'text',
          'text':
              '''You are a professional fashion stylist. Analyze this outfit and provide helpful feedback. Context: \$context Occasion: \$occasion Time: \$timeOfDay Weather: \$weather Provide: 1. Overall assessment (works/doesn't work) 2. Style score (1-10) 3. Specific suggestions for improvement 4. Color harmony analysis 5. Fit and proportion feedback''',
        },
        {
          'type': 'image_url',
          'image_url': {'url': imageUrl},
        },
      ];

      final requestData = {
        'model': 'gpt-5-mini',
        'messages': [
          {'role': 'user', 'content': content},
        ],
        'reasoning_effort': 'minimal',
        'verbosity': 'medium',
        'max_completion_tokens': 500,
      };

      final response = await dio.post('/chat/completions', data: requestData);
      final text = response.data['choices'][0]['message']['content'];

      return OutfitAnalysis.fromText(text);
    } on DioException catch (e) {
      throw OpenAIException(
        statusCode: e.response?.statusCode ?? 500,
        message:
            e.response?.data['error']['message'] ??
            e.message ??
            'Unknown error',
      );
    }
  }

  /// Generate outfit recommendations based on preferences
  Stream<String> generateOutfitSuggestions({
    required List<String> wardrobeItems,
    required String occasion,
    required String timeOfDay,
    required String weather,
    String? location,
  }) async* {
    try {
      final requestData = {
        'model': 'gpt-5-mini',
        'messages': [
          {
            'role': 'user',
            'content':
                '''Create outfit recommendations from these wardrobe items:
${wardrobeItems.join(', ')}

Occasion: $occasion
Time: $timeOfDay
Weather: $weather${location != null ? '\nLocation: $location' : ''}

Suggest 3 complete outfit combinations with reasoning.''',
          },
        ],
        'stream': true,
        'reasoning_effort': 'low',
        'verbosity': 'medium',
        'max_completion_tokens': 800,
      };

      final response = await dio.post(
        '/chat/completions',
        data: requestData,
        options: Options(responseType: ResponseType.stream),
      );

      final stream = response.data.stream;
      await for (var line in LineSplitter().bind(
        utf8.decoder.bind(stream.stream),
      )) {
        if (line.startsWith('data: ')) {
          final data = line.substring(6);
          if (data == '[DONE]') break;

          final json = jsonDecode(data) as Map<String, dynamic>;
          final delta = json['choices'][0]['delta'] as Map<String, dynamic>;
          final content = delta['content'] ?? '';

          if (content.isNotEmpty) {
            yield content;
          }

          final finishReason = json['choices'][0]['finish_reason'];
          if (finishReason != null) break;
        }
      }
    } on DioException catch (e) {
      throw OpenAIException(
        statusCode: e.response?.statusCode ?? 500,
        message:
            e.response?.data['error']['message'] ??
            e.message ??
            'Unknown error',
      );
    }
  }
}

/// Outfit analysis result model
class OutfitAnalysis {
  final bool worksWell;
  final int styleScore;
  final String feedback;
  final List<String> suggestions;

  OutfitAnalysis({
    required this.worksWell,
    required this.styleScore,
    required this.feedback,
    required this.suggestions,
  });

  factory OutfitAnalysis.fromText(String text) {
    // Parse AI response text
    final worksWell =
        text.toLowerCase().contains('works') ||
        text.toLowerCase().contains('good') ||
        text.toLowerCase().contains('appropriate');

    // Extract score if present
    final scoreMatch = RegExp(r'(\d+)(?:/10|\s*out of 10)').firstMatch(text);
    final styleScore = scoreMatch != null ? int.parse(scoreMatch.group(1)!) : 7;

    // Extract suggestions
    final suggestions = <String>[];
    final lines = text.split('\n');
    for (var line in lines) {
      if (line.trim().startsWith('-') ||
          line.trim().startsWith('•') ||
          RegExp(r'^\d+\.').hasMatch(line.trim())) {
        suggestions.add(
          line.trim().replaceFirst(RegExp(r'^[-•\d+\.]'), '').trim(),
        );
      }
    }

    return OutfitAnalysis(
      worksWell: worksWell,
      styleScore: styleScore,
      feedback: text,
      suggestions: suggestions.take(5).toList(),
    );
  }
}

/// OpenAI exception model
class OpenAIException implements Exception {
  final int statusCode;
  final String message;

  OpenAIException({required this.statusCode, required this.message});

  @override
  String toString() => 'OpenAIException: $statusCode - $message';
}
