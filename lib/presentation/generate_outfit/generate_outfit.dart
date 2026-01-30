import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:sizer/sizer.dart';
import 'package:image_picker/image_picker.dart';

import '../../core/app_export.dart';
import '../../widgets/custom_icon_widget.dart';
import '../../services/openai_service.dart';
import '../../services/openai_client.dart';
import '../../services/weather_service.dart';

/// Generate Outfit screen - AI-powered outfit generation with style preferences
/// Replaces camera tab with comprehensive outfit analysis and recommendations
class GenerateOutfit extends StatefulWidget {
  const GenerateOutfit({super.key});

  @override
  State<GenerateOutfit> createState() => _GenerateOutfitState();
}

class _GenerateOutfitState extends State<GenerateOutfit> {
  late OpenAIClient _aiClient;
  late WeatherService _weatherService;
  final ImagePicker _picker = ImagePicker();

  // Preference selections
  String? _selectedTimeOfDay;
  String? _selectedStyle;
  String? _selectedLocation;
  String? _selectedOccasion;
  String? _selectedCity;

  // Photo selections
  List<XFile> _selectedPhotos = [];
  bool _isAnalyzing = false;
  bool _isLoadingWeather = false;
  OutfitAnalysis? _analysis;
  Map<String, dynamic>? _weatherData;

  final List<String> _timesOfDay = [
    'Morning',
    'Afternoon',
    'Evening',
    'Night',
    'All Day',
  ];

  final List<String> _styles = [
    'Business',
    'Casual',
    'Date',
    'Club',
    'Dinner',
    'Interview',
    'Meeting',
    'Beach',
    'Athletic',
    'Formal',
  ];

  final List<String> _locations = [
    'Indoor',
    'Outdoor',
    'Partial Day Indoor',
    'Partial Day Outdoor',
  ];

  final List<String> _occasions = [
    'Everyday',
    'Special Event',
    'Work',
    'Social',
    'Recreation',
  ];

  // Add city selection list
  final List<Map<String, dynamic>> _cities = [
    {'name': 'San Francisco', 'lat': 37.7749, 'lon': -122.4194},
    {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
    {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
    {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298},
    {'name': 'Miami', 'lat': 25.7617, 'lon': -80.1918},
  ];

  @override
  void initState() {
    super.initState();
    try {
      _aiClient = OpenAIClient(OpenAIService().dio);
    } catch (e) {
      debugPrint('Failed to initialize OpenAI client: $e');
      // Create a fallback Dio instance
      _aiClient = OpenAIClient(Dio());
    }
    _weatherService = WeatherService.instance;
  }

  Future<void> _fetchWeatherForCity(String cityName) async {
    setState(() => _isLoadingWeather = true);

    try {
      final city = _cities.firstWhere((c) => c['name'] == cityName);
      final weather = await _weatherService.getWeatherForLocation(
        latitude: city['lat'],
        longitude: city['lon'],
      );

      setState(() {
        _weatherData = weather.toJson();
        _isLoadingWeather = false;
      });
    } catch (e) {
      setState(() => _isLoadingWeather = false);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to fetch weather: $e'),
            duration: const Duration(seconds: 2),
          ),
        );
      }
    }
  }

  Future<void> _pickPhotos() async {
    try {
      final photos = await _picker.pickMultiImage(
        maxWidth: 1800,
        maxHeight: 1800,
        imageQuality: 85,
      );

      if (photos.isNotEmpty) {
        setState(() {
          _selectedPhotos.addAll(photos);
          if (_selectedPhotos.length > 5) {
            _selectedPhotos = _selectedPhotos.sublist(0, 5);
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text(
                  'Maximum 5 photos allowed. Only first 5 selected.',
                ),
                duration: Duration(seconds: 2),
              ),
            );
          }
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to pick photos: $e'),
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  Future<void> _takePhoto() async {
    try {
      final photo = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1800,
        maxHeight: 1800,
        imageQuality: 85,
      );

      if (photo != null) {
        setState(() {
          _selectedPhotos.add(photo);
          if (_selectedPhotos.length > 3) {
            _selectedPhotos.removeLast();
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Maximum 3 camera photos allowed'),
                duration: Duration(seconds: 2),
              ),
            );
          }
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to take photo: $e'),
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  Future<void> _analyzeOutfit() async {
    if (OpenAIService.apiKey.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'AI analysis is disabled. Set OPENAI_API_KEY via --dart-define.',
          ),
          duration: Duration(seconds: 3),
        ),
      );
      return;
    }

    if (_selectedPhotos.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select at least one photo'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }

    if (_selectedTimeOfDay == null || _selectedStyle == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select time of day and style'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }

    setState(() => _isAnalyzing = true);

    try {
      // For demo, use first photo
      // In production, would upload to storage and get URL
      final demoImageUrl =
          'https://images.unsplash.com/photo-1618397351187-ee6afd732119';

      final weatherCondition = _weatherData?['condition'] ?? 'moderate';

      final analysis = await _aiClient.analyzeOutfit(
        imageUrl: demoImageUrl,
        context: 'User wardrobe photo for outfit analysis',
        occasion: _selectedOccasion ?? 'casual',
        timeOfDay: _selectedTimeOfDay!.toLowerCase().replaceAll(' ', '_'),
        weather: weatherCondition,
      );

      setState(() => _analysis = analysis);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Analysis failed: $e'),
          duration: const Duration(seconds: 3),
        ),
      );
    } finally {
      setState(() => _isAnalyzing = false);
    }
  }

  void _clearSelections() {
    setState(() {
      _selectedPhotos.clear();
      _analysis = null;
      _selectedTimeOfDay = null;
      _selectedStyle = null;
      _selectedLocation = null;
      _selectedOccasion = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Column(
      children: [
        // Custom app bar
        Container(
          padding: EdgeInsets.only(
            top: MediaQuery.of(context).padding.top,
            left: 4.w,
            right: 4.w,
            bottom: 2.h,
          ),
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            border: Border(
              bottom: BorderSide(
                color: theme.colorScheme.outline.withValues(alpha: 0.2),
                width: 1,
              ),
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Generate Outfit',
                style: theme.textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              if (_selectedPhotos.isNotEmpty || _analysis != null)
                TextButton(
                  onPressed: _clearSelections,
                  child: const Text('Clear'),
                ),
            ],
          ),
        ),

        // Scrollable content
        Expanded(
          child: SingleChildScrollView(
            padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Instructions
                Container(
                  padding: EdgeInsets.all(4.w),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.primaryContainer.withValues(
                      alpha: 0.3,
                    ),
                    borderRadius: BorderRadius.circular(12.0),
                  ),
                  child: Row(
                    children: [
                      CustomIconWidget(
                        iconName: 'info',
                        color: theme.colorScheme.primary,
                        size: 24,
                      ),
                      SizedBox(width: 3.w),
                      Expanded(
                        child: Text(
                          'Select your style preferences and upload photos to get AI-powered outfit recommendations',
                          style: theme.textTheme.bodyMedium,
                        ),
                      ),
                    ],
                  ),
                ),

                SizedBox(height: 3.h),

                // City selection for weather
                _buildPreferenceSection(
                  'Select City for Weather',
                  _cities.map((c) => c['name'] as String).toList(),
                  _selectedCity,
                  (value) {
                    setState(() => _selectedCity = value);
                    if (value != null) {
                      _fetchWeatherForCity(value);
                    }
                  },
                ),

                // Weather display
                if (_weatherData != null) ...[
                  SizedBox(height: 2.h),
                  Container(
                    padding: EdgeInsets.all(3.w),
                    decoration: BoxDecoration(
                      color: theme.colorScheme.primaryContainer.withValues(
                        alpha: 0.3,
                      ),
                      borderRadius: BorderRadius.circular(12.0),
                    ),
                    child: Row(
                      children: [
                        CustomIconWidget(
                          iconName: 'thermostat',
                          color: theme.colorScheme.primary,
                          size: 24,
                        ),
                        SizedBox(width: 2.w),
                        Text(
                          '${_weatherData!['temperature']}°F - ${_weatherData!['condition']}',
                          style: theme.textTheme.titleMedium,
                        ),
                      ],
                    ),
                  ),
                ] else if (_isLoadingWeather) ...[
                  SizedBox(height: 2.h),
                  Center(child: CircularProgressIndicator()),
                ],

                SizedBox(height: 2.h),

                // Time of Day selection
                _buildPreferenceSection(
                  'Time of Day',
                  _timesOfDay,
                  _selectedTimeOfDay,
                  (value) => setState(() => _selectedTimeOfDay = value),
                ),

                SizedBox(height: 2.h),

                // Style selection
                _buildPreferenceSection(
                  'Style',
                  _styles,
                  _selectedStyle,
                  (value) => setState(() => _selectedStyle = value),
                ),

                SizedBox(height: 2.h),

                // Location selection
                _buildPreferenceSection(
                  'Location',
                  _locations,
                  _selectedLocation,
                  (value) => setState(() => _selectedLocation = value),
                ),

                SizedBox(height: 2.h),

                // Occasion selection
                _buildPreferenceSection(
                  'Occasion',
                  _occasions,
                  _selectedOccasion,
                  (value) => setState(() => _selectedOccasion = value),
                ),

                SizedBox(height: 3.h),

                // Photo selection section
                Text(
                  'Upload Photos',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),

                SizedBox(height: 2.h),

                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: _pickPhotos,
                        icon: CustomIconWidget(
                          iconName: 'photo_library',
                          color: theme.colorScheme.primary,
                          size: 20,
                        ),
                        label: Text('Gallery (max 5)'),
                        style: OutlinedButton.styleFrom(
                          padding: EdgeInsets.symmetric(vertical: 2.h),
                        ),
                      ),
                    ),
                    SizedBox(width: 3.w),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: _takePhoto,
                        icon: CustomIconWidget(
                          iconName: 'photo_camera',
                          color: theme.colorScheme.primary,
                          size: 20,
                        ),
                        label: Text('Camera (max 3)'),
                        style: OutlinedButton.styleFrom(
                          padding: EdgeInsets.symmetric(vertical: 2.h),
                        ),
                      ),
                    ),
                  ],
                ),

                if (_selectedPhotos.isNotEmpty) ...[
                  SizedBox(height: 2.h),
                  SizedBox(
                    height: 15.h,
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      itemCount: _selectedPhotos.length,
                      itemBuilder: (context, index) {
                        return Container(
                          margin: EdgeInsets.only(right: 2.w),
                          width: 15.h,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(12.0),
                            border: Border.all(
                              color: theme.colorScheme.outline.withValues(
                                alpha: 0.3,
                              ),
                            ),
                          ),
                          child: Stack(
                            children: [
                              ClipRRect(
                                borderRadius: BorderRadius.circular(12.0),
                                child: Image.network(
                                  'https://images.unsplash.com/photo-1618397351187-ee6afd732119',
                                  width: 15.h,
                                  height: 15.h,
                                  fit: BoxFit.cover,
                                ),
                              ),
                              Positioned(
                                top: 1.w,
                                right: 1.w,
                                child: IconButton(
                                  icon: Container(
                                    padding: EdgeInsets.all(1.w),
                                    decoration: BoxDecoration(
                                      color: Colors.black.withValues(
                                        alpha: 0.5,
                                      ),
                                      shape: BoxShape.circle,
                                    ),
                                    child: const Icon(
                                      Icons.close,
                                      color: Colors.white,
                                      size: 16,
                                    ),
                                  ),
                                  onPressed: () {
                                    setState(() {
                                      _selectedPhotos.removeAt(index);
                                    });
                                  },
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                ],

                SizedBox(height: 4.h),

                // Analyze button
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: _isAnalyzing ? null : _analyzeOutfit,
                    icon: _isAnalyzing
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : CustomIconWidget(
                            iconName: 'auto_awesome',
                            color: theme.colorScheme.onPrimary,
                            size: 20,
                          ),
                    label: Text(
                      _isAnalyzing ? 'Analyzing...' : 'Analyze with AI',
                      style: theme.textTheme.labelLarge?.copyWith(
                        color: theme.colorScheme.onPrimary,
                      ),
                    ),
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(vertical: 2.5.h),
                      backgroundColor: theme.colorScheme.primary,
                    ),
                  ),
                ),

                // Analysis results
                if (_analysis != null) ...[
                  SizedBox(height: 4.h),
                  Container(
                    padding: EdgeInsets.all(4.w),
                    decoration: BoxDecoration(
                      color: theme.colorScheme.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(16.0),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            CustomIconWidget(
                              iconName: 'auto_awesome',
                              color: theme.colorScheme.primary,
                              size: 28,
                            ),
                            SizedBox(width: 3.w),
                            Text(
                              'AI Analysis',
                              style: theme.textTheme.titleLarge?.copyWith(
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),

                        SizedBox(height: 2.h),

                        // Style score
                        Container(
                          padding: EdgeInsets.all(3.w),
                          decoration: BoxDecoration(
                            color: _analysis!.styleScore >= 7
                                ? theme.colorScheme.primaryContainer
                                : theme.colorScheme.errorContainer,
                            borderRadius: BorderRadius.circular(12.0),
                          ),
                          child: Row(
                            children: [
                              Icon(
                                _analysis!.worksWell
                                    ? Icons.check_circle
                                    : Icons.error,
                                color: _analysis!.styleScore >= 7
                                    ? theme.colorScheme.primary
                                    : theme.colorScheme.error,
                              ),
                              SizedBox(width: 2.w),
                              Expanded(
                                child: Text(
                                  'Style Score: ${_analysis!.styleScore}/10',
                                  style: theme.textTheme.titleMedium?.copyWith(
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),

                        SizedBox(height: 2.h),

                        // Feedback
                        Text(
                          'Feedback',
                          style: theme.textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        SizedBox(height: 1.h),
                        Text(
                          _analysis!.feedback,
                          style: theme.textTheme.bodyMedium,
                        ),

                        if (_analysis!.suggestions.isNotEmpty) ...[
                          SizedBox(height: 2.h),
                          Text(
                            'Suggestions',
                            style: theme.textTheme.titleSmall?.copyWith(
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          SizedBox(height: 1.h),
                          ..._analysis!.suggestions.map(
                            (suggestion) => Padding(
                              padding: EdgeInsets.only(bottom: 1.h),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Icon(
                                    Icons.arrow_right,
                                    color: theme.colorScheme.primary,
                                    size: 20,
                                  ),
                                  SizedBox(width: 2.w),
                                  Expanded(
                                    child: Text(
                                      suggestion,
                                      style: theme.textTheme.bodyMedium,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ],

                SizedBox(height: 4.h),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildPreferenceSection(
    String title,
    List<String> options,
    String? selectedValue,
    Function(String?) onChanged,
  ) {
    final theme = Theme.of(context);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: theme.textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w600,
          ),
        ),
        SizedBox(height: 1.h),
        Wrap(
          spacing: 2.w,
          runSpacing: 1.h,
          children: options.map((option) {
            final isSelected = selectedValue == option;
            return ChoiceChip(
              label: Text(option),
              selected: isSelected,
              onSelected: (selected) => onChanged(selected ? option : null),
              selectedColor: theme.colorScheme.primaryContainer,
              backgroundColor: theme.colorScheme.surface,
              labelStyle: theme.textTheme.bodySmall?.copyWith(
                color: isSelected
                    ? theme.colorScheme.onPrimaryContainer
                    : theme.colorScheme.onSurface,
              ),
            );
          }).toList(),
        ),
      ],
    );
  }
}
