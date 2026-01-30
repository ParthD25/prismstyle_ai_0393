import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';
import 'package:image_picker/image_picker.dart';

import '../../core/app_export.dart';
import '../../widgets/custom_icon_widget.dart';
import '../../services/openai_service.dart';
import '../../services/openai_client.dart';
import '../../services/weather_service.dart';
import '../../services/location_service.dart';

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
  final LocationService _locationService = LocationService.instance;
  final ImagePicker _picker = ImagePicker();
  final TextEditingController _locationController = TextEditingController();

  // Preference selections
  String? _selectedTimeOfDay;
  String? _selectedStyle;
  String? _selectedLocation;
  String? _selectedOccasion;

  // Location data
  LocationData? _currentLocationData;
  String? _displayLocationName;
  bool _usingCurrentLocation = false;

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

    // Auto-detect location on page load
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _autoDetectLocation();
    });
  }

  /// Auto-detect user's location on page load
  Future<void> _autoDetectLocation() async {
    // Only auto-detect if we don't already have location data
    if (_currentLocationData != null) return;

    setState(() => _isLoadingWeather = true);

    try {
      final locationData = await _locationService.getLocationData();

      if (locationData != null && mounted) {
        setState(() {
          _currentLocationData = locationData;
          _displayLocationName = locationData.displayName;
          _usingCurrentLocation = true;
          _locationController.text = _displayLocationName!;
        });

        await _fetchWeatherForLocation(
          locationData.latitude,
          locationData.longitude,
        );
      } else {
        setState(() => _isLoadingWeather = false);
      }
    } catch (e) {
      debugPrint('Auto-detect location failed: $e');
      if (mounted) {
        setState(() => _isLoadingWeather = false);
      }
    }
  }

  @override
  void dispose() {
    _locationController.dispose();
    super.dispose();
  }

  /// Show modern location selection bottom sheet
  Future<void> _showLocationDialog() async {
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => _buildLocationBottomSheet(),
    );
  }

  Widget _buildLocationBottomSheet() {
    final theme = Theme.of(context);
    final textController = TextEditingController();

    return StatefulBuilder(
      builder: (context, setSheetState) {
        return Container(
          height: MediaQuery.of(context).size.height * 0.7,
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
          ),
          child: Column(
            children: [
              // Handle bar
              Container(
                margin: const EdgeInsets.only(top: 12),
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),

              Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Header
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.primaryContainer,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Icon(
                            Icons.location_on,
                            color: theme.colorScheme.primary,
                            size: 24,
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Set Your Location',
                                style: theme.textTheme.titleLarge?.copyWith(
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              Text(
                                'For weather-based outfit recommendations',
                                style: theme.textTheme.bodySmall?.copyWith(
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                              ),
                            ],
                          ),
                        ),
                        IconButton(
                          onPressed: () => Navigator.pop(context),
                          icon: Icon(
                            Icons.close,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 24),

                    // Current Location Button (Modern)
                    Material(
                      color: Colors.transparent,
                      child: InkWell(
                        onTap: () async {
                          Navigator.pop(context);
                          await _useCurrentLocation();
                        },
                        borderRadius: BorderRadius.circular(16),
                        child: Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                theme.colorScheme.primary.withValues(
                                  alpha: 0.1,
                                ),
                                theme.colorScheme.secondary.withValues(
                                  alpha: 0.05,
                                ),
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderRadius: BorderRadius.circular(16),
                            border: Border.all(
                              color: theme.colorScheme.primary.withValues(
                                alpha: 0.3,
                              ),
                            ),
                          ),
                          child: Row(
                            children: [
                              Container(
                                padding: const EdgeInsets.all(12),
                                decoration: BoxDecoration(
                                  color: theme.colorScheme.primary,
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: const Icon(
                                  Icons.my_location,
                                  color: Colors.white,
                                  size: 24,
                                ),
                              ),
                              const SizedBox(width: 16),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'Use Current Location',
                                      style: theme.textTheme.titleMedium
                                          ?.copyWith(
                                            fontWeight: FontWeight.w600,
                                          ),
                                    ),
                                    Text(
                                      'Automatically detect where you are',
                                      style: theme.textTheme.bodySmall
                                          ?.copyWith(
                                            color: theme
                                                .colorScheme
                                                .onSurfaceVariant,
                                          ),
                                    ),
                                  ],
                                ),
                              ),
                              Icon(
                                Icons.arrow_forward_ios,
                                size: 16,
                                color: theme.colorScheme.primary,
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),

                    const SizedBox(height: 20),

                    // Divider with text
                    Row(
                      children: [
                        Expanded(
                          child: Divider(
                            color: theme.colorScheme.outline.withValues(
                              alpha: 0.3,
                            ),
                          ),
                        ),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          child: Text(
                            'OR',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: theme.colorScheme.onSurfaceVariant,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                        Expanded(
                          child: Divider(
                            color: theme.colorScheme.outline.withValues(
                              alpha: 0.3,
                            ),
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 20),

                    // Search Location
                    Text(
                      'Search for a City',
                      style: theme.textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: textController,
                      decoration: InputDecoration(
                        hintText: 'e.g., Menlo Park, CA or Paris, France',
                        prefixIcon: Icon(
                          Icons.search,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        suffixIcon: IconButton(
                          onPressed: () async {
                            if (textController.text.isNotEmpty) {
                              Navigator.pop(context);
                              await _searchLocation(textController.text);
                            }
                          },
                          icon: Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: theme.colorScheme.primary,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: const Icon(
                              Icons.arrow_forward,
                              color: Colors.white,
                              size: 16,
                            ),
                          ),
                        ),
                        filled: true,
                        fillColor: theme.colorScheme.surfaceContainerHighest
                            .withValues(alpha: 0.5),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(16),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 16,
                        ),
                      ),
                      onSubmitted: (value) async {
                        if (value.isNotEmpty) {
                          Navigator.pop(context);
                          await _searchLocation(value);
                        }
                      },
                    ),

                    const SizedBox(height: 16),

                    // Tip box
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: theme.colorScheme.tertiaryContainer.withValues(
                          alpha: 0.3,
                        ),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            Icons.lightbulb_outline,
                            color: theme.colorScheme.tertiary,
                            size: 20,
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Text(
                              'Tip: Include state/country for accurate results (e.g., "Menlo Park, CA, USA")',
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: theme.colorScheme.onTertiaryContainer,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  /// Use device's current location
  Future<void> _useCurrentLocation() async {
    setState(() => _isLoadingWeather = true);

    try {
      final locationData = await _locationService.getLocationData();

      if (locationData != null) {
        setState(() {
          _currentLocationData = locationData;
          _displayLocationName = locationData.displayName;
          _usingCurrentLocation = true;
          _locationController.text = _displayLocationName!;
        });

        await _fetchWeatherForLocation(
          locationData.latitude,
          locationData.longitude,
        );
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text(
                'Unable to get current location. Please try manual entry.',
              ),
              duration: Duration(seconds: 3),
            ),
          );
        }
        setState(() => _isLoadingWeather = false);
      }
    } catch (e) {
      debugPrint('Error getting current location: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Location error: $e'),
            duration: const Duration(seconds: 3),
          ),
        );
      }
      setState(() => _isLoadingWeather = false);
    }
  }

  /// Search for a location by name and fetch weather
  Future<void> _searchLocation(String locationName) async {
    setState(() => _isLoadingWeather = true);

    try {
      final locationData = await _locationService.getLocationFromCity(
        locationName,
      );

      if (locationData != null) {
        setState(() {
          _currentLocationData = locationData;
          // Show full location name with state/country for clarity
          _displayLocationName = locationData.displayName;
          _usingCurrentLocation = false;
          _locationController.text = _displayLocationName!;
        });

        await _fetchWeatherForLocation(
          locationData.latitude,
          locationData.longitude,
        );
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                'Could not find "$locationName". Please try a different search.',
              ),
              duration: const Duration(seconds: 3),
            ),
          );
        }
        setState(() => _isLoadingWeather = false);
      }
    } catch (e) {
      debugPrint('Error searching location: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Search failed: $e'),
            duration: const Duration(seconds: 3),
          ),
        );
      }
      setState(() => _isLoadingWeather = false);
    }
  }

  /// Fetch weather for specific coordinates with retry
  Future<void> _fetchWeatherForLocation(
    double latitude,
    double longitude,
  ) async {
    setState(() => _isLoadingWeather = true);

    // Retry up to 3 times
    for (int attempt = 1; attempt <= 3; attempt++) {
      try {
        final weather = await _weatherService.getWeatherForLocation(
          latitude: latitude,
          longitude: longitude,
        );

        if (mounted) {
          setState(() {
            _weatherData = weather.toJson();
            _isLoadingWeather = false;
          });
        }
        return; // Success, exit retry loop
      } catch (e) {
        debugPrint('Weather fetch attempt $attempt failed: $e');

        if (attempt < 3) {
          // Wait before retry
          await Future.delayed(Duration(milliseconds: 500 * attempt));
        } else {
          // Final attempt failed
          if (mounted) {
            setState(() => _isLoadingWeather = false);
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Failed to fetch weather. Please try again.'),
                duration: Duration(seconds: 2),
              ),
            );
          }
        }
      }
    }
  }

  /// Get weather icon based on condition
  IconData _getWeatherIcon(String? condition) {
    if (condition == null) return Icons.cloud;
    final lowerCondition = condition.toLowerCase();

    if (lowerCondition.contains('sun') || lowerCondition.contains('clear')) {
      return Icons.wb_sunny;
    } else if (lowerCondition.contains('cloud') ||
        lowerCondition.contains('overcast')) {
      return Icons.cloud;
    } else if (lowerCondition.contains('rain') ||
        lowerCondition.contains('drizzle')) {
      return Icons.water_drop;
    } else if (lowerCondition.contains('snow') ||
        lowerCondition.contains('sleet')) {
      return Icons.ac_unit;
    } else if (lowerCondition.contains('thunder') ||
        lowerCondition.contains('storm')) {
      return Icons.thunderstorm;
    } else if (lowerCondition.contains('fog') ||
        lowerCondition.contains('mist')) {
      return Icons.foggy;
    } else if (lowerCondition.contains('wind')) {
      return Icons.air;
    } else if (lowerCondition.contains('partly')) {
      return Icons.cloud_queue;
    }
    return Icons.cloud;
  }

  /// Get weather color based on condition
  Color _getWeatherColor(String? condition) {
    if (condition == null) return Colors.grey;
    final lowerCondition = condition.toLowerCase();

    if (lowerCondition.contains('sun') || lowerCondition.contains('clear')) {
      return Colors.orange;
    } else if (lowerCondition.contains('cloud') ||
        lowerCondition.contains('overcast')) {
      return Colors.blueGrey;
    } else if (lowerCondition.contains('rain') ||
        lowerCondition.contains('drizzle')) {
      return Colors.blue;
    } else if (lowerCondition.contains('snow') ||
        lowerCondition.contains('sleet')) {
      return Colors.lightBlue;
    } else if (lowerCondition.contains('thunder') ||
        lowerCondition.contains('storm')) {
      return Colors.deepPurple;
    } else if (lowerCondition.contains('fog') ||
        lowerCondition.contains('mist')) {
      return Colors.grey;
    } else if (lowerCondition.contains('wind')) {
      return Colors.teal;
    } else if (lowerCondition.contains('partly')) {
      return Colors.amber;
    }
    return Colors.grey;
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
      if (!mounted) return;
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
      if (!mounted) return;
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

      // Build location context for AI analysis
      final locationContext = _currentLocationData != null
          ? 'Location: ${_displayLocationName ?? _currentLocationData!.city}'
          : '';

      final analysis = await _aiClient.analyzeOutfit(
        imageUrl: demoImageUrl,
        context: 'User wardrobe photo for outfit analysis. $locationContext',
        occasion: _selectedOccasion ?? 'casual',
        timeOfDay: _selectedTimeOfDay!.toLowerCase().replaceAll(' ', '_'),
        weather: weatherCondition,
      );

      setState(() => _analysis = analysis);
    } catch (e) {
      if (!mounted) return;
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
      _currentLocationData = null;
      _displayLocationName = null;
      _usingCurrentLocation = false;
      _weatherData = null;
      _locationController.clear();
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

                // Location selection for weather - Modern Card Design
                Container(
                  padding: EdgeInsets.all(4.w),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        theme.colorScheme.primaryContainer.withValues(
                          alpha: 0.4,
                        ),
                        theme.colorScheme.secondaryContainer.withValues(
                          alpha: 0.2,
                        ),
                      ],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: theme.colorScheme.primary.withValues(alpha: 0.2),
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: theme.colorScheme.primary.withValues(
                                alpha: 0.1,
                              ),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Icon(
                              Icons.location_on,
                              color: theme.colorScheme.primary,
                              size: 20,
                            ),
                          ),
                          SizedBox(width: 3.w),
                          Text(
                            'Weather Location',
                            style: theme.textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                      SizedBox(height: 2.h),
                      GestureDetector(
                        onTap: _showLocationDialog,
                        child: Container(
                          padding: EdgeInsets.symmetric(
                            horizontal: 4.w,
                            vertical: 1.5.h,
                          ),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.surface,
                            borderRadius: BorderRadius.circular(14),
                            boxShadow: [
                              BoxShadow(
                                color: theme.colorScheme.shadow.withValues(
                                  alpha: 0.05,
                                ),
                                blurRadius: 10,
                                offset: const Offset(0, 2),
                              ),
                            ],
                          ),
                          child: Row(
                            children: [
                              Container(
                                padding: const EdgeInsets.all(8),
                                decoration: BoxDecoration(
                                  color: _usingCurrentLocation
                                      ? theme.colorScheme.primary
                                      : theme.colorScheme.secondary,
                                  borderRadius: BorderRadius.circular(10),
                                ),
                                child: Icon(
                                  _usingCurrentLocation
                                      ? Icons.my_location
                                      : Icons.edit_location_alt,
                                  color: Colors.white,
                                  size: 18,
                                ),
                              ),
                              SizedBox(width: 3.w),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      _displayLocationName ??
                                          'Tap to set location',
                                      style: theme.textTheme.bodyLarge
                                          ?.copyWith(
                                            fontWeight: FontWeight.w500,
                                            color: _displayLocationName != null
                                                ? theme.colorScheme.onSurface
                                                : theme
                                                      .colorScheme
                                                      .onSurfaceVariant,
                                          ),
                                    ),
                                    if (_usingCurrentLocation &&
                                        _displayLocationName != null)
                                      Text(
                                        'Using GPS location',
                                        style: theme.textTheme.bodySmall
                                            ?.copyWith(
                                              color: theme.colorScheme.primary,
                                            ),
                                      ),
                                  ],
                                ),
                              ),
                              Container(
                                padding: const EdgeInsets.all(6),
                                decoration: BoxDecoration(
                                  color:
                                      theme.colorScheme.surfaceContainerHighest,
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                child: Icon(
                                  Icons.chevron_right,
                                  color: theme.colorScheme.onSurfaceVariant,
                                  size: 20,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),

                      // Weather display inline
                      if (_weatherData != null) ...[
                        SizedBox(height: 2.h),
                        Container(
                          padding: EdgeInsets.all(3.w),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.surface,
                            borderRadius: BorderRadius.circular(14),
                          ),
                          child: Row(
                            children: [
                              Container(
                                padding: const EdgeInsets.all(10),
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    colors: [
                                      Colors.orange.withValues(alpha: 0.8),
                                      Colors.amber.withValues(alpha: 0.8),
                                    ],
                                  ),
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: const Icon(
                                  Icons.thermostat,
                                  color: Colors.white,
                                  size: 22,
                                ),
                              ),
                              SizedBox(width: 3.w),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      '${_weatherData!['temperature']}°F',
                                      style: theme.textTheme.headlineSmall
                                          ?.copyWith(
                                            fontWeight: FontWeight.bold,
                                          ),
                                    ),
                                    Text(
                                      _weatherData!['condition'] ?? 'Unknown',
                                      style: theme.textTheme.bodyMedium
                                          ?.copyWith(
                                            color: theme
                                                .colorScheme
                                                .onSurfaceVariant,
                                          ),
                                    ),
                                  ],
                                ),
                              ),
                              Icon(
                                _getWeatherIcon(_weatherData!['condition']),
                                color: _getWeatherColor(
                                  _weatherData!['condition'],
                                ),
                                size: 32,
                              ),
                            ],
                          ),
                        ),
                      ] else if (_isLoadingWeather) ...[
                        SizedBox(height: 2.h),
                        Center(
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: theme.colorScheme.primary,
                                ),
                              ),
                              SizedBox(width: 3.w),
                              Text(
                                'Getting weather...',
                                style: theme.textTheme.bodyMedium?.copyWith(
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ],
                  ),
                ),

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
