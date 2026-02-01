import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';
import 'package:share_plus/share_plus.dart';

import '../../core/app_export.dart';
import '../../widgets/custom_icon_widget.dart';
import '../../widgets/shimmer_loading.dart';
import '../../services/location_service.dart';
import '../../services/weather_service.dart';
import './widgets/outfit_recommendation_card_widget.dart';
import './widgets/recent_favorite_card_widget.dart';
import './widgets/trending_style_card_widget.dart';
import './widgets/weather_card_widget.dart';

class HomeDashboardInitialPage extends StatefulWidget {
  const HomeDashboardInitialPage({super.key});

  @override
  State<HomeDashboardInitialPage> createState() =>
      _HomeDashboardInitialPageState();
}

class _HomeDashboardInitialPageState extends State<HomeDashboardInitialPage> {
  bool _isLoading = false;
  // ignore: unused_field - Reserved for weather loading indicator
  bool _isLoadingWeather = true;

  // Services
  final LocationService _locationService = LocationService.instance;
  final WeatherService _weatherService = WeatherService.instance;

  // Weather data - will be updated from API
  Map<String, dynamic> weatherData = {
    "temperature": "--°F",
    "condition": "Loading...",
    "location": "Getting location...",
    "icon": "https://images.unsplash.com/photo-1716152937181-892e450ed5af",
    "semanticLabel":
        "Partly cloudy sky with white fluffy clouds against blue background",
    "appropriateness": "Checking weather...",
  };

  // Mock data for today's recommendations
  final List<Map<String, dynamic>> todaysRecommendations = [
    {
      "id": 1,
      "title": "Casual Chic",
      "score": 95,
      "items": [
        {
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_1b4b25899-1766343998388.png",
          "semanticLabel":
              "Light blue denim jacket with silver buttons on white background",
        },
        {
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_19fe7ad69-1764712498074.png",
          "semanticLabel":
              "White cotton t-shirt with crew neck on neutral background",
        },
        {
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_166f8f848-1764637973754.png",
          "semanticLabel": "Dark blue slim-fit jeans on white background",
        },
      ],
      "description": "Perfect for a casual day out with friends",
    },
    {
      "id": 2,
      "title": "Business Casual",
      "score": 92,
      "items": [
        {
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_15f3767ae-1764657201731.png",
          "semanticLabel":
              "Navy blue blazer with notch lapels hanging on wooden hanger",
        },
        {
          "image":
              "https://images.unsplash.com/photo-1623658580851-3b25bf83b4ea",
          "semanticLabel": "Light pink button-up dress shirt folded neatly",
        },
        {
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_1479209f9-1764659037202.png",
          "semanticLabel": "Beige chino pants laid flat on white surface",
        },
      ],
      "description": "Ideal for office meetings and professional events",
    },
    {
      "id": 3,
      "title": "Weekend Comfort",
      "score": 88,
      "items": [
        {
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_1788c161e-1764649786630.png",
          "semanticLabel":
              "Gray cotton hoodie with drawstring hood on white background",
        },
        {
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_17b384088-1764678417657.png",
          "semanticLabel": "Black athletic joggers with elastic waistband",
        },
        {
          "image":
              "https://images.unsplash.com/photo-1591731714000-b51e6b48dac6",
          "semanticLabel":
              "White sneakers with gray accents on clean background",
        },
      ],
      "description": "Comfortable and stylish for relaxed weekends",
    },
  ];

  // Mock data for trending styles
  final List<Map<String, dynamic>> trendingStyles = [
    {
      "id": 1,
      "title": "Minimalist Monochrome",
      "image":
          "https://img.rocket.new/generatedImages/rocket_gen_img_134be3dbf-1767270953556.png",
      "semanticLabel":
          "Fashion model wearing all-black minimalist outfit with turtleneck and wide-leg pants",
      "likes": 1247,
    },
    {
      "id": 2,
      "title": "Vintage Denim",
      "image":
          "https://img.rocket.new/generatedImages/rocket_gen_img_13c03f70c-1764925788199.png",
      "semanticLabel":
          "Vintage light-wash denim jacket with distressed details on mannequin",
      "likes": 982,
    },
    {
      "id": 3,
      "title": "Athleisure Fusion",
      "image":
          "https://img.rocket.new/generatedImages/rocket_gen_img_10b9510cf-1764769074732.png",
      "semanticLabel":
          "Athletic wear styled with casual pieces, featuring black leggings and oversized sweater",
      "likes": 856,
    },
  ];

  // Mock data for recent favorites
  final List<Map<String, dynamic>> recentFavorites = [
    {
      "id": 1,
      "title": "Summer Breeze",
      "image": "https://images.unsplash.com/photo-1562459201-ac62b364d87c",
      "semanticLabel":
          "Light floral sundress in pastel colors hanging on outdoor clothesline",
      "date": "2 days ago",
    },
    {
      "id": 2,
      "title": "Urban Explorer",
      "image": "https://images.unsplash.com/photo-1571568727822-8db701e2179c",
      "semanticLabel":
          "Black leather jacket with cargo pants and combat boots outfit combination",
      "date": "5 days ago",
    },
    {
      "id": 3,
      "title": "Classic Elegance",
      "image": "https://images.unsplash.com/photo-1589363358751-ab05797e5629",
      "semanticLabel":
          "Elegant beige trench coat with white blouse and tailored trousers",
      "date": "1 week ago",
    },
  ];

  @override
  void initState() {
    super.initState();
    _loadWeatherData();
  }

  /// Load weather data based on user's location
  Future<void> _loadWeatherData() async {
    setState(() => _isLoadingWeather = true);

    try {
      // Get user's location
      final locationData = await _locationService.getLocationData();
      final location = locationData ?? _locationService.defaultLocation;

      // Fetch weather for location
      final weather = await _weatherService.getWeatherForLocation(
        latitude: location.latitude,
        longitude: location.longitude,
      );

      // Update weather data
      setState(() {
        weatherData = {
          "temperature": "${weather.temperature.round()}°F",
          "condition": weather.condition,
          "location": location.displayName,
          "icon": _getWeatherIcon(weather.condition),
          "semanticLabel": "${weather.condition} weather in ${location.city}",
          "appropriateness": _getOutfitSuggestion(weather),
          "humidity": weather.humidity,
          "windSpeed": weather.windSpeed,
        };
        _isLoadingWeather = false;
      });
    } catch (e) {
      debugPrint('Error loading weather: $e');
      // Use default location weather
      final weather = await _weatherService.getSanFranciscoWeather();
      setState(() {
        weatherData = {
          "temperature": "${weather.temperature.round()}°F",
          "condition": weather.condition,
          "location": "San Francisco, CA",
          "icon": _getWeatherIcon(weather.condition),
          "semanticLabel": "${weather.condition} weather in San Francisco",
          "appropriateness": _getOutfitSuggestion(weather),
        };
        _isLoadingWeather = false;
      });
    }
  }

  /// Get weather icon URL based on condition
  String _getWeatherIcon(String condition) {
    switch (condition.toLowerCase()) {
      case 'sunny':
        return 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64';
      case 'partly cloudy':
        return 'https://images.unsplash.com/photo-1716152937181-892e450ed5af';
      case 'cloudy':
        return 'https://images.unsplash.com/photo-1534088568595-a066f410bcda';
      case 'rainy':
        return 'https://images.unsplash.com/photo-1519692933481-e162a57d6721';
      case 'snowy':
        return 'https://images.unsplash.com/photo-1491002052546-bf38f186af56';
      case 'stormy':
        return 'https://images.unsplash.com/photo-1527482797697-8795b05a13fe';
      default:
        return 'https://images.unsplash.com/photo-1716152937181-892e450ed5af';
    }
  }

  /// Get outfit suggestion based on weather
  String _getOutfitSuggestion(WeatherData weather) {
    final temp = weather.temperature;
    final condition = weather.condition.toLowerCase();

    if (condition.contains('rain') || condition.contains('storm')) {
      return 'Bring an umbrella and waterproof jacket';
    } else if (condition.contains('snow')) {
      return 'Time for warm layers and boots';
    } else if (temp >= 85) {
      return 'Light, breathable fabrics recommended';
    } else if (temp >= 70) {
      return 'Perfect for light layers';
    } else if (temp >= 55) {
      return 'A light jacket would be ideal';
    } else if (temp >= 40) {
      return 'Layer up with sweaters and coats';
    } else {
      return 'Bundle up! Heavy winter wear needed';
    }
  }

  Future<void> _handleRefresh() async {
    setState(() => _isLoading = true);
    // Reload weather data
    await _loadWeatherData();
    // Simulate additional AI processing
    await Future.delayed(const Duration(seconds: 1));
    setState(() => _isLoading = false);
  }

  void _navigateToCameraCapture() {
    Navigator.of(context, rootNavigator: true).pushNamed('/camera-capture');
  }

  void _navigateToOutfitDetail(Map<String, dynamic> outfit) {
    Navigator.of(context, rootNavigator: true).pushNamed('/outfit-generator');
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: RefreshIndicator(
        onRefresh: _handleRefresh,
        child: SafeArea(
          bottom: false,
          child: CustomScrollView(
          slivers: [
            // Weather Card Section
            SliverToBoxAdapter(
              child: Padding(
                padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
                child: WeatherCardWidget(weatherData: weatherData),
              ),
            ),

            // Today's Recommendations Section
            SliverToBoxAdapter(
              child: Padding(
                padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 1.h),
                child: Text(
                  "Today's Recommendations",
                  style: theme.textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
              ),
            ),

            SliverToBoxAdapter(
              child: _isLoading
                  ? _buildSkeletonLoader()
                  : SizedBox(
                      height: 48.h,
                      child: ListView.separated(
                        scrollDirection: Axis.horizontal,
                        padding: EdgeInsets.symmetric(horizontal: 4.w),
                        itemCount: todaysRecommendations.length,
                        separatorBuilder: (context, index) =>
                            SizedBox(width: 3.w),
                        itemBuilder: (context, index) {
                          return OutfitRecommendationCardWidget(
                            outfit: todaysRecommendations[index],
                            onTap: () => _navigateToOutfitDetail(
                              todaysRecommendations[index],
                            ),
                            onLongPress: () => _showQuickActions(
                              context,
                              todaysRecommendations[index],
                            ),
                          );
                        },
                      ),
                    ),
            ),

            // Trending Styles Section
            SliverToBoxAdapter(
              child: Padding(
                padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
                child: Text(
                  "Trending Styles",
                  style: theme.textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
              ),
            ),

            SliverToBoxAdapter(
              child: SizedBox(
                height: 28.h,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  padding: EdgeInsets.symmetric(horizontal: 4.w),
                  itemCount: trendingStyles.length,
                  separatorBuilder: (context, index) => SizedBox(width: 3.w),
                  itemBuilder: (context, index) {
                    return TrendingStyleCardWidget(
                      style: trendingStyles[index],
                    );
                  },
                ),
              ),
            ),

            // Recent Favorites Section
            SliverToBoxAdapter(
              child: Padding(
                padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
                child: Text(
                  "Your Recent Favorites",
                  style: theme.textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
              ),
            ),

            SliverPadding(
              padding: EdgeInsets.symmetric(horizontal: 4.w),
              sliver: SliverList(
                delegate: SliverChildBuilderDelegate((context, index) {
                  return Padding(
                    padding: EdgeInsets.only(bottom: 2.h),
                    child: RecentFavoriteCardWidget(
                      favorite: recentFavorites[index],
                    ),
                  );
                }, childCount: recentFavorites.length),
              ),
            ),

            SliverToBoxAdapter(child: SizedBox(height: 10.h)),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _navigateToCameraCapture,
        icon: CustomIconWidget(
          iconName: 'camera_alt',
          color: theme.colorScheme.onPrimary,
          size: 24,
        ),
        label: Text(
          'Add Item',
          style: theme.textTheme.labelLarge?.copyWith(
            color: theme.colorScheme.onPrimary,
          ),
        ),
      ),
    );
  }

  Widget _buildSkeletonLoader() {
    return const ShimmerOutfitList(itemCount: 3);
  }

  void _showQuickActions(BuildContext context, Map<String, dynamic> outfit) {
    final theme = Theme.of(context);

    showModalBottomSheet(
      context: context,
      builder: (context) {
        // Corrected syntax: 'return' keyword is needed here.
        // The 'body:' from the diff was syntactically incorrect for a builder.
        return Container(
          // Instruction 2: Adjust FloatingNavbar padding.
          // This padding is for the bottom sheet, not a FloatingNavbar.
          // Assuming the instruction refers to the bottom padding of this modal sheet
          // to accommodate a potential FloatingNavbar or system navigation bar.
          // The diff snippet provided a line `bottom: MediaQuery.of(context).padding.bottom + 10,`
          // which is syntactically incorrect in its original position.
          // Applying it as bottom padding to the container.
          padding: EdgeInsets.fromLTRB(4.w, 4.w, 4.w, MediaQuery.of(context).padding.bottom + 10),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: CustomIconWidget(
                  iconName: 'favorite',
                  color: theme.colorScheme.primary,
                  size: 24,
                ),
                title: Text(
                  'Save to Favorites',
                  style: theme.textTheme.bodyLarge,
                ),
                onTap: () {
                  Navigator.pop(context);
                  ScaffoldMessenger.of(
                    context,
                  ).showSnackBar(SnackBar(content: Text('Added to favorites')));
                },
              ),
              ListTile(
                leading: CustomIconWidget(
                  iconName: 'share',
                  color: theme.colorScheme.primary,
                  size: 24,
                ),
                title: Text(
                  'Share with Friends',
                  style: theme.textTheme.bodyLarge,
                ),
                onTap: () {
                  Navigator.pop(context);
                  // Use share_plus for native sharing
                  Share.share(
                    'Check out this outfit recommendation from PrismStyle AI! ${outfit['title']}',
                    subject: 'PrismStyle AI - ${outfit['title']}',
                  );
                },
              ),
              ListTile(
                leading: CustomIconWidget(
                  iconName: 'thumb_down',
                  color: theme.colorScheme.error,
                  size: 24,
                ),
                title: Text('Not My Style', style: theme.textTheme.bodyLarge),
                onTap: () {
                  Navigator.pop(context);
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(
                        'Thanks for the feedback! AI will improve recommendations',
                      ),
                    ),
                  );
                },
              ),
            ],
          ),
        );
      },
    );
  }
}
