import 'dart:math';

import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../core/app_export.dart';
import '../../widgets/custom_icon_widget.dart';
import './widgets/action_bar_widget.dart';
import './widgets/category_carousel_widget.dart';
import './widgets/compatibility_score_widget.dart';
import './widgets/outfit_preview_widget.dart';
import './widgets/weather_widget.dart';
import '../../services/openai_service.dart';
import '../../services/openai_client.dart';
import '../../services/weather_service.dart';

class OutfitGenerator extends StatefulWidget {
  const OutfitGenerator({super.key});

  @override
  State<OutfitGenerator> createState() => _OutfitGeneratorState();
}

class _OutfitGeneratorState extends State<OutfitGenerator>
    with TickerProviderStateMixin {
  late AnimationController _assemblyController;
  late AnimationController _transitionController;

  int _currentOutfitIndex = 0;
  bool _isShuffling = false;

  // Mock wardrobe data
  final List<Map<String, dynamic>> _wardrobeItems = [
    {
      "category": "tops",
      "items": [
        {
          "id": 1,
          "name": "White Cotton Shirt",
          "image":
              "https://images.unsplash.com/photo-1605760719369-be714c32a7f6",
          "semanticLabel":
              "White button-up cotton shirt hanging on wooden hanger against neutral background",
          "color": "White",
          "pattern": "Solid",
          "style": "Casual",
          "weatherScore": 8,
          "tip": "Perfect for warm weather with breathable cotton fabric",
        },
        {
          "id": 2,
          "name": "Navy Blazer",
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_185bb065f-1764674679589.png",
          "semanticLabel":
              "Navy blue blazer jacket on display with structured shoulders and lapels",
          "color": "Navy",
          "pattern": "Solid",
          "style": "Formal",
          "weatherScore": 6,
          "tip":
              "Adds sophistication to any outfit, ideal for business settings",
        },
        {
          "id": 3,
          "name": "Gray Sweater",
          "image":
              "https://images.unsplash.com/photo-1731402232633-6eb20c0f1521",
          "semanticLabel":
              "Light gray knit sweater folded neatly showing soft texture",
          "color": "Gray",
          "pattern": "Solid",
          "style": "Casual",
          "weatherScore": 9,
          "tip": "Cozy and versatile for cooler temperatures",
        },
      ],
    },
    {
      "category": "bottoms",
      "items": [
        {
          "id": 4,
          "name": "Dark Denim Jeans",
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_14f04dc0e-1767404876256.png",
          "semanticLabel":
              "Dark blue denim jeans laid flat showing classic five-pocket design",
          "color": "Dark Blue",
          "pattern": "Solid",
          "style": "Casual",
          "weatherScore": 7,
          "tip": "Classic choice that pairs well with most tops",
        },
        {
          "id": 5,
          "name": "Black Trousers",
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_175fb7e4a-1764674676449.png",
          "semanticLabel":
              "Black dress trousers hanging with crisp creases and tailored fit",
          "color": "Black",
          "pattern": "Solid",
          "style": "Formal",
          "weatherScore": 6,
          "tip": "Professional and sleek for formal occasions",
        },
        {
          "id": 6,
          "name": "Khaki Chinos",
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_1d8f0de2e-1764674681289.png",
          "semanticLabel":
              "Khaki colored chino pants folded showing cotton twill fabric",
          "color": "Khaki",
          "pattern": "Solid",
          "style": "Smart Casual",
          "weatherScore": 8,
          "tip": "Versatile option for both casual and semi-formal settings",
        },
      ],
    },
    {
      "category": "shoes",
      "items": [
        {
          "id": 7,
          "name": "White Sneakers",
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_13ef60586-1767723958930.png",
          "semanticLabel":
              "Clean white leather sneakers with minimal design on white background",
          "color": "White",
          "pattern": "Solid",
          "style": "Casual",
          "weatherScore": 8,
          "tip": "Comfortable and stylish for everyday wear",
        },
        {
          "id": 8,
          "name": "Brown Leather Shoes",
          "image":
              "https://images.unsplash.com/photo-1664505504065-31f8937d2261",
          "semanticLabel":
              "Brown leather oxford dress shoes polished and positioned at angle",
          "color": "Brown",
          "pattern": "Solid",
          "style": "Formal",
          "weatherScore": 7,
          "tip": "Elevates formal outfits with classic elegance",
        },
        {
          "id": 9,
          "name": "Black Boots",
          "image":
              "https://images.unsplash.com/photo-1613673720017-56e42d90fee4",
          "semanticLabel":
              "Black leather ankle boots with laces standing upright showing side profile",
          "color": "Black",
          "pattern": "Solid",
          "style": "Casual",
          "weatherScore": 9,
          "tip": "Perfect for cooler weather and adds edge to outfits",
        },
      ],
    },
    {
      "category": "accessories",
      "items": [
        {
          "id": 10,
          "name": "Silver Watch",
          "image":
              "https://images.unsplash.com/photo-1621500600029-00ff71efde5b",
          "semanticLabel":
              "Silver metal wristwatch with round face and leather strap on dark surface",
          "color": "Silver",
          "pattern": "Solid",
          "style": "Formal",
          "weatherScore": 10,
          "tip": "Timeless accessory that complements any outfit",
        },
        {
          "id": 11,
          "name": "Brown Leather Belt",
          "image":
              "https://images.unsplash.com/photo-1664286074240-d7059e004dff",
          "semanticLabel":
              "Brown leather belt coiled showing textured grain and metal buckle",
          "color": "Brown",
          "pattern": "Solid",
          "style": "Casual",
          "weatherScore": 10,
          "tip": "Essential accessory for a polished look",
        },
        {
          "id": 12,
          "name": "Black Sunglasses",
          "image":
              "https://img.rocket.new/generatedImages/rocket_gen_img_12a42b63f-1767210538184.png",
          "semanticLabel":
              "Black framed sunglasses with dark lenses positioned on white background",
          "color": "Black",
          "pattern": "Solid",
          "style": "Casual",
          "weatherScore": 10,
          "tip": "Protects eyes while adding cool factor to any outfit",
        },
      ],
    },
  ];

  // Generated outfit combinations
  final List<Map<String, dynamic>> _outfitCombinations = [];
  Map<String, dynamic>? _currentOutfit;

  // Weather data - will be fetched from API
  Map<String, dynamic> _weatherData = {
    "temperature": 0,
    "condition": "Loading...",
    "icon": "partly_sunny",
    "humidity": 0,
    "windSpeed": 0,
  };

  late OpenAIClient _aiClient;
  late WeatherService _weatherService;
  bool _isGenerating = false;
  String _aiRecommendations = '';
  bool _isLoadingWeather = true;

  final List<String> _selectedCategories = [
    'tops',
    'bottoms',
    'shoes',
    'accessories',
  ];
  final String _selectedOccasion = 'casual';
  final String _selectedTimeOfDay = 'all_day';
  String? _selectedLocation;

  @override
  void initState() {
    super.initState();
    _assemblyController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    _transitionController = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );
    _aiClient = OpenAIClient(OpenAIService().dio);
    _weatherService = WeatherService.instance;
    _generateOutfitCombinations();
    _assembleOutfit();
    _fetchWeatherData();
  }

  Future<void> _fetchWeatherData() async {
    setState(() => _isLoadingWeather = true);

    try {
      final weather = await _weatherService.getSanFranciscoWeather();
      setState(() {
        _weatherData = weather.toJson();
        _isLoadingWeather = false;
      });
    } catch (e) {
      setState(() {
        _weatherData = {
          "temperature": 72,
          "condition": "Unavailable",
          "icon": "cloud_off",
          "humidity": 0,
          "windSpeed": 0,
        };
        _isLoadingWeather = false;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Weather data unavailable: $e'),
            duration: const Duration(seconds: 2),
          ),
        );
      }
    }
  }

  @override
  void dispose() {
    _assemblyController.dispose();
    _transitionController.dispose();
    super.dispose();
  }

  void _generateOutfitCombinations() {
    _outfitCombinations.clear();

    final tops =
        _wardrobeItems.firstWhere((cat) => cat["category"] == "tops")["items"]
            as List;
    final bottoms =
        _wardrobeItems.firstWhere(
              (cat) => cat["category"] == "bottoms",
            )["items"]
            as List;
    final shoes =
        _wardrobeItems.firstWhere((cat) => cat["category"] == "shoes")["items"]
            as List;
    final accessories =
        _wardrobeItems.firstWhere(
              (cat) => cat["category"] == "accessories",
            )["items"]
            as List;

    for (var top in tops) {
      for (var bottom in bottoms) {
        for (var shoe in shoes) {
          for (var accessory in accessories) {
            final compatibility = _calculateCompatibility(
              top as Map<String, dynamic>,
              bottom as Map<String, dynamic>,
              shoe as Map<String, dynamic>,
              accessory as Map<String, dynamic>,
            );

            _outfitCombinations.add({
              "top": top,
              "bottom": bottom,
              "shoes": shoe,
              "accessory": accessory,
              "compatibilityScore": compatibility["score"],
              "colorHarmony": compatibility["colorHarmony"],
              "styleMatch": compatibility["styleMatch"],
              "weatherScore": compatibility["weatherScore"],
            });
          }
        }
      }
    }

    _outfitCombinations.sort(
      (a, b) => (b["compatibilityScore"] as double).compareTo(
        a["compatibilityScore"] as double,
      ),
    );
  }

  Map<String, dynamic> _calculateCompatibility(
    Map<String, dynamic> top,
    Map<String, dynamic> bottom,
    Map<String, dynamic> shoes,
    Map<String, dynamic> accessory,
  ) {
    double colorScore = _calculateColorHarmony([
      top["color"] as String,
      bottom["color"] as String,
      shoes["color"] as String,
      accessory["color"] as String,
    ]);

    double styleScore = _calculateStyleMatch([
      top["style"] as String,
      bottom["style"] as String,
      shoes["style"] as String,
      accessory["style"] as String,
    ]);

    double weatherScore =
        ((top["weatherScore"] as int) +
            (bottom["weatherScore"] as int) +
            (shoes["weatherScore"] as int)) /
        3.0;

    double finalScore =
        (colorScore * 0.4) + (styleScore * 0.4) + (weatherScore * 0.2);

    return {
      "score": finalScore,
      "colorHarmony": colorScore,
      "styleMatch": styleScore,
      "weatherScore": weatherScore,
    };
  }

  double _calculateColorHarmony(List<String> colors) {
    final neutralColors = ["White", "Black", "Gray", "Navy", "Khaki", "Brown"];
    int neutralCount = colors.where((c) => neutralColors.contains(c)).length;

    if (neutralCount >= 3) return 9.0 + Random().nextDouble();
    if (neutralCount == 2) return 7.0 + Random().nextDouble() * 2;
    return 5.0 + Random().nextDouble() * 3;
  }

  double _calculateStyleMatch(List<String> styles) {
    final uniqueStyles = styles.toSet();

    if (uniqueStyles.length == 1) return 9.5 + Random().nextDouble() * 0.5;
    if (uniqueStyles.contains("Formal") && uniqueStyles.contains("Casual")) {
      return 6.0 + Random().nextDouble() * 2;
    }
    return 7.5 + Random().nextDouble() * 1.5;
  }

  void _assembleOutfit() {
    if (_outfitCombinations.isEmpty) return;

    setState(() {
      _currentOutfit = _outfitCombinations[_currentOutfitIndex];
    });

    _assemblyController.forward(from: 0.0);
  }

  void _shuffleOutfit() async {
    if (_isShuffling) return;

    setState(() => _isShuffling = true);

    await _transitionController.forward(from: 0.0);

    setState(() {
      _currentOutfitIndex = Random().nextInt(_outfitCombinations.length);
      _currentOutfit = _outfitCombinations[_currentOutfitIndex];
    });

    await _transitionController.reverse();
    _assemblyController.forward(from: 0.0);

    setState(() => _isShuffling = false);
  }

  void _cycleOutfit(bool forward) async {
    await _transitionController.forward(from: 0.0);

    setState(() {
      if (forward) {
        _currentOutfitIndex =
            (_currentOutfitIndex + 1) % _outfitCombinations.length;
      } else {
        _currentOutfitIndex =
            (_currentOutfitIndex - 1 + _outfitCombinations.length) %
            _outfitCombinations.length;
      }
      _currentOutfit = _outfitCombinations[_currentOutfitIndex];
    });

    await _transitionController.reverse();
    _assemblyController.forward(from: 0.0);
  }

  void _swapItem(String category, Map<String, dynamic> newItem) async {
    await _transitionController.forward(from: 0.0);

    setState(() {
      _currentOutfit![category] = newItem;

      final compatibility = _calculateCompatibility(
        _currentOutfit!["top"] as Map<String, dynamic>,
        _currentOutfit!["bottom"] as Map<String, dynamic>,
        _currentOutfit!["shoes"] as Map<String, dynamic>,
        _currentOutfit!["accessory"] as Map<String, dynamic>,
      );

      _currentOutfit!["compatibilityScore"] = compatibility["score"];
      _currentOutfit!["colorHarmony"] = compatibility["colorHarmony"];
      _currentOutfit!["styleMatch"] = compatibility["styleMatch"];
      _currentOutfit!["weatherScore"] = compatibility["weatherScore"];
    });

    await _transitionController.reverse();
    _assemblyController.forward(from: 0.0);
  }

  void _saveOutfit() {
    showDialog(context: context, builder: (context) => _buildSaveDialog());
  }

  Widget _buildSaveDialog() {
    final theme = Theme.of(context);
    final TextEditingController nameController = TextEditingController();

    return AlertDialog(
      title: Text('Save Outfit', style: theme.textTheme.titleLarge),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextField(
            controller: nameController,
            decoration: const InputDecoration(
              labelText: 'Outfit Name',
              hintText: 'Enter a name for this outfit',
            ),
          ),
          SizedBox(height: 2.h),
          Text(
            'Compatibility Score: ${(_currentOutfit!["compatibilityScore"] as double).toStringAsFixed(1)}/10',
            style: theme.textTheme.bodyMedium,
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        ElevatedButton(
          onPressed: () {
            Navigator.pop(context);
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  'Outfit "${nameController.text.isEmpty ? "Untitled" : nameController.text}" saved successfully!',
                ),
                duration: const Duration(seconds: 2),
              ),
            );
          },
          child: const Text('Save'),
        ),
      ],
    );
  }

  void _shareOutfit() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Outfit shared for social validation!'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  void _toggleFavorite() {
    setState(() {
      _currentOutfit!["isFavorite"] = !(_currentOutfit!["isFavorite"] ?? false);
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          _currentOutfit!["isFavorite"] == true
              ? 'Added to favorites!'
              : 'Removed from favorites',
        ),
        duration: const Duration(seconds: 1),
      ),
    );
  }

  Future<void> _generateAIRecommendations() async {
    if (_selectedCategories.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select at least one wardrobe category'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }

    setState(() {
      _isGenerating = true;
      _aiRecommendations = '';
    });

    try {
      final wardrobeItems = _selectedCategories
          .map((cat) => '$cat items')
          .toList();

      await for (final chunk in _aiClient.generateOutfitSuggestions(
        wardrobeItems: wardrobeItems,
        occasion: _selectedOccasion ?? 'casual',
        timeOfDay: _selectedTimeOfDay ?? 'all_day',
        weather: _weatherData['condition'] as String? ?? 'moderate',
        location: _selectedLocation,
      )) {
        setState(() {
          _aiRecommendations += chunk;
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to generate recommendations: $e'),
          duration: const Duration(seconds: 3),
        ),
      );
    } finally {
      setState(() => _isGenerating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Outfit Generator'),
        centerTitle: true,
        leading: IconButton(
          icon: CustomIconWidget(
            iconName: 'arrow_back',
            color: theme.colorScheme.onSurface,
            size: 24,
          ),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: _currentOutfit == null
          ? Center(
              child: CircularProgressIndicator(
                color: theme.colorScheme.primary,
              ),
            )
          : Column(
              children: [
                WeatherWidget(weatherData: _weatherData),
                Expanded(
                  child: GestureDetector(
                    onHorizontalDragEnd: (details) {
                      if (details.primaryVelocity! > 0) {
                        _cycleOutfit(false);
                      } else if (details.primaryVelocity! < 0) {
                        _cycleOutfit(true);
                      }
                    },
                    child: SingleChildScrollView(
                      child: Column(
                        children: [
                          OutfitPreviewWidget(
                            outfit: _currentOutfit!,
                            assemblyController: _assemblyController,
                            transitionController: _transitionController,
                          ),
                          SizedBox(height: 2.h),
                          CompatibilityScoreWidget(outfit: _currentOutfit!),
                          SizedBox(height: 2.h),
                          CategoryCarouselWidget(
                            category: "tops",
                            items:
                                _wardrobeItems.firstWhere(
                                      (cat) => cat["category"] == "tops",
                                    )["items"]
                                    as List,
                            selectedItem:
                                _currentOutfit!["top"] as Map<String, dynamic>,
                            onItemSelected: (item) => _swapItem("top", item),
                          ),
                          SizedBox(height: 1.h),
                          CategoryCarouselWidget(
                            category: "bottoms",
                            items:
                                _wardrobeItems.firstWhere(
                                      (cat) => cat["category"] == "bottoms",
                                    )["items"]
                                    as List,
                            selectedItem:
                                _currentOutfit!["bottom"]
                                    as Map<String, dynamic>,
                            onItemSelected: (item) => _swapItem("bottom", item),
                          ),
                          SizedBox(height: 1.h),
                          CategoryCarouselWidget(
                            category: "shoes",
                            items:
                                _wardrobeItems.firstWhere(
                                      (cat) => cat["category"] == "shoes",
                                    )["items"]
                                    as List,
                            selectedItem:
                                _currentOutfit!["shoes"]
                                    as Map<String, dynamic>,
                            onItemSelected: (item) => _swapItem("shoes", item),
                          ),
                          SizedBox(height: 1.h),
                          CategoryCarouselWidget(
                            category: "accessories",
                            items:
                                _wardrobeItems.firstWhere(
                                      (cat) => cat["category"] == "accessories",
                                    )["items"]
                                    as List,
                            selectedItem:
                                _currentOutfit!["accessory"]
                                    as Map<String, dynamic>,
                            onItemSelected: (item) =>
                                _swapItem("accessory", item),
                          ),
                          SizedBox(height: 2.h),
                        ],
                      ),
                    ),
                  ),
                ),
                ActionBarWidget(
                  onShuffle: _shuffleOutfit,
                  onFavorite: _toggleFavorite,
                  onShare: _shareOutfit,
                  onSave: _saveOutfit,
                  isFavorite: _currentOutfit!["isFavorite"] ?? false,
                  isShuffling: _isShuffling,
                ),
              ],
            ),
    );
  }
}
