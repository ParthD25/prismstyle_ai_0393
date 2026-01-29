import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';
import 'package:smooth_page_indicator/smooth_page_indicator.dart';

import './widgets/onboarding_page_widget.dart';

/// Onboarding Flow Screen
/// Introduces new users to AI-powered wardrobe management through interactive tutorial screens
class OnboardingFlow extends StatefulWidget {
  const OnboardingFlow({super.key});

  @override
  State<OnboardingFlow> createState() => _OnboardingFlowState();
}

class _OnboardingFlowState extends State<OnboardingFlow> {
  final PageController _pageController = PageController();
  int _currentPage = 0;

  final List<Map<String, dynamic>> _onboardingData = [
    {
      "title": "AI-Powered Wardrobe",
      "description":
          "Let AI analyze your clothing and create perfect outfit combinations tailored to your style",
      "image":
          "https://img.rocket.new/generatedImages/rocket_gen_img_1047ee3d3-1768182112858.png",
      "semanticLabel":
          "Illustration of AI analyzing clothing items with digital overlay showing style recommendations and outfit matching algorithms",
    },
    {
      "title": "Capture Your Wardrobe",
      "description":
          "Simply photograph your clothing items and let our AI organize and categorize them automatically",
      "image":
          "https://img.rocket.new/generatedImages/rocket_gen_img_13fff105e-1764765623905.png",
      "semanticLabel":
          "Smartphone camera viewfinder overlay capturing a clothing item with animated focus frame and AI recognition indicators",
    },
    {
      "title": "Smart Outfit Generation",
      "description":
          "Get weather-aware outfit suggestions that match your style and the day's conditions",
      "image":
          "https://img.rocket.new/generatedImages/rocket_gen_img_150ff7b7e-1768182112780.png",
      "semanticLabel":
          "Multiple clothing items morphing together with weather icons showing AI-generated outfit combinations for different conditions",
    },
    {
      "title": "Social Validation",
      "description":
          "Share your outfits with friends and get instant feedback to boost your style confidence",
      "image":
          "https://img.rocket.new/generatedImages/rocket_gen_img_1fd794cc8-1767715053727.png",
      "semanticLabel":
          "Group of diverse friend avatars giving thumbs up and thumbs down feedback on outfit combinations displayed on mobile screen",
    },
  ];

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  void _onPageChanged(int page) {
    setState(() {
      _currentPage = page;
    });
  }

  void _nextPage() {
    if (_currentPage < _onboardingData.length - 1) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    } else {
      _completeOnboarding();
    }
  }

  void _skipOnboarding() {
    _completeOnboarding();
  }

  void _completeOnboarding() {
    Navigator.of(
      context,
      rootNavigator: true,
    ).pushReplacementNamed('/home-dashboard');
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: theme.scaffoldBackgroundColor,
      body: SafeArea(
        child: Column(
          children: [
            _buildTopBar(theme),
            Expanded(
              child: PageView.builder(
                controller: _pageController,
                onPageChanged: _onPageChanged,
                itemCount: _onboardingData.length,
                itemBuilder: (context, index) {
                  return OnboardingPageWidget(
                    title: _onboardingData[index]["title"] as String,
                    description:
                        _onboardingData[index]["description"] as String,
                    imageUrl: _onboardingData[index]["image"] as String,
                    semanticLabel:
                        _onboardingData[index]["semanticLabel"] as String,
                  );
                },
              ),
            ),
            _buildBottomSection(theme),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar(ThemeData theme) {
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          if (_currentPage < _onboardingData.length - 1)
            TextButton(
              onPressed: _skipOnboarding,
              style: TextButton.styleFrom(
                padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 1.h),
              ),
              child: Text(
                'Skip',
                style: theme.textTheme.labelLarge?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildBottomSection(ThemeData theme) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 6.w, vertical: 3.h),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SmoothPageIndicator(
            controller: _pageController,
            count: _onboardingData.length,
            effect: ExpandingDotsEffect(
              activeDotColor: theme.colorScheme.primary,
              dotColor: theme.colorScheme.outline,
              dotHeight: 1.h,
              dotWidth: 2.w,
              expansionFactor: 3,
              spacing: 1.w,
            ),
          ),
          SizedBox(height: 3.h),
          SizedBox(
            width: double.infinity,
            height: 6.h,
            child: ElevatedButton(
              onPressed: _nextPage,
              style: ElevatedButton.styleFrom(
                backgroundColor: theme.colorScheme.primary,
                foregroundColor: theme.colorScheme.onPrimary,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(2.w),
                ),
              ),
              child: Text(
                _currentPage == _onboardingData.length - 1
                    ? 'Get Started'
                    : 'Next',
                style: theme.textTheme.labelLarge?.copyWith(
                  color: theme.colorScheme.onPrimary,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
