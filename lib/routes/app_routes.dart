import 'package:flutter/material.dart';
import '../presentation/outfit_generator/outfit_generator.dart';
import '../presentation/splash_screen/splash_screen.dart';
import '../presentation/home_dashboard/home_dashboard.dart';
import '../presentation/user_profile/user_profile.dart';
import '../presentation/onboarding_flow/onboarding_flow.dart';
import '../presentation/social_validation/social_validation.dart';
import '../presentation/wardrobe_management/wardrobe_management.dart';
import '../presentation/camera_capture/camera_capture.dart';
import '../presentation/generate_outfit/generate_outfit.dart';
import '../presentation/privacy_policy/privacy_policy_screen.dart';
import '../presentation/ai_test/ai_test_screen.dart';

class AppRoutes {
  static const String initial = '/';
  static const String outfitGenerator = '/outfit-generator';
  static const String splash = '/splash-screen';
  static const String homeDashboard = '/home-dashboard';
  static const String userProfile = '/user-profile';
  static const String onboardingFlow = '/onboarding-flow';
  static const String socialValidation = '/social-validation';
  static const String wardrobeManagement = '/wardrobe-management';
  static const String cameraCapture = '/camera-capture';
  static const String generateOutfit = '/generate-outfit';
  static const String privacyPolicy = '/privacy-policy';
  static const String aiTest = '/ai-test';

  static Map<String, WidgetBuilder> routes = {
    initial: (context) => const SplashScreen(),
    outfitGenerator: (context) => const OutfitGenerator(),
    splash: (context) => const SplashScreen(),
    homeDashboard: (context) => const HomeDashboard(),
    userProfile: (context) => const UserProfile(),
    onboardingFlow: (context) => const OnboardingFlow(),
    socialValidation: (context) => const SocialValidation(),
    wardrobeManagement: (context) => const WardrobeManagement(),
    cameraCapture: (context) => const CameraCapture(),
    generateOutfit: (context) => const GenerateOutfit(),
    privacyPolicy: (context) => const PrivacyPolicyScreen(),
    aiTest: (context) => const AITestScreen(),
  };
}
