# Getting Started

<cite>
**Referenced Files in This Document**
- [README.md](file://README.md)
- [pubspec.yaml](file://pubspec.yaml)
- [env.json](file://env.json)
- [supabase_schema.sql](file://supabase_schema.sql)
- [scripts/setup_supabase.py](file://scripts/setup_supabase.py)
- [ios/Podfile](file://ios/Podfile)
- [ios/Runner/AppDelegate.swift](file://ios/Runner/AppDelegate.swift)
- [ios/Runner/AppleVisionHandler.swift](file://ios/Runner/AppleVisionHandler.swift)
- [android/app/build.gradle](file://android/app/build.gradle)
- [assets/models/model_config.json](file://assets/models/model_config.json)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Development Environment Setup](#development-environment-setup)
4. [Production Environment Setup](#production-environment-setup)
5. [Initial Configuration](#initial-configuration)
6. [Platform-Specific Setup](#platform-specific-setup)
7. [Verification Steps](#verification-steps)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)

## Introduction
This guide provides a complete getting started walkthrough for PrismStyle AI, covering prerequisites, environment setup, configuration, platform-specific requirements, and verification steps. PrismStyle AI is an AI-powered fashion assistant for iOS 18+ and Android, featuring smart camera capture, digital wardrobe management, color detection, ensemble AI combining TFLite, Apple Vision, and Core ML, weather integration, outfit generation, and privacy-focused design.

## Prerequisites
Before installing PrismStyle AI, ensure your development environment meets the following requirements:

- Flutter SDK 3.38.4 and Dart SDK 3.9.0+ for cross-platform development
- Android Studio for Android development and Android 8.0 (API 26)+ devices/emulators
- Xcode 15.0+ for iOS development and iOS 18.0+ devices/emulators
- Python 3.x for Supabase setup automation (optional but recommended)
- CocoaPods for iOS dependency management

These requirements are essential for building and running the application on both platforms.

**Section sources**
- [README.md](file://README.md#L26-L46)

## Development Environment Setup
Follow these steps to set up your development environment:

1. Install Flutter SDK 3.38.4 and configure your PATH so the `flutter` command is available.
2. Install Dart SDK 3.9.0+ as required by the project.
3. Install Android Studio and set up an Android virtual device or connect a physical Android device (API 26+).
4. Install Xcode 15.0+ and ensure CocoaPods is available for iOS.
5. Install Python 3.x to run the Supabase setup script (optional).

Once installed, verify your setup by running:
- `flutter doctor` to check Flutter and platform configurations
- `flutter pub get` to fetch Dart dependencies

**Section sources**
- [README.md](file://README.md#L37-L46)
- [pubspec.yaml](file://pubspec.yaml#L6-L7)

## Production Environment Setup
For production builds, ensure you have:

- iOS 18.0+ deployment target and Xcode 15.0+ for iOS distribution
- Android API 23+ for Android distribution
- Properly configured signing keys for Android app bundles
- Supabase credentials configured for backend services

Build commands:
- iOS: `flutter build ios --release`
- Android: `flutter build appbundle --release`

**Section sources**
- [README.md](file://README.md#L181-L195)
- [ios/Podfile](file://ios/Podfile#L1-L2)

## Initial Configuration
Configure the application by setting up environment variables and Supabase:

1. Configure environment variables:
   - Copy the template file and update the Supabase URL, anonymous key, and secret key
   - Set OPEN_METEO_API_URL for weather integration
   - Leave other API keys blank if not using external providers

2. Set up Supabase:
   - Use the provided Python script to validate connection and display setup instructions
   - Run the script with options to check connection, show setup instructions, create buckets, or verify tables
   - Execute the SQL schema in the Supabase SQL Editor
   - Create storage buckets: clothing-images, outfit-images, profile-avatars
   - Enable Realtime subscriptions for specific tables
   - Verify Row Level Security (RLS) policies are applied

3. Deploy database schema:
   - Open Supabase SQL Editor
   - Paste and run the contents of the schema file
   - Confirm tables, indexes, triggers, and policies are created

**Section sources**
- [env.json](file://env.json#L1-L13)
- [scripts/setup_supabase.py](file://scripts/setup_supabase.py#L15-L19)
- [scripts/setup_supabase.py](file://scripts/setup_supabase.py#L94-L146)
- [supabase_schema.sql](file://supabase_schema.sql#L1-L319)

## Platform-Specific Setup
Configure platform-specific features for optimal performance:

### iOS (Apple Vision Framework and Core ML)
- Deployment target: iOS 18.0+
- Xcode 15.0+ with CocoaPods
- Apple Vision Framework and Core ML are integrated via Flutter method channels
- The app initializes Apple Vision and Core ML handlers during startup
- Real device testing is required for Apple Vision and Core ML features

Key implementation details:
- Method channels for Apple Vision and Core ML are registered in the AppDelegate
- Apple Vision uses Vision framework for built-in image classification
- Core ML integrates custom model optimization for enhanced performance

**Section sources**
- [README.md](file://README.md#L28-L31)
- [ios/Podfile](file://ios/Podfile#L1-L2)
- [ios/Runner/AppDelegate.swift](file://ios/Runner/AppDelegate.swift#L11-L27)
- [ios/Runner/AppDelegate.swift](file://ios/Runner/AppDelegate.swift#L35-L99)
- [ios/Runner/AppleVisionHandler.swift](file://ios/Runner/AppleVisionHandler.swift#L12-L17)

### Android (TFLite Model Integration)
- Android 8.0 (API 26)+ with Android Studio
- TensorFlow Lite GPU delegate for hardware acceleration
- Multi-architecture support with ABI filters for ARM, x86, and x86_64
- TFLite model files are included in assets and marked as non-compressed

Important build settings:
- MultiDex enabled for larger applications
- Desugaring for Java 8 language features
- TFLite GPU delegate plugin for performance optimization

**Section sources**
- [README.md](file://README.md#L33-L35)
- [android/app/build.gradle](file://android/app/build.gradle#L23-L42)
- [android/app/build.gradle](file://android/app/build.gradle#L63-L66)

## Verification Steps
Perform these checks to ensure proper installation and configuration:

1. Flutter environment verification:
   - Run `flutter doctor` and resolve any warnings
   - Execute `flutter pub get` to confirm dependency resolution

2. Supabase connection verification:
   - Use the setup script to check connection: `python scripts/setup_supabase.py --check`
   - Verify all required tables exist using: `python scripts/setup_supabase.py --verify`
   - Confirm storage buckets are created and accessible

3. Platform-specific verification:
   - iOS: Build and run on a physical iOS device (18.0+) to test Apple Vision and Core ML
   - Android: Build and run on a physical Android device (API 23+) to test TFLite integration
   - Camera capture should work on both platforms

4. AI model verification:
   - Ensure TFLite model files are present in assets/models/
   - Verify asset configuration in pubspec.yaml includes the models directory

**Section sources**
- [README.md](file://README.md#L161-L180)
- [scripts/setup_supabase.py](file://scripts/setup_supabase.py#L238-L271)
- [pubspec.yaml](file://pubspec.yaml#L63-L76)

## Troubleshooting Guide
Common issues and solutions:

### Flutter Command Issues
- If `flutter` command is not found, use FVM prefix for commands:
  - Example: `fvm flutter pub get`, `fvm flutter run`

### iOS Build Failures
- Clean and reinstall dependencies:
  - `cd ios && pod deintegrate && pod install && cd ..`
  - `flutter clean && flutter pub get`

### TFLite Model Loading Problems
- Ensure model files are placed in `assets/models/`
- Verify pubspec.yaml includes the models asset directory:
  - `assets/models/` under the assets section

### Apple Framework Issues
- Apple Vision and Core ML require iOS 13.0+ (project targets iOS 18.0+)
- Test on real devices, not simulators
- Check Xcode console for detailed error logs

### Android TFLite Performance
- Verify GPU delegate is properly configured
- Ensure ABI filters match target architectures
- Confirm TFLite files are marked as non-compressed in build configuration

**Section sources**
- [README.md](file://README.md#L210-L240)

## Conclusion
You now have all the necessary steps to set up PrismStyle AI for both development and production. By following the prerequisites, environment setup, initial configuration, platform-specific requirements, and verification procedures outlined above, you can successfully develop and deploy this AI-powered fashion assistant. For advanced features like model training and custom AI models, refer to the additional documentation links provided in the project README.