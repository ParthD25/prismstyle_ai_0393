# PrismStyle AI

A Flutter application for AI-powered fashion recommendations and wardrobe management.

## Setup & Build Fix

### iOS Build Issue Fix

Due to a compatibility issue with the `app_links` plugin (Swift-only plugin), you need to run the fix script after `flutter pub get`:

```bash
./fix_app_links.sh
```

This fixes the "Module 'app_links' not found" error in `GeneratedPluginRegistrant.m`.

### Getting Started

1. Install dependencies:
   ```bash
   flutter pub get
   ```

2. Fix iOS plugin registration:
   ```bash
   ./fix_app_links.sh
   ```

3. Run the app:
   ```bash
   flutter run  # Android
   flutter build ios --no-codesign  # iOS build
   ```

## Features

- AI-powered outfit recommendations
- Wardrobe management
- Fashion style analysis
- Local AI backend support
- Cross-platform (iOS/Android)

## Resources

- [Flutter Documentation](https://docs.flutter.dev/)
- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)
