import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sizer/sizer.dart';
import 'package:permission_handler/permission_handler.dart';

import '../core/app_export.dart';
import '../widgets/custom_error_widget.dart';
import './services/supabase_service.dart';
import './services/local_notification_service.dart';
import './services/clothing_classifier_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize Supabase (handles auth + database + real-time)
  try {
    await SupabaseService.initialize();
    debugPrint('Supabase initialized successfully');
  } catch (e) {
    debugPrint('Failed to initialize Supabase: $e');
  }

  // Initialize Local Notification Service (No Firebase needed)
  try {
    await LocalNotificationService.instance.initialize();
    debugPrint('Local notification service initialized successfully');
  } catch (e) {
    debugPrint('Failed to initialize notification service: $e');
  }

  // Initialize AI Clothing Classifier with Apple frameworks
  try {
    await ClothingClassifierService.instance.initialize();
    debugPrint('Clothing classifier initialized successfully');
  } catch (e) {
    debugPrint('Failed to initialize clothing classifier: $e');
  }

  // Request necessary permissions
  await _requestAppPermissions();

  bool hasShownError = false;

  // 🚨 CRITICAL: Custom error handling - DO NOT REMOVE
  ErrorWidget.builder = (FlutterErrorDetails details) {
    if (!hasShownError) {
      hasShownError = true;

      // Reset flag after 3 seconds to allow error widget on new screens
      Future.delayed(Duration(seconds: 5), () {
        hasShownError = false;
      });

      return CustomErrorWidget(errorDetails: details);
    }
    return SizedBox.shrink();
  };

  // 🚨 CRITICAL: Device orientation lock - DO NOT REMOVE
  Future.wait([
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]),
  ]).then((value) {
    runApp(MyApp());
  });
}

/// Request app permissions at startup
Future<void> _requestAppPermissions() async {
  // Request location permission for weather-based recommendations
  final locationStatus = await Permission.locationWhenInUse.request();
  if (locationStatus.isGranted) {
    debugPrint('Location permission granted');
  } else {
    debugPrint('Location permission denied');
  }

  // Request camera permission for wardrobe capture
  final cameraStatus = await Permission.camera.request();
  if (cameraStatus.isGranted) {
    debugPrint('Camera permission granted');
  } else {
    debugPrint('Camera permission denied');
  }

  // Request photo library permission
  final photosStatus = await Permission.photos.request();
  if (photosStatus.isGranted) {
    debugPrint('Photos permission granted');
  } else {
    debugPrint('Photos permission denied');
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return Sizer(
      builder: (context, orientation, screenType) {
        return MaterialApp(
          title: 'prismstyle_ai',
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          themeMode: ThemeMode.light,
          // 🚨 CRITICAL: NEVER REMOVE OR MODIFY
          builder: (context, child) {
            return MediaQuery(
              data: MediaQuery.of(
                context,
              ).copyWith(textScaler: TextScaler.linear(1.0)),
              child: child!,
            );
          },
          // 🚨 END CRITICAL SECTION
          debugShowCheckedModeBanner: false,
          routes: AppRoutes.routes,
          initialRoute: AppRoutes.initial,
        );
      },
    );
  }
}
