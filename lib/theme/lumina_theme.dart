import 'package:flutter/material.dart';
import 'dart:ui';

class LuminaTheme {
  static const Color backgroundBlack = Color(0xFF0F0F0F);
  static const Color accentPurple = Color(0xFF6B21A8); // purple-900
  static const Color accentBlue = Color(0xFF1E3A8A); // blue-900
  static const Color textWhite = Colors.white;
  static const Color textGrey = Color(0xFF9CA3AF); // gray-400

  // Gradients
  static const RadialGradient backgroundGradient = RadialGradient(
    center: Alignment.topLeft,
    radius: 1.5,
    colors: [
      Color(0xFF2E1065), // deep purple
      Color(0xFF0F0F0F), // black
      Color(0xFF172554), // deep blue
    ],
    stops: [0.0, 0.5, 1.0],
  );

  static BoxDecoration glassDecoration = BoxDecoration(
    color: Colors.white.withOpacity(0.1),
    borderRadius: BorderRadius.circular(24),
    border: Border.all(color: Colors.white.withOpacity(0.2)),
    boxShadow: [
      BoxShadow(
        color: Colors.black.withOpacity(0.2),
        blurRadius: 16,
        offset: const Offset(0, 4),
      ),
    ],
  );

  static ThemeData get themeData {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: backgroundBlack,
      primaryColor: accentPurple,
      textTheme: const TextTheme(
        displayLarge: TextStyle(
          fontFamily: 'Serif', // Placeholder, using default serif
          fontSize: 32,
          fontWeight: FontWeight.bold,
          color: textWhite,
        ),
        bodyLarge: TextStyle(
          fontFamily: 'Sans', // Placeholder
          fontSize: 16,
          color: textWhite,
        ),
      ),
      useMaterial3: true,
    );
  }
}
