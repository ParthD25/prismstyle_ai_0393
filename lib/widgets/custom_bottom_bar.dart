import 'package:flutter/material.dart';

/// Custom bottom navigation bar for the fashion app
/// Implements thumb-optimized bottom placement for one-handed operation
/// Follows the Mobile Navigation Hierarchy from design specifications
class CustomBottomBar extends StatelessWidget {
  /// Current selected index
  final int currentIndex;

  /// Callback when a navigation item is tapped
  final Function(int) onTap;

  const CustomBottomBar({
    super.key,
    required this.currentIndex,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return BottomNavigationBar(
      currentIndex: currentIndex,
      onTap: onTap,
      type: BottomNavigationBarType.fixed,
      elevation: 8.0,
      items: const [
        // Home - Weather-based outfit recommendations
        BottomNavigationBarItem(
          icon: Icon(Icons.auto_awesome_outlined, size: 24),
          activeIcon: Icon(Icons.auto_awesome, size: 24),
          label: 'Home',
          tooltip: 'Weather-based outfit recommendations',
        ),

        // Wardrobe - Clothing inventory management
        BottomNavigationBarItem(
          icon: Icon(Icons.checkroom_outlined, size: 24),
          activeIcon: Icon(Icons.checkroom, size: 24),
          label: 'Wardrobe',
          tooltip: 'Manage your clothing inventory',
        ),

        // Generate - AI-powered outfit generation (replaces Camera)
        BottomNavigationBarItem(
          icon: Icon(Icons.auto_fix_high_outlined, size: 28),
          activeIcon: Icon(Icons.auto_fix_high, size: 28),
          label: 'Generate',
          tooltip: 'Generate outfit with AI preferences',
        ),

        // Social - Outfit sharing and validation
        BottomNavigationBarItem(
          icon: Icon(Icons.favorite_outline, size: 24),
          activeIcon: Icon(Icons.favorite, size: 24),
          label: 'Social',
          tooltip: 'Share and validate outfits',
        ),

        // Profile - Account and preference management
        BottomNavigationBarItem(
          icon: Icon(Icons.person_outline, size: 24),
          activeIcon: Icon(Icons.person, size: 24),
          label: 'Profile',
          tooltip: 'Manage account and preferences',
        ),
      ],
    );
  }
}
