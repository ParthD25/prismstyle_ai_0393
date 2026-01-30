import 'package:flutter/material.dart';

import '../../routes/app_routes.dart';
import '../../widgets/custom_bottom_bar.dart';
import '../generate_outfit/generate_outfit.dart';
import '../social_validation/social_validation.dart';
import '../user_profile/user_profile.dart';
import '../wardrobe_management/wardrobe_management.dart';
import './home_dashboard_initial_page.dart';

class HomeDashboard extends StatefulWidget {
  const HomeDashboard({super.key});

  @override
  HomeDashboardState createState() => HomeDashboardState();
}

class HomeDashboardState extends State<HomeDashboard> {
  final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();
  int currentIndex = 0;

  // ALL CustomBottomBar routes in EXACT order matching CustomBottomBar items
  final List<String> routes = [
    '/home-dashboard', // index 0 - Home
    '/wardrobe-management', // index 1 - Wardrobe
    '/generate-outfit', // index 2 - Generate Outfit
    '/social-validation', // index 3 - Social
    '/user-profile', // index 4 - Profile
  ];

  // ignore: unused_field - Available for tab-based navigation
  final List<Widget> _screens = [
    const HomeDashboardInitialPage(),
    const WardrobeManagement(),
    const GenerateOutfit(),
    const SocialValidation(),
    const UserProfile(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Navigator(
        key: navigatorKey,
        initialRoute: '/home-dashboard',
        onGenerateRoute: (settings) {
          switch (settings.name) {
            case '/home-dashboard' || '/':
              return MaterialPageRoute(
                builder: (context) => const HomeDashboardInitialPage(),
                settings: settings,
              );
            default:
              // Check AppRoutes.routes for all other routes
              if (AppRoutes.routes.containsKey(settings.name)) {
                return MaterialPageRoute(
                  builder: AppRoutes.routes[settings.name]!,
                  settings: settings,
                );
              }
              return null;
          }
        },
      ),
      bottomNavigationBar: CustomBottomBar(
        currentIndex: currentIndex,
        onTap: (index) {
          // For the routes that are not in the AppRoutes.routes, do not navigate to them.
          if (!AppRoutes.routes.containsKey(routes[index])) {
            return;
          }
          if (currentIndex != index) {
            setState(() => currentIndex = index);
            navigatorKey.currentState?.pushReplacementNamed(routes[index]);
          }
        },
      ),
    );
  }
}
