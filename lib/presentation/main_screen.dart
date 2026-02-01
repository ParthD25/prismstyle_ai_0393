import 'package:flutter/material.dart';
import 'dart:ui';
import 'package:prismstyle_ai/theme/lumina_theme.dart';
import 'package:prismstyle_ai/presentation/widgets/floating_navbar.dart';
import 'package:prismstyle_ai/presentation/home_dashboard/home_dashboard_initial_page.dart';
import 'package:prismstyle_ai/presentation/stylist/stylist_screen.dart';
import 'package:prismstyle_ai/presentation/lookbook/lookbook_screen.dart';
import 'package:prismstyle_ai/presentation/ai/ask_ai_screen.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({Key? key}) : super(key: key);

  static const String routeName = '/main-screen';

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;

  final List<Widget> _pages = [
    // 0: Wardrobe (Home)
    const HomeDashboardInitialPage(), 
    
    // 1: Stylist
    const StylistScreen(),
    
    // 2: Lookbook
    const LookbookScreen(),
    
    // 3: Ask AI
    const AskAIScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBody: true, // Important for floating navbar
      body: Container(
        decoration: const BoxDecoration(
          gradient: LuminaTheme.backgroundGradient,
        ),
        child: Stack(
          children: [
            // Background ambient blur effects
            Positioned(
              top: -100,
              left: -100,
              child: Container(
                width: 300,
                height: 300,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: LuminaTheme.accentPurple.withOpacity(0.3),
                ),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 100, sigmaY: 100),
                  child: Container(color: Colors.transparent),
                ),
              ),
            ),
             Positioned(
              bottom: -100,
              right: -100,
              child: Container(
                width: 300,
                height: 300,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: LuminaTheme.accentBlue.withOpacity(0.3),
                ),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 100, sigmaY: 100),
                  child: Container(color: Colors.transparent),
                ),
              ),
            ),
            
            // Main Content
            IndexedStack(
              index: _currentIndex,
              children: _pages,
            ),
            
            // Floating Navbar
            FloatingNavbar(
              currentIndex: _currentIndex,
              onTap: (index) => setState(() => _currentIndex = index),
            ),
          ],
        ),
      ),
    );
  }
}
