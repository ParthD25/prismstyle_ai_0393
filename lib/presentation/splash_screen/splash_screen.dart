import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../core/app_export.dart';
import '../../widgets/custom_icon_widget.dart';

/// Splash Screen - AI-powered fashion recommendation app launch
/// Displays branded experience while initializing AI models and checking auth status
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;

  // ignore: unused_field - Reserved for initialization state tracking
  bool _isInitializing = true;
  String _loadingMessage = 'Preparing your style assistant';

  @override
  void initState() {
    super.initState();
    _setupAnimations();
    _initializeApp();
  }

  void _setupAnimations() {
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: const Interval(0.0, 0.5, curve: Curves.easeIn),
      ),
    );

    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: const Interval(0.0, 0.7, curve: Curves.elasticOut),
      ),
    );

    _animationController.forward();
  }

  Future<void> _initializeApp() async {
    try {
      // Simulate AI model loading and initialization
      await Future.delayed(const Duration(milliseconds: 800));

      if (mounted) {
        setState(() {
          _loadingMessage = 'Loading AI models';
        });
      }

      await Future.delayed(const Duration(milliseconds: 1000));

      if (mounted) {
        setState(() {
          _loadingMessage = 'Syncing wardrobe data';
        });
      }

      await Future.delayed(const Duration(milliseconds: 700));

      if (mounted) {
        setState(() {
          _isInitializing = false;
        });
      }

      // Navigate based on authentication status
      await Future.delayed(const Duration(milliseconds: 500));

      if (mounted) {
        _navigateToNextScreen();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _loadingMessage = 'Initialization failed. Retrying...';
        });
        await Future.delayed(const Duration(seconds: 2));
        _initializeApp();
      }
    }
  }

  Future<void> _navigateToNextScreen() async {
    // Check if first time user
    await _checkFirstTimeUser();
    final bool isAuthenticated = _checkAuthStatus();

    String nextRoute;
    if (_isFirstTimeUser) {
      nextRoute = '/onboarding-flow';
    } else if (isAuthenticated) {
      nextRoute = '/home-dashboard';
    } else {
      nextRoute = '/home-dashboard';
    }

    if (mounted) {
      Navigator.of(
        context,
        rootNavigator: true,
      ).pushReplacementNamed(nextRoute);
    }
  }

  bool _checkAuthStatus() {
    // Simulate auth check - returns false for new users
    return false;
  }

  bool _isFirstTimeUser = true;

  Future<void> _checkFirstTimeUser() async {
    final prefs = await SharedPreferences.getInstance();
    _isFirstTimeUser = !(prefs.getBool('hasCompletedOnboarding') ?? false);
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              theme.colorScheme.primary,
              theme.colorScheme.primary.withValues(alpha: 0.8),
              theme.colorScheme.tertiary.withValues(alpha: 0.6),
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Spacer(flex: 2),
              _buildLogo(theme),
              SizedBox(height: 6.h),
              _buildLoadingIndicator(theme),
              SizedBox(height: 2.h),
              _buildLoadingMessage(theme),
              const Spacer(flex: 3),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLogo(ThemeData theme) {
    return FadeTransition(
      opacity: _fadeAnimation,
      child: ScaleTransition(
        scale: _scaleAnimation,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 30.w,
              height: 30.w,
              decoration: BoxDecoration(
                color: theme.colorScheme.surface,
                borderRadius: BorderRadius.circular(4.w),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.2),
                    blurRadius: 20,
                    offset: const Offset(0, 10),
                  ),
                ],
              ),
              child: Center(
                child: CustomIconWidget(
                  iconName: 'auto_awesome',
                  size: 15.w,
                  color: theme.colorScheme.primary,
                ),
              ),
            ),
            SizedBox(height: 3.h),
            Text(
              'PrismStyle AI',
              style: theme.textTheme.headlineMedium?.copyWith(
                color: theme.colorScheme.surface,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.2,
              ),
            ),
            SizedBox(height: 1.h),
            Text(
              'Your Personal Style Assistant',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.surface.withValues(alpha: 0.9),
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingIndicator(ThemeData theme) {
    return SizedBox(
      width: 12.w,
      height: 12.w,
      child: CircularProgressIndicator(
        strokeWidth: 3,
        valueColor: AlwaysStoppedAnimation<Color>(theme.colorScheme.surface),
      ),
    );
  }

  Widget _buildLoadingMessage(ThemeData theme) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 300),
      child: Text(
        _loadingMessage,
        key: ValueKey<String>(_loadingMessage),
        style: theme.textTheme.bodyMedium?.copyWith(
          color: theme.colorScheme.surface.withValues(alpha: 0.9),
          letterSpacing: 0.5,
        ),
        textAlign: TextAlign.center,
      ),
    );
  }
}
