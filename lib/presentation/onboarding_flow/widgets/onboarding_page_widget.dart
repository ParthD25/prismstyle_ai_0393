import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';
import '../../../widgets/custom_image_widget.dart';

/// Individual onboarding page widget
/// Displays hero visual, headline, and descriptive text
class OnboardingPageWidget extends StatelessWidget {
  final String title;
  final String description;
  final String imageUrl;
  final String semanticLabel;

  const OnboardingPageWidget({
    super.key,
    required this.title,
    required this.description,
    required this.imageUrl,
    required this.semanticLabel,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return LayoutBuilder(
      builder: (context, constraints) {
        final maxHeight = constraints.maxHeight.isFinite
            ? constraints.maxHeight
            : 600.0;
        final imageHeight =
            (maxHeight * 0.45).clamp(180.0, maxHeight * 0.6);

        return Padding(
          padding: EdgeInsets.symmetric(horizontal: 6.w),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(
                height: imageHeight,
                child: _buildHeroImage(theme, imageHeight),
              ),
              SizedBox(height: 3.h),
              Flexible(child: _buildTitle(theme)),
              SizedBox(height: 1.h),
              Flexible(child: _buildDescription(theme)),
            ],
          ),
        );
      },
    );
  }

  Widget _buildHeroImage(ThemeData theme, double height) {
    return Container(
      height: height,
      width: double.infinity,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(4.w),
        boxShadow: [
          BoxShadow(
            color: theme.colorScheme.shadow.withValues(alpha: 0.1),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(4.w),
        child: CustomImageWidget(
          imageUrl: imageUrl,
          width: double.infinity,
          height: height,
          fit: BoxFit.cover,
          semanticLabel: semanticLabel,
        ),
      ),
    );
  }

  Widget _buildTitle(ThemeData theme) {
    return Text(
      title,
      textAlign: TextAlign.center,
      style: theme.textTheme.headlineMedium?.copyWith(
        color: theme.colorScheme.onSurface,
        fontWeight: FontWeight.w700,
      ),
    );
  }

  Widget _buildDescription(ThemeData theme) {
    return Text(
      description,
      textAlign: TextAlign.center,
      style: theme.textTheme.bodyLarge?.copyWith(
        color: theme.colorScheme.onSurfaceVariant,
        height: 1.5,
      ),
    );
  }
}
