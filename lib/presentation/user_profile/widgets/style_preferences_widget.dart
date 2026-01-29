import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';
import '../../../widgets/custom_icon_widget.dart';

class StylePreferencesWidget extends StatelessWidget {
  final Map<String, dynamic> userData;

  const StylePreferencesWidget({super.key, required this.userData});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: EdgeInsets.all(4.w),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(
                  'Style Preferences',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              TextButton.icon(
                onPressed: () => _navigateToEditStylePreferences(context),
                icon: CustomIconWidget(
                  iconName: 'edit',
                  color: theme.colorScheme.primary,
                  size: 18,
                ),
                label: Text('Edit'),
              ),
            ],
          ),
          SizedBox(height: 1.h),
          Wrap(
            spacing: 2.w,
            runSpacing: 1.h,
            children: (userData["stylePreferences"] as List)
                .map(
                  (pref) => Chip(
                    label: Text(pref),
                    backgroundColor: theme.colorScheme.primaryContainer
                        .withValues(alpha: 0.5),
                    labelStyle: theme.textTheme.bodySmall,
                  ),
                )
                .toList(),
          ),
        ],
      ),
    );
  }

  void _navigateToEditStylePreferences(BuildContext context) {
    // Navigate to style preferences editing screen
    Navigator.pushNamed(context, '/edit-style-preferences');
  }
}
