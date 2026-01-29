import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';
import '../../../widgets/custom_icon_widget.dart';

/// Horizontal metrics cards showing wardrobe statistics
class MetricsCardWidget extends StatelessWidget {
  final Map<String, dynamic> userData;

  const MetricsCardWidget({super.key, required this.userData});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Padding(
      padding: EdgeInsets.symmetric(horizontal: 4.w),
      child: Row(
        children: [
          Expanded(
            child: _buildMetricCard(
              context: context,
              icon: 'checkroom',
              label: 'Wardrobe Items',
              value: userData['wardrobeCount'].toString(),
              theme: theme,
            ),
          ),
          SizedBox(width: 3.w),
          Expanded(
            child: _buildMetricCard(
              context: context,
              icon: 'auto_awesome',
              label: 'Outfits Created',
              value: userData['outfitsCreated'].toString(),
              theme: theme,
            ),
          ),
          SizedBox(width: 3.w),
          Expanded(
            child: _buildMetricCard(
              context: context,
              icon: 'star',
              label: 'Validation Score',
              value: userData['validationScore'].toString(),
              theme: theme,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricCard({
    required BuildContext context,
    required String icon,
    required String label,
    required String value,
    required ThemeData theme,
  }) {
    return Container(
      padding: EdgeInsets.all(3.w),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: theme.colorScheme.outline.withValues(alpha: 0.2),
          width: 1,
        ),
      ),
      child: Column(
        children: [
          CustomIconWidget(
            iconName: icon,
            color: theme.colorScheme.primary,
            size: 28,
          ),
          SizedBox(height: 1.h),
          Text(
            value,
            style: theme.textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.w700,
            ),
          ),
          SizedBox(height: 0.5.h),
          Text(
            label,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
            textAlign: TextAlign.center,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}
