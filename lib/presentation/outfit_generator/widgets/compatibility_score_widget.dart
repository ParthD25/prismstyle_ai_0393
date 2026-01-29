import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';
import '../../../widgets/custom_icon_widget.dart';

class CompatibilityScoreWidget extends StatelessWidget {
  final Map<String, dynamic> outfit;

  const CompatibilityScoreWidget({super.key, required this.outfit});

  Color _getScoreColor(double score, ThemeData theme) {
    if (score >= 8.5) return theme.colorScheme.tertiary;
    if (score >= 7.0) return Colors.green;
    if (score >= 5.5) return Colors.orange;
    return theme.colorScheme.error;
  }

  String _getScoreLabel(double score) {
    if (score >= 8.5) return 'Excellent';
    if (score >= 7.0) return 'Good';
    if (score >= 5.5) return 'Fair';
    return 'Poor';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final score = outfit["compatibilityScore"] as double;
    final colorHarmony = outfit["colorHarmony"] as double;
    final styleMatch = outfit["styleMatch"] as double;
    final weatherScore = outfit["weatherScore"] as double;

    return Container(
      margin: EdgeInsets.symmetric(horizontal: 4.w),
      padding: EdgeInsets.all(4.w),
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
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Compatibility Score',
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              Container(
                padding: EdgeInsets.symmetric(horizontal: 3.w, vertical: 1.h),
                decoration: BoxDecoration(
                  color: _getScoreColor(score, theme).withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: _getScoreColor(score, theme),
                    width: 2,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      score.toStringAsFixed(1),
                      style: theme.textTheme.titleLarge?.copyWith(
                        color: _getScoreColor(score, theme),
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      '/10',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: _getScoreColor(score, theme),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          SizedBox(height: 1.h),
          Text(
            _getScoreLabel(score),
            style: theme.textTheme.bodyMedium?.copyWith(
              color: _getScoreColor(score, theme),
              fontWeight: FontWeight.w500,
            ),
          ),
          SizedBox(height: 2.h),
          _buildIndicator(
            context,
            'Color Harmony',
            colorHarmony,
            Icons.palette,
          ),
          SizedBox(height: 1.h),
          _buildIndicator(context, 'Style Match', styleMatch, Icons.checkroom),
          SizedBox(height: 1.h),
          _buildIndicator(
            context,
            'Weather Appropriate',
            weatherScore,
            Icons.wb_sunny,
          ),
        ],
      ),
    );
  }

  Widget _buildIndicator(
    BuildContext context,
    String label,
    double value,
    IconData icon,
  ) {
    final theme = Theme.of(context);
    final percentage = value / 10.0;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            CustomIconWidget(
              iconName: icon.codePoint.toRadixString(16),
              color: theme.colorScheme.onSurfaceVariant,
              size: 16,
            ),
            SizedBox(width: 2.w),
            Text(label, style: theme.textTheme.bodyMedium),
            const Spacer(),
            Text(
              value.toStringAsFixed(1),
              style: theme.textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
        SizedBox(height: 0.5.h),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: percentage,
            backgroundColor: theme.colorScheme.outline.withValues(alpha: 0.2),
            valueColor: AlwaysStoppedAnimation<Color>(
              _getScoreColor(value, theme),
            ),
            minHeight: 8,
          ),
        ),
      ],
    );
  }
}
