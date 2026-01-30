import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

class OutfitRecommendationCardWidget extends StatelessWidget {
  final Map<String, dynamic> outfit;
  final VoidCallback onTap;
  final VoidCallback onLongPress;

  const OutfitRecommendationCardWidget({
    super.key,
    required this.outfit,
    required this.onTap,
    required this.onLongPress,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final items = outfit["items"] as List;

    return GestureDetector(
      onTap: onTap,
      onLongPress: onLongPress,
      child: Container(
        width: 70.w,
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: theme.colorScheme.shadow,
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Outfit Images Stack
            Expanded(
              flex: 5,
              child: Container(
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerHighest,
                  borderRadius: const BorderRadius.vertical(
                    top: Radius.circular(16),
                  ),
                ),
                child: Stack(
                  children: [
                    Positioned(
                      left: 4.w,
                      top: 2.h,
                      child: _buildItemImage(items[0], theme, 0),
                    ),
                    items.length > 1
                        ? Positioned(
                            left: 20.w,
                            top: 6.h,
                            child: _buildItemImage(items[1], theme, 1),
                          )
                        : const SizedBox.shrink(),
                    items.length > 2
                        ? Positioned(
                            left: 36.w,
                            top: 10.h,
                            child: _buildItemImage(items[2], theme, 2),
                          )
                        : const SizedBox.shrink(),
                  ],
                ),
              ),
            ),

            // Outfit Details
            Expanded(
              flex: 3,
              child: Padding(
                padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.w),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Text(
                            outfit["title"] as String,
                            style: theme.textTheme.titleMedium,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        Container(
                          padding: EdgeInsets.symmetric(
                            horizontal: 2.w,
                            vertical: 0.3.h,
                          ),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.tertiary.withValues(
                              alpha: 0.1,
                            ),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              CustomIconWidget(
                                iconName: 'star',
                                color: theme.colorScheme.tertiary,
                                size: 12,
                              ),
                              SizedBox(width: 1.w),
                              Text(
                                '${outfit["score"]}',
                                style: theme.textTheme.labelSmall?.copyWith(
                                  color: theme.colorScheme.tertiary,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 0.5.h),
                    Flexible(
                      child: Text(
                        outfit["description"] as String,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    const Spacer(),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: onTap,
                        style: ElevatedButton.styleFrom(
                          padding: EdgeInsets.symmetric(vertical: 1.h),
                        ),
                        child: Text('Try This Look'),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildItemImage(
    Map<String, dynamic> item,
    ThemeData theme,
    int index,
  ) {
    return Container(
      width: 24.w,
      height: 24.w,
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: theme.colorScheme.shadow,
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: CustomImageWidget(
          imageUrl: item["image"] as String,
          width: 24.w,
          height: 24.w,
          fit: BoxFit.cover,
          semanticLabel: item["semanticLabel"] as String,
        ),
      ),
    );
  }
}
