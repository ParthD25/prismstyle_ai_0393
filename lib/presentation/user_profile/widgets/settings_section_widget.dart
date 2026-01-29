import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';
import '../../../widgets/custom_icon_widget.dart';

/// Settings section widget with grouped list items
class SettingsSectionWidget extends StatelessWidget {
  final String title;
  final List<Map<String, dynamic>> items;

  const SettingsSectionWidget({
    super.key,
    required this.title,
    required this.items,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: EdgeInsets.symmetric(horizontal: 4.w),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          SizedBox(height: 1.h),
          Container(
            decoration: BoxDecoration(
              color: theme.colorScheme.surface,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: theme.colorScheme.outline.withValues(alpha: 0.2),
                width: 1,
              ),
            ),
            child: ListView.separated(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: items.length,
              separatorBuilder: (context, index) => Divider(
                height: 1,
                color: theme.colorScheme.outline.withValues(alpha: 0.1),
              ),
              itemBuilder: (context, index) {
                final item = items[index];
                return ListTile(
                  leading: CustomIconWidget(
                    iconName: item['icon'] as String,
                    color: theme.colorScheme.onSurface,
                    size: 24,
                  ),
                  title: Text(
                    item['title'] as String,
                    style: theme.textTheme.bodyLarge,
                  ),
                  subtitle: item['subtitle'] != null
                      ? Text(
                          item['subtitle'] as String,
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        )
                      : null,
                  trailing: CustomIconWidget(
                    iconName: 'chevron_right',
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
                    size: 24,
                  ),
                  onTap: () {
                    // Use the onTap callback from item if provided
                    final onTapCallback = item['onTap'];
                    if (onTapCallback != null && onTapCallback is VoidCallback) {
                      onTapCallback();
                    } else {
                      // Fallback to Coming Soon
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Feature coming soon'),
                          duration: Duration(seconds: 2),
                        ),
                      );
                    }
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
