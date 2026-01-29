import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

class CategoryCarouselWidget extends StatelessWidget {
  final String category;
  final List items;
  final Map<String, dynamic> selectedItem;
  final Function(Map<String, dynamic>) onItemSelected;

  const CategoryCarouselWidget({
    super.key,
    required this.category,
    required this.items,
    required this.selectedItem,
    required this.onItemSelected,
  });

  String _getCategoryDisplayName() {
    switch (category) {
      case 'tops':
        return 'Tops';
      case 'bottoms':
        return 'Bottoms';
      case 'shoes':
        return 'Shoes';
      case 'accessories':
        return 'Accessories';
      default:
        return category;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: EdgeInsets.symmetric(horizontal: 4.w),
          child: Text(
            _getCategoryDisplayName(),
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        SizedBox(height: 1.h),
        SizedBox(
          height: 20.h,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            padding: EdgeInsets.symmetric(horizontal: 4.w),
            itemCount: items.length,
            itemBuilder: (context, index) {
              final item = items[index] as Map<String, dynamic>;
              final isSelected = item["id"] == selectedItem["id"];

              return GestureDetector(
                onTap: () => onItemSelected(item),
                child: Container(
                  width: 30.w,
                  margin: EdgeInsets.only(right: 3.w),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: isSelected
                          ? theme.colorScheme.primary
                          : theme.colorScheme.outline.withValues(alpha: 0.3),
                      width: isSelected ? 3 : 1,
                    ),
                    boxShadow: isSelected
                        ? [
                            BoxShadow(
                              color: theme.colorScheme.primary.withValues(
                                alpha: 0.3,
                              ),
                              blurRadius: 8,
                              offset: const Offset(0, 2),
                            ),
                          ]
                        : null,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(
                        child: ClipRRect(
                          borderRadius: const BorderRadius.vertical(
                            top: Radius.circular(12),
                          ),
                          child: CustomImageWidget(
                            imageUrl: item["image"] as String,
                            width: 30.w,
                            height: double.infinity,
                            fit: BoxFit.cover,
                            semanticLabel: item["semanticLabel"] as String,
                          ),
                        ),
                      ),
                      Container(
                        padding: EdgeInsets.all(2.w),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              item["name"] as String,
                              style: theme.textTheme.labelMedium?.copyWith(
                                fontWeight: isSelected
                                    ? FontWeight.w600
                                    : FontWeight.w400,
                              ),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                            SizedBox(height: 0.5.h),
                            Row(
                              children: [
                                CustomIconWidget(
                                  iconName: 'circle',
                                  color: _getColorFromName(
                                    item["color"] as String,
                                  ),
                                  size: 12,
                                ),
                                SizedBox(width: 1.w),
                                Expanded(
                                  child: Text(
                                    item["color"] as String,
                                    style: theme.textTheme.labelSmall?.copyWith(
                                      color: theme.colorScheme.onSurfaceVariant,
                                    ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Color _getColorFromName(String colorName) {
    switch (colorName.toLowerCase()) {
      case 'white':
        return Colors.white;
      case 'black':
        return Colors.black;
      case 'gray':
        return Colors.grey;
      case 'navy':
        return Colors.blue.shade900;
      case 'dark blue':
        return Colors.blue.shade800;
      case 'khaki':
        return const Color(0xFFC3B091);
      case 'brown':
        return Colors.brown;
      case 'silver':
        return Colors.grey.shade400;
      default:
        return Colors.grey;
    }
  }
}
