import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';
import '../../../widgets/custom_icon_widget.dart';

class EmptyCategoryWidget extends StatelessWidget {
  final String category;

  const EmptyCategoryWidget({super.key, required this.category});

  String _getEmptyMessage() {
    if (category == 'All') {
      return 'Your wardrobe is empty';
    }
    return 'No $category found';
  }

  String _getEmptyDescription() {
    if (category == 'All') {
      return 'Start building your digital wardrobe by adding your first item';
    }
    return 'Add your first ${category.toLowerCase()} to get started';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Center(
      child: Padding(
        padding: EdgeInsets.symmetric(horizontal: 8.w),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 30.w,
              height: 30.w,
              decoration: BoxDecoration(
                color: theme.colorScheme.primaryContainer.withValues(
                  alpha: 0.3,
                ),
                shape: BoxShape.circle,
              ),
              child: Center(
                child: CustomIconWidget(
                  iconName: 'checkroom',
                  color: theme.colorScheme.primary,
                  size: 15.w,
                ),
              ),
            ),
            SizedBox(height: 3.h),
            Text(
              _getEmptyMessage(),
              style: theme.textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.w600,
              ),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 1.h),
            Text(
              _getEmptyDescription(),
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 4.h),
            ElevatedButton.icon(
              onPressed: () {
                Navigator.of(
                  context,
                  rootNavigator: true,
                ).pushNamed('/camera-capture');
              },
              icon: CustomIconWidget(
                iconName: 'add_a_photo',
                color: theme.colorScheme.onPrimary,
                size: 20,
              ),
              label: Text(
                'Add Your First ${category == 'All' ? 'Item' : category}',
              ),
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(horizontal: 6.w, vertical: 1.5.h),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
