import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';
import '../../../widgets/custom_icon_widget.dart';

class ActionBarWidget extends StatelessWidget {
  final VoidCallback onShuffle;
  final VoidCallback onFavorite;
  final VoidCallback onShare;
  final VoidCallback onSave;
  final bool isFavorite;
  final bool isShuffling;

  const ActionBarWidget({
    super.key,
    required this.onShuffle,
    required this.onFavorite,
    required this.onShare,
    required this.onSave,
    required this.isFavorite,
    required this.isShuffling,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      width: double.infinity,
      padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        border: Border(
          top: BorderSide(
            color: theme.colorScheme.outline.withValues(alpha: 0.2),
            width: 1,
          ),
        ),
        boxShadow: [
          BoxShadow(
            color: theme.colorScheme.shadow,
            blurRadius: 8,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _buildActionButton(
            context,
            icon: Icons.shuffle,
            label: 'Shuffle',
            onTap: onShuffle,
            isPrimary: true,
            isLoading: isShuffling,
          ),
          _buildActionButton(
            context,
            icon: isFavorite ? Icons.favorite : Icons.favorite_border,
            label: 'Favorite',
            onTap: onFavorite,
            color: isFavorite ? Colors.red : null,
          ),
          _buildActionButton(
            context,
            icon: Icons.share,
            label: 'Share',
            onTap: onShare,
          ),
          _buildActionButton(
            context,
            icon: Icons.save_alt,
            label: 'Save',
            onTap: onSave,
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton(
    BuildContext context, {
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    bool isPrimary = false,
    bool isLoading = false,
    Color? color,
  }) {
    final theme = Theme.of(context);
    final buttonColor =
        color ??
        (isPrimary
            ? theme.colorScheme.primary
            : theme.colorScheme.onSurfaceVariant);

    return InkWell(
      onTap: isLoading ? null : onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 3.w, vertical: 1.h),
        decoration: BoxDecoration(
          color: isPrimary
              ? theme.colorScheme.primary.withValues(alpha: 0.1)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
          border: isPrimary
              ? Border.all(color: theme.colorScheme.primary, width: 1)
              : null,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            isLoading
                ? SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(buttonColor),
                    ),
                  )
                : CustomIconWidget(
                    iconName: icon.codePoint.toRadixString(16),
                    color: buttonColor,
                    size: 24,
                  ),
            SizedBox(height: 0.5.h),
            Text(
              label,
              style: theme.textTheme.labelSmall?.copyWith(
                color: buttonColor,
                fontWeight: isPrimary ? FontWeight.w600 : FontWeight.w400,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
