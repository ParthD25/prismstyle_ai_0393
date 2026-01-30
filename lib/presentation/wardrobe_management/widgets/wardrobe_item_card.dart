import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

class WardrobeItemCard extends StatefulWidget {
  final Map<String, dynamic> item;
  final bool isSelected;
  final bool isSelectionMode;
  final VoidCallback onTap;
  final VoidCallback onLongPress;
  final VoidCallback onFavoriteToggle;
  final VoidCallback onContextMenu;

  const WardrobeItemCard({
    super.key,
    required this.item,
    required this.isSelected,
    required this.isSelectionMode,
    required this.onTap,
    required this.onLongPress,
    required this.onFavoriteToggle,
    required this.onContextMenu,
  });

  @override
  State<WardrobeItemCard> createState() => _WardrobeItemCardState();
}

class _WardrobeItemCardState extends State<WardrobeItemCard> {
  double _rotationX = 0;
  double _rotationY = 0;
  bool _isHovering = false;

  IconData _getCategoryIcon(String category) {
    switch (category) {
      case 'Tops':
        return Icons.checkroom;
      case 'Bottoms':
        return Icons.dry_cleaning;
      case 'Dresses':
        return Icons.woman;
      case 'Shoes':
        return Icons.shopping_bag;
      case 'Accessories':
        return Icons.watch;
      default:
        return Icons.category;
    }
  }

  void _onPanUpdate(DragUpdateDetails details, Size size) {
    setState(() {
      // Calculate rotation based on drag position
      _rotationY = (details.localPosition.dx - size.width / 2) / size.width * 0.3;
      _rotationX = -(details.localPosition.dy - size.height / 2) / size.height * 0.3;
      _isHovering = true;
    });
  }

  void _onPanEnd(DragEndDetails details) {
    setState(() {
      _rotationX = 0;
      _rotationY = 0;
      _isHovering = false;
    });
  }

  void _onPanCancel() {
    setState(() {
      _rotationX = 0;
      _rotationY = 0;
      _isHovering = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isFavorite = widget.item['isFavorite'] as bool;

    return LayoutBuilder(
      builder: (context, constraints) {
        return GestureDetector(
          onTap: widget.onTap,
          onLongPress: widget.onLongPress,
          onPanUpdate: (details) => _onPanUpdate(details, constraints.biggest),
          onPanEnd: _onPanEnd,
          onPanCancel: _onPanCancel,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 150),
            transform: Matrix4.identity()
              ..setEntry(3, 2, 0.001) // perspective
              ..rotateX(_rotationX)
              ..rotateY(_rotationY)
              ..scale(_isHovering ? 1.02 : 1.0),
            transformAlignment: Alignment.center,
            child: Card(
              elevation: _isHovering ? 8 : (widget.isSelected ? 4 : 2),
              shadowColor: theme.colorScheme.primary.withValues(alpha: _isHovering ? 0.4 : 0.2),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
                side: widget.isSelected
                    ? BorderSide(color: theme.colorScheme.primary, width: 2)
                    : BorderSide.none,
              ),
              child: Stack(
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(
                        child: ClipRRect(
                          borderRadius: const BorderRadius.vertical(
                            top: Radius.circular(12),
                          ),
                          child: CustomImageWidget(
                            imageUrl: widget.item['imageUrl'] as String,
                            width: double.infinity,
                            height: double.infinity,
                            fit: BoxFit.cover,
                            semanticLabel: widget.item['semanticLabel'] as String,
                          ),
                        ),
                      ),
                      Padding(
                        padding: EdgeInsets.all(2.w),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              widget.item['name'] as String,
                              style: theme.textTheme.bodyMedium?.copyWith(
                                fontWeight: FontWeight.w600,
                              ),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                            SizedBox(height: 0.5.h),
                            Text(
                              widget.item['category'] as String,
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: theme.colorScheme.onSurfaceVariant,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  Positioned(
                    top: 2.w,
                    left: 2.w,
                    child: Container(
                      padding: EdgeInsets.all(1.w),
                      decoration: BoxDecoration(
                        color: theme.colorScheme.surface.withValues(alpha: 0.9),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: CustomIconWidget(
                        iconName: _getCategoryIcon(
                          widget.item['category'] as String,
                        ).codePoint.toRadixString(16),
                        color: theme.colorScheme.primary,
                        size: 16,
                      ),
                    ),
                  ),
                  Positioned(
                    top: 2.w,
                    right: 2.w,
                    child: GestureDetector(
                      onTap: widget.onFavoriteToggle,
                      child: Container(
                        padding: EdgeInsets.all(1.w),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.surface.withValues(alpha: 0.9),
                          shape: BoxShape.circle,
                        ),
                        child: CustomIconWidget(
                          iconName: isFavorite ? 'favorite' : 'favorite_border',
                          color: isFavorite
                              ? theme.colorScheme.error
                              : theme.colorScheme.onSurfaceVariant,
                          size: 20,
                        ),
                      ),
                    ),
                  ),
                  if (widget.isSelectionMode)
                    Positioned(
                      bottom: 2.w,
                      right: 2.w,
                      child: Container(
                        padding: EdgeInsets.all(0.5.w),
                        decoration: BoxDecoration(
                          color: widget.isSelected
                              ? theme.colorScheme.primary
                              : theme.colorScheme.surface.withValues(alpha: 0.9),
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: theme.colorScheme.primary,
                            width: 2,
                          ),
                        ),
                        child: widget.isSelected
                            ? CustomIconWidget(
                                iconName: 'check',
                                color: theme.colorScheme.onPrimary,
                                size: 16,
                              )
                            : SizedBox(width: 16, height: 16),
                      ),
                    ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
