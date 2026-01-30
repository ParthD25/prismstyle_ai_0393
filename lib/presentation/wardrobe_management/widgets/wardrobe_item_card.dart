import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

/// 2.5D Parallax Wardrobe Card
/// Uses device sensors to create depth-based parallax effect similar to:
/// - iOS Portrait mode depth photos
/// - Android ARCore depth effect
/// The card image shifts based on device tilt, creating a "wiggle/look-around" effect
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

class _WardrobeItemCardState extends State<WardrobeItemCard>
    with SingleTickerProviderStateMixin {
  // Device tilt values (-1 to 1) from accelerometer
  double _tiltX = 0; // Left/right tilt (roll)
  double _tiltY = 0; // Forward/back tilt (pitch)
  
  // Smoothed values for fluid animation
  double _smoothTiltX = 0;
  double _smoothTiltY = 0;
  
  // Parallax configuration
  static const double _parallaxAmount = 0.04; // How much the image shifts (tune: 0.02-0.06)
  static const double _rotationAmount = 0.08; // 3D rotation amount
  static const double _smoothingFactor = 0.15; // Animation smoothing (lower = smoother)
  
  StreamSubscription<AccelerometerEvent>? _accelerometerSubscription;
  late AnimationController _animationController;
  
  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 16), // ~60fps
    )..addListener(_updateSmoothedValues);
    
    _startListeningToSensors();
  }

  void _startListeningToSensors() {
    // Listen to accelerometer for device tilt (like the iOS/Android code you provided)
    _accelerometerSubscription = accelerometerEventStream(
      samplingPeriod: const Duration(milliseconds: 16), // 60Hz
    ).listen((AccelerometerEvent event) {
      // Map accelerometer values to -1..1 range
      // Similar to: orientations[2] / (Math.PI / 12) in Android
      // Or: g.x * 1.5 in iOS
      setState(() {
        // X axis = roll (left/right tilt) - maps to horizontal parallax
        _tiltX = (event.x / 9.8 * 1.5).clamp(-1.0, 1.0);
        // Y axis = pitch (forward/back tilt) - maps to vertical parallax
        _tiltY = ((event.y - 4.9) / 9.8 * 1.5).clamp(-1.0, 1.0); // Offset for typical phone holding angle
      });
      
      if (!_animationController.isAnimating) {
        _animationController.forward(from: 0);
      }
    });
  }
  
  void _updateSmoothedValues() {
    // Smooth interpolation for fluid movement (like the iOS Core Image approach)
    setState(() {
      _smoothTiltX += (_tiltX - _smoothTiltX) * _smoothingFactor;
      _smoothTiltY += (_tiltY - _smoothTiltY) * _smoothingFactor;
    });
  }

  @override
  void dispose() {
    _accelerometerSubscription?.cancel();
    _animationController.dispose();
    super.dispose();
  }

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

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isFavorite = widget.item['isFavorite'] as bool;

    // Calculate parallax offsets based on device tilt
    // This mimics the shader: offset = (depth - 0.5) * tilt * amount
    // We simulate depth layers: background (image) moves opposite to foreground (UI elements)
    final double imageOffsetX = _smoothTiltX * _parallaxAmount * 100; // Background layer
    final double imageOffsetY = _smoothTiltY * _parallaxAmount * 50;
    final double foregroundOffsetX = -_smoothTiltX * _parallaxAmount * 30; // Foreground (UI) layer
    final double foregroundOffsetY = -_smoothTiltY * _parallaxAmount * 15;

    return GestureDetector(
      onTap: widget.onTap,
      onLongPress: widget.onLongPress,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 50),
        // 3D rotation based on tilt (like the Matrix4 transform in iOS)
        transform: Matrix4.identity()
          ..setEntry(3, 2, 0.002) // Perspective
          ..rotateX(_smoothTiltY * _rotationAmount) // Pitch rotation
          ..rotateY(-_smoothTiltX * _rotationAmount), // Roll rotation
        transformAlignment: Alignment.center,
        child: Card(
          elevation: 4 + (_smoothTiltX.abs() + _smoothTiltY.abs()) * 4, // Dynamic elevation
          shadowColor: theme.colorScheme.primary.withValues(alpha: 0.3),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: widget.isSelected
                ? BorderSide(color: theme.colorScheme.primary, width: 2)
                : BorderSide.none,
          ),
          clipBehavior: Clip.antiAlias,
          child: Stack(
            children: [
              // BACKGROUND LAYER: Image with parallax offset (moves WITH tilt)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: ClipRRect(
                      borderRadius: const BorderRadius.vertical(
                        top: Radius.circular(12),
                      ),
                      child: Transform.translate(
                        // Parallax offset for depth effect
                        // Like the shader: uv2 = uv + vec2(offset, 0.0)
                        offset: Offset(imageOffsetX, imageOffsetY),
                        child: Transform.scale(
                          // Slightly scale up to hide edges during parallax
                          scale: 1.1,
                          child: CustomImageWidget(
                            imageUrl: widget.item['imageUrl'] as String,
                            width: double.infinity,
                            height: double.infinity,
                            fit: BoxFit.cover,
                            semanticLabel:
                                widget.item['semanticLabel'] as String,
                          ),
                        ),
                      ),
                    ),
                  ),
                  // Text section (part of background layer)
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
              
              // FOREGROUND LAYER: UI elements with inverse parallax (moves AGAINST tilt)
              // This creates depth separation between image and floating UI
              Transform.translate(
                offset: Offset(foregroundOffsetX, foregroundOffsetY),
                child: Stack(
                  children: [
                    // Category icon (foreground)
                    Positioned(
                      top: 2.w,
                      left: 2.w,
                      child: Container(
                        padding: EdgeInsets.all(1.w),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.surface.withValues(alpha: 0.95),
                          borderRadius: BorderRadius.circular(8),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withValues(alpha: 0.2),
                              blurRadius: 4,
                              offset: Offset(_smoothTiltX * 2, _smoothTiltY * 2),
                            ),
                          ],
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
                    
                    // Favorite button (foreground)
                    Positioned(
                      top: 2.w,
                      right: 2.w,
                      child: GestureDetector(
                        onTap: widget.onFavoriteToggle,
                        child: Container(
                          padding: EdgeInsets.all(1.w),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.surface.withValues(alpha: 0.95),
                            shape: BoxShape.circle,
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withValues(alpha: 0.2),
                                blurRadius: 4,
                                offset: Offset(_smoothTiltX * 2, _smoothTiltY * 2),
                              ),
                            ],
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
                    
                    // Selection checkbox (foreground)
                    if (widget.isSelectionMode)
                      Positioned(
                        bottom: 2.w,
                        right: 2.w,
                        child: Container(
                          padding: EdgeInsets.all(0.5.w),
                          decoration: BoxDecoration(
                            color: widget.isSelected
                                ? theme.colorScheme.primary
                                : theme.colorScheme.surface.withValues(alpha: 0.95),
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: theme.colorScheme.primary,
                              width: 2,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withValues(alpha: 0.2),
                                blurRadius: 4,
                                offset: Offset(_smoothTiltX * 2, _smoothTiltY * 2),
                              ),
                            ],
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
            ],
          ),
        ),
      ),
    );
  }
}
