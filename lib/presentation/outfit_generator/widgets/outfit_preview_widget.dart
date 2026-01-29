import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

class OutfitPreviewWidget extends StatelessWidget {
  final Map<String, dynamic> outfit;
  final AnimationController assemblyController;
  final AnimationController transitionController;

  const OutfitPreviewWidget({
    super.key,
    required this.outfit,
    required this.assemblyController,
    required this.transitionController,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      width: double.infinity,
      height: 50.h,
      margin: EdgeInsets.symmetric(horizontal: 4.w),
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
      child: Stack(
        children: [
          Positioned.fill(
            child: CustomPaint(
              painter: _BodySilhouettePainter(
                color: theme.colorScheme.outline.withValues(alpha: 0.1),
              ),
            ),
          ),
          AnimatedBuilder(
            animation: assemblyController,
            builder: (context, child) {
              return FadeTransition(
                opacity: transitionController.drive(
                  Tween<double>(begin: 1.0, end: 0.0),
                ),
                child: Stack(
                  children: [
                    _buildItemLayer(
                      outfit["top"] as Map<String, dynamic>,
                      0.15,
                      0.2,
                      0.0,
                    ),
                    _buildItemLayer(
                      outfit["bottom"] as Map<String, dynamic>,
                      0.15,
                      0.45,
                      0.25,
                    ),
                    _buildItemLayer(
                      outfit["shoes"] as Map<String, dynamic>,
                      0.15,
                      0.7,
                      0.5,
                    ),
                    _buildItemLayer(
                      outfit["accessory"] as Map<String, dynamic>,
                      0.7,
                      0.25,
                      0.75,
                    ),
                  ],
                ),
              );
            },
          ),
          Positioned(
            top: 2.h,
            right: 2.w,
            child: Container(
              padding: EdgeInsets.symmetric(horizontal: 3.w, vertical: 1.h),
              decoration: BoxDecoration(
                color: theme.colorScheme.primary.withValues(alpha: 0.9),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CustomIconWidget(
                    iconName: 'auto_awesome',
                    color: theme.colorScheme.onPrimary,
                    size: 16,
                  ),
                  SizedBox(width: 1.w),
                  Text(
                    'AI Generated',
                    style: theme.textTheme.labelSmall?.copyWith(
                      color: theme.colorScheme.onPrimary,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildItemLayer(
    Map<String, dynamic> item,
    double leftFraction,
    double topFraction,
    double animationDelay,
  ) {
    return AnimatedBuilder(
      animation: assemblyController,
      builder: (context, child) {
        final progress = (assemblyController.value - animationDelay).clamp(
          0.0,
          1.0,
        );
        final opacity = progress;
        final scale = 0.8 + (progress * 0.2);

        return Positioned(
          left: leftFraction * 100.w,
          top: topFraction * 50.h,
          child: Opacity(
            opacity: opacity,
            child: Transform.scale(
              scale: scale,
              child: GestureDetector(
                onLongPress: () => _showItemDetails(item),
                child: Container(
                  width: 25.w,
                  height: 25.w,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.2),
                        blurRadius: 8,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: CustomImageWidget(
                      imageUrl: item["image"] as String,
                      width: 25.w,
                      height: 25.w,
                      fit: BoxFit.cover,
                      semanticLabel: item["semanticLabel"] as String,
                    ),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  void _showItemDetails(Map<String, dynamic> item) {
    // This would show a dialog with item details in a real implementation
  }
}

class _BodySilhouettePainter extends CustomPainter {
  final Color color;

  _BodySilhouettePainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final path = Path();

    final centerX = size.width / 2;
    final headRadius = size.width * 0.08;

    path.addOval(
      Rect.fromCircle(
        center: Offset(centerX, size.height * 0.15),
        radius: headRadius,
      ),
    );

    path.moveTo(centerX, size.height * 0.15 + headRadius);
    path.lineTo(centerX, size.height * 0.5);

    path.moveTo(centerX - size.width * 0.15, size.height * 0.25);
    path.lineTo(centerX + size.width * 0.15, size.height * 0.25);

    path.moveTo(centerX, size.height * 0.5);
    path.lineTo(centerX - size.width * 0.1, size.height * 0.75);

    path.moveTo(centerX, size.height * 0.5);
    path.lineTo(centerX + size.width * 0.1, size.height * 0.75);

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
