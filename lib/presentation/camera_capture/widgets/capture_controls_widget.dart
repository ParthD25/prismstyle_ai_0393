import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

class CaptureControlsWidget extends StatelessWidget {
  final VoidCallback onCapture;
  final VoidCallback onGalleryTap;
  final VoidCallback onLightingTips;
  final XFile? lastGalleryImage;

  const CaptureControlsWidget({
    super.key,
    required this.onCapture,
    required this.onGalleryTap,
    required this.onLightingTips,
    this.lastGalleryImage,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      bottom: 4.h,
      left: 0,
      right: 0,
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 8.w),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            _buildGalleryButton(context),
            _buildCaptureButton(context),
            _buildLightingTipsButton(context),
          ],
        ),
      ),
    );
  }

  Widget _buildGalleryButton(BuildContext context) {
    return GestureDetector(
      onTap: onGalleryTap,
      child: Container(
        width: 16.w,
        height: 8.h,
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.2),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: Colors.white.withValues(alpha: 0.5),
            width: 2,
          ),
        ),
        child: lastGalleryImage != null
            ? ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: CustomImageWidget(
                  imageUrl: lastGalleryImage!.path,
                  width: 16.w,
                  height: 8.h,
                  fit: BoxFit.cover,
                  semanticLabel: 'Last captured wardrobe item thumbnail',
                ),
              )
            : Center(
                child: CustomIconWidget(
                  iconName: 'photo_library',
                  color: Colors.white,
                  size: 28,
                ),
              ),
      ),
    );
  }

  Widget _buildCaptureButton(BuildContext context) {
    return GestureDetector(
      onTap: onCapture,
      child: Container(
        width: 20.w,
        height: 10.h,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: Colors.white,
          border: Border.all(
            color: Theme.of(context).colorScheme.tertiary,
            width: 4,
          ),
        ),
        child: Center(
          child: Container(
            width: 16.w,
            height: 8.h,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Theme.of(context).colorScheme.tertiary,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLightingTipsButton(BuildContext context) {
    return GestureDetector(
      onTap: onLightingTips,
      child: Container(
        width: 16.w,
        height: 8.h,
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.5),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Center(
          child: CustomIconWidget(
            iconName: 'lightbulb_outline',
            color: Colors.white,
            size: 28,
          ),
        ),
      ),
    );
  }
}
