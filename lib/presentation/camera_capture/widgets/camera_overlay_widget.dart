import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';
import '../../../widgets/custom_icon_widget.dart';

class CameraOverlayWidget extends StatelessWidget {
  final String aiGuidance;
  final Color guidanceColor;
  final bool isFlashOn;
  final VoidCallback onFlashToggle;
  final VoidCallback onFlipCamera;
  final VoidCallback onClose;
  final bool showFlash;

  const CameraOverlayWidget({
    super.key,
    required this.aiGuidance,
    required this.guidanceColor,
    required this.isFlashOn,
    required this.onFlashToggle,
    required this.onFlipCamera,
    required this.onClose,
    this.showFlash = true,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _buildTopControls(context),
        const Spacer(),
        _buildFrameGuide(context),
        const Spacer(),
        _buildAIGuidance(context),
        SizedBox(height: 2.h),
      ],
    );
  }

  Widget _buildTopControls(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          if (showFlash)
            _buildControlButton(
              context,
              icon: isFlashOn ? 'flash_on' : 'flash_off',
              onTap: onFlashToggle,
            )
          else
            SizedBox(width: 12.w),
          _buildControlButton(
            context,
            icon: 'flip_camera_android',
            onTap: onFlipCamera,
          ),
          _buildControlButton(context, icon: 'close', onTap: onClose),
        ],
      ),
    );
  }

  Widget _buildControlButton(
    BuildContext context, {
    required String icon,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 12.w,
        height: 6.h,
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.5),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Center(
          child: CustomIconWidget(
            iconName: icon,
            color: Colors.white,
            size: 24,
          ),
        ),
      ),
    );
  }

  Widget _buildFrameGuide(BuildContext context) {
    return Container(
      width: 80.w,
      height: 50.h,
      decoration: BoxDecoration(
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.7),
          width: 2,
        ),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Stack(
        children: [
          _buildCornerIndicator(top: 0, left: 0),
          _buildCornerIndicator(top: 0, right: 0),
          _buildCornerIndicator(bottom: 0, left: 0),
          _buildCornerIndicator(bottom: 0, right: 0),
          Center(
            child: Container(
              width: 2,
              height: 20.h,
              color: Colors.white.withValues(alpha: 0.3),
            ),
          ),
          Center(
            child: Container(
              width: 40.w,
              height: 2,
              color: Colors.white.withValues(alpha: 0.3),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCornerIndicator({
    double? top,
    double? bottom,
    double? left,
    double? right,
  }) {
    return Positioned(
      top: top,
      bottom: bottom,
      left: left,
      right: right,
      child: Container(
        width: 8.w,
        height: 4.h,
        decoration: BoxDecoration(
          border: Border(
            top: top != null
                ? BorderSide(color: Colors.white, width: 3)
                : BorderSide.none,
            bottom: bottom != null
                ? BorderSide(color: Colors.white, width: 3)
                : BorderSide.none,
            left: left != null
                ? BorderSide(color: Colors.white, width: 3)
                : BorderSide.none,
            right: right != null
                ? BorderSide(color: Colors.white, width: 3)
                : BorderSide.none,
          ),
        ),
      ),
    );
  }

  Widget _buildAIGuidance(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 1.5.h),
      margin: EdgeInsets.symmetric(horizontal: 8.w),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.7),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          CustomIconWidget(
            iconName: 'auto_awesome',
            color: guidanceColor,
            size: 20,
          ),
          SizedBox(width: 2.w),
          Flexible(
            child: Text(
              aiGuidance,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: guidanceColor,
                fontWeight: FontWeight.w500,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }
}
