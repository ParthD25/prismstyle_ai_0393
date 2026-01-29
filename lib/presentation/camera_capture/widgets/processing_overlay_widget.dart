import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

class ProcessingOverlayWidget extends StatelessWidget {
  const ProcessingOverlayWidget({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      color: Colors.black.withValues(alpha: 0.8),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(
              width: 20.w,
              height: 10.h,
              child: CircularProgressIndicator(
                color: theme.colorScheme.tertiary,
                strokeWidth: 4,
              ),
            ),
            SizedBox(height: 3.h),
            Text(
              'Processing Image',
              style: theme.textTheme.titleLarge?.copyWith(color: Colors.white),
            ),
            SizedBox(height: 1.h),
            Text(
              'Analyzing clothing attributes...',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: Colors.white.withValues(alpha: 0.7),
              ),
            ),
            SizedBox(height: 3.h),
            Container(
              padding: EdgeInsets.symmetric(horizontal: 8.w),
              child: Column(
                children: [
                  _buildProcessingStep(context, 'Background removal', true),
                  SizedBox(height: 1.h),
                  _buildProcessingStep(context, 'Color detection', true),
                  SizedBox(height: 1.h),
                  _buildProcessingStep(context, 'Pattern recognition', false),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProcessingStep(
    BuildContext context,
    String label,
    bool isComplete,
  ) {
    final theme = Theme.of(context);
    return Row(
      children: [
        Icon(
          isComplete ? Icons.check_circle : Icons.radio_button_unchecked,
          color: isComplete
              ? theme.colorScheme.tertiary
              : Colors.white.withValues(alpha: 0.5),
          size: 20,
        ),
        SizedBox(width: 2.w),
        Text(
          label,
          style: theme.textTheme.bodyMedium?.copyWith(
            color: isComplete
                ? Colors.white
                : Colors.white.withValues(alpha: 0.5),
          ),
        ),
      ],
    );
  }
}
