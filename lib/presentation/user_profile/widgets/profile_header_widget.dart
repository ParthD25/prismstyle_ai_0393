import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

/// Profile header widget displaying user's profile photo, name, and style statistics
class ProfileHeaderWidget extends StatelessWidget {
  final Map<String, dynamic> userData;

  const ProfileHeaderWidget({super.key, required this.userData});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: EdgeInsets.all(4.w),
      child: Column(
        children: [
          // Profile Photo
          Container(
            width: 25.w,
            height: 25.w,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: theme.colorScheme.primary, width: 3),
            ),
            child: ClipOval(
              child: CustomImageWidget(
                imageUrl: userData['profilePhoto'] as String,
                width: 25.w,
                height: 25.w,
                fit: BoxFit.cover,
                semanticLabel: userData['semanticLabel'] as String,
              ),
            ),
          ),

          SizedBox(height: 2.h),

          // User Name
          Text(
            userData['name'] as String,
            style: theme.textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.w700,
            ),
            textAlign: TextAlign.center,
          ),

          SizedBox(height: 0.5.h),

          // Email
          Text(
            userData['email'] as String,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
            textAlign: TextAlign.center,
          ),

          SizedBox(height: 2.h),

          // Edit Profile Button
          OutlinedButton.icon(
            onPressed: () {
              // Navigate to edit profile
            },
            icon: CustomIconWidget(
              iconName: 'edit',
              color: theme.colorScheme.primary,
              size: 18,
            ),
            label: Text('Edit Profile'),
            style: OutlinedButton.styleFrom(
              padding: EdgeInsets.symmetric(horizontal: 6.w, vertical: 1.5.h),
            ),
          ),
        ],
      ),
    );
  }
}
