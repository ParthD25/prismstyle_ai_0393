import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../core/app_export.dart';
import '../../widgets/custom_icon_widget.dart';
import './widgets/metrics_card_widget.dart';
import './widgets/profile_header_widget.dart';
import './widgets/settings_section_widget.dart';
import './widgets/style_preferences_widget.dart';

/// User Profile screen for comprehensive account management and personalization
/// Implements bottom tab navigation with Profile tab active
class UserProfile extends StatefulWidget {
  const UserProfile({super.key});

  @override
  State<UserProfile> createState() => _UserProfileState();
}

class _UserProfileState extends State<UserProfile> {
  // Mock user data
  final Map<String, dynamic> userData = {
    "name": "Emma Rodriguez",
    "email": "emma.rodriguez@example.com",
    "profilePhoto":
        "https://img.rocket.new/generatedImages/rocket_gen_img_1ef28082b-1763294062175.png",
    "semanticLabel":
        "Profile photo of a woman with long brown hair wearing a white blouse, smiling at the camera",
    "wardrobeCount": 127,
    "outfitsCreated": 43,
    "validationScore": 4.8,
    "stylePreferences": ["Casual", "Minimalist", "Streetwear", "Vintage"],
    "notificationsEnabled": true,
    "weatherAlertsEnabled": true,
    "socialSharingEnabled": false,
    "biometricEnabled": false,
    "measurementUnit": "Imperial",
    "weatherLocation": "San Francisco, CA",
  };

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            // Custom AppBar content
            Container(
              padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 1.5.h),
              decoration: BoxDecoration(
                color: theme.scaffoldBackgroundColor,
                border: Border(
                  bottom: BorderSide(
                    color: theme.colorScheme.outline.withValues(alpha: 0.2),
                    width: 1,
                  ),
                ),
              ),
              child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Profile',
                  style: theme.textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                IconButton(
                  icon: CustomIconWidget(
                    iconName: 'settings',
                    color: theme.colorScheme.onSurface,
                    size: 24,
                  ),
                  onPressed: () => _showSettingsSheet(context),
                ),
              ],
            ),
            ),

            // Scrollable content
            Expanded(
          child: SingleChildScrollView(
            physics: const BouncingScrollPhysics(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Profile Header
                ProfileHeaderWidget(userData: userData),

                SizedBox(height: 2.h),

                // Metrics Cards
                MetricsCardWidget(userData: userData),

                SizedBox(height: 3.h),

                // Style Preferences Section
                StylePreferencesWidget(userData: userData),

                SizedBox(height: 2.h),

                // Privacy Controls Section
                SettingsSectionWidget(
                  title: 'Privacy Controls',
                  items: [
                    {
                      'icon': 'share',
                      'title': 'Social Sharing',
                      'subtitle': 'Control outfit sharing permissions',
                      'onTap': () => _navigateToSocialSettings(),
                    },
                    {
                      'icon': 'data_usage',
                      'title': 'Data Usage',
                      'subtitle': 'Manage your data preferences',
                      'onTap': () => _navigateToDataSettings(),
                    },
                    {
                      'icon': 'fingerprint',
                      'title': 'Biometric Authentication',
                      'subtitle': userData['biometricEnabled'] == true
                          ? 'Enabled'
                          : 'Disabled',
                      'onTap': () => _toggleBiometric(),
                    },
                  ],
                ),

                SizedBox(height: 2.h),

                // App Preferences Section
                SettingsSectionWidget(
                  title: 'App Preferences',
                  items: [
                    {
                      'icon': 'straighten',
                      'title': 'Measurement Units',
                      'subtitle': userData['measurementUnit'] as String,
                      'onTap': () => _navigateToMeasurementSettings(),
                    },
                    {
                      'icon': 'location_on',
                      'title': 'Weather Location',
                      'subtitle': userData['weatherLocation'] as String,
                      'onTap': () => _navigateToLocationSettings(),
                    },
                  ],
                ),

                SizedBox(height: 3.h),

                // Sign Out Button
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 4.w),
                  child: SizedBox(
                    width: double.infinity,
                    child: OutlinedButton(
                      onPressed: () => _showSignOutDialog(),
                      style: OutlinedButton.styleFrom(
                        padding: EdgeInsets.symmetric(vertical: 2.h),
                        side: BorderSide(
                          color: theme.colorScheme.error,
                          width: 1.5,
                        ),
                      ),
                      child: Text(
                        'Sign Out',
                        style: theme.textTheme.labelLarge?.copyWith(
                          color: theme.colorScheme.error,
                        ),
                      ),
                    ),
                  ),
                ),

                SizedBox(height: 2.h),

                // Delete Account Button
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 4.w),
                  child: TextButton(
                    onPressed: () => _showDeleteAccountDialog(),
                    child: Text(
                      'Delete Account',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: theme.colorScheme.error.withValues(alpha: 0.7),
                        decoration: TextDecoration.underline,
                      ),
                    ),
                  ),
                ),

                SizedBox(height: 4.h),
              ],
            ),
          ),
        ),
          ],
        ),
      ),
    );
  }

  void _navigateToEmailSettings() {
    // Navigate to email settings
  }

  void _navigateToPasswordSettings() {
    // Navigate to password settings
  }

  void _navigateToNotificationSettings() {
    // Navigate to notification settings
  }

  void _navigateToSocialSettings() {
    // Navigate to social sharing settings
  }

  void _navigateToDataSettings() {
    // Navigate to data usage settings
  }

  void _toggleBiometric() {
    setState(() {
      userData['biometricEnabled'] = !(userData['biometricEnabled'] as bool);
    });
  }

  void _navigateToMeasurementSettings() {
    // Navigate to measurement unit settings
  }

  void _navigateToLocationSettings() {
    // Navigate to weather location settings
  }

  void _showSettingsSheet(BuildContext context) {
    final theme = Theme.of(context);
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: theme.colorScheme.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Container(
        padding: EdgeInsets.all(4.w),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Handle bar
            Center(
              child: Container(
                width: 12.w,
                height: 4,
                margin: EdgeInsets.only(bottom: 2.h),
                decoration: BoxDecoration(
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            Text(
              'Account Settings',
              style: theme.textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            SizedBox(height: 2.h),
            // Settings items
            _buildSettingsTile(
              context,
              Icons.email_outlined,
              'Email',
              userData['email'] as String,
              _navigateToEmailSettings,
            ),
            _buildSettingsTile(
              context,
              Icons.lock_outline,
              'Password',
              'Change your password',
              _navigateToPasswordSettings,
            ),
            _buildSettingsTile(
              context,
              Icons.notifications_outlined,
              'Notifications',
              'Manage notification preferences',
              _navigateToNotificationSettings,
            ),
            SizedBox(height: 2.h),
          ],
        ),
      ),
    );
  }

  Widget _buildSettingsTile(
    BuildContext context,
    IconData icon,
    String title,
    String subtitle,
    VoidCallback onTap,
  ) {
    final theme = Theme.of(context);
    return ListTile(
      contentPadding: EdgeInsets.symmetric(horizontal: 2.w),
      leading: Icon(icon, color: theme.colorScheme.onSurface, size: 24),
      title: Text(title, style: theme.textTheme.bodyLarge),
      subtitle: Text(
        subtitle,
        style: theme.textTheme.bodySmall?.copyWith(
          color: theme.colorScheme.onSurfaceVariant,
        ),
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
      ),
      trailing: Icon(
        Icons.chevron_right,
        color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
        size: 24,
      ),
      onTap: () {
        Navigator.pop(context);
        onTap();
      },
    );
  }

  void _showSignOutDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        final theme = Theme.of(context);
        return AlertDialog(
          title: Text('Sign Out', style: theme.textTheme.titleLarge),
          content: Text(
            'Are you sure you want to sign out?',
            style: theme.textTheme.bodyMedium,
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop();
                Navigator.of(
                  context,
                  rootNavigator: true,
                ).pushNamed('/splash-screen');
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: theme.colorScheme.error,
              ),
              child: Text('Sign Out'),
            ),
          ],
        );
      },
    );
  }

  void _showDeleteAccountDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        final theme = Theme.of(context);
        return AlertDialog(
          title: Text(
            'Delete Account',
            style: theme.textTheme.titleLarge?.copyWith(
              color: theme.colorScheme.error,
            ),
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'This action cannot be undone. All your data will be permanently deleted.',
                style: theme.textTheme.bodyMedium,
              ),
              SizedBox(height: 2.h),
              Text(
                'Are you absolutely sure?',
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop();
                _showDeleteConfirmationDialog();
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: theme.colorScheme.error,
              ),
              child: Text('Delete'),
            ),
          ],
        );
      },
    );
  }

  void _showDeleteConfirmationDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        final theme = Theme.of(context);
        return AlertDialog(
          title: Text(
            'Final Confirmation',
            style: theme.textTheme.titleLarge?.copyWith(
              color: theme.colorScheme.error,
            ),
          ),
          content: Text(
            'Type "DELETE" to confirm account deletion.',
            style: theme.textTheme.bodyMedium,
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop();
                Navigator.of(
                  context,
                  rootNavigator: true,
                ).pushNamed('/splash-screen');
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: theme.colorScheme.error,
              ),
              child: Text('Confirm Delete'),
            ),
          ],
        );
      },
    );
  }
}
