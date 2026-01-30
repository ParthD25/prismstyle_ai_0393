import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';
import 'package:share_plus/share_plus.dart';

import '../../widgets/custom_app_bar.dart';
import '../../widgets/custom_icon_widget.dart';

/// Privacy Policy Screen for App Store Compliance
/// Source: App Privacy Policy Generator - https://privacypolicygenerator.info
class PrivacyPolicyScreen extends StatelessWidget {
  const PrivacyPolicyScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: PreferredSize(
        preferredSize: Size.fromHeight(7.h),
        child: CustomAppBar(
          title: 'Privacy Policy',
          variant: CustomAppBarVariant.withBack,
          actions: [
            IconButton(
              icon: CustomIconWidget(
                iconName: 'share',
                color: theme.colorScheme.onSurface,
                size: 24,
              ),
              onPressed: () => _sharePrivacyPolicy(context),
            ),
          ],
        ),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(4.w),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(theme),
            SizedBox(height: 3.h),
            _buildSection(
              theme,
              'Information We Collect',
              _informationCollected,
            ),
            _buildSection(theme, 'How We Use Your Information', _howWeUseInfo),
            _buildSection(theme, 'Data Storage and Security', _dataStorage),
            _buildSection(theme, 'Third-Party Services', _thirdPartyServices),
            _buildSection(theme, 'Your Rights', _yourRights),
            _buildSection(theme, 'Camera and Photo Access', _cameraAccess),
            _buildSection(theme, 'Location Data', _locationData),
            _buildSection(theme, 'Push Notifications', _pushNotifications),
            _buildSection(theme, 'Children\'s Privacy', _childrenPrivacy),
            _buildSection(theme, 'Changes to This Policy', _policyChanges),
            _buildSection(theme, 'Contact Us', _contactUs),
            SizedBox(height: 4.h),
            _buildFooter(theme),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(ThemeData theme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'PrismStyle AI Privacy Policy',
          style: theme.textTheme.headlineMedium?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        SizedBox(height: 1.h),
        Container(
          padding: EdgeInsets.symmetric(horizontal: 3.w, vertical: 1.h),
          decoration: BoxDecoration(
            color: theme.colorScheme.primaryContainer.withValues(alpha: 0.3),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            'Last Updated: January 2026',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.primary,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
        SizedBox(height: 2.h),
        Text(
          'Welcome to PrismStyle AI. We are committed to protecting your privacy '
          'and ensuring you have a positive experience using our app. This policy '
          'outlines how we collect, use, and protect your personal information.',
          style: theme.textTheme.bodyLarge,
        ),
      ],
    );
  }

  Widget _buildSection(ThemeData theme, String title, String content) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(height: 2.h),
        Text(
          title,
          style: theme.textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.bold,
            color: theme.colorScheme.primary,
          ),
        ),
        SizedBox(height: 1.h),
        Text(content, style: theme.textTheme.bodyMedium?.copyWith(height: 1.6)),
      ],
    );
  }

  Widget _buildFooter(ThemeData theme) {
    return Container(
      padding: EdgeInsets.all(4.w),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: theme.colorScheme.outline.withValues(alpha: 0.2),
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              CustomIconWidget(
                iconName: 'verified_user',
                color: theme.colorScheme.primary,
                size: 24,
              ),
              SizedBox(width: 2.w),
              Expanded(
                child: Text(
                  'Your data is protected',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          SizedBox(height: 1.h),
          Text(
            'PrismStyle AI uses industry-standard security measures to protect '
            'your personal information. We never sell your data to third parties.',
            style: theme.textTheme.bodySmall,
          ),
        ],
      ),
    );
  }

  void _sharePrivacyPolicy(BuildContext context) {
    Share.share(
      'PrismStyle AI Privacy Policy\n\n'
      'Read our full privacy policy at: https://prismstyle.ai/privacy',
      subject: 'PrismStyle AI Privacy Policy',
    );
  }

  // Privacy Policy Content Sections

  static const String _informationCollected = '''
We collect the following types of information:

• Account Information: Email address, name, and profile preferences when you create an account.

• Wardrobe Data: Photos of clothing items you upload, along with AI-detected attributes (color, category, pattern, style).

• Usage Data: How you interact with the app, including outfit combinations, saved items, and feature usage.

• Location Data: With your permission, we access your location to provide weather-based outfit recommendations.

• Device Information: Device type, operating system, and app version for improving compatibility.''';

  static const String _howWeUseInfo = '''
Your information helps us:

• Generate personalized outfit recommendations based on your wardrobe and preferences.

• Provide weather-appropriate styling suggestions using your location data.

• Improve our AI algorithms to better categorize and match clothing items.

• Enable social features like sharing outfits and receiving feedback from friends.

• Send relevant notifications about outfit suggestions, weather changes, and style tips.

• Analyze usage patterns to enhance app performance and user experience.''';

  static const String _dataStorage = '''
We prioritize the security of your data:

• All data is stored securely on Supabase cloud infrastructure with encryption at rest and in transit.

• Images are stored in secure cloud storage with access controls.

• We use industry-standard SSL/TLS encryption for all data transmission.

• Regular security audits and updates ensure ongoing protection.

• You can request deletion of your data at any time through the app settings.''';

  static const String _thirdPartyServices = '''
We integrate with trusted third-party services:

• Supabase: Database and authentication services (PostgreSQL-based).

• Open-Meteo: Weather data for outfit recommendations (free, no tracking).

• TensorFlow Lite: On-device machine learning for clothing classification.

• Apple Vision Framework: iOS native image classification (on-device only).

• Apple Core ML: iOS native machine learning (on-device only).

These services have their own privacy policies, and we recommend reviewing them. All AI processing happens on your device - your photos are never sent to external servers for classification.''';

  static const String _yourRights = '''
You have the right to:

• Access: Request a copy of all personal data we hold about you.

• Rectification: Update or correct your personal information.

• Deletion: Request deletion of your account and associated data.

• Portability: Export your wardrobe data in a standard format.

• Opt-out: Disable notifications and location services at any time.

• Withdraw Consent: Revoke any permissions previously granted.

To exercise these rights, contact us at privacy@prismstyle.ai or use the in-app settings.''';

  static const String _cameraAccess = '''
Camera and photo library access is used to:

• Capture photos of clothing items to add to your digital wardrobe.

• Import existing photos from your gallery for AI analysis.

• Take outfit photos for sharing with friends.

This access is optional and can be revoked at any time in your device settings. Photos are processed on-device when possible, and uploaded images are stored securely.''';

  static const String _locationData = '''
Location access enables:

• Real-time weather data for outfit recommendations.

• Localized style trends and suggestions.

• Weather-based notifications about appropriate clothing choices.

Location data is:
• Used only when the app is active (no background tracking).
• Cached locally to minimize requests.
• Never shared with advertisers or third parties.
• Completely optional - the app works without location access.''';

  static const String _pushNotifications = '''
Push notifications inform you about:

• Daily outfit suggestions based on weather and schedule.

• Weather changes that may affect your outfit choice.

• Social interactions (comments, likes on shared outfits).

• Style tips and fashion trends.

You can customize notification preferences or disable them entirely in the app settings or through your device's notification settings.''';

  static const String _childrenPrivacy = '''
PrismStyle AI is not intended for children under 13 years of age:

• We do not knowingly collect personal information from children.

• If you believe a child has provided us with personal information, please contact us immediately.

• We will take steps to delete such information from our servers.

Parents and guardians should supervise their children's online activities.''';

  static const String _policyChanges = '''
We may update this Privacy Policy periodically:

• Changes will be posted in the app and on our website.

• We will notify you of significant changes via email or in-app notification.

• Continued use of the app after changes constitutes acceptance of the new policy.

• We recommend reviewing this policy regularly for updates.''';

  static const String _contactUs = '''
For privacy-related questions or concerns:

Email: privacy@prismstyle.ai
Support: support@prismstyle.ai

Mailing Address:
PrismStyle AI
Privacy Team
[Your Company Address]

We aim to respond to all privacy inquiries within 48 hours.''';
}
