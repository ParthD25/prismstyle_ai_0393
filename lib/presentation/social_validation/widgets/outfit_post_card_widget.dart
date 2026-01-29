import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

/// Widget for displaying individual outfit posts in the social feed
/// Implements photo-first card design with voting and interaction features
class OutfitPostCardWidget extends StatefulWidget {
  final Map<String, dynamic> post;
  final VoidCallback onVote;
  final VoidCallback onComment;
  final VoidCallback onSave;

  const OutfitPostCardWidget({
    super.key,
    required this.post,
    required this.onVote,
    required this.onComment,
    required this.onSave,
  });

  @override
  State<OutfitPostCardWidget> createState() => _OutfitPostCardWidgetState();
}

class _OutfitPostCardWidgetState extends State<OutfitPostCardWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _scaleAnimation;
  int _currentVote = 0; // -1 for thumbs down, 0 for no vote, 1 for thumbs up

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 200),
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 1.0, end: 1.2).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  void _handleVote(int voteType) {
    setState(() {
      _currentVote = _currentVote == voteType ? 0 : voteType;
    });
    _animationController.forward().then((_) => _animationController.reverse());
    widget.onVote();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      margin: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
      elevation: 2.0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20.0)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // User header
          Padding(
            padding: EdgeInsets.all(3.w),
            child: Row(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(20.0),
                  child: CustomImageWidget(
                    imageUrl: widget.post["userAvatar"] as String,
                    width: 40,
                    height: 40,
                    fit: BoxFit.cover,
                    semanticLabel: widget.post["userAvatarLabel"] as String,
                  ),
                ),
                SizedBox(width: 3.w),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        widget.post["userName"] as String,
                        style: theme.textTheme.titleMedium,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      Text(
                        widget.post["timestamp"] as String,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: CustomIconWidget(
                    iconName: 'more_vert',
                    color: theme.colorScheme.onSurface,
                    size: 24,
                  ),
                  onPressed: () => _showContextMenu(context),
                ),
              ],
            ),
          ),

          // Outfit image with rounded corners
          ClipRRect(
            borderRadius: BorderRadius.circular(20.0),
            child: GestureDetector(
              onDoubleTap: () => _handleVote(1),
              child: CustomImageWidget(
                imageUrl: widget.post["outfitImage"] as String,
                width: double.infinity,
                height: 50.h,
                fit: BoxFit.cover,
                semanticLabel: widget.post["outfitImageLabel"] as String,
              ),
            ),
          ),

          // Caption
          widget.post["caption"] != null &&
                  (widget.post["caption"] as String).isNotEmpty
              ? Padding(
                  padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
                  child: Text(
                    widget.post["caption"] as String,
                    style: theme.textTheme.bodyMedium,
                    maxLines: 3,
                    overflow: TextOverflow.ellipsis,
                  ),
                )
              : const SizedBox.shrink(),

          // Interaction bar
          Padding(
            padding: EdgeInsets.symmetric(horizontal: 2.w, vertical: 1.h),
            child: Row(
              children: [
                // Thumbs up
                ScaleTransition(
                  scale: _currentVote == 1
                      ? _scaleAnimation
                      : const AlwaysStoppedAnimation(1.0),
                  child: IconButton(
                    icon: CustomIconWidget(
                      iconName: _currentVote == 1
                          ? 'thumb_up'
                          : 'thumb_up_outlined',
                      color: _currentVote == 1
                          ? theme.colorScheme.tertiary
                          : theme.colorScheme.onSurface,
                      size: 24,
                    ),
                    onPressed: () => _handleVote(1),
                  ),
                ),
                Text(
                  '${widget.post["thumbsUpCount"]}',
                  style: theme.textTheme.bodyMedium,
                ),
                SizedBox(width: 4.w),

                // Thumbs down
                ScaleTransition(
                  scale: _currentVote == -1
                      ? _scaleAnimation
                      : const AlwaysStoppedAnimation(1.0),
                  child: IconButton(
                    icon: CustomIconWidget(
                      iconName: _currentVote == -1
                          ? 'thumb_down'
                          : 'thumb_down_outlined',
                      color: _currentVote == -1
                          ? theme.colorScheme.error
                          : theme.colorScheme.onSurface,
                      size: 24,
                    ),
                    onPressed: () => _handleVote(-1),
                  ),
                ),
                Text(
                  '${widget.post["thumbsDownCount"]}',
                  style: theme.textTheme.bodyMedium,
                ),
                SizedBox(width: 4.w),

                // Comments
                IconButton(
                  icon: CustomIconWidget(
                    iconName: 'chat_bubble_outline',
                    color: theme.colorScheme.onSurface,
                    size: 24,
                  ),
                  onPressed: widget.onComment,
                ),
                Text(
                  '${widget.post["commentCount"]}',
                  style: theme.textTheme.bodyMedium,
                ),

                const Spacer(),

                // Share to social media button
                IconButton(
                  icon: CustomIconWidget(
                    iconName: 'share',
                    color: theme.colorScheme.onSurface,
                    size: 24,
                  ),
                  onPressed: () => _showShareOptions(context),
                ),

                // Save
                IconButton(
                  icon: CustomIconWidget(
                    iconName: 'bookmark_border',
                    color: theme.colorScheme.onSurface,
                    size: 24,
                  ),
                  onPressed: widget.onSave,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _showShareOptions(BuildContext context) {
    final theme = Theme.of(context);
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20.0)),
      ),
      builder: (context) => Container(
        padding: EdgeInsets.symmetric(vertical: 3.h, horizontal: 4.w),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Share to Social Media',
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            SizedBox(height: 3.h),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildShareOption(
                  context,
                  'Instagram',
                  Icons.camera_alt,
                  theme.colorScheme.primary,
                ),
                _buildShareOption(
                  context,
                  'Facebook',
                  Icons.facebook,
                  const Color(0xFF1877F2),
                ),
                _buildShareOption(
                  context,
                  'Twitter',
                  Icons.tag,
                  const Color(0xFF1DA1F2),
                ),
                _buildShareOption(
                  context,
                  'Pinterest',
                  Icons.help_outline,
                  const Color(0xFFE60023),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildShareOption(
    BuildContext context,
    String platform,
    IconData icon,
    Color color,
  ) {
    return InkWell(
      onTap: () {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Sharing to $platform...'),
            duration: const Duration(seconds: 2),
          ),
        );
      },
      borderRadius: BorderRadius.circular(12.0),
      child: Container(
        padding: EdgeInsets.all(3.w),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: EdgeInsets.all(3.w),
              decoration: BoxDecoration(
                color: color.withAlpha(26),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, color: color, size: 28),
            ),
            SizedBox(height: 1.h),
            Text(platform, style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      ),
    );
  }

  void _showContextMenu(BuildContext context) {
    final theme = Theme.of(context);
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16.0)),
      ),
      builder: (context) => Container(
        padding: EdgeInsets.symmetric(vertical: 2.h),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: CustomIconWidget(
                iconName: 'bookmark_border',
                color: theme.colorScheme.onSurface,
                size: 24,
              ),
              title: Text('Save Outfit', style: theme.textTheme.bodyLarge),
              onTap: () {
                Navigator.pop(context);
                widget.onSave();
              },
            ),
            ListTile(
              leading: CustomIconWidget(
                iconName: 'flag_outlined',
                color: theme.colorScheme.error,
                size: 24,
              ),
              title: Text('Report', style: theme.textTheme.bodyLarge),
              onTap: () {
                Navigator.pop(context);
                _showReportDialog(context);
              },
            ),
            ListTile(
              leading: CustomIconWidget(
                iconName: 'visibility_off_outlined',
                color: theme.colorScheme.onSurface,
                size: 24,
              ),
              title: Text('Hide User', style: theme.textTheme.bodyLarge),
              onTap: () {
                Navigator.pop(context);
                _showHideUserDialog(context);
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showReportDialog(BuildContext context) {
    final theme = Theme.of(context);
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Report Post', style: theme.textTheme.titleLarge),
        content: Text(
          'Are you sure you want to report this post? Our team will review it.',
          style: theme.textTheme.bodyMedium,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancel', style: theme.textTheme.labelLarge),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text('Post reported successfully'),
                  duration: const Duration(seconds: 2),
                ),
              );
            },
            child: Text('Report'),
          ),
        ],
      ),
    );
  }

  void _showHideUserDialog(BuildContext context) {
    final theme = Theme.of(context);
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Hide User', style: theme.textTheme.titleLarge),
        content: Text(
          'You will no longer see posts from ${widget.post["userName"]}. You can undo this in settings.',
          style: theme.textTheme.bodyMedium,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancel', style: theme.textTheme.labelLarge),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text('User hidden successfully'),
                  duration: const Duration(seconds: 2),
                ),
              );
            },
            child: Text('Hide'),
          ),
        ],
      ),
    );
  }
}
