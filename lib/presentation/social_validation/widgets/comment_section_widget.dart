import 'package:emoji_picker_flutter/emoji_picker_flutter.dart' as emoji_picker;
import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

/// Widget for displaying and managing comments on outfit posts
/// Implements expandable comment section with emoji reactions
class CommentSectionWidget extends StatefulWidget {
  final List<Map<String, dynamic>> comments;
  final Function(String) onCommentSubmit;

  const CommentSectionWidget({
    super.key,
    required this.comments,
    required this.onCommentSubmit,
  });

  @override
  State<CommentSectionWidget> createState() => _CommentSectionWidgetState();
}

class _CommentSectionWidgetState extends State<CommentSectionWidget> {
  final TextEditingController _commentController = TextEditingController();
  final FocusNode _commentFocusNode = FocusNode();
  bool _showEmojiPicker = false;

  @override
  void dispose() {
    _commentController.dispose();
    _commentFocusNode.dispose();
    super.dispose();
  }

  void _submitComment() {
    if (_commentController.text.trim().isNotEmpty) {
      widget.onCommentSubmit(_commentController.text.trim());
      _commentController.clear();
      setState(() => _showEmojiPicker = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(16.0)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Handle bar
          Container(
            margin: EdgeInsets.symmetric(vertical: 1.h),
            width: 12.w,
            height: 4,
            decoration: BoxDecoration(
              color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.3),
              borderRadius: BorderRadius.circular(2),
            ),
          ),

          // Comments header
          Padding(
            padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 1.h),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Comments (${widget.comments.length})',
                  style: theme.textTheme.titleMedium,
                ),
                IconButton(
                  icon: CustomIconWidget(
                    iconName: 'close',
                    color: theme.colorScheme.onSurface,
                    size: 24,
                  ),
                  onPressed: () => Navigator.pop(context),
                ),
              ],
            ),
          ),

          Divider(
            height: 1,
            color: theme.colorScheme.outline.withValues(alpha: 0.2),
          ),

          // Comments list
          Expanded(
            child: widget.comments.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CustomIconWidget(
                          iconName: 'chat_bubble_outline',
                          color: theme.colorScheme.onSurfaceVariant.withValues(
                            alpha: 0.5,
                          ),
                          size: 48,
                        ),
                        SizedBox(height: 2.h),
                        Text(
                          'No comments yet',
                          style: theme.textTheme.bodyLarge?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                        SizedBox(height: 1.h),
                        Text(
                          'Be the first to comment!',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
                  )
                : ListView.separated(
                    padding: EdgeInsets.symmetric(
                      horizontal: 4.w,
                      vertical: 2.h,
                    ),
                    itemCount: widget.comments.length,
                    separatorBuilder: (context, index) => SizedBox(height: 2.h),
                    itemBuilder: (context, index) {
                      final comment = widget.comments[index];
                      return _buildCommentItem(context, comment);
                    },
                  ),
          ),

          // Emoji picker
          _showEmojiPicker
              ? SizedBox(
                  height: 30.h,
                  child: emoji_picker.EmojiPicker(
                    onEmojiSelected: (category, emoji) {
                      _commentController.text += emoji.emoji;
                    },
                    config: emoji_picker.Config(
                      height: 30.h,
                      checkPlatformCompatibility: true,
                      emojiViewConfig: emoji_picker.EmojiViewConfig(
                        emojiSizeMax: 28,
                        backgroundColor: theme.colorScheme.surface,
                      ),
                      categoryViewConfig: emoji_picker.CategoryViewConfig(
                        backgroundColor: theme.colorScheme.surface,
                        iconColor: theme.colorScheme.onSurfaceVariant,
                        iconColorSelected: theme.colorScheme.primary,
                      ),
                      bottomActionBarConfig: emoji_picker.BottomActionBarConfig(
                        backgroundColor: theme.colorScheme.surface,
                      ),
                    ),
                  ),
                )
              : const SizedBox.shrink(),

          // Comment input
          Container(
            padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 1.h),
            decoration: BoxDecoration(
              color: theme.colorScheme.surface,
              border: Border(
                top: BorderSide(
                  color: theme.colorScheme.outline.withValues(alpha: 0.2),
                  width: 1,
                ),
              ),
            ),
            child: SafeArea(
              child: Row(
                children: [
                  IconButton(
                    icon: CustomIconWidget(
                      iconName: _showEmojiPicker
                          ? 'keyboard'
                          : 'emoji_emotions_outlined',
                      color: theme.colorScheme.onSurface,
                      size: 24,
                    ),
                    onPressed: () {
                      setState(() => _showEmojiPicker = !_showEmojiPicker);
                      if (!_showEmojiPicker) {
                        _commentFocusNode.requestFocus();
                      }
                    },
                  ),
                  Expanded(
                    child: TextField(
                      controller: _commentController,
                      focusNode: _commentFocusNode,
                      decoration: InputDecoration(
                        hintText: 'Add a comment...',
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.symmetric(horizontal: 3.w),
                      ),
                      maxLines: null,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _submitComment(),
                      onTap: () {
                        if (_showEmojiPicker) {
                          setState(() => _showEmojiPicker = false);
                        }
                      },
                    ),
                  ),
                  IconButton(
                    icon: CustomIconWidget(
                      iconName: 'send',
                      color: _commentController.text.trim().isEmpty
                          ? theme.colorScheme.onSurfaceVariant.withValues(
                              alpha: 0.5,
                            )
                          : theme.colorScheme.primary,
                      size: 24,
                    ),
                    onPressed: _submitComment,
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCommentItem(BuildContext context, Map<String, dynamic> comment) {
    final theme = Theme.of(context);

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(16.0),
          child: CustomImageWidget(
            imageUrl: comment["userAvatar"] as String,
            width: 32,
            height: 32,
            fit: BoxFit.cover,
            semanticLabel: comment["userAvatarLabel"] as String,
          ),
        ),
        SizedBox(width: 3.w),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text(
                    comment["userName"] as String,
                    style: theme.textTheme.titleSmall,
                  ),
                  SizedBox(width: 2.w),
                  Text(
                    comment["timestamp"] as String,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
              SizedBox(height: 0.5.h),
              Text(
                comment["text"] as String,
                style: theme.textTheme.bodyMedium,
              ),
              SizedBox(height: 1.h),
              Row(
                children: [
                  InkWell(
                    onTap: () {},
                    child: Text(
                      'Like',
                      style: theme.textTheme.labelMedium?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ),
                  SizedBox(width: 4.w),
                  InkWell(
                    onTap: () {},
                    child: Text(
                      'Reply',
                      style: theme.textTheme.labelMedium?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ),
                  comment["likeCount"] != null &&
                          (comment["likeCount"] as int) > 0
                      ? Padding(
                          padding: EdgeInsets.only(left: 4.w),
                          child: Text(
                            '${comment["likeCount"]} likes',
                            style: theme.textTheme.labelSmall?.copyWith(
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                          ),
                        )
                      : const SizedBox.shrink(),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }
}