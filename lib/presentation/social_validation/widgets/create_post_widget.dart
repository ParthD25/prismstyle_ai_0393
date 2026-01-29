import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

/// Widget for creating and sharing new outfit posts
/// Implements outfit selection, caption input, and visibility settings
class CreatePostWidget extends StatefulWidget {
  final Function(Map<String, dynamic>) onPostCreate;

  const CreatePostWidget({super.key, required this.onPostCreate});

  @override
  State<CreatePostWidget> createState() => _CreatePostWidgetState();
}

class _CreatePostWidgetState extends State<CreatePostWidget> {
  final TextEditingController _captionController = TextEditingController();
  String? _selectedOutfitImage;
  String _visibility = 'friends'; // 'friends' or 'public'
  bool _isLoading = false;

  final List<Map<String, dynamic>> _savedOutfits = [
    {
      "id": 1,
      "image":
          "https://images.unsplash.com/photo-1618397351187-ee6afd732119",
      "semanticLabel":
          "Casual outfit with white t-shirt and blue jeans laid out on wooden floor",
      "name": "Casual Friday",
    },
    {
      "id": 2,
      "image":
          "https://img.rocket.new/generatedImages/rocket_gen_img_11c7650d6-1764746226516.png",
      "semanticLabel":
          "Formal business outfit with navy blazer and dress pants on hanger",
      "name": "Business Meeting",
    },
    {
      "id": 3,
      "image":
          "https://images.unsplash.com/photo-1676284303477-7233d4fcdbd0",
      "semanticLabel":
          "Summer outfit with floral dress and sandals on white background",
      "name": "Summer Vibes",
    },
    {
      "id": 4,
      "image":
          "https://images.unsplash.com/photo-1726195222148-fc8a7e7f37fa",
      "semanticLabel":
          "Athletic wear with black leggings and sports top on mannequin",
      "name": "Gym Ready",
    },
  ];

  @override
  void dispose() {
    _captionController.dispose();
    super.dispose();
  }

  void _handlePostCreate() async {
    if (_selectedOutfitImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select an outfit to share'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }

    setState(() => _isLoading = true);

    // Simulate post creation
    await Future.delayed(const Duration(seconds: 1));

    final selectedOutfit = _savedOutfits.firstWhere(
      (outfit) => outfit["image"] == _selectedOutfitImage,
    );

    widget.onPostCreate({
      "outfitImage": _selectedOutfitImage,
      "outfitImageLabel": selectedOutfit["semanticLabel"],
      "caption": _captionController.text.trim(),
      "visibility": _visibility,
      "timestamp": DateTime.now(),
    });

    setState(() => _isLoading = false);

    if (mounted) {
      Navigator.pop(context);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Outfit posted successfully!'),
          duration: Duration(seconds: 2),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      height: 85.h,
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(16.0)),
      ),
      child: Column(
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

          // Header
          Padding(
            padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 1.h),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: Text('Cancel', style: theme.textTheme.labelLarge),
                ),
                Text('Create Post', style: theme.textTheme.titleMedium),
                TextButton(
                  onPressed: _isLoading ? null : _handlePostCreate,
                  child: _isLoading
                      ? SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: theme.colorScheme.primary,
                          ),
                        )
                      : Text('Post', style: theme.textTheme.labelLarge),
                ),
              ],
            ),
          ),

          Divider(
            height: 1,
            color: theme.colorScheme.outline.withValues(alpha: 0.2),
          ),

          Expanded(
            child: SingleChildScrollView(
              padding: EdgeInsets.all(4.w),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Caption input
                  TextField(
                    controller: _captionController,
                    decoration: InputDecoration(
                      hintText: 'Write a caption...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8.0),
                      ),
                      contentPadding: EdgeInsets.all(4.w),
                    ),
                    maxLines: 3,
                    maxLength: 200,
                  ),

                  SizedBox(height: 3.h),

                  // Outfit selection
                  Text('Select Outfit', style: theme.textTheme.titleMedium),
                  SizedBox(height: 2.h),

                  GridView.builder(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 2,
                      crossAxisSpacing: 3.w,
                      mainAxisSpacing: 2.h,
                      childAspectRatio: 0.75,
                    ),
                    itemCount: _savedOutfits.length,
                    itemBuilder: (context, index) {
                      final outfit = _savedOutfits[index];
                      final isSelected =
                          _selectedOutfitImage == outfit["image"];

                      return GestureDetector(
                        onTap: () {
                          setState(() {
                            _selectedOutfitImage = outfit["image"] as String;
                          });
                        },
                        child: Container(
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(12.0),
                            border: Border.all(
                              color: isSelected
                                  ? theme.colorScheme.primary
                                  : theme.colorScheme.outline.withValues(
                                      alpha: 0.3,
                                    ),
                              width: isSelected ? 3 : 1,
                            ),
                          ),
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(12.0),
                            child: Stack(
                              children: [
                                CustomImageWidget(
                                  imageUrl: outfit["image"] as String,
                                  width: double.infinity,
                                  height: double.infinity,
                                  fit: BoxFit.cover,
                                  semanticLabel:
                                      outfit["semanticLabel"] as String,
                                ),
                                isSelected
                                    ? Positioned(
                                        top: 8,
                                        right: 8,
                                        child: Container(
                                          padding: const EdgeInsets.all(4),
                                          decoration: BoxDecoration(
                                            color: theme.colorScheme.primary,
                                            shape: BoxShape.circle,
                                          ),
                                          child: CustomIconWidget(
                                            iconName: 'check',
                                            color: theme.colorScheme.onPrimary,
                                            size: 16,
                                          ),
                                        ),
                                      )
                                    : const SizedBox.shrink(),
                                Positioned(
                                  bottom: 0,
                                  left: 0,
                                  right: 0,
                                  child: Container(
                                    padding: EdgeInsets.all(2.w),
                                    decoration: BoxDecoration(
                                      gradient: LinearGradient(
                                        begin: Alignment.bottomCenter,
                                        end: Alignment.topCenter,
                                        colors: [
                                          Colors.black.withValues(alpha: 0.7),
                                          Colors.transparent,
                                        ],
                                      ),
                                    ),
                                    child: Text(
                                      outfit["name"] as String,
                                      style: theme.textTheme.labelMedium
                                          ?.copyWith(color: Colors.white),
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),

                  SizedBox(height: 3.h),

                  // Visibility settings
                  Text('Visibility', style: theme.textTheme.titleMedium),
                  SizedBox(height: 2.h),

                  Container(
                    decoration: BoxDecoration(
                      border: Border.all(
                        color: theme.colorScheme.outline.withValues(alpha: 0.3),
                      ),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                    child: Column(
                      children: [
                        RadioListTile<String>(
                          value: 'friends',
                          groupValue: _visibility,
                          onChanged: (value) {
                            setState(() => _visibility = value!);
                          },
                          title: Text(
                            'Friends Only',
                            style: theme.textTheme.bodyLarge,
                          ),
                          subtitle: Text(
                            'Only your friends can see this post',
                            style: theme.textTheme.bodySmall,
                          ),
                        ),
                        Divider(
                          height: 1,
                          color: theme.colorScheme.outline.withValues(
                            alpha: 0.2,
                          ),
                        ),
                        RadioListTile<String>(
                          value: 'public',
                          groupValue: _visibility,
                          onChanged: (value) {
                            setState(() => _visibility = value!);
                          },
                          title: Text(
                            'Public',
                            style: theme.textTheme.bodyLarge,
                          ),
                          subtitle: Text(
                            'Anyone can see this post',
                            style: theme.textTheme.bodySmall,
                          ),
                        ),
                      ],
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
}
