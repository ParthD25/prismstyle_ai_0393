import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

/// Widget for creating and sharing new outfit posts
/// Instagram-style layout: Photo first, then caption
class CreatePostWidget extends StatefulWidget {
  final Function(Map<String, dynamic>) onPostCreate;

  const CreatePostWidget({super.key, required this.onPostCreate});

  @override
  State<CreatePostWidget> createState() => _CreatePostWidgetState();
}

class _CreatePostWidgetState extends State<CreatePostWidget> {
  final TextEditingController _captionController = TextEditingController();
  final ImagePicker _imagePicker = ImagePicker();
  String? _selectedOutfitImage;
  File? _selectedPhotoFromGallery;
  String _visibility = 'friends'; // 'friends' or 'public'
  bool _isLoading = false;
  int _selectedTab = 0; // 0 = From Gallery, 1 = From Wardrobe

  final List<Map<String, dynamic>> _savedOutfits = [
    {
      "id": 1,
      "image": "https://images.unsplash.com/photo-1618397351187-ee6afd732119",
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
      "image": "https://images.unsplash.com/photo-1676284303477-7233d4fcdbd0",
      "semanticLabel":
          "Summer outfit with floral dress and sandals on white background",
      "name": "Summer Vibes",
    },
    {
      "id": 4,
      "image": "https://images.unsplash.com/photo-1726195222148-fc8a7e7f37fa",
      "semanticLabel":
          "Athletic wear with black leggings and sports top on mannequin",
      "name": "Gym Ready",
    },
  ];

  Future<void> _pickImageFromGallery() async {
    final XFile? image = await _imagePicker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 1080,
      maxHeight: 1080,
      imageQuality: 85,
    );
    if (image != null) {
      setState(() {
        _selectedPhotoFromGallery = File(image.path);
        _selectedOutfitImage = null; // Clear wardrobe selection
      });
    }
  }

  Future<void> _takePhoto() async {
    final XFile? image = await _imagePicker.pickImage(
      source: ImageSource.camera,
      maxWidth: 1080,
      maxHeight: 1080,
      imageQuality: 85,
    );
    if (image != null) {
      setState(() {
        _selectedPhotoFromGallery = File(image.path);
        _selectedOutfitImage = null; // Clear wardrobe selection
      });
    }
  }

  @override
  void dispose() {
    _captionController.dispose();
    super.dispose();
  }

  void _handlePostCreate() async {
    if (_selectedOutfitImage == null && _selectedPhotoFromGallery == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select a photo or outfit to share'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }

    setState(() => _isLoading = true);

    // Simulate post creation
    await Future.delayed(const Duration(seconds: 1));

    String? outfitLabel;
    if (_selectedOutfitImage != null) {
      final selectedOutfit = _savedOutfits.firstWhere(
        (outfit) => outfit["image"] == _selectedOutfitImage,
      );
      outfitLabel = selectedOutfit["semanticLabel"] as String;
    }

    widget.onPostCreate({
      "outfitImage": _selectedOutfitImage ?? _selectedPhotoFromGallery?.path,
      "outfitImageLabel": outfitLabel ?? 'User uploaded photo',
      "isLocalImage": _selectedPhotoFromGallery != null,
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
    final hasSelectedImage = _selectedOutfitImage != null || _selectedPhotoFromGallery != null;

    return Container(
      height: 90.h,
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
                Text('New Post', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
                TextButton(
                  onPressed: _isLoading || !hasSelectedImage ? null : _handlePostCreate,
                  child: _isLoading
                      ? SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: theme.colorScheme.primary,
                          ),
                        )
                      : Text(
                          'Share',
                          style: theme.textTheme.labelLarge?.copyWith(
                            color: hasSelectedImage ? theme.colorScheme.primary : theme.colorScheme.onSurface.withValues(alpha: 0.4),
                            fontWeight: FontWeight.w600,
                          ),
                        ),
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
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // PHOTO PREVIEW AREA (Instagram-style - Photo first!)
                  Container(
                    width: double.infinity,
                    height: 45.h,
                    color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
                    child: _buildPhotoPreview(theme),
                  ),

                  // Caption input (below photo like Instagram)
                  Padding(
                    padding: EdgeInsets.all(4.w),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // User avatar
                        CircleAvatar(
                          radius: 5.w,
                          backgroundColor: theme.colorScheme.primary.withValues(alpha: 0.2),
                          child: CustomIconWidget(
                            iconName: 'person',
                            color: theme.colorScheme.primary,
                            size: 24,
                          ),
                        ),
                        SizedBox(width: 3.w),
                        Expanded(
                          child: TextField(
                            controller: _captionController,
                            decoration: InputDecoration(
                              hintText: 'Write a caption...',
                              border: InputBorder.none,
                              hintStyle: theme.textTheme.bodyLarge?.copyWith(
                                color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
                              ),
                            ),
                            style: theme.textTheme.bodyLarge,
                            maxLines: 3,
                            maxLength: 200,
                          ),
                        ),
                      ],
                    ),
                  ),

                  Divider(color: theme.colorScheme.outline.withValues(alpha: 0.2)),

                  // Source selection tabs
                  Padding(
                    padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Choose Photo', style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600)),
                        SizedBox(height: 1.5.h),
                        
                        // Tab selection
                        Row(
                          children: [
                            Expanded(
                              child: _buildSourceTab(0, 'Gallery', Icons.photo_library_rounded, theme),
                            ),
                            SizedBox(width: 2.w),
                            Expanded(
                              child: _buildSourceTab(1, 'Wardrobe', Icons.checkroom_rounded, theme),
                            ),
                          ],
                        ),
                        
                        SizedBox(height: 2.h),
                        
                        // Content based on selected tab
                        _selectedTab == 0
                            ? _buildGalleryOptions(theme)
                            : _buildWardrobeGrid(theme),
                      ],
                    ),
                  ),

                  // Visibility settings
                  Padding(
                    padding: EdgeInsets.symmetric(horizontal: 4.w),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Visibility', style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600)),
                        SizedBox(height: 1.h),
                        _buildVisibilityOption('friends', 'Friends Only', Icons.group_rounded, theme),
                        _buildVisibilityOption('public', 'Public', Icons.public_rounded, theme),
                      ],
                    ),
                  ),
                  
                  SizedBox(height: 4.h),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPhotoPreview(ThemeData theme) {
    if (_selectedPhotoFromGallery != null) {
      return Stack(
        fit: StackFit.expand,
        children: [
          Image.file(
            _selectedPhotoFromGallery!,
            fit: BoxFit.cover,
          ),
          Positioned(
            top: 8,
            right: 8,
            child: GestureDetector(
              onTap: () => setState(() => _selectedPhotoFromGallery = null),
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.6),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.close, color: Colors.white, size: 20),
              ),
            ),
          ),
        ],
      );
    } else if (_selectedOutfitImage != null) {
      return Stack(
        fit: StackFit.expand,
        children: [
          CustomImageWidget(
            imageUrl: _selectedOutfitImage!,
            width: double.infinity,
            height: double.infinity,
            fit: BoxFit.cover,
            semanticLabel: 'Selected outfit',
          ),
          Positioned(
            top: 8,
            right: 8,
            child: GestureDetector(
              onTap: () => setState(() => _selectedOutfitImage = null),
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.6),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.close, color: Colors.white, size: 20),
              ),
            ),
          ),
        ],
      );
    }
    
    // Empty state
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          Icons.add_photo_alternate_rounded,
          size: 60,
          color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
        ),
        SizedBox(height: 2.h),
        Text(
          'Select a photo to share',
          style: theme.textTheme.bodyLarge?.copyWith(
            color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
          ),
        ),
      ],
    );
  }

  Widget _buildSourceTab(int index, String label, IconData icon, ThemeData theme) {
    final isSelected = _selectedTab == index;
    return GestureDetector(
      onTap: () => setState(() => _selectedTab = index),
      child: Container(
        padding: EdgeInsets.symmetric(vertical: 1.5.h),
        decoration: BoxDecoration(
          color: isSelected ? theme.colorScheme.primary : theme.colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 20,
              color: isSelected ? theme.colorScheme.onPrimary : theme.colorScheme.onSurface,
            ),
            SizedBox(width: 2.w),
            Text(
              label,
              style: theme.textTheme.labelLarge?.copyWith(
                color: isSelected ? theme.colorScheme.onPrimary : theme.colorScheme.onSurface,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildGalleryOptions(ThemeData theme) {
    return Row(
      children: [
        Expanded(
          child: GestureDetector(
            onTap: _pickImageFromGallery,
            child: Container(
              padding: EdgeInsets.symmetric(vertical: 3.h),
              decoration: BoxDecoration(
                border: Border.all(color: theme.colorScheme.outline.withValues(alpha: 0.3)),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                children: [
                  Icon(Icons.photo_library_rounded, size: 32, color: theme.colorScheme.primary),
                  SizedBox(height: 1.h),
                  Text('Choose from Album', style: theme.textTheme.labelMedium),
                ],
              ),
            ),
          ),
        ),
        SizedBox(width: 3.w),
        Expanded(
          child: GestureDetector(
            onTap: _takePhoto,
            child: Container(
              padding: EdgeInsets.symmetric(vertical: 3.h),
              decoration: BoxDecoration(
                border: Border.all(color: theme.colorScheme.outline.withValues(alpha: 0.3)),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                children: [
                  Icon(Icons.camera_alt_rounded, size: 32, color: theme.colorScheme.primary),
                  SizedBox(height: 1.h),
                  Text('Take Photo', style: theme.textTheme.labelMedium),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildWardrobeGrid(ThemeData theme) {
    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        crossAxisSpacing: 2.w,
        mainAxisSpacing: 2.w,
        childAspectRatio: 0.8,
      ),
      itemCount: _savedOutfits.length,
      itemBuilder: (context, index) {
        final outfit = _savedOutfits[index];
        final isSelected = _selectedOutfitImage == outfit["image"];

        return GestureDetector(
          onTap: () {
            setState(() {
              _selectedOutfitImage = outfit["image"] as String;
              _selectedPhotoFromGallery = null; // Clear gallery selection
            });
          },
          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: isSelected
                    ? theme.colorScheme.primary
                    : Colors.transparent,
                width: isSelected ? 3 : 0,
              ),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CustomImageWidget(
                    imageUrl: outfit["image"] as String,
                    width: double.infinity,
                    height: double.infinity,
                    fit: BoxFit.cover,
                    semanticLabel: outfit["semanticLabel"] as String,
                  ),
                  if (isSelected)
                    Container(
                      color: theme.colorScheme.primary.withValues(alpha: 0.3),
                      child: Center(
                        child: Container(
                          padding: const EdgeInsets.all(4),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.primary,
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            Icons.check,
                            color: theme.colorScheme.onPrimary,
                            size: 16,
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildVisibilityOption(String value, String label, IconData icon, ThemeData theme) {
    final isSelected = _visibility == value;
    return GestureDetector(
      onTap: () => setState(() => _visibility = value),
      child: Container(
        margin: EdgeInsets.only(bottom: 1.h),
        padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 1.5.h),
        decoration: BoxDecoration(
          color: isSelected ? theme.colorScheme.primary.withValues(alpha: 0.1) : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: isSelected ? theme.colorScheme.primary : theme.colorScheme.outline.withValues(alpha: 0.2),
          ),
        ),
        child: Row(
          children: [
            Icon(icon, size: 20, color: isSelected ? theme.colorScheme.primary : theme.colorScheme.onSurface),
            SizedBox(width: 3.w),
            Expanded(child: Text(label, style: theme.textTheme.bodyMedium)),
            if (isSelected)
              Icon(Icons.check_circle, size: 20, color: theme.colorScheme.primary),
          ],
        ),
      ),
    );
  }
}
