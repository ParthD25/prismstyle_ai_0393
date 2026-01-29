import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../../core/app_export.dart';

class ItemDetailsWidget extends StatefulWidget {
  final XFile capturedImage;
  final Map<String, dynamic> detectedAttributes;
  final VoidCallback onConfirm;
  final VoidCallback onRetake;

  const ItemDetailsWidget({
    super.key,
    required this.capturedImage,
    required this.detectedAttributes,
    required this.onConfirm,
    required this.onRetake,
  });

  @override
  State<ItemDetailsWidget> createState() => _ItemDetailsWidgetState();
}

class _ItemDetailsWidgetState extends State<ItemDetailsWidget> {
  late String _selectedCategory;
  late List<String> _selectedTags;
  final List<String> _availableCategories = [
    'T-Shirt',
    'Shirt',
    'Jeans',
    'Pants',
    'Dress',
    'Jacket',
    'Sweater',
    'Skirt',
    'Shorts',
    'Shoes',
    'Accessories',
  ];
  final List<String> _availableTags = [
    'Casual',
    'Formal',
    'Summer',
    'Winter',
    'Everyday',
    'Party',
    'Work',
    'Sport',
    'Classic',
    'Trendy',
  ];

  @override
  void initState() {
    super.initState();
    _selectedCategory = widget.detectedAttributes['category'] ?? 'T-Shirt';
    _selectedTags = List<String>.from(
      widget.detectedAttributes['suggestedTags'] ?? [],
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      appBar: AppBar(
        title: const Text('Item Details'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: widget.onRetake,
        ),
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildImagePreview(context),
            SizedBox(height: 2.h),
            _buildDetectedAttributes(context),
            SizedBox(height: 2.h),
            _buildCategorySelector(context),
            SizedBox(height: 2.h),
            _buildTagSelector(context),
            SizedBox(height: 2.h),
            _buildActionButtons(context),
            SizedBox(height: 2.h),
          ],
        ),
      ),
    );
  }

  Widget _buildImagePreview(BuildContext context) {
    return Container(
      width: double.infinity,
      height: 40.h,
      color: Colors.black,
      child: kIsWeb
          ? CustomImageWidget(
              imageUrl: widget.capturedImage.path,
              width: double.infinity,
              height: 40.h,
              fit: BoxFit.contain,
              semanticLabel: 'Captured wardrobe item preview',
            )
          : Image.file(File(widget.capturedImage.path), fit: BoxFit.contain),
    );
  }

  Widget _buildDetectedAttributes(BuildContext context) {
    final theme = Theme.of(context);
    final confidence = widget.detectedAttributes['confidence'] ?? 0.0;
    return Container(
      padding: EdgeInsets.all(4.w),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              CustomIconWidget(
                iconName: 'auto_awesome',
                color: theme.colorScheme.tertiary,
                size: 24,
              ),
              SizedBox(width: 2.w),
              Text(
                'AI Detected Attributes',
                style: theme.textTheme.titleMedium,
              ),
              const Spacer(),
              Container(
                padding: EdgeInsets.symmetric(horizontal: 3.w, vertical: 0.5.h),
                decoration: BoxDecoration(
                  color: theme.colorScheme.tertiary.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  '${(confidence * 100).toInt()}% confident',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.tertiary,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          SizedBox(height: 2.h),
          _buildAttributeRow(
            context,
            'Primary Color',
            widget.detectedAttributes['primaryColor'] ?? 'Unknown',
          ),
          _buildAttributeRow(
            context,
            'Secondary Color',
            widget.detectedAttributes['secondaryColor'] ?? 'None',
          ),
          _buildAttributeRow(
            context,
            'Pattern',
            widget.detectedAttributes['pattern'] ?? 'Unknown',
          ),
          _buildAttributeRow(
            context,
            'Material',
            widget.detectedAttributes['material'] ?? 'Unknown',
          ),
        ],
      ),
    );
  }

  Widget _buildAttributeRow(BuildContext context, String label, String value) {
    final theme = Theme.of(context);
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 1.h),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
          Text(
            value,
            style: theme.textTheme.bodyMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCategorySelector(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: EdgeInsets.all(4.w),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Category', style: theme.textTheme.titleMedium),
          SizedBox(height: 1.h),
          DropdownButtonFormField<String>(
            initialValue: _selectedCategory,
            decoration: InputDecoration(
              contentPadding: EdgeInsets.symmetric(
                horizontal: 4.w,
                vertical: 1.5.h,
              ),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
            items: _availableCategories.map((category) {
              return DropdownMenuItem(value: category, child: Text(category));
            }).toList(),
            onChanged: (value) {
              if (value != null) {
                setState(() {
                  _selectedCategory = value;
                });
              }
            },
          ),
        ],
      ),
    );
  }

  Widget _buildTagSelector(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: EdgeInsets.all(4.w),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Tags', style: theme.textTheme.titleMedium),
          SizedBox(height: 1.h),
          Wrap(
            spacing: 2.w,
            runSpacing: 1.h,
            children: _availableTags.map((tag) {
              final isSelected = _selectedTags.contains(tag);
              return GestureDetector(
                onTap: () {
                  setState(() {
                    isSelected
                        ? _selectedTags.remove(tag)
                        : _selectedTags.add(tag);
                  });
                },
                child: Container(
                  padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 1.h),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? theme.colorScheme.tertiary
                        : theme.colorScheme.surface,
                    border: Border.all(
                      color: isSelected
                          ? theme.colorScheme.tertiary
                          : theme.colorScheme.outline,
                    ),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    tag,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: isSelected
                          ? theme.colorScheme.onTertiary
                          : theme.colorScheme.onSurface,
                      fontWeight: isSelected
                          ? FontWeight.w600
                          : FontWeight.w400,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 4.w),
      child: Row(
        children: [
          Expanded(
            child: OutlinedButton(
              onPressed: widget.onRetake,
              style: OutlinedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 2.h),
              ),
              child: const Text('Retake Photo'),
            ),
          ),
          SizedBox(width: 4.w),
          Expanded(
            child: ElevatedButton(
              onPressed: widget.onConfirm,
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 2.h),
              ),
              child: const Text('Save to Wardrobe'),
            ),
          ),
        ],
      ),
    );
  }
}
