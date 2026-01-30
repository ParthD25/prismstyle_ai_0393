import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

import '../../core/app_export.dart';
import '../../widgets/custom_app_bar.dart';
import '../../widgets/shimmer_loading.dart';
import './widgets/category_filter_chip.dart';
import './widgets/empty_category_widget.dart';
import './widgets/wardrobe_item_card.dart';

class WardrobeManagement extends StatefulWidget {
  const WardrobeManagement({super.key});

  @override
  State<WardrobeManagement> createState() => _WardrobeManagementState();
}

class _WardrobeManagementState extends State<WardrobeManagement> {
  final TextEditingController _searchController = TextEditingController();
  String _selectedCategory = 'All';
  final Set<String> _selectedItems = {};
  bool _isSelectionMode = false;
  bool _isLoading = true;
  // ignore: unused_field - Reserved for pull-to-refresh feature
  bool _isRefreshing = false;

  final List<String> _categories = [
    'All',
    'Tops',
    'Bottoms',
    'Dresses',
    'Shoes',
    'Accessories',
  ];

  final List<Map<String, dynamic>> _wardrobeItems = [
    {
      "id": "1",
      "name": "White Cotton T-Shirt",
      "category": "Tops",
      "imageUrl":
          "https://img.rocket.new/generatedImages/rocket_gen_img_1d49fa61f-1766582359298.png",
      "semanticLabel":
          "White cotton crew neck t-shirt laid flat on neutral background",
      "color": "White",
      "pattern": "Solid",
      "season": "All Season",
      "occasion": "Casual",
      "isFavorite": true,
      "tags": ["basic", "versatile", "cotton"],
    },
    {
      "id": "2",
      "name": "Blue Denim Jeans",
      "category": "Bottoms",
      "imageUrl":
          "https://img.rocket.new/generatedImages/rocket_gen_img_102029895-1767024608521.png",
      "semanticLabel":
          "Classic blue denim jeans with straight leg cut on white background",
      "color": "Blue",
      "pattern": "Denim",
      "season": "All Season",
      "occasion": "Casual",
      "isFavorite": false,
      "tags": ["denim", "classic", "everyday"],
    },
    {
      "id": "3",
      "name": "Black Leather Jacket",
      "category": "Tops",
      "imageUrl":
          "https://img.rocket.new/generatedImages/rocket_gen_img_144defeba-1766797906480.png",
      "semanticLabel":
          "Black leather motorcycle jacket with silver zippers hanging on rack",
      "color": "Black",
      "pattern": "Solid",
      "season": "Fall/Winter",
      "occasion": "Casual/Evening",
      "isFavorite": true,
      "tags": ["leather", "edgy", "outerwear"],
    },
    {
      "id": "4",
      "name": "Floral Summer Dress",
      "category": "Dresses",
      "imageUrl":
          "https://img.rocket.new/generatedImages/rocket_gen_img_1c06252f6-1766727266144.png",
      "semanticLabel":
          "Light pink floral print sundress with thin straps on hanger",
      "color": "Pink",
      "pattern": "Floral",
      "season": "Spring/Summer",
      "occasion": "Casual/Formal",
      "isFavorite": false,
      "tags": ["floral", "feminine", "summer"],
    },
    {
      "id": "5",
      "name": "White Sneakers",
      "category": "Shoes",
      "imageUrl":
          "https://img.rocket.new/generatedImages/rocket_gen_img_13ef60586-1767723958930.png",
      "semanticLabel":
          "Clean white leather sneakers with minimal design on white surface",
      "color": "White",
      "pattern": "Solid",
      "season": "All Season",
      "occasion": "Casual",
      "isFavorite": true,
      "tags": ["sneakers", "comfortable", "minimalist"],
    },
    {
      "id": "6",
      "name": "Brown Leather Belt",
      "category": "Accessories",
      "imageUrl":
          "https://images.unsplash.com/photo-1664286074240-d7059e004dff",
      "semanticLabel":
          "Brown leather belt with silver buckle coiled on wooden surface",
      "color": "Brown",
      "pattern": "Solid",
      "season": "All Season",
      "occasion": "Casual/Formal",
      "isFavorite": false,
      "tags": ["leather", "accessory", "classic"],
    },
    {
      "id": "7",
      "name": "Striped Button-Up Shirt",
      "category": "Tops",
      "imageUrl":
          "https://images.unsplash.com/photo-1611244565482-602f7182e346",
      "semanticLabel":
          "Blue and white vertical striped button-up shirt on hanger",
      "color": "Blue/White",
      "pattern": "Striped",
      "season": "Spring/Summer",
      "occasion": "Casual/Business",
      "isFavorite": false,
      "tags": ["striped", "button-up", "versatile"],
    },
    {
      "id": "8",
      "name": "Black Ankle Boots",
      "category": "Shoes",
      "imageUrl":
          "https://images.unsplash.com/photo-1642596885960-c536ace4deac",
      "semanticLabel":
          "Black leather ankle boots with low heel on neutral background",
      "color": "Black",
      "pattern": "Solid",
      "season": "Fall/Winter",
      "occasion": "Casual/Formal",
      "isFavorite": true,
      "tags": ["boots", "leather", "versatile"],
    },
  ];

  List<Map<String, dynamic>> get _filteredItems {
    List<Map<String, dynamic>> filtered = _wardrobeItems;

    if (_selectedCategory != 'All') {
      filtered = filtered
          .where((item) => item['category'] == _selectedCategory)
          .toList();
    }

    if (_searchController.text.isNotEmpty) {
      final searchTerm = _searchController.text.toLowerCase();
      filtered = filtered.where((item) {
        final name = (item['name'] as String).toLowerCase();
        final color = (item['color'] as String).toLowerCase();
        final tags = (item['tags'] as List).join(' ').toLowerCase();
        return name.contains(searchTerm) ||
            color.contains(searchTerm) ||
            tags.contains(searchTerm);
      }).toList();
    }

    return filtered;
  }

  @override
  void initState() {
    super.initState();
    _loadWardrobeItems();
  }

  Future<void> _loadWardrobeItems() async {
    // Simulate loading wardrobe items
    await Future.delayed(const Duration(milliseconds: 800));
    if (mounted) {
      setState(() => _isLoading = false);
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _handleRefresh() async {
    setState(() => _isRefreshing = true);
    await Future.delayed(const Duration(seconds: 2));
    setState(() => _isRefreshing = false);
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Wardrobe synced successfully'),
          duration: Duration(seconds: 2),
        ),
      );
    }
  }

  void _toggleFavorite(String itemId) {
    setState(() {
      final index = _wardrobeItems.indexWhere((item) => item['id'] == itemId);
      if (index != -1) {
        _wardrobeItems[index]['isFavorite'] =
            !(_wardrobeItems[index]['isFavorite'] as bool);
      }
    });
  }

  void _handleItemLongPress(String itemId) {
    setState(() {
      _isSelectionMode = true;
      _selectedItems.add(itemId);
    });
  }

  void _handleItemTap(String itemId) {
    if (_isSelectionMode) {
      setState(() {
        _selectedItems.contains(itemId)
            ? _selectedItems.remove(itemId)
            : _selectedItems.add(itemId);
        if (_selectedItems.isEmpty) {
          _isSelectionMode = false;
        }
      });
    } else {
      _showItemDetails(itemId);
    }
  }

  void _showItemDetails(String itemId) {
    final item = _wardrobeItems.firstWhere((item) => item['id'] == itemId);
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => _buildItemDetailsSheet(item),
    );
  }

  Widget _buildItemDetailsSheet(Map<String, dynamic> item) {
    final theme = Theme.of(context);
    return Container(
      height: 70.h,
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: Column(
        children: [
          SizedBox(height: 1.h),
          Container(
            width: 12.w,
            height: 0.5.h,
            decoration: BoxDecoration(
              color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.3),
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          SizedBox(height: 2.h),
          Expanded(
            child: SingleChildScrollView(
              padding: EdgeInsets.symmetric(horizontal: 5.w),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: CustomImageWidget(
                      imageUrl: item['imageUrl'] as String,
                      width: double.infinity,
                      height: 35.h,
                      fit: BoxFit.cover,
                      semanticLabel: item['semanticLabel'] as String,
                    ),
                  ),
                  SizedBox(height: 2.h),
                  Text(
                    item['name'] as String,
                    style: theme.textTheme.headlineSmall,
                  ),
                  SizedBox(height: 2.h),
                  _buildDetailRow(
                    theme,
                    'Category',
                    item['category'] as String,
                  ),
                  _buildDetailRow(theme, 'Color', item['color'] as String),
                  _buildDetailRow(theme, 'Pattern', item['pattern'] as String),
                  _buildDetailRow(theme, 'Season', item['season'] as String),
                  _buildDetailRow(
                    theme,
                    'Occasion',
                    item['occasion'] as String,
                  ),
                  SizedBox(height: 2.h),
                  Text('Tags', style: theme.textTheme.titleMedium),
                  SizedBox(height: 1.h),
                  Wrap(
                    spacing: 2.w,
                    runSpacing: 1.h,
                    children: (item['tags'] as List)
                        .map(
                          (tag) => Chip(
                            label: Text(tag as String),
                            backgroundColor: theme.colorScheme.surface,
                            side: BorderSide(
                              color: theme.colorScheme.outline.withValues(
                                alpha: 0.3,
                              ),
                            ),
                          ),
                        )
                        .toList(),
                  ),
                  SizedBox(height: 3.h),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDetailRow(ThemeData theme, String label, String value) {
    return Padding(
      padding: EdgeInsets.only(bottom: 1.h),
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

  void _showContextMenu(String itemId) {
    final theme = Theme.of(context);
    showModalBottomSheet(
      context: context,
      backgroundColor: theme.colorScheme.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: CustomIconWidget(
                iconName: 'edit',
                color: theme.colorScheme.onSurface,
                size: 24,
              ),
              title: const Text('Edit Details'),
              onTap: () {
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Edit functionality')),
                );
              },
            ),
            ListTile(
              leading: CustomIconWidget(
                iconName: 'drive_file_move',
                color: theme.colorScheme.onSurface,
                size: 24,
              ),
              title: const Text('Move to Category'),
              onTap: () {
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Move functionality')),
                );
              },
            ),
            ListTile(
              leading: CustomIconWidget(
                iconName: 'favorite',
                color: theme.colorScheme.onSurface,
                size: 24,
              ),
              title: const Text('Mark as Favorite'),
              onTap: () {
                Navigator.pop(context);
                _toggleFavorite(itemId);
              },
            ),
            ListTile(
              leading: CustomIconWidget(
                iconName: 'delete',
                color: theme.colorScheme.error,
                size: 24,
              ),
              title: Text(
                'Delete',
                style: TextStyle(color: theme.colorScheme.error),
              ),
              onTap: () {
                Navigator.pop(context);
                _showDeleteConfirmation(itemId);
              },
            ),
            SizedBox(height: 2.h),
          ],
        ),
      ),
    );
  }

  void _showDeleteConfirmation(String itemId) {
    final theme = Theme.of(context);
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Item'),
        content: const Text('Are you sure you want to delete this item?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              setState(() {
                _wardrobeItems.removeWhere((item) => item['id'] == itemId);
              });
              Navigator.pop(context);
              ScaffoldMessenger.of(
                context,
              ).showSnackBar(const SnackBar(content: Text('Item deleted')));
            },
            child: Text(
              'Delete',
              style: TextStyle(color: theme.colorScheme.error),
            ),
          ),
        ],
      ),
    );
  }

  void _handleBatchOperation(String operation) {
    setState(() {
      _isSelectionMode = false;
      _selectedItems.clear();
    });
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text('$operation completed')));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final filteredItems = _filteredItems;

    return Column(
      children: [
        CustomAppBar(
          title: 'Wardrobe',
          variant: CustomAppBarVariant.standard,
          actions: [
            if (_isSelectionMode)
              IconButton(
                icon: CustomIconWidget(
                  iconName: 'close',
                  color: theme.colorScheme.onSurface,
                  size: 24,
                ),
                onPressed: () {
                  setState(() {
                    _isSelectionMode = false;
                    _selectedItems.clear();
                  });
                },
              )
            else
              IconButton(
                icon: CustomIconWidget(
                  iconName: 'filter_list',
                  color: theme.colorScheme.onSurface,
                  size: 24,
                ),
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Advanced filters')),
                  );
                },
              ),
          ],
        ),
        Expanded(
          child: RefreshIndicator(
            onRefresh: _handleRefresh,
            child: CustomScrollView(
              slivers: [
                SliverToBoxAdapter(
                  child: Column(
                    children: [
                      Padding(
                        padding: EdgeInsets.symmetric(
                          horizontal: 4.w,
                          vertical: 1.h,
                        ),
                        child: TextField(
                          controller: _searchController,
                          onChanged: (value) => setState(() {}),
                          decoration: InputDecoration(
                            hintText: 'Search by name, color, or tags...',
                            prefixIcon: CustomIconWidget(
                              iconName: 'search',
                              color: theme.colorScheme.onSurfaceVariant,
                              size: 24,
                            ),
                            suffixIcon: _searchController.text.isNotEmpty
                                ? IconButton(
                                    icon: CustomIconWidget(
                                      iconName: 'clear',
                                      color: theme.colorScheme.onSurfaceVariant,
                                      size: 20,
                                    ),
                                    onPressed: () {
                                      _searchController.clear();
                                      setState(() {});
                                    },
                                  )
                                : null,
                          ),
                        ),
                      ),
                      SizedBox(
                        height: 6.h,
                        child: ListView.builder(
                          scrollDirection: Axis.horizontal,
                          padding: EdgeInsets.symmetric(horizontal: 4.w),
                          itemCount: _categories.length,
                          itemBuilder: (context, index) {
                            return CategoryFilterChip(
                              label: _categories[index],
                              isSelected:
                                  _selectedCategory == _categories[index],
                              onSelected: () {
                                setState(() {
                                  _selectedCategory = _categories[index];
                                });
                              },
                            );
                          },
                        ),
                      ),
                      if (_isSelectionMode)
                        Container(
                          padding: EdgeInsets.symmetric(
                            horizontal: 4.w,
                            vertical: 1.h,
                          ),
                          color: theme.colorScheme.primaryContainer.withValues(
                            alpha: 0.3,
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                '${_selectedItems.length} selected',
                                style: theme.textTheme.titleMedium,
                              ),
                              Row(
                                children: [
                                  TextButton.icon(
                                    onPressed: () =>
                                        _handleBatchOperation('Move'),
                                    icon: CustomIconWidget(
                                      iconName: 'drive_file_move',
                                      color: theme.colorScheme.primary,
                                      size: 20,
                                    ),
                                    label: const Text('Move'),
                                  ),
                                  TextButton.icon(
                                    onPressed: () =>
                                        _handleBatchOperation('Delete'),
                                    icon: CustomIconWidget(
                                      iconName: 'delete',
                                      color: theme.colorScheme.error,
                                      size: 20,
                                    ),
                                    label: Text(
                                      'Delete',
                                      style: TextStyle(
                                        color: theme.colorScheme.error,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                ),
                _isLoading
                    ? SliverPadding(
                        padding: EdgeInsets.symmetric(
                          horizontal: 4.w,
                          vertical: 2.h,
                        ),
                        sliver: SliverGrid(
                          gridDelegate:
                              SliverGridDelegateWithFixedCrossAxisCount(
                                crossAxisCount:
                                    MediaQuery.of(context).orientation ==
                                        Orientation.portrait
                                    ? 2
                                    : 3,
                                crossAxisSpacing: 3.w,
                                mainAxisSpacing: 2.h,
                                childAspectRatio: 0.75,
                              ),
                          delegate: SliverChildBuilderDelegate(
                            (context, index) => const ShimmerWardrobeCard(),
                            childCount: 6,
                          ),
                        ),
                      )
                    : filteredItems.isEmpty
                    ? SliverFillRemaining(
                        child: EmptyCategoryWidget(category: _selectedCategory),
                      )
                    : SliverPadding(
                        padding: EdgeInsets.symmetric(
                          horizontal: 4.w,
                          vertical: 2.h,
                        ),
                        sliver: SliverGrid(
                          gridDelegate:
                              SliverGridDelegateWithFixedCrossAxisCount(
                                crossAxisCount:
                                    MediaQuery.of(context).orientation ==
                                        Orientation.portrait
                                    ? 2
                                    : 3,
                                crossAxisSpacing: 3.w,
                                mainAxisSpacing: 2.h,
                                childAspectRatio: 0.75,
                              ),
                          delegate: SliverChildBuilderDelegate((
                            context,
                            index,
                          ) {
                            final item = filteredItems[index];
                            final itemId = item['id'] as String;
                            return WardrobeItemCard(
                              item: item,
                              isSelected: _selectedItems.contains(itemId),
                              isSelectionMode: _isSelectionMode,
                              onTap: () => _handleItemTap(itemId),
                              onLongPress: () => _handleItemLongPress(itemId),
                              onFavoriteToggle: () => _toggleFavorite(itemId),
                              onContextMenu: () => _showContextMenu(itemId),
                            );
                          }, childCount: filteredItems.length),
                        ),
                      ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
