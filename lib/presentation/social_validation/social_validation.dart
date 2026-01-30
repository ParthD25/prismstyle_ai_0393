import 'package:flutter/material.dart';
import 'package:flutter_slidable/flutter_slidable.dart';
import 'package:sizer/sizer.dart';

import '../../core/app_export.dart';
import '../../widgets/custom_icon_widget.dart';
import './widgets/comment_section_widget.dart';
import './widgets/create_post_widget.dart';
import './widgets/outfit_post_card_widget.dart';

/// Social Validation screen for outfit sharing and feedback
/// Implements tab navigation with My Posts and Friends' Feed
class SocialValidation extends StatefulWidget {
  const SocialValidation({super.key});

  @override
  State<SocialValidation> createState() => _SocialValidationState();
}

class _SocialValidationState extends State<SocialValidation>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  // ignore: unused_field - Reserved for pull-to-refresh feature
  bool _isRefreshing = false;

  final List<Map<String, dynamic>> _myPosts = [
    {
      "id": 1,
      "userName": "You",
      "userAvatar":
          "https://img.rocket.new/generatedImages/rocket_gen_img_1e5c54426-1763299106654.png",
      "userAvatarLabel":
          "Profile photo of a woman with long brown hair wearing a white blouse",
      "outfitImage":
          "https://images.unsplash.com/photo-1618397351187-ee6afd732119",
      "outfitImageLabel":
          "Casual outfit with white t-shirt and blue jeans laid out on wooden floor",
      "caption": "Perfect outfit for a casual Friday! What do you think? 👗",
      "timestamp": "2 hours ago",
      "thumbsUpCount": 24,
      "thumbsDownCount": 2,
      "commentCount": 8,
      "comments": [
        {
          "userName": "Sarah Johnson",
          "userAvatar":
              "https://img.rocket.new/generatedImages/rocket_gen_img_1c2d4d37e-1763298686580.png",
          "userAvatarLabel": "Profile photo of a woman with short blonde hair",
          "text": "Love this combination! The colors work so well together 💙",
          "timestamp": "1h ago",
          "likeCount": 3,
        },
        {
          "userName": "Mike Chen",
          "userAvatar":
              "https://img.rocket.new/generatedImages/rocket_gen_img_10612235a-1763296225971.png",
          "userAvatarLabel":
              "Profile photo of a man with black hair and glasses",
          "text": "Very stylish! Where did you get those jeans?",
          "timestamp": "45m ago",
          "likeCount": 1,
        },
      ],
    },
    {
      "id": 2,
      "userName": "You",
      "userAvatar":
          "https://img.rocket.new/generatedImages/rocket_gen_img_1e5c54426-1763299106654.png",
      "userAvatarLabel":
          "Profile photo of a woman with long brown hair wearing a white blouse",
      "outfitImage":
          "https://img.rocket.new/generatedImages/rocket_gen_img_11c7650d6-1764746226516.png",
      "outfitImageLabel":
          "Formal business outfit with navy blazer and dress pants on hanger",
      "caption": "Ready for that important meeting! 💼",
      "timestamp": "1 day ago",
      "thumbsUpCount": 42,
      "thumbsDownCount": 1,
      "commentCount": 15,
      "comments": [],
    },
  ];

  final List<Map<String, dynamic>> _friendsPosts = [
    {
      "id": 3,
      "userName": "Emma Wilson",
      "userAvatar":
          "https://images.unsplash.com/photo-1673529300380-9db1ccaa42ea",
      "userAvatarLabel":
          "Profile photo of a woman with curly red hair and freckles",
      "outfitImage":
          "https://images.unsplash.com/photo-1676284303477-7233d4fcdbd0",
      "outfitImageLabel":
          "Summer outfit with floral dress and sandals on white background",
      "caption":
          "Summer vibes! ☀️ Loving this new dress from the boutique downtown",
      "timestamp": "3 hours ago",
      "thumbsUpCount": 67,
      "thumbsDownCount": 3,
      "commentCount": 22,
      "comments": [
        {
          "userName": "Lisa Anderson",
          "userAvatar":
              "https://img.rocket.new/generatedImages/rocket_gen_img_155101c03-1763301787020.png",
          "userAvatarLabel":
              "Profile photo of a woman with straight black hair",
          "text": "Gorgeous! That dress is perfect for you! 😍",
          "timestamp": "2h ago",
          "likeCount": 5,
        },
      ],
    },
    {
      "id": 4,
      "userName": "Alex Rodriguez",
      "userAvatar":
          "https://img.rocket.new/generatedImages/rocket_gen_img_1ba2cea14-1763296014154.png",
      "userAvatarLabel":
          "Profile photo of a man with short brown hair and beard",
      "outfitImage":
          "https://images.unsplash.com/photo-1726195222148-fc8a7e7f37fa",
      "outfitImageLabel":
          "Athletic wear with black leggings and sports top on mannequin",
      "caption": "Gym outfit on point! 💪 Ready to crush this workout",
      "timestamp": "5 hours ago",
      "thumbsUpCount": 38,
      "thumbsDownCount": 2,
      "commentCount": 12,
      "comments": [],
    },
    {
      "id": 5,
      "userName": "Sophie Martinez",
      "userAvatar":
          "https://img.rocket.new/generatedImages/rocket_gen_img_13e86e864-1765273185495.png",
      "userAvatarLabel":
          "Profile photo of a woman with long dark hair and brown eyes",
      "outfitImage":
          "https://images.unsplash.com/photo-1684254221933-eed11d11ce53",
      "outfitImageLabel":
          "Trendy street style outfit with leather jacket and ripped jeans",
      "caption":
          "Street style for the weekend! What's your go-to casual look? 🖤",
      "timestamp": "8 hours ago",
      "thumbsUpCount": 91,
      "thumbsDownCount": 4,
      "commentCount": 34,
      "comments": [],
    },
  ];

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _handleRefresh() async {
    setState(() => _isRefreshing = true);
    await Future.delayed(const Duration(seconds: 1));
    setState(() => _isRefreshing = false);

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Feed refreshed'),
          duration: Duration(seconds: 1),
        ),
      );
    }
  }

  void _showComments(BuildContext context, Map<String, dynamic> post) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.7,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        builder: (context, scrollController) => CommentSectionWidget(
          comments: (post["comments"] as List).cast<Map<String, dynamic>>(),
          onCommentSubmit: (comment) {
            setState(() {
              (post["comments"] as List).add({
                "userName": "You",
                "userAvatar":
                    "https://cdn.pixabay.com/photo/2015/03/04/22/35/avatar-659652_640.png",
                "userAvatarLabel":
                    "Profile photo of a woman with long brown hair wearing a white blouse",
                "text": comment,
                "timestamp": "Just now",
                "likeCount": 0,
              });
              post["commentCount"] = (post["commentCount"] as int) + 1;
            });
          },
        ),
      ),
    );
  }

  void _showCreatePost() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => CreatePostWidget(
        onPostCreate: (postData) {
          setState(() {
            _myPosts.insert(0, {
              "id": DateTime.now().millisecondsSinceEpoch,
              "userName": "You",
              "userAvatar":
                  "https://cdn.pixabay.com/photo/2015/03/04/22/35/avatar-659652_640.png",
              "userAvatarLabel":
                  "Profile photo of a woman with long brown hair wearing a white blouse",
              "outfitImage": postData["outfitImage"],
              "outfitImageLabel": postData["outfitImageLabel"],
              "caption": postData["caption"],
              "timestamp": "Just now",
              "thumbsUpCount": 0,
              "thumbsDownCount": 0,
              "commentCount": 0,
              "comments": [],
            });
          });
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Column(
      children: [
        // Custom app bar with SafeArea
        SafeArea(
          bottom: false,
          child: Padding(
            padding: EdgeInsets.symmetric(horizontal: 5.w, vertical: 1.5.h),
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Padding(
                      padding: EdgeInsets.only(left: 1.w),
                      child: Text(
                        'Social',
                        style: theme.textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    IconButton(
                      icon: CustomIconWidget(
                        iconName: 'notifications_outlined',
                        color: theme.colorScheme.onSurface,
                        size: 24,
                      ),
                      onPressed: () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('No new notifications'),
                            duration: Duration(seconds: 2),
                          ),
                        );
                      },
                    ),
                  ],
                ),
                SizedBox(height: 1.h),
                Container(
                  padding: EdgeInsets.all(0.5.w),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surfaceContainerHighest.withValues(
                      alpha: 0.5,
                    ),
                    borderRadius: BorderRadius.circular(12.0),
                  ),
                  child: TabBar(
                    controller: _tabController,
                    indicator: BoxDecoration(
                      color: theme.colorScheme.primary,
                      borderRadius: BorderRadius.circular(10.0),
                    ),
                    indicatorSize: TabBarIndicatorSize.tab,
                    labelColor: theme.colorScheme.onPrimary,
                    unselectedLabelColor: theme.colorScheme.onSurfaceVariant,
                    dividerColor: Colors.transparent,
                    labelStyle: theme.textTheme.labelLarge?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                    unselectedLabelStyle: theme.textTheme.labelLarge,
                    tabs: const [
                      Tab(text: 'My Posts'),
                      Tab(text: 'Friends\' Feed'),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),

        // Tab content
        Expanded(
          child: TabBarView(
            controller: _tabController,
            children: [
              // My Posts tab
              _buildPostsList(_myPosts),

              // Friends' Feed tab
              _buildPostsList(_friendsPosts),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildPostsList(List<Map<String, dynamic>> posts) {
    return RefreshIndicator(
      onRefresh: _handleRefresh,
      child: posts.isEmpty
          ? _buildEmptyState()
          : Column(
              children: [
                // Add Instagram-style post creation button for My Posts tab only
                if (_tabController.index == 0)
                  Container(
                    margin: EdgeInsets.all(4.w),
                    child: OutlinedButton.icon(
                      onPressed: _showCreatePost,
                      icon: CustomIconWidget(
                        iconName: 'add_photo_alternate',
                        color: Theme.of(context).colorScheme.primary,
                        size: 24,
                      ),
                      label: const Text('Create New Post'),
                      style: OutlinedButton.styleFrom(
                        padding: EdgeInsets.symmetric(
                          horizontal: 6.w,
                          vertical: 2.h,
                        ),
                        side: BorderSide(
                          color: Theme.of(context).colorScheme.primary,
                          width: 2,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(20.0),
                        ),
                      ),
                    ),
                  ),

                // Posts list
                Expanded(
                  child: ListView.builder(
                    padding: EdgeInsets.only(top: 1.h, bottom: 10.h),
                    itemCount: posts.length,
                    itemBuilder: (context, index) {
                      final post = posts[index];
                      return Slidable(
                        key: ValueKey(post["id"]),
                        startActionPane: ActionPane(
                          motion: const ScrollMotion(),
                          children: [
                            SlidableAction(
                              onPressed: (context) {
                                setState(() {
                                  post["thumbsUpCount"] =
                                      (post["thumbsUpCount"] as int) + 1;
                                });
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(
                                    content: Text('Thumbs up! 👍'),
                                    duration: Duration(seconds: 1),
                                  ),
                                );
                              },
                              backgroundColor: Theme.of(
                                context,
                              ).colorScheme.tertiary,
                              foregroundColor: Colors.white,
                              icon: Icons.thumb_up,
                              label: 'Like',
                              borderRadius: BorderRadius.circular(20.0),
                            ),
                          ],
                        ),
                        endActionPane: ActionPane(
                          motion: const ScrollMotion(),
                          children: [
                            SlidableAction(
                              onPressed: (context) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(
                                    content: Text('Outfit saved to favorites'),
                                    duration: Duration(seconds: 2),
                                  ),
                                );
                              },
                              backgroundColor: Theme.of(
                                context,
                              ).colorScheme.secondary,
                              foregroundColor: Colors.white,
                              icon: Icons.bookmark,
                              label: 'Save',
                              borderRadius: BorderRadius.circular(20.0),
                            ),
                          ],
                        ),
                        child: OutfitPostCardWidget(
                          post: post,
                          onVote: () {},
                          onComment: () => _showComments(context, post),
                          onSave: () {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text('Outfit saved to favorites'),
                                duration: Duration(seconds: 2),
                              ),
                            );
                          },
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
    );
  }

  Widget _buildEmptyState() {
    final theme = Theme.of(context);
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CustomIconWidget(
            iconName: 'photo_camera_outlined',
            color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.5),
            size: 64,
          ),
          SizedBox(height: 2.h),
          Text(
            'No posts yet',
            style: theme.textTheme.titleLarge?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
          SizedBox(height: 1.h),
          Text(
            'Share your first outfit to get feedback!',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
            textAlign: TextAlign.center,
          ),
          SizedBox(height: 3.h),
          ElevatedButton.icon(
            onPressed: _showCreatePost,
            icon: CustomIconWidget(
              iconName: 'add',
              color: theme.colorScheme.onPrimary,
              size: 20,
            ),
            label: const Text('Create Post'),
          ),
        ],
      ),
    );
  }
}
