import 'package:flutter/material.dart';

/// Custom app bar variants for the fashion app
/// Implements clean, photo-centric interface with minimal design
enum CustomAppBarVariant {
  /// Standard app bar with title and optional actions
  standard,

  /// App bar with back button and title
  withBack,

  /// App bar with search functionality
  withSearch,

  /// Transparent app bar for photo-first screens
  transparent,

  /// App bar with custom leading widget
  custom,
}

/// Custom app bar for the fashion application
/// Follows Contemporary Minimalist Fashion design principles
class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  /// Title text to display
  final String? title;

  /// Leading widget (typically back button or menu)
  final Widget? leading;

  /// Action widgets displayed on the right
  final List<Widget>? actions;

  /// App bar variant to use
  final CustomAppBarVariant variant;

  /// Whether to center the title
  final bool centerTitle;

  /// Custom background color (overrides theme)
  final Color? backgroundColor;

  /// Custom foreground color (overrides theme)
  final Color? foregroundColor;

  /// Elevation of the app bar
  final double? elevation;

  /// Whether to show bottom border
  final bool showBottomBorder;

  /// Search controller for search variant
  final TextEditingController? searchController;

  /// Search hint text
  final String? searchHint;

  /// Callback when search text changes
  final ValueChanged<String>? onSearchChanged;

  const CustomAppBar({
    super.key,
    this.title,
    this.leading,
    this.actions,
    this.variant = CustomAppBarVariant.standard,
    this.centerTitle = true,
    this.backgroundColor,
    this.foregroundColor,
    this.elevation,
    this.showBottomBorder = false,
    this.searchController,
    this.searchHint,
    this.onSearchChanged,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final appBarTheme = theme.appBarTheme;

    final effectiveBackgroundColor =
        backgroundColor ??
        (variant == CustomAppBarVariant.transparent
            ? Colors.transparent
            : appBarTheme.backgroundColor);

    final effectiveForegroundColor =
        foregroundColor ??
        appBarTheme.foregroundColor ??
        theme.colorScheme.onSurface;

    final effectiveElevation =
        elevation ??
        (variant == CustomAppBarVariant.transparent
            ? 0.0
            : appBarTheme.elevation ?? 0.0);

    Widget? titleWidget;
    Widget? leadingWidget = leading;

    // Build title based on variant
    switch (variant) {
      case CustomAppBarVariant.withSearch:
        titleWidget = _buildSearchField(context, effectiveForegroundColor);
        break;
      case CustomAppBarVariant.withBack:
        leadingWidget =
            leading ??
            IconButton(
              icon: const Icon(Icons.arrow_back),
              onPressed: () => Navigator.of(context).pop(),
              tooltip: 'Back',
            );
        titleWidget = title != null ? Text(title!) : null;
        break;
      default:
        titleWidget = title != null ? Text(title!) : null;
    }

    return Container(
      decoration: showBottomBorder
          ? BoxDecoration(
              border: Border(
                bottom: BorderSide(
                  color: theme.colorScheme.outline.withValues(alpha: 0.2),
                  width: 1,
                ),
              ),
            )
          : null,
      child: AppBar(
        title: titleWidget,
        leading: leadingWidget,
        actions: actions,
        centerTitle: centerTitle,
        backgroundColor: effectiveBackgroundColor,
        foregroundColor: effectiveForegroundColor,
        elevation: effectiveElevation,
        automaticallyImplyLeading: variant == CustomAppBarVariant.withBack,
      ),
    );
  }

  Widget _buildSearchField(BuildContext context, Color foregroundColor) {
    final theme = Theme.of(context);

    return Container(
      height: 40,
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(8.0),
        border: Border.all(
          color: theme.colorScheme.outline.withValues(alpha: 0.3),
          width: 1,
        ),
      ),
      child: TextField(
        controller: searchController,
        onChanged: onSearchChanged,
        style: theme.textTheme.bodyMedium?.copyWith(color: foregroundColor),
        decoration: InputDecoration(
          hintText: searchHint ?? 'Search...',
          hintStyle: theme.textTheme.bodyMedium?.copyWith(
            color: foregroundColor.withValues(alpha: 0.6),
          ),
          prefixIcon: Icon(
            Icons.search,
            color: foregroundColor.withValues(alpha: 0.6),
            size: 20,
          ),
          suffixIcon: searchController?.text.isNotEmpty ?? false
              ? IconButton(
                  icon: Icon(
                    Icons.clear,
                    color: foregroundColor.withValues(alpha: 0.6),
                    size: 20,
                  ),
                  onPressed: () {
                    searchController?.clear();
                    onSearchChanged?.call('');
                  },
                )
              : null,
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 16,
            vertical: 8,
          ),
        ),
      ),
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}
