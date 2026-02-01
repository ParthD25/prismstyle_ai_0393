import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import 'package:prismstyle_ai/theme/lumina_theme.dart';

class StylistScreen extends StatefulWidget {
  const StylistScreen({Key? key}) : super(key: key);

  @override
  State<StylistScreen> createState() => _StylistScreenState();
}

class _StylistScreenState extends State<StylistScreen> with SingleTickerProviderStateMixin {
  // Placeholder data - connect to CompatibilityEngine later
  final List<String> _demoImages = [
    'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=800', // Reliable Fashion Image 1
    'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=800', // Reliable Fashion Image 2 
    'https://images.unsplash.com/photo-1487222477894-8943e31ef7b2?w=800', // Reliable Fashion Image 3
  ];

  late AnimationController _controller;
  int _currentIndex = 0;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _handleSwipe(bool liked) {
    // Animate and remove card
    // Integrate with UserLearning logic here
    setState(() {
      _currentIndex++;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_currentIndex >= _demoImages.length) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text("You're all caught up!", style: LuminaTheme.themeData.textTheme.displayLarge),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () => setState(() => _currentIndex = 0),
              style: ElevatedButton.styleFrom(
                backgroundColor: LuminaTheme.accentPurple,
                foregroundColor: Colors.white,
              ),
              child: const Text('Reset Styling Session'),
            ),
          ],
        ),
      );
    }

    return Stack(
      children: [
        // Background - handled by MainScreen usually, but valid here too
        
        // Main Card Stack
        Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
               Text(
                'Today\'s Pick',
                style: LuminaTheme.themeData.textTheme.displayLarge?.copyWith(fontSize: 24),
              ),
              const SizedBox(height: 20),
              
              // The Outfit Card
              Container(
                width: MediaQuery.of(context).size.width * 0.85,
                height: MediaQuery.of(context).size.height * 0.6,
                decoration: LuminaTheme.glassDecoration.copyWith(
                  image: DecorationImage(
                    image: NetworkImage(_demoImages[_currentIndex]),
                    fit: BoxFit.cover,
                  ),
                ),
                child: Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(24),
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [Colors.transparent, Colors.black.withOpacity(0.8)],
                      stops: const [0.7, 1.0],
                    ),
                  ),
                  padding: const EdgeInsets.all(24),
                  alignment: Alignment.bottomLeft,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.end,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Business Casual Mix',
                        style: LuminaTheme.themeData.textTheme.titleLarge?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Navy Blazer + Chinos',
                        style: LuminaTheme.themeData.textTheme.bodyMedium?.copyWith(color: Colors.white70),
                      ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: 32),
              
              // Action Buttons
              // Action Buttons
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                   _ActionButton(
                    icon: Icons.close,
                    color: Colors.redAccent,
                    onTap: () => _handleSwipe(false),
                  ),
                  const SizedBox(width: 24),
                  
                  // Share Button
                   _ActionButton(
                    icon: Icons.share,
                    color: Colors.blueAccent,
                    onTap: () {
                      Share.share(
                        'Check out this outfit I found on PrismStyle AI! #LuminaStyle',
                        subject: 'Style Inspiration',
                      );
                    },
                   ),
                  
                  const SizedBox(width: 24),
                   _ActionButton(
                    icon: Icons.favorite,
                    color: Colors.greenAccent,
                    onTap: () => _handleSwipe(true),
                  ),
                ],
              ),
              
              const SizedBox(height: 20),
              
              // Custom Generator Link
              TextButton.icon(
                onPressed: () {
                   Navigator.pushNamed(context, '/generator');
                },
                icon: const Icon(Icons.tune, color: Colors.white70),
                label: const Text(
                  'Custom Generator',
                  style: TextStyle(color: Colors.white70),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const _ActionButton({required this.icon, required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.1),
          shape: BoxShape.circle,
          border: Border.all(color: color.withOpacity(0.5), width: 2),
          boxShadow: [
            BoxShadow(
              color: color.withOpacity(0.2),
              blurRadius: 12,
              spreadRadius: 2,
            )
          ],
        ),
        child: Icon(icon, color: color, size: 32),
      ),
    );
  }
}
