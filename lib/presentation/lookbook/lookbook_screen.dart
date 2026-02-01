import 'package:flutter/material.dart';
import 'package:prismstyle_ai/theme/lumina_theme.dart';

class LookbookScreen extends StatelessWidget {
  const LookbookScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('My Lookbooks', style: LuminaTheme.themeData.textTheme.displayLarge),
              const SizedBox(height: 20),
              
              Expanded(
                child: GridView.builder(
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    mainAxisSpacing: 16,
                    crossAxisSpacing: 16,
                    childAspectRatio: 0.8,
                  ),
                  itemCount: 4, // Placeholder
                  itemBuilder: (context, index) {
                    return Container(
                      decoration: LuminaTheme.glassDecoration,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Expanded(
                            flex: 3,
                            child: ClipRRect(
                              borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
                              child: Image.network(
                                [
                                  'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=800',
                                  'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=800', 
                                  'https://images.unsplash.com/photo-1487222477894-8943e31ef7b2?w=800',
                                  'https://images.unsplash.com/photo-1483985988355-763728e1935b?w=800',
                                ][index],
                                fit: BoxFit.cover,
                              ),
                            ),
                          ),
                          Expanded(
                            flex: 1,
                            child: Padding(
                              padding: const EdgeInsets.all(12.0),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Text(
                                    ['Favorites', 'Workwear', 'Summer 2024', 'Date Night'][index],
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  Text(
                                    '${(index + 1) * 3} items',
                                    style: const TextStyle(
                                      color: Colors.grey,
                                      fontSize: 12,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
