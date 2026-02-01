import 'package:flutter/material.dart';
import 'package:prismstyle_ai/theme/lumina_theme.dart';

class AskAIScreen extends StatefulWidget {
  const AskAIScreen({Key? key}) : super(key: key);

  @override
  State<AskAIScreen> createState() => _AskAIScreenState();
}

class _AskAIScreenState extends State<AskAIScreen> {
  final TextEditingController _controller = TextEditingController();
  final List<Map<String, String>> _messages = [
    {'role': 'ai', 'text': 'Hello! I am your personal AI Stylist. How can I help you look your best today?'},
  ];

    final userText = _controller.text;
    if (userText.isEmpty) return;
    
    setState(() {
      _messages.add({'role': 'user', 'text': userText});
      _controller.clear();
      
      // Simulate AI thinking and responding with improved logic
      Future.delayed(const Duration(seconds: 1), () {
        String aiResponse;
        final lowerText = userText.toLowerCase();

        if (lowerText.contains('cotton') || lowerText.contains('polyester')) {
          aiResponse = "Cotton is generally better for breathability and comfort, making it ideal for summer and everyday wear. Polyester is durable and wrinkle-resistant but can trap heat. For winter, cotton layering is good, but wool or synthetic blends (like polyester fleece) often provide better insulation.";
        } else if (lowerText.contains('winter')) {
          aiResponse = "For winter, layering is key! Start with a moisture-wicking base layer, add an insulating middle layer (like wool or fleece), and finish with a wind/waterproof outer shell. Fabrics like wool, cashmere, and down are excellent choices.";
        } else if (lowerText.contains('summer')) {
          aiResponse = "In summer, opt for lightweight, breathable fabrics like linen, cotton, and seersucker. Light colors help reflect sunlight and keep you cool. Consider loose-fitting clothes for better airflow.";
        } else {
          aiResponse = "That sounds interesting! Based on your wardrobe, I'd suggest experimenting with different textures. Could you tell me more about the occasion?";
        }

        if (mounted) {
          setState(() {
            _messages.add({
              'role': 'ai',
              'text': aiResponse
            });
          });
        }
      });
    });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Text('Lumina AI Stylist', style: LuminaTheme.themeData.textTheme.displayLarge?.copyWith(fontSize: 24)),
          ),
          
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final msg = _messages[index];
                final isUser = msg['role'] == 'user';
                
                return Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Row(
                    mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
                    children: [
                      if (!isUser) ...[
                        CircleAvatar(
                          backgroundColor: LuminaTheme.accentPurple,
                          radius: 16,
                          child: const Icon(Icons.auto_awesome, size: 16, color: Colors.white),
                        ),
                        const SizedBox(width: 8),
                      ],
                      Flexible(
                        child: Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: isUser 
                                ? Colors.white 
                                : Colors.white.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(20).copyWith(
                              topLeft: !isUser ? Radius.zero : null,
                              topRight: isUser ? Radius.zero : null,
                            ),
                            border: !isUser ? Border.all(color: Colors.white.withOpacity(0.2)) : null,
                          ),
                          child: Text(
                            msg['text']!,
                            style: TextStyle(
                              color: isUser ? Colors.black : Colors.white,
                              fontSize: 15,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
          
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Container(
              decoration: LuminaTheme.glassDecoration.copyWith(
                borderRadius: BorderRadius.circular(30),
              ),
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      style: const TextStyle(color: Colors.black),
                      decoration: const InputDecoration(
                        hintText: 'Ask for style advice...',
                        hintStyle: TextStyle(color: Colors.grey),
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.symmetric(horizontal: 20),
                      ),
                      onSubmitted: (_) => _sendMessage(),
                    ),
                  ),
                  IconButton(
                    onPressed: _sendMessage,
                    icon: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: const BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.arrow_upward, color: Colors.black, size: 20),
                    ),
                  ),
                ],
              ),
            ),
          ),
          // Space for Floating Nav
          const SizedBox(height: 70),
        ],
      ),
    );
  }
}
