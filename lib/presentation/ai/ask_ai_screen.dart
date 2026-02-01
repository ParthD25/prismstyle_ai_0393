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

  void _sendMessage() {
    if (_controller.text.isEmpty) return;
    
    setState(() {
      _messages.add({'role': 'user', 'text': _controller.text});
      // Simulate AI thinking and responding
      Future.delayed(const Duration(seconds: 1), () {
        if (mounted) {
          setState(() {
            _messages.add({
              'role': 'ai',
              'text': 'That sounds like a great idea! Based on your wardrobe, I recommend pairing the Navy Blazer with the Beige Chinos for a smart casual look.'
            });
          });
        }
      });
      _controller.clear();
    });
  }

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
                      style: const TextStyle(color: Colors.white),
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
