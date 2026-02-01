import 'package:flutter/material.dart';
import 'package:prismstyle_ai/theme/lumina_theme.dart';

class GeneratorScreen extends StatefulWidget {
  const GeneratorScreen({Key? key}) : super(key: key);

  static const String routeName = '/generator';

  @override
  State<GeneratorScreen> createState() => _GeneratorScreenState();
}

class _GeneratorScreenState extends State<GeneratorScreen> {
  final _formKey = GlobalKey<FormState>();
  String _location = '';
  String _event = '';
  String _style = 'Casual';

  void _generateOutfit() {
    if (_formKey.currentState!.validate()) {
      _formKey.currentState!.save();
      // Logic to generate outfit would go here
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Generating outfit for $_event in $_location...')),
      );
      Navigator.pop(context); // Go back to Stylist or show results
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text('Custom Generator', style: TextStyle(color: Colors.white)),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LuminaTheme.backgroundGradient,
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Form(
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Create Your Look',
                    style: LuminaTheme.themeData.textTheme.displayLarge,
                  ),
                  const SizedBox(height: 32),
                  
                  // Location Input
                  _buildGlassInput(
                    label: 'Location',
                    icon: Icons.location_on,
                    onSaved: (value) => _location = value ?? '',
                    validator: (value) => value!.isEmpty ? 'Please enter a location' : null,
                  ),
                  const SizedBox(height: 20),
                  
                  // Event Input
                  _buildGlassInput(
                    label: 'Occasion/Event',
                    icon: Icons.event,
                    onSaved: (value) => _event = value ?? '',
                    validator: (value) => value!.isEmpty ? 'Please enter an occasion' : null,
                  ),
                  const SizedBox(height: 20),
                  
                  // Style Dropdown (Simulated as input for now)
                  _buildGlassInput(
                    label: 'Style Preference',
                    icon: Icons.style,
                    initialValue: 'Casual',
                    onSaved: (value) => _style = value ?? 'Casual',
                  ),
                  
                  const Spacer(),
                  
                  SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: ElevatedButton(
                      onPressed: _generateOutfit,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: LuminaTheme.accentPurple,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                      ),
                      child: const Text(
                        'Generate Outfit',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildGlassInput({
    required String label,
    required IconData icon,
    String? initialValue,
    required Function(String?) onSaved,
    String? Function(String?)? validator,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.2)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: TextFormField(
        initialValue: initialValue,
        style: const TextStyle(color: Colors.white),
        decoration: InputDecoration(
          icon: Icon(icon, color: Colors.white70),
          labelText: label,
          labelStyle: const TextStyle(color: Colors.white60),
          border: InputBorder.none,
          enabledBorder: InputBorder.none,
          focusedBorder: InputBorder.none,
        ),
        validator: validator,
        onSaved: onSaved,
      ),
    );
  }
}
