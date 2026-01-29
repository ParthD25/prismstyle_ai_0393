import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:sizer/sizer.dart';

import '../../services/ensemble_ai_service.dart';
import '../../services/color_detection_service.dart';
import '../../widgets/custom_app_bar.dart';

/// AI Test Screen for validating model accuracy
/// Use this screen to test clothing classification on sample images
class AITestScreen extends StatefulWidget {
  const AITestScreen({super.key});

  @override
  State<AITestScreen> createState() => _AITestScreenState();
}

class _AITestScreenState extends State<AITestScreen> {
  final EnsembleAIService _aiService = EnsembleAIService.instance;
  final ColorDetectionService _colorService = ColorDetectionService.instance;
  final ImagePicker _picker = ImagePicker();

  File? _selectedImage;
  Uint8List? _imageBytes;
  bool _isProcessing = false;
  EnsembleClassificationResult? _classificationResult;
  ColorExtractionResult? _colorResult;
  Map<String, bool>? _modelStatus;
  int _processingTimeMs = 0;

  @override
  void initState() {
    super.initState();
    _initializeAI();
  }

  Future<void> _initializeAI() async {
    await _aiService.initialize();
    setState(() {
      _modelStatus = _aiService.getModelStatus();
    });
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? image = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 90,
      );

      if (image != null) {
        final file = File(image.path);
        final bytes = await file.readAsBytes();

        setState(() {
          _selectedImage = file;
          _imageBytes = bytes;
          _classificationResult = null;
          _colorResult = null;
        });
      }
    } catch (e) {
      _showError('Failed to pick image: $e');
    }
  }

  Future<void> _classifyImage() async {
    if (_imageBytes == null) {
      _showError('Please select an image first');
      return;
    }

    setState(() {
      _isProcessing = true;
    });

    final stopwatch = Stopwatch()..start();

    try {
      // Run classification
      final result = await _aiService.classifyClothing(_imageBytes!);

      // Run color detection
      final colorResult = await _colorService.extractColorsFromBytes(
        _imageBytes!,
      );

      stopwatch.stop();

      setState(() {
        _classificationResult = result;
        _colorResult = colorResult;
        _processingTimeMs = stopwatch.elapsedMilliseconds;
        _isProcessing = false;
      });
    } catch (e) {
      stopwatch.stop();
      setState(() {
        _isProcessing = false;
      });
      _showError('Classification failed: $e');
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: PreferredSize(
        preferredSize: Size.fromHeight(7.h),
        child: const CustomAppBar(
          title: 'AI Test Lab',
          variant: CustomAppBarVariant.withBack,
        ),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(4.w),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Model Status Card
            _buildModelStatusCard(theme),
            SizedBox(height: 2.h),

            // Image Selection
            _buildImageSection(theme),
            SizedBox(height: 2.h),

            // Classify Button
            ElevatedButton.icon(
              onPressed: _isProcessing ? null : _classifyImage,
              icon: _isProcessing
                  ? SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: theme.colorScheme.onPrimary,
                      ),
                    )
                  : const Icon(Icons.auto_awesome),
              label: Text(_isProcessing ? 'Processing...' : 'Classify Image'),
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 2.h),
              ),
            ),
            SizedBox(height: 2.h),

            // Results
            if (_classificationResult != null) ...[
              _buildClassificationResultCard(theme),
              SizedBox(height: 2.h),
            ],

            if (_colorResult != null && _colorResult!.isNotEmpty) ...[
              _buildColorResultCard(theme),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildModelStatusCard(ThemeData theme) {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(3.w),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.memory, color: theme.colorScheme.primary),
                SizedBox(width: 2.w),
                Text(
                  'Model Status',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            SizedBox(height: 1.h),
            if (_modelStatus != null) ...[
              _buildStatusRow('TFLite Model', _modelStatus!['tflite'] ?? false),
              _buildStatusRow(
                'Apple Vision',
                _modelStatus!['apple_vision'] ?? false,
              ),
              _buildStatusRow('Core ML', _modelStatus!['coreml'] ?? false),
              _buildStatusRow('Heuristics', _modelStatus!['heuristic'] ?? true),
            ] else
              const Center(child: CircularProgressIndicator()),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusRow(String name, bool active) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 0.5.h),
      child: Row(
        children: [
          Icon(
            active ? Icons.check_circle : Icons.cancel,
            color: active ? Colors.green : Colors.grey,
            size: 18,
          ),
          SizedBox(width: 2.w),
          Text(name),
          const Spacer(),
          Text(
            active ? 'Active' : 'Inactive',
            style: TextStyle(
              color: active ? Colors.green : Colors.grey,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageSection(ThemeData theme) {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(3.w),
        child: Column(
          children: [
            if (_selectedImage != null)
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.file(
                  _selectedImage!,
                  height: 30.h,
                  width: double.infinity,
                  fit: BoxFit.cover,
                ),
              )
            else
              Container(
                height: 30.h,
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.image_outlined,
                        size: 50,
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                      SizedBox(height: 1.h),
                      Text(
                        'Select an image to test',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            SizedBox(height: 2.h),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                ),
                SizedBox(width: 2.w),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildClassificationResultCard(ThemeData theme) {
    final result = _classificationResult!;

    return Card(
      child: Padding(
        padding: EdgeInsets.all(3.w),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.category, color: theme.colorScheme.primary),
                SizedBox(width: 2.w),
                Text(
                  'Classification Result',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const Spacer(),
                Container(
                  padding: EdgeInsets.symmetric(
                    horizontal: 2.w,
                    vertical: 0.5.h,
                  ),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.primaryContainer,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${_processingTimeMs}ms',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onPrimaryContainer,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 2.h),

            // Primary category
            Container(
              padding: EdgeInsets.all(3.w),
              decoration: BoxDecoration(
                color: theme.colorScheme.primaryContainer.withOpacity(0.3),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.check_circle,
                    color: theme.colorScheme.primary,
                    size: 32,
                  ),
                  SizedBox(width: 3.w),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          result.primaryCategory,
                          style: theme.textTheme.titleLarge?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text(
                          'Confidence: ${(result.confidence * 100).toStringAsFixed(1)}%',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 2.h),

            // All predictions
            Text(
              'All Predictions',
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 1.h),
            ...result.predictions.map(
              (pred) => Padding(
                padding: EdgeInsets.only(bottom: 1.h),
                child: Row(
                  children: [
                    Expanded(flex: 2, child: Text(pred.category)),
                    Expanded(
                      flex: 3,
                      child: LinearProgressIndicator(
                        value: pred.confidence,
                        backgroundColor:
                            theme.colorScheme.surfaceContainerHighest,
                      ),
                    ),
                    SizedBox(width: 2.w),
                    SizedBox(
                      width: 50,
                      child: Text(
                        '${(pred.confidence * 100).toStringAsFixed(1)}%',
                        textAlign: TextAlign.right,
                        style: theme.textTheme.bodySmall,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            SizedBox(height: 1.h),
            Text(
              'Models used: ${result.contributingModels.join(", ")}',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildColorResultCard(ThemeData theme) {
    final colors = _colorResult!.colors;

    return Card(
      child: Padding(
        padding: EdgeInsets.all(3.w),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.palette, color: theme.colorScheme.primary),
                SizedBox(width: 2.w),
                Text(
                  'Color Analysis',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            SizedBox(height: 2.h),
            Wrap(
              spacing: 2.w,
              runSpacing: 1.h,
              children: colors.take(5).map((color) {
                return Container(
                  padding: EdgeInsets.symmetric(horizontal: 3.w, vertical: 1.h),
                  decoration: BoxDecoration(
                    color: Color.fromRGBO(color.r, color.g, color.b, 1),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.grey.shade300),
                  ),
                  child: Text(
                    '${color.name} (${color.percentage.toStringAsFixed(0)}%)',
                    style: TextStyle(
                      color: _getContrastColor(color.r, color.g, color.b),
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  Color _getContrastColor(int r, int g, int b) {
    final brightness = (r * 299 + g * 587 + b * 114) / 1000;
    return brightness > 128 ? Colors.black : Colors.white;
  }
}
