import 'dart:io';
import 'dart:math' as math;
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

/// ONNX Runtime Clothing Classifier Service
/// Cross-platform inference using ONNX Runtime Mobile
/// Works on both iOS and Android with the same model
class ONNXClassifierService {
  static ONNXClassifierService? _instance;
  static ONNXClassifierService get instance =>
      _instance ??= ONNXClassifierService._();

  ONNXClassifierService._();

  OrtSession? _session;
  Map<String, dynamic>? _modelConfig;
  bool _isInitialized = false;

  /// Check if the service is ready
  bool get isReady => _isInitialized && _session != null;

  /// Initialize ONNX Runtime and load the model
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Load model configuration
      await _loadModelConfig();

      // Initialize ONNX Runtime environment
      OrtEnv.instance.init();

      // Load the ONNX model from assets
      await _loadModel();

      _isInitialized = true;
      debugPrint('✅ ONNX Classifier initialized successfully');
    } catch (e) {
      debugPrint('❌ Failed to initialize ONNX Classifier: $e');
      _isInitialized = false;
    }
  }

  /// Load model configuration from assets
  Future<void> _loadModelConfig() async {
    try {
      final configString = await rootBundle.loadString(
        'assets/models/model_config.json',
      );
      _modelConfig = json.decode(configString);
      debugPrint('Model config loaded: ${_modelConfig?['model_name']}');
    } catch (e) {
      debugPrint('Failed to load model config: $e');
      throw Exception('Model configuration not found');
    }
  }

  /// Load ONNX model from assets
  Future<void> _loadModel() async {
    try {
      final modelPath =
          _modelConfig?['model_files']?['onnx'] ?? 'clothing_classifier.onnx';

      // Copy model files to a real file path for external data support.
      final modelFile = await _copyAssetToFile('assets/models/$modelPath');
      await _tryCopyExternalData('$modelPath.data');

      final sessionOptions = OrtSessionOptions();
      _session = OrtSession.fromFile(modelFile, sessionOptions);

      debugPrint('✅ ONNX model loaded: ${modelFile.path}');
      debugPrint('   Input: ${_session!.inputNames}');
      debugPrint('   Output: ${_session!.outputNames}');
    } catch (e) {
      debugPrint('❌ Failed to load ONNX model: $e');
      throw Exception('Failed to load ONNX model: $e');
    }
  }

  Future<File> _copyAssetToFile(String assetPath) async {
    final supportDir = await getApplicationSupportDirectory();
    final fileName = path.basename(assetPath);
    final outputFile = File(path.join(supportDir.path, fileName));
    if (await outputFile.exists()) {
      return outputFile;
    }

    final data = await rootBundle.load(assetPath);
    final bytes = data.buffer.asUint8List();
    await outputFile.writeAsBytes(bytes, flush: true);
    return outputFile;
  }

  Future<void> _tryCopyExternalData(String dataFileName) async {
    final assetPath = 'assets/models/$dataFileName';
    try {
      await _copyAssetToFile(assetPath);
    } catch (_) {
      // External data is optional for some models.
    }
  }

  /// Classify clothing from image bytes
  Future<ONNXClassificationResult> classifyImageBytes(
    Uint8List imageBytes,
  ) async {
    if (!isReady) {
      throw Exception('ONNX Classifier not initialized');
    }

    try {
      // Decode image
      final image = img.decodeImage(imageBytes);
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // Preprocess image
      final inputTensor = _preprocessImage(image);

      // Run inference
      final outputs = await _runInference(inputTensor);

      // Post-process results
      final result = _postProcessResults(outputs);

      return result;
    } catch (e) {
      debugPrint('Classification error: $e');
      return ONNXClassificationResult.error();
    }
  }

  /// Classify clothing from file path
  Future<ONNXClassificationResult> classifyImage(String imagePath) async {
    try {
      final file = File(imagePath);
      if (!await file.exists()) {
        throw Exception('Image file not found');
      }

      final bytes = await file.readAsBytes();
      return classifyImageBytes(bytes);
    } catch (e) {
      debugPrint('Classification error: $e');
      return ONNXClassificationResult.error();
    }
  }

  /// Preprocess image for model input
  Float32List _preprocessImage(img.Image image) {
    // Get config parameters
    final inputSize = (_modelConfig?['input_size'] as num?)?.toInt() ?? 224;
    final mean = List<double>.from(
      _modelConfig?['preprocessing']?['normalization']?['mean'] ??
          [0.485, 0.456, 0.406],
    );
    final std = List<double>.from(
      _modelConfig?['preprocessing']?['normalization']?['std'] ??
          [0.229, 0.224, 0.225],
    );

    // Resize image
    final resized = img.copyResize(
      image,
      width: inputSize,
      height: inputSize,
      interpolation: img.Interpolation.linear,
    );

    // Convert to float array with normalization (NCHW format)
    final inputData = Float32List(1 * 3 * inputSize * inputSize);
    int pixelIndex = 0;

    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = resized.getPixel(x, y);
          double value;

          if (c == 0) {
            value = pixel.r / 255.0; // Red channel
          } else if (c == 1) {
            value = pixel.g / 255.0; // Green channel
          } else {
            value = pixel.b / 255.0; // Blue channel
          }

          // Apply ImageNet normalization
          value = (value - mean[c]) / std[c];
          inputData[pixelIndex++] = value;
        }
      }
    }

    return inputData;
  }

  /// Run inference on preprocessed input
  Future<List<double>> _runInference(Float32List inputData) async {
    try {
      final inputSize = _modelConfig?['input_size'] ?? 224;

      // Create input tensor
      final inputOrt = OrtValueTensor.createTensorWithDataList(inputData, [
        1,
        3,
        inputSize,
        inputSize,
      ]);

      // Create inputs map
      final inputs = {_session!.inputNames.first: inputOrt};

      // Run inference
      final runOptions = OrtRunOptions();
      final outputs = await _session!.runAsync(runOptions, inputs);

      // Release input tensor
      inputOrt.release();
      runOptions.release();

      // Extract output
      final outputTensor = outputs?.first as OrtValueTensor?;
      if (outputTensor == null) {
        throw Exception('No output from model');
      }

      // Get output data
      final outputData = outputTensor.value as List<List<double>>;
      final probabilities = outputData[0];

      // Release output
      outputTensor.release();
      outputs?.forEach((element) => element?.release());

      return probabilities;
    } catch (e) {
      debugPrint('Inference error: $e');
      rethrow;
    }
  }

  /// Post-process model outputs to classification result
  ONNXClassificationResult _postProcessResults(List<double> probabilities) {
    final categories = List<String>.from(_modelConfig?['categories'] ?? []);
    final categoryMapping = Map<String, String>.from(
      _modelConfig?['category_to_app_category'] ?? {},
    );

    if (categories.isEmpty) {
      return ONNXClassificationResult.error();
    }

    // Find top prediction
    int maxIndex = 0;
    double maxProb = probabilities[0];

    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }

    // Apply softmax to get proper probabilities
    final softmax = _applySoftmax(probabilities);

    // Get predicted category
    final predictedCategory = categories[maxIndex];
    final appCategory = categoryMapping[predictedCategory] ?? 'Tops';
    final confidence = softmax[maxIndex];

    // Get all predictions sorted by confidence
    final allPredictions = <String, double>{};
    for (int i = 0; i < categories.length; i++) {
      allPredictions[categories[i]] = softmax[i];
    }

    return ONNXClassificationResult(
      category: predictedCategory,
      appCategory: appCategory,
      confidence: confidence,
      allPredictions: allPredictions,
    );
  }

  /// Apply softmax to convert logits to probabilities
  List<double> _applySoftmax(List<double> logits) {
    // Find max for numerical stability
    double maxLogit = logits.reduce((a, b) => a > b ? a : b);

    // Compute exp(x - max)
    final expValues = logits.map((x) => exp(x - maxLogit)).toList();

    // Compute sum
    final sumExp = expValues.reduce((a, b) => a + b);

    // Normalize
    return expValues.map((x) => x / sumExp).toList();
  }

  /// Simple exponential function
  double exp(double x) {
    return math.exp(x);
  }

  /// Get model status information
  Map<String, dynamic> getModelStatus() {
    return {
      'initialized': _isInitialized,
      'model_loaded': _session != null,
      'model_name': _modelConfig?['model_name'] ?? 'Unknown',
      'framework': 'ONNX',
      'platform': Platform.isIOS ? 'iOS' : 'Android',
    };
  }

  /// Dispose resources
  void dispose() {
    _session?.release();
    _session = null;
    _isInitialized = false;
    debugPrint('ONNX Classifier disposed');
  }
}

/// ONNX Classification Result
class ONNXClassificationResult {
  final String category;
  final String appCategory;
  final double confidence;
  final Map<String, double> allPredictions;
  final bool isError;

  ONNXClassificationResult({
    required this.category,
    required this.appCategory,
    required this.confidence,
    required this.allPredictions,
    this.isError = false,
  });

  factory ONNXClassificationResult.error() {
    return ONNXClassificationResult(
      category: 'Unknown',
      appCategory: 'Tops',
      confidence: 0.0,
      allPredictions: {},
      isError: true,
    );
  }

  bool get isConfident => confidence >= 0.7;

  Map<String, dynamic> toJson() {
    return {
      'category': category,
      'appCategory': appCategory,
      'confidence': confidence,
      'isConfident': isConfident,
      'allPredictions': allPredictions,
    };
  }
}

