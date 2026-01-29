import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:sizer/sizer.dart';

import '../../core/app_export.dart';
import '../../widgets/custom_icon_widget.dart';
import '../../services/clothing_classifier_service.dart';
import '../../services/color_detection_service.dart';
import '../../services/wardrobe_service.dart';
import './widgets/camera_overlay_widget.dart';
import './widgets/capture_controls_widget.dart';
import './widgets/item_details_widget.dart';
import './widgets/processing_overlay_widget.dart';

class CameraCapture extends StatefulWidget {
  const CameraCapture({super.key});

  @override
  State<CameraCapture> createState() => _CameraCaptureState();
}

class _CameraCaptureState extends State<CameraCapture>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isCameraInitialized = false;
  bool _isProcessing = false;
  bool _showItemDetails = false;
  XFile? _capturedImage;
  XFile? _lastGalleryImage;
  String _aiGuidance = 'Position clothing in frame';
  Color _guidanceColor = Colors.white;
  bool _isFlashOn = false;
  double _currentZoom = 1.0;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;

  // AI Services
  final ClothingClassifierService _classifierService =
      ClothingClassifierService.instance;
  final ColorDetectionService _colorService = ColorDetectionService.instance;
  final WardrobeService _wardrobeService = WardrobeService.instance;

  // Detected attributes for captured image
  Map<String, dynamic> _detectedAttributes = {};

  // Real-time classification state
  final bool _isRealTimeEnabled = false;
  DateTime? _lastClassificationTime;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
    _loadLastGalleryImage();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? cameraController = _cameraController;
    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      cameraController.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _initializeCamera() async {
    try {
      if (!await _requestCameraPermission()) {
        _showPermissionDeniedDialog();
        return;
      }

      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        _showNoCameraDialog();
        return;
      }

      final camera = kIsWeb
          ? _cameras.firstWhere(
              (c) => c.lensDirection == CameraLensDirection.front,
              orElse: () => _cameras.first,
            )
          : _cameras.firstWhere(
              (c) => c.lensDirection == CameraLensDirection.back,
              orElse: () => _cameras.first,
            );

      _cameraController = CameraController(
        camera,
        kIsWeb ? ResolutionPreset.medium : ResolutionPreset.high,
        enableAudio: false,
      );

      await _cameraController!.initialize();

      if (!kIsWeb) {
        _minZoom = await _cameraController!.getMinZoomLevel();
        _maxZoom = await _cameraController!.getMaxZoomLevel();
      }

      await _applySettings();

      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
      }
    } catch (e) {
      debugPrint('Camera initialization error: $e');
      if (mounted) {
        _showCameraErrorDialog();
      }
    }
  }

  Future<bool> _requestCameraPermission() async {
    if (kIsWeb) return true;
    final status = await Permission.camera.request();
    return status.isGranted;
  }

  Future<void> _applySettings() async {
    if (_cameraController == null) return;
    try {
      await _cameraController!.setFocusMode(FocusMode.auto);
    } catch (e) {
      debugPrint('Focus mode error: $e');
    }
    if (!kIsWeb) {
      try {
        await _cameraController!.setFlashMode(FlashMode.off);
      } catch (e) {
        debugPrint('Flash mode error: $e');
      }
    }
  }

  Future<void> _loadLastGalleryImage() async {
    try {
      final ImagePicker picker = ImagePicker();
      final List<XFile> images = await picker.pickMultiImage();
      if (images.isNotEmpty && mounted) {
        setState(() {
          _lastGalleryImage = images.last;
        });
      }
    } catch (e) {
      debugPrint('Gallery load error: $e');
    }
  }

  Future<void> _capturePhoto() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    try {
      setState(() {
        _isProcessing = true;
        _aiGuidance = 'Analyzing clothing...';
        _guidanceColor = Colors.blue;
      });

      final XFile photo = await _cameraController!.takePicture();

      // Process image with AI services
      final attributes = await _processImageWithAI(photo);

      if (mounted) {
        setState(() {
          _capturedImage = photo;
          _detectedAttributes = attributes;
          _isProcessing = false;
          _showItemDetails = true;
          _lastGalleryImage = photo;
        });
      }
    } catch (e) {
      debugPrint('Capture error: $e');
      if (mounted) {
        setState(() {
          _isProcessing = false;
          _aiGuidance = 'Capture failed. Try again.';
          _guidanceColor = Colors.red;
        });
      }
    }
  }

  /// Process image with AI clothing classifier and color detection
  Future<Map<String, dynamic>> _processImageWithAI(XFile imageFile) async {
    try {
      // Read image bytes
      final Uint8List imageBytes = await imageFile.readAsBytes();

      // Run clothing classification
      final classification = await _classifierService.classifyImageBytes(
        imageBytes,
      );

      // Run color detection
      final colors = await _colorService.extractColorsFromBytes(imageBytes);

      // Build attributes map with real AI results
      return {
        'category': classification.appCategory,
        'detailedCategory': classification.category,
        'primaryColor':
            colors.primaryColor?.name ?? classification.primaryColor,
        'secondaryColor': colors.secondaryColor?.name ?? 'None',
        'colorHex': colors.primaryColor?.hex ?? '#000000',
        'pattern': classification.pattern ?? 'Solid',
        'material': classification.material ?? 'Unknown',
        'suggestedTags': classification.suggestedTags,
        'confidence': classification.confidence,
        'allColors': colors.colors.map((c) => c.toJson()).toList(),
        'aiPredictions': classification.allPredictions,
      };
    } catch (e) {
      debugPrint('AI processing error: $e');
      // Return fallback attributes if AI fails
      return {
        'category': 'Tops',
        'primaryColor': 'Unknown',
        'secondaryColor': 'None',
        'pattern': 'Solid',
        'material': 'Unknown',
        'suggestedTags': ['unclassified'],
        'confidence': 0.0,
      };
    }
  }

  Future<void> _pickFromGallery() async {
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? image = await picker.pickImage(source: ImageSource.gallery);

      if (image != null) {
        setState(() {
          _isProcessing = true;
          _aiGuidance = 'Analyzing clothing...';
          _guidanceColor = Colors.blue;
        });

        // Process image with AI services
        final attributes = await _processImageWithAI(image);

        if (mounted) {
          setState(() {
            _capturedImage = image;
            _detectedAttributes = attributes;
            _isProcessing = false;
            _showItemDetails = true;
            _lastGalleryImage = image;
          });
        }
      }
    } catch (e) {
      debugPrint('Gallery pick error: $e');
      if (mounted) {
        setState(() {
          _isProcessing = false;
          _aiGuidance = 'Failed to load image';
          _guidanceColor = Colors.red;
        });
      }
    }
  }

  Future<void> _toggleFlash() async {
    if (_cameraController == null || kIsWeb) return;
    try {
      setState(() {
        _isFlashOn = !_isFlashOn;
      });
      await _cameraController!.setFlashMode(
        _isFlashOn ? FlashMode.torch : FlashMode.off,
      );
    } catch (e) {
      debugPrint('Flash toggle error: $e');
    }
  }

  Future<void> _flipCamera() async {
    if (_cameras.length < 2) return;

    try {
      final currentCamera = _cameraController!.description;
      final newCamera = _cameras.firstWhere(
        (camera) => camera.lensDirection != currentCamera.lensDirection,
        orElse: () => _cameras.first,
      );

      await _cameraController?.dispose();

      _cameraController = CameraController(
        newCamera,
        kIsWeb ? ResolutionPreset.medium : ResolutionPreset.high,
        enableAudio: false,
      );

      await _cameraController!.initialize();
      await _applySettings();

      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
      }
    } catch (e) {
      debugPrint('Camera flip error: $e');
    }
  }

  void _handleZoom(ScaleUpdateDetails details) {
    if (kIsWeb || _cameraController == null) return;
    final newZoom = (_currentZoom * details.scale).clamp(_minZoom, _maxZoom);
    _cameraController!.setZoomLevel(newZoom);
    setState(() {
      _currentZoom = newZoom;
    });
  }

  void _handleTapToFocus(TapDownDetails details, BoxConstraints constraints) {
    if (_cameraController == null) return;
    final offset = Offset(
      details.localPosition.dx / constraints.maxWidth,
      details.localPosition.dy / constraints.maxHeight,
    );
    _cameraController!.setFocusPoint(offset);
    _cameraController!.setExposurePoint(offset);
  }

  void _showLightingTips() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
        ),
        padding: EdgeInsets.all(4.w),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Lighting Tips',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 2.h),
            _buildTipItem('Use natural daylight when possible'),
            _buildTipItem('Avoid harsh shadows'),
            _buildTipItem('Position clothing flat or on hanger'),
            _buildTipItem('Ensure even lighting across item'),
            _buildTipItem('Avoid direct flash for best colors'),
            SizedBox(height: 2.h),
          ],
        ),
      ),
    );
  }

  Widget _buildTipItem(String tip) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 1.h),
      child: Row(
        children: [
          CustomIconWidget(
            iconName: 'check_circle',
            color: Theme.of(context).colorScheme.tertiary,
            size: 20,
          ),
          SizedBox(width: 2.w),
          Expanded(
            child: Text(tip, style: Theme.of(context).textTheme.bodyMedium),
          ),
        ],
      ),
    );
  }

  void _confirmAndSave() {
    Navigator.of(
      context,
      rootNavigator: true,
    ).pop({'image': _capturedImage, 'attributes': _detectedAttributes});
  }

  void _retakePhoto() {
    setState(() {
      _capturedImage = null;
      _detectedAttributes = {};
      _showItemDetails = false;
      _aiGuidance = 'Position clothing in frame';
      _guidanceColor = Colors.white;
    });
  }

  void _showPermissionDeniedDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Camera Permission Required'),
        content: const Text(
          'Please grant camera permission to capture wardrobe items.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              openAppSettings();
            },
            child: const Text('Open Settings'),
          ),
        ],
      ),
    );
  }

  void _showNoCameraDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('No Camera Available'),
        content: const Text('No camera was detected on this device.'),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.of(context, rootNavigator: true).pop();
            },
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showCameraErrorDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Camera Error'),
        content: const Text('Failed to initialize camera. Please try again.'),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.of(context, rootNavigator: true).pop();
            },
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    if (_showItemDetails && _capturedImage != null) {
      return ItemDetailsWidget(
        capturedImage: _capturedImage!,
        detectedAttributes: _detectedAttributes,
        onConfirm: _confirmAndSave,
        onRetake: _retakePhoto,
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: _isCameraInitialized
            ? Stack(
                children: [
                  Positioned.fill(
                    child: GestureDetector(
                      onScaleUpdate: _handleZoom,
                      child: LayoutBuilder(
                        builder: (context, constraints) {
                          return GestureDetector(
                            onTapDown: (details) =>
                                _handleTapToFocus(details, constraints),
                            child: CameraPreview(_cameraController!),
                          );
                        },
                      ),
                    ),
                  ),
                  CameraOverlayWidget(
                    aiGuidance: _aiGuidance,
                    guidanceColor: _guidanceColor,
                    isFlashOn: _isFlashOn,
                    onFlashToggle: _toggleFlash,
                    onFlipCamera: _flipCamera,
                    onClose: () =>
                        Navigator.of(context, rootNavigator: true).pop(),
                    showFlash: !kIsWeb,
                  ),
                  if (_isProcessing)
                    const ProcessingOverlayWidget()
                  else
                    CaptureControlsWidget(
                      onCapture: _capturePhoto,
                      onGalleryTap: _pickFromGallery,
                      onLightingTips: _showLightingTips,
                      lastGalleryImage: _lastGalleryImage,
                    ),
                ],
              )
            : Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(
                      color: theme.colorScheme.tertiary,
                    ),
                    SizedBox(height: 2.h),
                    Text(
                      'Initializing camera...',
                      style: theme.textTheme.bodyLarge?.copyWith(
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
      ),
    );
  }
}
