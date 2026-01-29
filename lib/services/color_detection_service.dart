import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

/// Color Detection Service for PrismStyle AI
/// Uses K-means clustering algorithm for dominant color extraction
/// 
/// Source: OpenCV color extraction algorithm adapted for Flutter
/// Algorithm: K-means clustering for RGB values
class ColorDetectionService {
  static ColorDetectionService? _instance;
  static ColorDetectionService get instance =>
      _instance ??= ColorDetectionService._();

  ColorDetectionService._();

  /// Color database for name mapping
  static const Map<String, List<int>> colorDatabase = {
    // Neutrals
    'White': [255, 255, 255],
    'Black': [0, 0, 0],
    'Gray': [128, 128, 128],
    'Light Gray': [192, 192, 192],
    'Dark Gray': [64, 64, 64],
    'Charcoal': [54, 69, 79],
    
    // Reds
    'Red': [255, 0, 0],
    'Crimson': [220, 20, 60],
    'Maroon': [128, 0, 0],
    'Burgundy': [128, 0, 32],
    'Coral': [255, 127, 80],
    'Salmon': [250, 128, 114],
    
    // Oranges
    'Orange': [255, 165, 0],
    'Peach': [255, 218, 185],
    'Tangerine': [255, 154, 0],
    'Rust': [183, 65, 14],
    
    // Yellows
    'Yellow': [255, 255, 0],
    'Gold': [255, 215, 0],
    'Mustard': [225, 173, 1],
    'Cream': [255, 253, 208],
    'Beige': [245, 245, 220],
    'Khaki': [195, 176, 145],
    
    // Greens
    'Green': [0, 128, 0],
    'Lime': [0, 255, 0],
    'Forest Green': [34, 139, 34],
    'Olive': [128, 128, 0],
    'Mint': [152, 255, 152],
    'Teal': [0, 128, 128],
    'Sage': [188, 184, 138],
    
    // Blues
    'Blue': [0, 0, 255],
    'Navy': [0, 0, 128],
    'Royal Blue': [65, 105, 225],
    'Sky Blue': [135, 206, 235],
    'Light Blue': [173, 216, 230],
    'Powder Blue': [176, 224, 230],
    'Denim': [21, 96, 189],
    'Indigo': [75, 0, 130],
    
    // Purples
    'Purple': [128, 0, 128],
    'Lavender': [230, 230, 250],
    'Violet': [238, 130, 238],
    'Plum': [221, 160, 221],
    'Magenta': [255, 0, 255],
    'Mauve': [224, 176, 255],
    
    // Pinks
    'Pink': [255, 192, 203],
    'Hot Pink': [255, 105, 180],
    'Blush': [222, 93, 131],
    'Rose': [255, 0, 127],
    'Dusty Rose': [199, 134, 147],
    
    // Browns
    'Brown': [139, 69, 19],
    'Tan': [210, 180, 140],
    'Chocolate': [123, 63, 0],
    'Camel': [193, 154, 107],
    'Taupe': [72, 60, 50],
    'Espresso': [56, 30, 17],
  };

  /// Extract dominant colors from an image file
  Future<ColorExtractionResult> extractColorsFromFile(
    String imagePath, {
    int numColors = 5,
  }) async {
    try {
      final file = File(imagePath);
      if (!await file.exists()) {
        throw Exception('Image file not found: $imagePath');
      }

      final bytes = await file.readAsBytes();
      return extractColorsFromBytes(bytes, numColors: numColors);
    } catch (e) {
      debugPrint('Error extracting colors from file: $e');
      return ColorExtractionResult.empty();
    }
  }

  /// Extract dominant colors from image bytes
  Future<ColorExtractionResult> extractColorsFromBytes(
    Uint8List imageBytes, {
    int numColors = 5,
  }) async {
    try {
      final image = img.decodeImage(imageBytes);
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      return await _extractColors(image, numColors);
    } catch (e) {
      debugPrint('Error extracting colors: $e');
      return ColorExtractionResult.empty();
    }
  }

  /// Main color extraction using K-means clustering
  Future<ColorExtractionResult> _extractColors(
    img.Image image,
    int numColors,
  ) async {
    // Resize image for faster processing
    final resized = img.copyResize(
      image,
      width: min(image.width, 100),
      height: min(image.height, 100),
    );

    // Extract all pixel colors
    final pixels = <List<int>>[];
    for (int y = 0; y < resized.height; y++) {
      for (int x = 0; x < resized.width; x++) {
        final pixel = resized.getPixel(x, y);
        final r = pixel.r.toInt();
        final g = pixel.g.toInt();
        final b = pixel.b.toInt();
        
        // Skip very bright (white background) and very dark pixels
        final brightness = (r + g + b) / 3;
        if (brightness > 20 && brightness < 240) {
          pixels.add([r, g, b]);
        }
      }
    }

    if (pixels.isEmpty) {
      return ColorExtractionResult.empty();
    }

    // Run K-means clustering
    final clusters = await _kMeansClustering(pixels, numColors);
    
    // Convert clusters to color results
    final extractedColors = <ExtractedColor>[];
    for (var cluster in clusters) {
      final colorName = _findClosestColorName(
        cluster.centroid[0],
        cluster.centroid[1],
        cluster.centroid[2],
      );
      
      extractedColors.add(ExtractedColor(
        r: cluster.centroid[0],
        g: cluster.centroid[1],
        b: cluster.centroid[2],
        name: colorName,
        percentage: cluster.percentage,
        hex: _rgbToHex(
          cluster.centroid[0],
          cluster.centroid[1],
          cluster.centroid[2],
        ),
      ));
    }

    // Sort by percentage (dominant first)
    extractedColors.sort((a, b) => b.percentage.compareTo(a.percentage));

    return ColorExtractionResult(
      colors: extractedColors,
      primaryColor: extractedColors.isNotEmpty ? extractedColors.first : null,
      secondaryColor: extractedColors.length > 1 ? extractedColors[1] : null,
    );
  }

  /// K-means clustering algorithm
  Future<List<ColorCluster>> _kMeansClustering(
    List<List<int>> pixels,
    int k,
  ) async {
    if (pixels.isEmpty || k <= 0) return [];

    final random = Random();
    
    // Initialize centroids randomly
    var centroids = <List<int>>[];
    for (int i = 0; i < k; i++) {
      centroids.add(List.from(pixels[random.nextInt(pixels.length)]));
    }

    List<List<List<int>>> clusters = List.generate(k, (_) => <List<int>>[]);
    
    // Iterate until convergence or max iterations
    const maxIterations = 20;
    for (int iteration = 0; iteration < maxIterations; iteration++) {
      // Clear clusters
      clusters = List.generate(k, (_) => <List<int>>[]);

      // Assign pixels to nearest centroid
      for (var pixel in pixels) {
        int nearestIdx = 0;
        double minDist = double.infinity;

        for (int i = 0; i < k; i++) {
          final dist = _colorDistance(pixel, centroids[i]);
          if (dist < minDist) {
            minDist = dist;
            nearestIdx = i;
          }
        }

        clusters[nearestIdx].add(pixel);
      }

      // Update centroids
      var converged = true;
      for (int i = 0; i < k; i++) {
        if (clusters[i].isEmpty) continue;

        final newCentroid = _calculateCentroid(clusters[i]);
        if (_colorDistance(centroids[i], newCentroid) > 1) {
          converged = false;
        }
        centroids[i] = newCentroid;
      }

      if (converged) break;
    }

    // Convert to ColorCluster objects
    final totalPixels = pixels.length;
    final result = <ColorCluster>[];

    for (int i = 0; i < k; i++) {
      if (clusters[i].isNotEmpty) {
        result.add(ColorCluster(
          centroid: centroids[i],
          percentage: clusters[i].length / totalPixels * 100,
          pixelCount: clusters[i].length,
        ));
      }
    }

    return result;
  }

  /// Calculate Euclidean distance between two colors
  double _colorDistance(List<int> c1, List<int> c2) {
    return sqrt(
      pow(c1[0] - c2[0], 2) +
      pow(c1[1] - c2[1], 2) +
      pow(c1[2] - c2[2], 2),
    );
  }

  /// Calculate centroid of a cluster
  List<int> _calculateCentroid(List<List<int>> cluster) {
    if (cluster.isEmpty) return [0, 0, 0];

    int sumR = 0, sumG = 0, sumB = 0;
    for (var pixel in cluster) {
      sumR += pixel[0];
      sumG += pixel[1];
      sumB += pixel[2];
    }

    return [
      sumR ~/ cluster.length,
      sumG ~/ cluster.length,
      sumB ~/ cluster.length,
    ];
  }

  /// Find closest color name from database
  String _findClosestColorName(int r, int g, int b) {
    String closestName = 'Unknown';
    double minDistance = double.infinity;

    colorDatabase.forEach((name, rgb) {
      final distance = _colorDistance([r, g, b], rgb);
      if (distance < minDistance) {
        minDistance = distance;
        closestName = name;
      }
    });

    return closestName;
  }

  /// Convert RGB to hex string
  String _rgbToHex(int r, int g, int b) {
    return '#${r.toRadixString(16).padLeft(2, '0')}'
        '${g.toRadixString(16).padLeft(2, '0')}'
        '${b.toRadixString(16).padLeft(2, '0')}'.toUpperCase();
  }

  /// Check if two colors are compatible (for outfit matching)
  bool areColorsCompatible(ExtractedColor color1, ExtractedColor color2) {
    // Define compatible color combinations
    final compatiblePairs = <List<String>>[
      ['White', 'Black', 'Navy', 'Gray', 'Charcoal'],
      ['Blue', 'White', 'Khaki', 'Tan', 'Beige'],
      ['Navy', 'White', 'Cream', 'Gold', 'Mustard'],
      ['Black', 'White', 'Red', 'Pink', 'Gold'],
      ['Gray', 'Pink', 'Blue', 'Purple', 'Lavender'],
      ['Beige', 'Brown', 'Navy', 'Burgundy', 'Olive'],
      ['Olive', 'Cream', 'White', 'Brown', 'Tan'],
    ];

    // Check if colors are in compatible pairs
    for (var pair in compatiblePairs) {
      if (pair.contains(color1.name) && pair.contains(color2.name)) {
        return true;
      }
    }

    // Also check color wheel compatibility
    return _checkColorWheelCompatibility(color1, color2);
  }

  /// Check color wheel compatibility (complementary, analogous, etc.)
  bool _checkColorWheelCompatibility(
    ExtractedColor color1,
    ExtractedColor color2,
  ) {
    // Convert RGB to HSL for color wheel analysis
    final hsl1 = _rgbToHsl(color1.r, color1.g, color1.b);
    final hsl2 = _rgbToHsl(color2.r, color2.g, color2.b);

    final hueDiff = (hsl1[0] - hsl2[0]).abs();

    // Complementary colors (opposite on color wheel)
    if (hueDiff > 150 && hueDiff < 210) return true;

    // Analogous colors (adjacent on color wheel)
    if (hueDiff < 30 || hueDiff > 330) return true;

    // Triadic colors (120 degrees apart)
    if ((hueDiff > 110 && hueDiff < 130) || (hueDiff > 230 && hueDiff < 250)) {
      return true;
    }

    // Neutral colors are compatible with everything
    if (hsl1[1] < 0.15 || hsl2[1] < 0.15) return true;

    return false;
  }

  /// Convert RGB to HSL
  List<double> _rgbToHsl(int r, int g, int b) {
    final rNorm = r / 255;
    final gNorm = g / 255;
    final bNorm = b / 255;

    final maxVal = max(max(rNorm, gNorm), bNorm);
    final minVal = min(min(rNorm, gNorm), bNorm);
    final delta = maxVal - minVal;

    final l = (maxVal + minVal) / 2;

    double h = 0;
    double s = 0;

    if (delta != 0) {
      s = l > 0.5 ? delta / (2 - maxVal - minVal) : delta / (maxVal + minVal);

      if (maxVal == rNorm) {
        h = ((gNorm - bNorm) / delta) % 6;
      } else if (maxVal == gNorm) {
        h = (bNorm - rNorm) / delta + 2;
      } else {
        h = (rNorm - gNorm) / delta + 4;
      }

      h *= 60;
      if (h < 0) h += 360;
    }

    return [h, s, l];
  }

  /// Get suggested color combinations for an outfit
  List<String> getSuggestedColors(ExtractedColor baseColor) {
    final suggestions = <String>[];
    
    // Always suggest neutrals
    suggestions.addAll(['White', 'Black', 'Gray']);
    
    // Get complementary color
    final hsl = _rgbToHsl(baseColor.r, baseColor.g, baseColor.b);
    final compHue = (hsl[0] + 180) % 360;
    
    // Find colors close to complementary hue
    colorDatabase.forEach((name, rgb) {
      final colorHsl = _rgbToHsl(rgb[0], rgb[1], rgb[2]);
      final hueDiff = (colorHsl[0] - compHue).abs();
      if (hueDiff < 30 || hueDiff > 330) {
        if (!suggestions.contains(name)) {
          suggestions.add(name);
        }
      }
    });
    
    return suggestions.take(8).toList();
  }
}

/// Color cluster result from K-means
class ColorCluster {
  final List<int> centroid;
  final double percentage;
  final int pixelCount;

  ColorCluster({
    required this.centroid,
    required this.percentage,
    required this.pixelCount,
  });
}

/// Single extracted color
class ExtractedColor {
  final int r;
  final int g;
  final int b;
  final String name;
  final double percentage;
  final String hex;

  ExtractedColor({
    required this.r,
    required this.g,
    required this.b,
    required this.name,
    required this.percentage,
    required this.hex,
  });

  Map<String, dynamic> toJson() {
    return {
      'r': r,
      'g': g,
      'b': b,
      'name': name,
      'percentage': percentage.toStringAsFixed(1),
      'hex': hex,
    };
  }
}

/// Color extraction result
class ColorExtractionResult {
  final List<ExtractedColor> colors;
  final ExtractedColor? primaryColor;
  final ExtractedColor? secondaryColor;

  ColorExtractionResult({
    required this.colors,
    this.primaryColor,
    this.secondaryColor,
  });

  factory ColorExtractionResult.empty() {
    return ColorExtractionResult(colors: []);
  }

  bool get isEmpty => colors.isEmpty;
  bool get isNotEmpty => colors.isNotEmpty;

  Map<String, dynamic> toJson() {
    return {
      'colors': colors.map((c) => c.toJson()).toList(),
      'primaryColor': primaryColor?.toJson(),
      'secondaryColor': secondaryColor?.toJson(),
    };
  }
}
