import Foundation
import UIKit
import Accelerate

/// CIELAB Color Theory Engine for PrismStyle AI
/// Provides perceptually uniform color distance calculations using ΔE 2000
/// and color harmony analysis for outfit compatibility scoring.
///
/// Key Features:
/// - CIELAB ΔE 2000 color distance (industry standard)
/// - Color harmony detection (complementary, analogous, triadic, etc.)
/// - Seasonal color analysis
/// - Fashion-specific color rules
///
/// Reference: CIE 15:2004, CIEDE2000 formula
@available(iOS 15.0, *)
public final class ColorTheory {
    
    // MARK: - Singleton
    
    public static let shared = ColorTheory()
    private init() {}
    
    // MARK: - Types
    
    /// CIELAB color representation
    public struct LabColor: Equatable, Codable {
        public let L: Double  // Lightness (0-100)
        public let a: Double  // Green-Red axis (-128 to +127)
        public let b: Double  // Blue-Yellow axis (-128 to +127)
        
        public init(L: Double, a: Double, b: Double) {
            self.L = L
            self.a = a
            self.b = b
        }
        
        /// Chroma (colorfulness) - distance from neutral axis
        public var chroma: Double {
            return sqrt(a * a + b * b)
        }
        
        /// Hue angle in radians
        public var hueAngle: Double {
            return atan2(b, a)
        }
        
        /// Hue angle in degrees (0-360)
        public var hueDegrees: Double {
            var h = hueAngle * 180.0 / .pi
            if h < 0 { h += 360 }
            return h
        }
    }
    
    /// Color harmony types
    public enum HarmonyType: String, CaseIterable {
        case complementary      // 180° apart
        case analogous          // 30° apart
        case triadic            // 120° apart
        case splitComplementary // 150° and 210°
        case tetradic           // 90° apart (square)
        case monochromatic      // Same hue, different L/C
        case neutral            // Low chroma (grays)
    }
    
    /// Color temperature
    public enum ColorTemperature {
        case warm       // Reds, oranges, yellows
        case cool       // Blues, greens, purples
        case neutral    // Grays, browns
    }
    
    /// Seasonal color palette
    public enum SeasonalPalette: String, CaseIterable {
        case spring     // Warm, bright, clear
        case summer     // Cool, soft, muted
        case autumn     // Warm, muted, rich
        case winter     // Cool, bright, clear
    }
    
    // MARK: - Color Conversion
    
    /// Convert RGB (0-255) to CIELAB
    public func rgbToLab(r: Int, g: Int, b: Int) -> LabColor {
        // Step 1: RGB to XYZ (sRGB with D65 illuminant)
        let xyz = rgbToXYZ(r: r, g: g, b: b)
        
        // Step 2: XYZ to Lab
        return xyzToLab(x: xyz.x, y: xyz.y, z: xyz.z)
    }
    
    /// Convert UIColor to CIELAB
    public func uiColorToLab(_ color: UIColor) -> LabColor {
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        color.getRed(&r, green: &g, blue: &b, alpha: &a)
        return rgbToLab(r: Int(r * 255), g: Int(g * 255), b: Int(b * 255))
    }
    
    /// Convert hex string to CIELAB
    public func hexToLab(_ hex: String) -> LabColor? {
        var hexSanitized = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        hexSanitized = hexSanitized.replacingOccurrences(of: "#", with: "")
        
        guard hexSanitized.count == 6,
              let rgb = UInt32(hexSanitized, radix: 16) else {
            return nil
        }
        
        let r = Int((rgb >> 16) & 0xFF)
        let g = Int((rgb >> 8) & 0xFF)
        let b = Int(rgb & 0xFF)
        
        return rgbToLab(r: r, g: g, b: b)
    }
    
    /// Convert CIELAB to RGB (0-255)
    public func labToRGB(_ lab: LabColor) -> (r: Int, g: Int, b: Int) {
        // Lab to XYZ
        let xyz = labToXYZ(lab)
        
        // XYZ to RGB
        return xyzToRGB(xyz)
    }
    
    // MARK: - Private Conversion Helpers
    
    private func rgbToXYZ(r: Int, g: Int, b: Int) -> (x: Double, y: Double, z: Double) {
        // Normalize to 0-1 and apply sRGB gamma correction
        func linearize(_ v: Double) -> Double {
            return v > 0.04045 ? pow((v + 0.055) / 1.055, 2.4) : v / 12.92
        }
        
        let rLin = linearize(Double(r) / 255.0)
        let gLin = linearize(Double(g) / 255.0)
        let bLin = linearize(Double(b) / 255.0)
        
        // sRGB to XYZ matrix (D65 illuminant)
        let x = rLin * 0.4124564 + gLin * 0.3575761 + bLin * 0.1804375
        let y = rLin * 0.2126729 + gLin * 0.7151522 + bLin * 0.0721750
        let z = rLin * 0.0193339 + gLin * 0.1191920 + bLin * 0.9503041
        
        return (x * 100, y * 100, z * 100)
    }
    
    private func xyzToLab(x: Double, y: Double, z: Double) -> LabColor {
        // D65 reference white
        let refX = 95.047
        let refY = 100.000
        let refZ = 108.883
        
        func f(_ t: Double) -> Double {
            let delta = 6.0 / 29.0
            return t > pow(delta, 3) ? pow(t, 1.0/3.0) : t / (3 * delta * delta) + 4.0/29.0
        }
        
        let fx = f(x / refX)
        let fy = f(y / refY)
        let fz = f(z / refZ)
        
        let L = 116.0 * fy - 16.0
        let a = 500.0 * (fx - fy)
        let b = 200.0 * (fy - fz)
        
        return LabColor(L: L, a: a, b: b)
    }
    
    private func labToXYZ(_ lab: LabColor) -> (x: Double, y: Double, z: Double) {
        let refX = 95.047
        let refY = 100.000
        let refZ = 108.883
        
        let fy = (lab.L + 16.0) / 116.0
        let fx = lab.a / 500.0 + fy
        let fz = fy - lab.b / 200.0
        
        let delta = 6.0 / 29.0
        
        func fInv(_ t: Double) -> Double {
            return t > delta ? pow(t, 3) : 3 * delta * delta * (t - 4.0/29.0)
        }
        
        return (fInv(fx) * refX, fInv(fy) * refY, fInv(fz) * refZ)
    }
    
    private func xyzToRGB(_ xyz: (x: Double, y: Double, z: Double)) -> (r: Int, g: Int, b: Int) {
        let x = xyz.x / 100.0
        let y = xyz.y / 100.0
        let z = xyz.z / 100.0
        
        // XYZ to sRGB matrix
        var r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
        var g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
        var b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252
        
        // Apply sRGB gamma
        func gammaCorrect(_ v: Double) -> Double {
            return v > 0.0031308 ? 1.055 * pow(v, 1.0/2.4) - 0.055 : 12.92 * v
        }
        
        r = gammaCorrect(r)
        g = gammaCorrect(g)
        b = gammaCorrect(b)
        
        // Clamp and convert to 0-255
        return (
            Int(max(0, min(255, r * 255)).rounded()),
            Int(max(0, min(255, g * 255)).rounded()),
            Int(max(0, min(255, b * 255)).rounded())
        )
    }
    
    // MARK: - ΔE 2000 Color Distance
    
    /// Calculate CIEDE2000 color difference
    /// This is the industry-standard perceptual color distance metric.
    /// Returns a value where:
    /// - ΔE < 1: Not perceptible by human eyes
    /// - ΔE 1-2: Perceptible through close observation
    /// - ΔE 2-10: Perceptible at a glance
    /// - ΔE 11-49: Colors are more similar than opposite
    /// - ΔE 50-100: Colors are opposite
    public func deltaE2000(_ lab1: LabColor, _ lab2: LabColor) -> Double {
        let kL: Double = 1.0
        let kC: Double = 1.0
        let kH: Double = 1.0
        
        let L1 = lab1.L, a1 = lab1.a, b1 = lab1.b
        let L2 = lab2.L, a2 = lab2.a, b2 = lab2.b
        
        // Step 1: Calculate C'
        let C1 = sqrt(a1 * a1 + b1 * b1)
        let C2 = sqrt(a2 * a2 + b2 * b2)
        let Cbar = (C1 + C2) / 2.0
        
        let Cbar7 = pow(Cbar, 7)
        let G = 0.5 * (1.0 - sqrt(Cbar7 / (Cbar7 + pow(25.0, 7))))
        
        let a1Prime = a1 * (1.0 + G)
        let a2Prime = a2 * (1.0 + G)
        
        let C1Prime = sqrt(a1Prime * a1Prime + b1 * b1)
        let C2Prime = sqrt(a2Prime * a2Prime + b2 * b2)
        
        // Step 2: Calculate h'
        func hPrime(_ aPrime: Double, _ b: Double) -> Double {
            if aPrime == 0 && b == 0 { return 0 }
            var h = atan2(b, aPrime) * 180.0 / .pi
            if h < 0 { h += 360.0 }
            return h
        }
        
        let h1Prime = hPrime(a1Prime, b1)
        let h2Prime = hPrime(a2Prime, b2)
        
        // Step 3: Calculate ΔL', ΔC', ΔH'
        let deltaLPrime = L2 - L1
        let deltaCPrime = C2Prime - C1Prime
        
        var deltahPrime: Double
        if C1Prime * C2Prime == 0 {
            deltahPrime = 0
        } else {
            var dhp = h2Prime - h1Prime
            if dhp > 180 { dhp -= 360 }
            if dhp < -180 { dhp += 360 }
            deltahPrime = dhp
        }
        
        let deltaHPrime = 2.0 * sqrt(C1Prime * C2Prime) * sin(deltahPrime * .pi / 360.0)
        
        // Step 4: Calculate CIEDE2000
        let LbarPrime = (L1 + L2) / 2.0
        let CbarPrime = (C1Prime + C2Prime) / 2.0
        
        var HbarPrime: Double
        if C1Prime * C2Prime == 0 {
            HbarPrime = h1Prime + h2Prime
        } else {
            if abs(h1Prime - h2Prime) <= 180 {
                HbarPrime = (h1Prime + h2Prime) / 2.0
            } else if h1Prime + h2Prime < 360 {
                HbarPrime = (h1Prime + h2Prime + 360) / 2.0
            } else {
                HbarPrime = (h1Prime + h2Prime - 360) / 2.0
            }
        }
        
        let T = 1.0 - 0.17 * cos((HbarPrime - 30) * .pi / 180)
                    + 0.24 * cos(2 * HbarPrime * .pi / 180)
                    + 0.32 * cos((3 * HbarPrime + 6) * .pi / 180)
                    - 0.20 * cos((4 * HbarPrime - 63) * .pi / 180)
        
        let deltaTheta = 30 * exp(-pow((HbarPrime - 275) / 25, 2))
        
        let CbarPrime7 = pow(CbarPrime, 7)
        let RC = 2.0 * sqrt(CbarPrime7 / (CbarPrime7 + pow(25.0, 7)))
        
        let LbarPrimeMinus50Sq = pow(LbarPrime - 50, 2)
        let SL = 1.0 + (0.015 * LbarPrimeMinus50Sq) / sqrt(20 + LbarPrimeMinus50Sq)
        let SC = 1.0 + 0.045 * CbarPrime
        let SH = 1.0 + 0.015 * CbarPrime * T
        
        let RT = -sin(2 * deltaTheta * .pi / 180) * RC
        
        let deltaE = sqrt(
            pow(deltaLPrime / (kL * SL), 2) +
            pow(deltaCPrime / (kC * SC), 2) +
            pow(deltaHPrime / (kH * SH), 2) +
            RT * (deltaCPrime / (kC * SC)) * (deltaHPrime / (kH * SH))
        )
        
        return deltaE
    }
    
    /// Calculate color similarity score (0-1)
    /// Higher = more similar
    public func colorSimilarity(_ lab1: LabColor, _ lab2: LabColor) -> Double {
        let deltaE = deltaE2000(lab1, lab2)
        // Use exponential decay: similarity drops off as ΔE increases
        // ΔE of ~10 gives ~0.5 similarity, ΔE of ~30 gives ~0.1
        return exp(-deltaE / 15.0)
    }
    
    // MARK: - Color Harmony Analysis
    
    /// Detect harmony type between two colors
    public func detectHarmony(_ lab1: LabColor, _ lab2: LabColor) -> HarmonyType {
        // Check for neutral colors (low chroma)
        if lab1.chroma < 10 || lab2.chroma < 10 {
            return .neutral
        }
        
        // Check for monochromatic (same hue, different L/C)
        let hueDiff = abs(lab1.hueDegrees - lab2.hueDegrees)
        let normalizedHueDiff = min(hueDiff, 360 - hueDiff)
        
        if normalizedHueDiff < 15 {
            return .monochromatic
        } else if abs(normalizedHueDiff - 30) < 15 {
            return .analogous
        } else if abs(normalizedHueDiff - 90) < 15 {
            return .tetradic
        } else if abs(normalizedHueDiff - 120) < 15 {
            return .triadic
        } else if abs(normalizedHueDiff - 150) < 15 || abs(normalizedHueDiff - 210) < 15 {
            return .splitComplementary
        } else if abs(normalizedHueDiff - 180) < 15 {
            return .complementary
        }
        
        // Default to closest match
        if normalizedHueDiff < 45 {
            return .analogous
        } else if normalizedHueDiff > 135 {
            return .complementary
        }
        
        return .triadic
    }
    
    /// Calculate harmony score for two colors (0-1)
    /// Higher = better harmony according to color theory
    public func harmonyScore(_ lab1: LabColor, _ lab2: LabColor) -> Double {
        let harmony = detectHarmony(lab1, lab2)
        
        // Base scores for each harmony type (fashion-appropriate)
        let baseScores: [HarmonyType: Double] = [
            .monochromatic: 0.95,       // Safe, always works
            .analogous: 0.90,           // Natural, pleasing
            .complementary: 0.85,       // Bold, eye-catching
            .neutral: 0.88,             // Versatile
            .triadic: 0.75,             // Requires skill
            .splitComplementary: 0.80,  // Vibrant but balanced
            .tetradic: 0.70             // Complex, can be overwhelming
        ]
        
        var score = baseScores[harmony] ?? 0.75
        
        // Adjust based on lightness contrast (moderate contrast is good)
        let lightnessDiff = abs(lab1.L - lab2.L)
        if lightnessDiff > 20 && lightnessDiff < 60 {
            score += 0.05  // Good contrast bonus
        } else if lightnessDiff > 80 {
            score -= 0.05  // Too extreme penalty
        }
        
        // Adjust based on chroma balance
        let chromaDiff = abs(lab1.chroma - lab2.chroma)
        if chromaDiff < 20 {
            score += 0.03  // Similar saturation bonus
        }
        
        return min(1.0, max(0.0, score))
    }
    
    /// Calculate harmony score for multiple colors (outfit)
    public func outfitHarmonyScore(_ colors: [LabColor]) -> Double {
        guard colors.count >= 2 else { return 1.0 }
        
        var totalScore: Double = 0
        var pairCount = 0
        
        // Calculate pairwise harmony scores
        for i in 0..<colors.count {
            for j in (i+1)..<colors.count {
                totalScore += harmonyScore(colors[i], colors[j])
                pairCount += 1
            }
        }
        
        let avgPairScore = pairCount > 0 ? totalScore / Double(pairCount) : 0.5
        
        // Penalty for too many colors
        let colorCountPenalty: Double
        switch colors.count {
        case 1...3: colorCountPenalty = 0
        case 4: colorCountPenalty = 0.05
        case 5: colorCountPenalty = 0.10
        default: colorCountPenalty = 0.15
        }
        
        return max(0, avgPairScore - colorCountPenalty)
    }
    
    // MARK: - Color Temperature
    
    /// Determine color temperature
    public func colorTemperature(_ lab: LabColor) -> ColorTemperature {
        // Low chroma = neutral
        if lab.chroma < 15 {
            return .neutral
        }
        
        let hue = lab.hueDegrees
        
        // Warm colors: 0-60° (red to yellow) and 300-360° (red-magenta)
        if hue < 60 || hue > 300 {
            return .warm
        }
        
        // Cool colors: 180-300° (cyan to purple)
        if hue > 180 && hue <= 300 {
            return .cool
        }
        
        // Transitional zones
        if hue >= 60 && hue <= 120 {
            return lab.b > 0 ? .warm : .neutral  // Yellow-green area
        }
        
        return .cool
    }
    
    /// Check if colors have compatible temperatures
    public func temperatureCompatibility(_ lab1: LabColor, _ lab2: LabColor) -> Double {
        let temp1 = colorTemperature(lab1)
        let temp2 = colorTemperature(lab2)
        
        // Same temperature = high compatibility
        if temp1 == temp2 {
            return 1.0
        }
        
        // Neutral pairs well with anything
        if temp1 == .neutral || temp2 == .neutral {
            return 0.9
        }
        
        // Warm + Cool can work but needs balance
        return 0.65
    }
    
    // MARK: - Seasonal Color Analysis
    
    /// Determine which seasonal palette a color belongs to
    public func seasonalPalette(_ lab: LabColor) -> SeasonalPalette {
        let isWarm = colorTemperature(lab) == .warm
        let isBright = lab.L > 60 && lab.chroma > 30
        let isMuted = lab.chroma < 40
        let isDeep = lab.L < 50
        
        if isWarm {
            if isBright { return .spring }
            if isMuted || isDeep { return .autumn }
            return .autumn
        } else {
            if isBright { return .winter }
            if isMuted { return .summer }
            return .winter
        }
    }
    
    /// Calculate seasonal compatibility score
    public func seasonalCompatibility(_ colors: [LabColor]) -> Double {
        guard colors.count >= 2 else { return 1.0 }
        
        let palettes = colors.map { seasonalPalette($0) }
        let uniquePalettes = Set(palettes)
        
        // All same season = perfect
        if uniquePalettes.count == 1 {
            return 1.0
        }
        
        // Compatible seasons (Spring-Autumn both warm, Summer-Winter both cool)
        let hasWarm = palettes.contains(.spring) || palettes.contains(.autumn)
        let hasCool = palettes.contains(.summer) || palettes.contains(.winter)
        
        if hasWarm && hasCool {
            return 0.6  // Mixed temperature palette
        }
        
        return 0.85  // Same temperature family
    }
    
    // MARK: - Fashion-Specific Rules
    
    /// Check if colors follow the "no more than 3 colors" rule
    public func followsThreeColorRule(_ colors: [LabColor]) -> Bool {
        // Filter out near-neutral colors
        let significantColors = colors.filter { $0.chroma > 15 }
        return significantColors.count <= 3
    }
    
    /// Calculate neutral ratio (good outfits often have neutral anchors)
    public func neutralRatio(_ colors: [LabColor]) -> Double {
        guard !colors.isEmpty else { return 0.5 }
        let neutralCount = colors.filter { $0.chroma < 15 }.count
        return Double(neutralCount) / Double(colors.count)
    }
    
    /// Calculate overall fashion color score for an outfit
    public func fashionColorScore(_ colors: [LabColor]) -> Double {
        guard colors.count >= 2 else { return 0.8 }
        
        var score: Double = 0
        
        // 1. Base harmony (40% weight)
        score += outfitHarmonyScore(colors) * 0.4
        
        // 2. Temperature compatibility (20% weight)
        var tempScore: Double = 0
        var tempPairs = 0
        for i in 0..<colors.count {
            for j in (i+1)..<colors.count {
                tempScore += temperatureCompatibility(colors[i], colors[j])
                tempPairs += 1
            }
        }
        score += (tempPairs > 0 ? tempScore / Double(tempPairs) : 0.5) * 0.2
        
        // 3. Seasonal compatibility (15% weight)
        score += seasonalCompatibility(colors) * 0.15
        
        // 4. Three-color rule (10% weight)
        score += (followsThreeColorRule(colors) ? 1.0 : 0.6) * 0.1
        
        // 5. Neutral anchoring (15% weight)
        let neutral = neutralRatio(colors)
        let neutralScore = neutral > 0.2 && neutral < 0.7 ? 1.0 : 0.7
        score += neutralScore * 0.15
        
        return min(1.0, max(0.0, score))
    }
}
