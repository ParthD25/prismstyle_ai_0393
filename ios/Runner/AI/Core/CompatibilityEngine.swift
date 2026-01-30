import Foundation
import UIKit

/// Multi-Factor Compatibility Engine for PrismStyle AI
/// Combines color harmony, occasion matching, style coherence, and user preferences
/// into a unified outfit scoring system.
///
/// Key Features:
/// - Multi-factor weighted scoring
/// - Fashion occasion rules
/// - Style coherence analysis
/// - Integration with UserLearning for personalization
///
/// Usage:
/// let score = CompatibilityEngine.shared.calculateOutfitScore(items, occasion: .casual)
@available(iOS 15.0, *)
public final class CompatibilityEngine {
    
    // MARK: - Singleton
    
    public static let shared = CompatibilityEngine()
    private init() {}
    
    // MARK: - Dependencies
    
    private let colorTheory = ColorTheory.shared
    
    // MARK: - Types
    
    /// Fashion occasions with associated style rules
    public enum Occasion: String, CaseIterable, Codable {
        case casual
        case business
        case formal
        case athletic
        case dateNight
        case outdoor
        case party
        case workFromHome
        case travel
        case beach
        
        /// Required formality level (0-1)
        public var formalityLevel: Double {
            switch self {
            case .casual, .beach: return 0.2
            case .workFromHome, .outdoor, .travel: return 0.3
            case .athletic: return 0.1
            case .business: return 0.7
            case .dateNight, .party: return 0.6
            case .formal: return 0.95
            }
        }
        
        /// Preferred style keywords
        public var styleKeywords: [String] {
            switch self {
            case .casual: return ["relaxed", "comfortable", "everyday"]
            case .business: return ["professional", "polished", "structured"]
            case .formal: return ["elegant", "sophisticated", "dressy"]
            case .athletic: return ["sporty", "functional", "breathable"]
            case .dateNight: return ["attractive", "stylish", "put-together"]
            case .outdoor: return ["practical", "durable", "layered"]
            case .party: return ["fun", "bold", "trendy"]
            case .workFromHome: return ["comfortable", "presentable", "relaxed"]
            case .travel: return ["versatile", "comfortable", "practical"]
            case .beach: return ["light", "breathable", "casual"]
            }
        }
    }
    
    /// Clothing style categories
    public enum StyleCategory: String, CaseIterable, Codable {
        case classic
        case modern
        case bohemian
        case streetwear
        case minimalist
        case preppy
        case romantic
        case edgy
        case sporty
        case vintage
    }
    
    /// Wardrobe item with all scoring attributes
    public struct WardrobeItem: Codable {
        public let id: String
        public let category: String           // Tops, Bottoms, Dresses, etc.
        public let subcategory: String        // T-shirt, Jeans, etc.
        public let dominantColor: ColorTheory.LabColor
        public let accentColors: [ColorTheory.LabColor]
        public let formalityScore: Double     // 0-1
        public let styleCategories: [StyleCategory]
        public let seasonality: [String]      // spring, summer, fall, winter
        public let patterns: [String]         // solid, striped, floral, etc.
        public let materials: [String]        // cotton, silk, denim, etc.
        public let fit: String                // slim, regular, loose, oversized
        
        public init(
            id: String,
            category: String,
            subcategory: String,
            dominantColor: ColorTheory.LabColor,
            accentColors: [ColorTheory.LabColor] = [],
            formalityScore: Double = 0.5,
            styleCategories: [StyleCategory] = [],
            seasonality: [String] = ["spring", "summer", "fall", "winter"],
            patterns: [String] = ["solid"],
            materials: [String] = [],
            fit: String = "regular"
        ) {
            self.id = id
            self.category = category
            self.subcategory = subcategory
            self.dominantColor = dominantColor
            self.accentColors = accentColors
            self.formalityScore = formalityScore
            self.styleCategories = styleCategories
            self.seasonality = seasonality
            self.patterns = patterns
            self.materials = materials
            self.fit = fit
        }
    }
    
    /// Outfit compatibility result
    public struct CompatibilityResult: Codable {
        public let overallScore: Double       // 0-1, final weighted score
        public let colorScore: Double         // Color harmony score
        public let occasionScore: Double      // Occasion appropriateness
        public let styleScore: Double         // Style coherence
        public let seasonScore: Double        // Seasonal appropriateness
        public let patternScore: Double       // Pattern mixing score
        public let formalityScore: Double     // Formality match
        
        public let explanation: String        // Human-readable explanation
        public let suggestions: [String]      // Improvement suggestions
        public let harmonyType: String        // Detected color harmony
        
        public init(
            overallScore: Double,
            colorScore: Double,
            occasionScore: Double,
            styleScore: Double,
            seasonScore: Double,
            patternScore: Double,
            formalityScore: Double,
            explanation: String,
            suggestions: [String],
            harmonyType: String
        ) {
            self.overallScore = overallScore
            self.colorScore = colorScore
            self.occasionScore = occasionScore
            self.styleScore = styleScore
            self.seasonScore = seasonScore
            self.patternScore = patternScore
            self.formalityScore = formalityScore
            self.explanation = explanation
            self.suggestions = suggestions
            self.harmonyType = harmonyType
        }
    }
    
    // MARK: - Scoring Weights (can be personalized via UserLearning)
    
    /// Default scoring weights
    public struct ScoringWeights: Codable {
        public var color: Double = 0.30        // Color harmony
        public var occasion: Double = 0.20     // Occasion match
        public var style: Double = 0.20        // Style coherence
        public var formality: Double = 0.15    // Formality match
        public var season: Double = 0.10       // Seasonal appropriateness
        public var pattern: Double = 0.05      // Pattern mixing
        
        public init() {}
        
        public mutating func normalize() {
            let total = color + occasion + style + formality + season + pattern
            if total > 0 {
                color /= total
                occasion /= total
                style /= total
                formality /= total
                season /= total
                pattern /= total
            }
        }
    }
    
    private var weights = ScoringWeights()
    
    /// Update weights (called by UserLearning)
    public func updateWeights(_ newWeights: ScoringWeights) {
        var w = newWeights
        w.normalize()
        self.weights = w
    }
    
    // MARK: - Main Scoring API
    
    /// Calculate comprehensive outfit compatibility score
    public func calculateOutfitScore(
        items: [WardrobeItem],
        occasion: Occasion = .casual,
        currentSeason: String? = nil
    ) -> CompatibilityResult {
        
        guard items.count >= 2 else {
            return CompatibilityResult(
                overallScore: 0.5,
                colorScore: 1.0,
                occasionScore: 1.0,
                styleScore: 1.0,
                seasonScore: 1.0,
                patternScore: 1.0,
                formalityScore: 1.0,
                explanation: "Need at least 2 items for outfit scoring",
                suggestions: ["Add more items to complete the outfit"],
                harmonyType: "N/A"
            )
        }
        
        // Calculate individual scores
        let colorResult = calculateColorScore(items)
        let occasionScore = calculateOccasionScore(items, occasion: occasion)
        let styleScore = calculateStyleScore(items)
        let seasonScore = calculateSeasonScore(items, currentSeason: currentSeason)
        let patternScore = calculatePatternScore(items)
        let formalityScore = calculateFormalityScore(items, occasion: occasion)
        
        // Weighted combination
        let overallScore = 
            colorResult.score * weights.color +
            occasionScore * weights.occasion +
            styleScore * weights.style +
            formalityScore * weights.formality +
            seasonScore * weights.season +
            patternScore * weights.pattern
        
        // Generate explanation
        let explanation = generateExplanation(
            colorScore: colorResult.score,
            harmonyType: colorResult.harmonyType,
            occasionScore: occasionScore,
            styleScore: styleScore,
            occasion: occasion
        )
        
        // Generate suggestions
        let suggestions = generateSuggestions(
            colorScore: colorResult.score,
            occasionScore: occasionScore,
            styleScore: styleScore,
            formalityScore: formalityScore,
            occasion: occasion,
            items: items
        )
        
        return CompatibilityResult(
            overallScore: min(1.0, max(0.0, overallScore)),
            colorScore: colorResult.score,
            occasionScore: occasionScore,
            styleScore: styleScore,
            seasonScore: seasonScore,
            patternScore: patternScore,
            formalityScore: formalityScore,
            explanation: explanation,
            suggestions: suggestions,
            harmonyType: colorResult.harmonyType.rawValue
        )
    }
    
    // MARK: - Color Scoring
    
    private func calculateColorScore(_ items: [WardrobeItem]) -> (score: Double, harmonyType: ColorTheory.HarmonyType) {
        // Collect all colors
        var allColors: [ColorTheory.LabColor] = items.map { $0.dominantColor }
        
        // Include accent colors with less weight
        for item in items {
            allColors.append(contentsOf: item.accentColors)
        }
        
        // Get fashion color score from ColorTheory
        let fashionScore = colorTheory.fashionColorScore(Array(allColors.prefix(6))) // Limit for performance
        
        // Detect primary harmony type
        var harmonyType: ColorTheory.HarmonyType = .neutral
        if items.count >= 2 {
            harmonyType = colorTheory.detectHarmony(items[0].dominantColor, items[1].dominantColor)
        }
        
        return (fashionScore, harmonyType)
    }
    
    // MARK: - Occasion Scoring
    
    private func calculateOccasionScore(_ items: [WardrobeItem], occasion: Occasion) -> Double {
        // Check if items are appropriate for the occasion
        var scores: [Double] = []
        
        for item in items {
            let categoryScore = occasionCategoryScore(item.category, item.subcategory, occasion: occasion)
            scores.append(categoryScore)
        }
        
        return scores.isEmpty ? 0.5 : scores.reduce(0, +) / Double(scores.count)
    }
    
    private func occasionCategoryScore(_ category: String, _ subcategory: String, occasion: Occasion) -> Double {
        // Define occasion-appropriate items
        let occasionAppropriateItems: [Occasion: [String: Double]] = [
            .casual: [
                "Tops": 0.9, "Bottoms": 0.9, "Dresses": 0.8, "Outerwear": 0.85,
                "t-shirt": 1.0, "jeans": 1.0, "sneakers": 1.0, "shorts": 0.9
            ],
            .business: [
                "Tops": 0.7, "Bottoms": 0.8, "Dresses": 0.9, "Outerwear": 0.9,
                "blazer": 1.0, "dress shirt": 1.0, "trousers": 1.0, "skirt": 0.9,
                "t-shirt": 0.3, "jeans": 0.4, "sneakers": 0.3
            ],
            .formal: [
                "Dresses": 1.0, "Outerwear": 0.8,
                "suit": 1.0, "gown": 1.0, "dress shoes": 1.0,
                "t-shirt": 0.1, "jeans": 0.1, "sneakers": 0.1
            ],
            .athletic: [
                "Tops": 0.8, "Bottoms": 0.8,
                "athletic": 1.0, "sporty": 1.0, "leggings": 1.0, "shorts": 0.9,
                "dress": 0.2, "blazer": 0.1
            ],
            .dateNight: [
                "Tops": 0.8, "Bottoms": 0.8, "Dresses": 1.0, "Outerwear": 0.85,
                "dress": 1.0, "blouse": 0.9, "heels": 0.9
            ]
        ]
        
        let items = occasionAppropriateItems[occasion] ?? [:]
        
        // Check subcategory first, then category
        if let subScore = items[subcategory.lowercased()] {
            return subScore
        }
        if let catScore = items[category] {
            return catScore
        }
        
        // Default moderate score
        return 0.6
    }
    
    // MARK: - Style Scoring
    
    private func calculateStyleScore(_ items: [WardrobeItem]) -> Double {
        guard items.count >= 2 else { return 1.0 }
        
        // Collect all style categories
        var styleCounts: [StyleCategory: Int] = [:]
        for item in items {
            for style in item.styleCategories {
                styleCounts[style, default: 0] += 1
            }
        }
        
        // Check style coherence
        if styleCounts.isEmpty {
            return 0.7  // No style info available
        }
        
        // Find dominant style
        let maxCount = styleCounts.values.max() ?? 0
        let dominantStyles = styleCounts.filter { $0.value == maxCount }.map { $0.key }
        
        // Calculate coherence based on style distribution
        let totalStyleMentions = styleCounts.values.reduce(0, +)
        let coherenceRatio = Double(maxCount) / Double(max(1, totalStyleMentions))
        
        // Check for compatible styles
        var compatibilityBonus: Double = 0
        let compatiblePairs: [(StyleCategory, StyleCategory)] = [
            (.classic, .minimalist),
            (.modern, .minimalist),
            (.streetwear, .edgy),
            (.bohemian, .romantic),
            (.preppy, .classic),
            (.sporty, .streetwear)
        ]
        
        for (style1, style2) in compatiblePairs {
            if styleCounts[style1] != nil && styleCounts[style2] != nil {
                compatibilityBonus += 0.1
            }
        }
        
        // Check for clashing styles
        var clashPenalty: Double = 0
        let clashingPairs: [(StyleCategory, StyleCategory)] = [
            (.formal, .streetwear),
            (.bohemian, .minimalist),
            (.edgy, .preppy),
            (.sporty, .romantic)
        ]
        
        for (style1, style2) in clashingPairs {
            if styleCounts[style1] != nil && styleCounts[style2] != nil {
                clashPenalty += 0.15
            }
        }
        
        return min(1.0, max(0.0, coherenceRatio + compatibilityBonus - clashPenalty))
    }
    
    // MARK: - Season Scoring
    
    private func calculateSeasonScore(_ items: [WardrobeItem], currentSeason: String?) -> Double {
        let season = currentSeason ?? getCurrentSeason()
        
        var appropriateCount = 0
        for item in items {
            if item.seasonality.contains(season.lowercased()) {
                appropriateCount += 1
            }
        }
        
        return Double(appropriateCount) / Double(max(1, items.count))
    }
    
    private func getCurrentSeason() -> String {
        let month = Calendar.current.component(.month, from: Date())
        switch month {
        case 3...5: return "spring"
        case 6...8: return "summer"
        case 9...11: return "fall"
        default: return "winter"
        }
    }
    
    // MARK: - Pattern Scoring
    
    private func calculatePatternScore(_ items: [WardrobeItem]) -> Double {
        var patternCounts: [String: Int] = [:]
        for item in items {
            for pattern in item.patterns {
                patternCounts[pattern.lowercased(), default: 0] += 1
            }
        }
        
        let hasMultiplePatterns = patternCounts.filter { $0.key != "solid" && $0.value > 0 }.count
        
        // Pattern mixing rules
        switch hasMultiplePatterns {
        case 0, 1:
            return 1.0  // Solid or one pattern is safe
        case 2:
            // Check if patterns are compatible
            let patterns = Array(patternCounts.keys.filter { $0 != "solid" })
            if arePatternCompatible(patterns) {
                return 0.85
            }
            return 0.6
        default:
            return 0.4  // Too many patterns
        }
    }
    
    private func arePatternCompatible(_ patterns: [String]) -> Bool {
        // Simple compatibility rules
        let compatibleSets: [[String]] = [
            ["stripes", "polka dots"],
            ["plaid", "stripes"],
            ["floral", "stripes"]
        ]
        
        for set in compatibleSets {
            if patterns.allSatisfy({ set.contains($0) }) {
                return true
            }
        }
        
        return false
    }
    
    // MARK: - Formality Scoring
    
    private func calculateFormalityScore(_ items: [WardrobeItem], occasion: Occasion) -> Double {
        let targetFormality = occasion.formalityLevel
        
        let avgFormality = items.map { $0.formalityScore }.reduce(0, +) / Double(max(1, items.count))
        
        // Calculate how close we are to target
        let diff = abs(targetFormality - avgFormality)
        
        // Penalize under-dressing more than over-dressing for formal occasions
        if occasion.formalityLevel > 0.5 && avgFormality < targetFormality {
            return max(0, 1.0 - diff * 1.5)
        }
        
        return max(0, 1.0 - diff)
    }
    
    // MARK: - Explanation Generation
    
    private func generateExplanation(
        colorScore: Double,
        harmonyType: ColorTheory.HarmonyType,
        occasionScore: Double,
        styleScore: Double,
        occasion: Occasion
    ) -> String {
        var parts: [String] = []
        
        // Color explanation
        if colorScore >= 0.8 {
            parts.append("Colors work beautifully together with \(harmonyType.rawValue) harmony")
        } else if colorScore >= 0.6 {
            parts.append("Colors are reasonably coordinated")
        } else {
            parts.append("Color combination could be improved")
        }
        
        // Occasion explanation
        if occasionScore >= 0.8 {
            parts.append("Perfect for \(occasion.rawValue) occasions")
        } else if occasionScore >= 0.6 {
            parts.append("Suitable for \(occasion.rawValue)")
        } else {
            parts.append("May not be ideal for \(occasion.rawValue)")
        }
        
        // Style explanation
        if styleScore >= 0.8 {
            parts.append("Style is cohesive and well-matched")
        } else if styleScore < 0.6 {
            parts.append("Consider items with more unified style")
        }
        
        return parts.joined(separator: ". ") + "."
    }
    
    // MARK: - Suggestion Generation
    
    private func generateSuggestions(
        colorScore: Double,
        occasionScore: Double,
        styleScore: Double,
        formalityScore: Double,
        occasion: Occasion,
        items: [WardrobeItem]
    ) -> [String] {
        var suggestions: [String] = []
        
        if colorScore < 0.7 {
            suggestions.append("Try adding a neutral piece to balance the colors")
        }
        
        if occasionScore < 0.7 {
            suggestions.append("Consider swapping an item for something more \(occasion.rawValue)-appropriate")
        }
        
        if styleScore < 0.7 {
            suggestions.append("Stick to items from the same style family for better cohesion")
        }
        
        if formalityScore < 0.7 {
            if occasion.formalityLevel > 0.5 {
                suggestions.append("Dress up with a more formal piece")
            } else {
                suggestions.append("You might be overdressed - consider something more casual")
            }
        }
        
        if suggestions.isEmpty {
            suggestions.append("Great outfit! No changes needed.")
        }
        
        return suggestions
    }
    
    // MARK: - Quick Compatibility Check
    
    /// Quick check if two items go well together
    public func quickPairCheck(
        _ item1: WardrobeItem,
        _ item2: WardrobeItem
    ) -> Double {
        // Fast color check
        let colorScore = colorTheory.harmonyScore(item1.dominantColor, item2.dominantColor)
        
        // Fast formality check
        let formalityDiff = abs(item1.formalityScore - item2.formalityScore)
        let formalityScore = max(0, 1.0 - formalityDiff)
        
        // Fast style check
        let sharedStyles = Set(item1.styleCategories).intersection(Set(item2.styleCategories))
        let styleScore = sharedStyles.isEmpty ? 0.6 : 0.9
        
        return (colorScore * 0.5 + formalityScore * 0.3 + styleScore * 0.2)
    }
}
