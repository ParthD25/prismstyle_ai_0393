import Foundation
import UIKit

/// Unified AI Container for PrismStyle AI
/// This is the single entry point for all AI functionality,
/// replacing the scattered "Brain" logic pattern.
///
/// Key Benefits:
/// - Single source of truth for all AI logic
/// - Consistent API across all screens
/// - Explainable AI with reasoning traces
/// - Safe evolution without breaking changes
/// - Full testability
///
/// Architecture:
/// ┌─────────────────────────────────────────┐
/// │            AIContainer                   │
/// │  (Unified Entry Point)                  │
/// ├─────────────────────────────────────────┤
/// │  ColorTheory → Color harmony & ΔE 2000  │
/// │  CompatibilityEngine → Outfit scoring   │
/// │  UserLearning → EWMA preferences        │
/// │  ImageScoring → Quality assessment      │
/// └─────────────────────────────────────────┘
@available(iOS 15.0, *)
public final class AIContainer {
    
    // MARK: - Singleton
    
    public static let shared = AIContainer()
    
    private init() {
        // Initialize on first access
        setupModules()
    }
    
    // MARK: - Module References
    
    /// Color theory engine (CIELAB ΔE 2000)
    public let colorTheory = ColorTheory.shared
    
    /// Multi-factor compatibility scoring
    public let compatibility = CompatibilityEngine.shared
    
    /// User preference learning (EWMA)
    public let learning = UserLearning.shared
    
    /// Image quality scoring
    public let imageScoring = ImageScoring.shared
    
    // MARK: - State
    
    private var isInitialized = false
    private var featureFlags: [String: Bool] = [
        "enableLearning": true,
        "enableExplanations": true,
        "enableDepthParallax": false,  // Feature-flagged for iPhone 12+ Pro
        "enableAdvancedColorTheory": true
    ]
    
    // MARK: - Initialization
    
    private func setupModules() {
        // Sync learning weights to compatibility engine
        let personalizedWeights = learning.getPersonalizedWeights()
        compatibility.updateWeights(personalizedWeights)
        
        isInitialized = true
        print("✅ AIContainer initialized with all modules")
    }
    
    /// Reinitialize with fresh state (useful for testing)
    public func reinitialize() {
        setupModules()
    }
    
    // MARK: - Feature Flags
    
    public func isFeatureEnabled(_ feature: String) -> Bool {
        return featureFlags[feature] ?? false
    }
    
    public func setFeatureFlag(_ feature: String, enabled: Bool) {
        featureFlags[feature] = enabled
    }
    
    // MARK: - High-Level API
    
    /// Score an outfit with full analysis
    /// This is the main entry point for outfit recommendations
    public func scoreOutfit(
        items: [CompatibilityEngine.WardrobeItem],
        occasion: CompatibilityEngine.Occasion = .casual,
        includeExplanation: Bool = true
    ) -> OutfitScore {
        // Get base compatibility score
        let compatResult = compatibility.calculateOutfitScore(
            items: items,
            occasion: occasion
        )
        
        // Apply personalization if enabled and available
        var personalizedScore = compatResult.overallScore
        if featureFlags["enableLearning"] == true && learning.hasEnoughDataForPersonalization {
            // Apply item-specific modifiers
            var itemModifiers: Double = 0
            for item in items {
                itemModifiers += learning.getPersonalizedModifier(for: item.id)
            }
            let avgModifier = items.isEmpty ? 1.0 : itemModifiers / Double(items.count)
            
            // Blend base score with personalized modifier (80% base, 20% personalized)
            personalizedScore = compatResult.overallScore * 0.8 + (compatResult.overallScore * avgModifier) * 0.2
        }
        
        // Generate explanation if requested
        var explanation: OutfitExplanation?
        if includeExplanation && featureFlags["enableExplanations"] == true {
            explanation = generateExplanation(
                items: items,
                occasion: occasion,
                compatResult: compatResult
            )
        }
        
        return OutfitScore(
            score: min(1.0, max(0.0, personalizedScore)),
            rawScore: compatResult.overallScore,
            components: ScoreComponents(
                colorHarmony: compatResult.colorScore,
                occasionMatch: compatResult.occasionScore,
                styleCoherence: compatResult.styleScore,
                formalityMatch: compatResult.formalityScore,
                seasonalFit: compatResult.seasonScore,
                patternBalance: compatResult.patternScore
            ),
            explanation: explanation,
            suggestions: compatResult.suggestions
        )
    }
    
    /// Quick compatibility check between two items
    public func quickPairScore(_ item1: CompatibilityEngine.WardrobeItem, _ item2: CompatibilityEngine.WardrobeItem) -> Double {
        return compatibility.quickPairCheck(item1, item2)
    }
    
    /// Analyze image quality before adding to wardrobe
    public func analyzeImageQuality(_ image: UIImage) -> ImageScoring.QualityResult {
        return imageScoring.analyzeImage(image)
    }
    
    /// Get color harmony analysis between colors
    public func analyzeColorHarmony(_ colors: [UIColor]) -> ColorHarmonyResult {
        let labColors = colors.map { colorTheory.uiColorToLab($0) }
        
        guard labColors.count >= 2 else {
            return ColorHarmonyResult(
                overallScore: 1.0,
                harmonyType: .neutral,
                temperatureCompatible: true,
                seasonallyCoherent: true,
                followsRules: true
            )
        }
        
        let fashionScore = colorTheory.fashionColorScore(labColors)
        let harmony = colorTheory.detectHarmony(labColors[0], labColors[1])
        
        var tempCompatible = true
        for i in 0..<labColors.count {
            for j in (i+1)..<labColors.count {
                if colorTheory.temperatureCompatibility(labColors[i], labColors[j]) < 0.7 {
                    tempCompatible = false
                    break
                }
            }
        }
        
        let seasonalScore = colorTheory.seasonalCompatibility(labColors)
        let followsRules = colorTheory.followsThreeColorRule(labColors)
        
        return ColorHarmonyResult(
            overallScore: fashionScore,
            harmonyType: harmony,
            temperatureCompatible: tempCompatible,
            seasonallyCoherent: seasonalScore > 0.7,
            followsRules: followsRules
        )
    }
    
    /// Record user interaction for learning
    public func recordInteraction(
        type: UserLearning.InteractionType,
        itemIds: [String],
        itemMetadata: [[String: Any]] = [],
        occasion: String? = nil
    ) {
        guard featureFlags["enableLearning"] == true else { return }
        
        let context = UserLearning.InteractionContext(occasion: occasion)
        learning.recordInteraction(
            type: type,
            itemIds: itemIds,
            itemMetadata: itemMetadata,
            context: context
        )
        
        // Update compatibility weights after learning
        let updatedWeights = learning.getPersonalizedWeights()
        compatibility.updateWeights(updatedWeights)
    }
    
    // MARK: - Explanation Generation
    
    private func generateExplanation(
        items: [CompatibilityEngine.WardrobeItem],
        occasion: CompatibilityEngine.Occasion,
        compatResult: CompatibilityEngine.CompatibilityResult
    ) -> OutfitExplanation {
        var reasons: [ExplanationReason] = []
        
        // Color explanation
        let colorReason = ExplanationReason(
            factor: "Color Harmony",
            score: compatResult.colorScore,
            description: describeColorScore(compatResult.colorScore, harmonyType: compatResult.harmonyType),
            impact: .high
        )
        reasons.append(colorReason)
        
        // Occasion explanation
        let occasionReason = ExplanationReason(
            factor: "Occasion Match",
            score: compatResult.occasionScore,
            description: describeOccasionScore(compatResult.occasionScore, occasion: occasion),
            impact: .high
        )
        reasons.append(occasionReason)
        
        // Style explanation
        let styleReason = ExplanationReason(
            factor: "Style Coherence",
            score: compatResult.styleScore,
            description: describeStyleScore(compatResult.styleScore),
            impact: .medium
        )
        reasons.append(styleReason)
        
        // Formality explanation
        let formalityReason = ExplanationReason(
            factor: "Formality Level",
            score: compatResult.formalityScore,
            description: describeFormalityScore(compatResult.formalityScore, occasion: occasion),
            impact: .medium
        )
        reasons.append(formalityReason)
        
        // Generate summary
        let summary = generateSummary(
            overallScore: compatResult.overallScore,
            topReasons: reasons.sorted { $0.score < $1.score }.prefix(2).map { $0.factor }
        )
        
        return OutfitExplanation(
            summary: summary,
            reasons: reasons,
            confidenceLevel: compatResult.overallScore > 0.8 ? .high : (compatResult.overallScore > 0.6 ? .medium : .low)
        )
    }
    
    private func describeColorScore(_ score: Double, harmonyType: String) -> String {
        if score >= 0.85 {
            return "Excellent \(harmonyType) color combination"
        } else if score >= 0.7 {
            return "Good color coordination with \(harmonyType) harmony"
        } else if score >= 0.5 {
            return "Colors work reasonably well together"
        } else {
            return "Color combination could be improved"
        }
    }
    
    private func describeOccasionScore(_ score: Double, occasion: CompatibilityEngine.Occasion) -> String {
        if score >= 0.85 {
            return "Perfect choice for \(occasion.rawValue)"
        } else if score >= 0.7 {
            return "Suitable for \(occasion.rawValue) occasions"
        } else if score >= 0.5 {
            return "Acceptable but not ideal for \(occasion.rawValue)"
        } else {
            return "May not be appropriate for \(occasion.rawValue)"
        }
    }
    
    private func describeStyleScore(_ score: Double) -> String {
        if score >= 0.85 {
            return "Perfectly cohesive style"
        } else if score >= 0.7 {
            return "Well-matched style elements"
        } else if score >= 0.5 {
            return "Mixed style elements"
        } else {
            return "Clashing style categories"
        }
    }
    
    private func describeFormalityScore(_ score: Double, occasion: CompatibilityEngine.Occasion) -> String {
        if score >= 0.85 {
            return "Formality level matches perfectly"
        } else if score >= 0.7 {
            return "Appropriate formality level"
        } else if occasion.formalityLevel > 0.5 {
            return "Could be dressed up more"
        } else {
            return "Might be overdressed"
        }
    }
    
    private func generateSummary(overallScore: Double, topReasons: [String]) -> String {
        if overallScore >= 0.85 {
            return "This is an excellent outfit choice!"
        } else if overallScore >= 0.7 {
            return "This outfit works well together."
        } else if overallScore >= 0.5 {
            let weakAreas = topReasons.joined(separator: " and ")
            return "This outfit is okay, but \(weakAreas) could be improved."
        } else {
            return "Consider making some changes for a better look."
        }
    }
    
    // MARK: - Types
    
    /// Comprehensive outfit score result
    public struct OutfitScore {
        public let score: Double              // Final personalized score (0-1)
        public let rawScore: Double           // Score before personalization
        public let components: ScoreComponents
        public let explanation: OutfitExplanation?
        public let suggestions: [String]
    }
    
    /// Individual score components
    public struct ScoreComponents {
        public let colorHarmony: Double
        public let occasionMatch: Double
        public let styleCoherence: Double
        public let formalityMatch: Double
        public let seasonalFit: Double
        public let patternBalance: Double
    }
    
    /// Color harmony analysis result
    public struct ColorHarmonyResult {
        public let overallScore: Double
        public let harmonyType: ColorTheory.HarmonyType
        public let temperatureCompatible: Bool
        public let seasonallyCoherent: Bool
        public let followsRules: Bool
    }
    
    /// Explainable AI result
    public struct OutfitExplanation {
        public let summary: String
        public let reasons: [ExplanationReason]
        public let confidenceLevel: ConfidenceLevel
    }
    
    /// Individual explanation reason
    public struct ExplanationReason {
        public let factor: String
        public let score: Double
        public let description: String
        public let impact: Impact
        
        public enum Impact {
            case high, medium, low
        }
    }
    
    /// Confidence level for explanations
    public enum ConfidenceLevel {
        case high, medium, low
    }
    
    // MARK: - Diagnostics
    
    /// Get diagnostic information for debugging
    public func getDiagnostics() -> [String: Any] {
        return [
            "isInitialized": isInitialized,
            "featureFlags": featureFlags,
            "learningStatus": learning.getLearningStatus(),
            "hasPersonalization": learning.hasEnoughDataForPersonalization,
            "totalInteractions": learning.totalInteractions
        ]
    }
    
    /// Reset all learning data (for testing or user request)
    public func resetLearning() {
        learning.resetLearning()
        
        // Reset compatibility weights to defaults
        compatibility.updateWeights(CompatibilityEngine.ScoringWeights())
        
        print("✅ AIContainer: All learning data reset")
    }
}
