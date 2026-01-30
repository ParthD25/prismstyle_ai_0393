import Foundation

/// User Preference Learning System for PrismStyle AI
/// Uses Exponentially Weighted Moving Average (EWMA) to learn and adapt
/// to user preferences over time.
///
/// Key Features:
/// - EWMA weight updates for gradual learning
/// - Interaction tracking with decay
/// - Preference persistence across sessions
/// - Privacy-preserving on-device learning
///
/// The system learns from:
/// - Outfit selections (positive signal)
/// - Item dismissals (negative signal)
/// - Wear frequency
/// - Rating patterns
@available(iOS 15.0, *)
public final class UserLearning {
    
    // MARK: - Singleton
    
    public static let shared = UserLearning()
    private init() {
        loadPreferences()
    }
    
    // MARK: - Types
    
    /// User interaction types
    public enum InteractionType: String, Codable {
        case selected           // User chose this outfit/item
        case dismissed          // User swiped away / rejected
        case worn              // User marked as worn
        case favorited         // User added to favorites
        case unfavorited       // User removed from favorites
        case rated             // User explicitly rated
        case viewed            // User viewed details (weak signal)
    }
    
    /// Learned preference dimensions
    public struct PreferenceDimensions: Codable {
        // Style preferences (0-1)
        public var stylePreferences: [String: Double] = [:]  // StyleCategory -> preference
        
        // Color preferences (0-1)
        public var colorPreferences: [String: Double] = [:]  // Color name -> preference
        
        // Category preferences
        public var categoryPreferences: [String: Double] = [:] // Category -> preference
        
        // Occasion preferences
        public var occasionPreferences: [String: Double] = [:] // Occasion -> preference
        
        // Formality preference (0-1 scale, 0.5 = balanced)
        public var formalityPreference: Double = 0.5
        
        // Color adventurousness (0-1, higher = more willing to try bold combinations)
        public var colorAdventurousness: Double = 0.5
        
        // Pattern tolerance (0-1, higher = more pattern mixing accepted)
        public var patternTolerance: Double = 0.5
        
        // Comfort vs style balance (0-1, higher = prioritize style)
        public var styleOverComfort: Double = 0.5
        
        public init() {}
    }
    
    /// Interaction record for learning
    public struct InteractionRecord: Codable {
        public let timestamp: Date
        public let type: InteractionType
        public let itemIds: [String]
        public let context: InteractionContext
        public let strength: Double  // Signal strength (0-1)
        
        public init(
            timestamp: Date = Date(),
            type: InteractionType,
            itemIds: [String],
            context: InteractionContext,
            strength: Double = 1.0
        ) {
            self.timestamp = timestamp
            self.type = type
            self.itemIds = itemIds
            self.context = context
            self.strength = strength
        }
    }
    
    /// Context for an interaction
    public struct InteractionContext: Codable {
        public let occasion: String?
        public let season: String?
        public let weather: String?
        public let dayOfWeek: Int
        public let timeOfDay: String  // morning, afternoon, evening
        
        public init(
            occasion: String? = nil,
            season: String? = nil,
            weather: String? = nil,
            dayOfWeek: Int = Calendar.current.component(.weekday, from: Date()),
            timeOfDay: String = UserLearning.currentTimeOfDay()
        ) {
            self.occasion = occasion
            self.season = season
            self.weather = weather
            self.dayOfWeek = dayOfWeek
            self.timeOfDay = timeOfDay
        }
    }
    
    /// Scoring weight adjustments
    public struct WeightAdjustments: Codable {
        public var colorWeight: Double = 1.0
        public var occasionWeight: Double = 1.0
        public var styleWeight: Double = 1.0
        public var formalityWeight: Double = 1.0
        public var seasonWeight: Double = 1.0
        public var patternWeight: Double = 1.0
        
        public init() {}
    }
    
    // MARK: - Properties
    
    /// Learning rate for EWMA (0-1, lower = slower adaptation)
    public var learningRate: Double = 0.15
    
    /// Decay factor for old interactions (0-1, lower = faster forgetting)
    public var decayFactor: Double = 0.995
    
    /// Minimum interactions before personalization kicks in
    public var minimumInteractions: Int = 5
    
    /// Current learned preferences
    private(set) var preferences = PreferenceDimensions()
    
    /// Weight adjustments for scoring
    private(set) var weightAdjustments = WeightAdjustments()
    
    /// Recent interaction history (capped at 1000)
    private var interactionHistory: [InteractionRecord] = []
    
    /// Interaction counts for statistics
    private var interactionCounts: [InteractionType: Int] = [:]
    
    /// Item preference scores (item ID -> learned score)
    private var itemScores: [String: Double] = [:]
    
    // MARK: - Persistence Keys
    
    private let preferencesKey = "com.prismstyle.userlearning.preferences"
    private let historyKey = "com.prismstyle.userlearning.history"
    private let weightsKey = "com.prismstyle.userlearning.weights"
    private let scoresKey = "com.prismstyle.userlearning.itemscores"
    
    // MARK: - Core Learning API
    
    /// Record a user interaction and update preferences
    public func recordInteraction(
        type: InteractionType,
        itemIds: [String],
        itemMetadata: [[String: Any]] = [],
        context: InteractionContext = InteractionContext()
    ) {
        // Create record
        let strength = signalStrength(for: type)
        let record = InteractionRecord(
            type: type,
            itemIds: itemIds,
            context: context,
            strength: strength
        )
        
        // Add to history
        interactionHistory.append(record)
        if interactionHistory.count > 1000 {
            interactionHistory.removeFirst()
        }
        
        // Update counts
        interactionCounts[type, default: 0] += 1
        
        // Update item scores
        updateItemScores(for: record, metadata: itemMetadata)
        
        // Update preferences using EWMA
        updatePreferences(from: record, metadata: itemMetadata)
        
        // Update weight adjustments
        updateWeightAdjustments(from: record)
        
        // Persist changes
        savePreferences()
        
        print("📊 UserLearning: Recorded \(type.rawValue) interaction")
    }
    
    /// Get personalized score modifier for an item
    public func getPersonalizedModifier(for itemId: String) -> Double {
        guard totalInteractions >= minimumInteractions else {
            return 1.0  // No personalization yet
        }
        
        // Base modifier from item score
        let itemScore = itemScores[itemId] ?? 0.5
        
        // Convert to modifier (0.8 to 1.2 range)
        return 0.8 + (itemScore * 0.4)
    }
    
    /// Get personalized scoring weights for CompatibilityEngine
    public func getPersonalizedWeights() -> CompatibilityEngine.ScoringWeights {
        var weights = CompatibilityEngine.ScoringWeights()
        
        guard totalInteractions >= minimumInteractions else {
            return weights  // Return defaults
        }
        
        // Apply learned adjustments
        weights.color *= weightAdjustments.colorWeight
        weights.occasion *= weightAdjustments.occasionWeight
        weights.style *= weightAdjustments.styleWeight
        weights.formality *= weightAdjustments.formalityWeight
        weights.season *= weightAdjustments.seasonWeight
        weights.pattern *= weightAdjustments.patternWeight
        
        weights.normalize()
        return weights
    }
    
    /// Get preference for a specific style
    public func stylePreference(for style: String) -> Double {
        return preferences.stylePreferences[style] ?? 0.5
    }
    
    /// Get preference for a specific color
    public func colorPreference(for color: String) -> Double {
        return preferences.colorPreferences[color.lowercased()] ?? 0.5
    }
    
    // MARK: - EWMA Updates
    
    private func updateItemScores(for record: InteractionRecord, metadata: [[String: Any]]) {
        let signalValue = interactionSignalValue(record.type)
        
        for itemId in record.itemIds {
            let currentScore = itemScores[itemId] ?? 0.5
            
            // EWMA update: new_score = α * signal + (1-α) * old_score
            let newScore = learningRate * signalValue + (1 - learningRate) * currentScore
            itemScores[itemId] = max(0, min(1, newScore))
        }
        
        // Apply decay to all other items
        applyDecayToItems(excluding: Set(record.itemIds))
    }
    
    private func updatePreferences(from record: InteractionRecord, metadata: [[String: Any]]) {
        let signalValue = interactionSignalValue(record.type)
        
        // Extract and update style preferences from metadata
        for meta in metadata {
            if let styles = meta["styles"] as? [String] {
                for style in styles {
                    let current = preferences.stylePreferences[style] ?? 0.5
                    preferences.stylePreferences[style] = ewmaUpdate(current, signalValue)
                }
            }
            
            if let color = meta["dominantColor"] as? String {
                let current = preferences.colorPreferences[color.lowercased()] ?? 0.5
                preferences.colorPreferences[color.lowercased()] = ewmaUpdate(current, signalValue)
            }
            
            if let category = meta["category"] as? String {
                let current = preferences.categoryPreferences[category] ?? 0.5
                preferences.categoryPreferences[category] = ewmaUpdate(current, signalValue)
            }
            
            if let formality = meta["formality"] as? Double {
                // Update formality preference toward the item's formality
                let direction = signalValue > 0.5 ? formality : (1 - formality)
                preferences.formalityPreference = ewmaUpdate(preferences.formalityPreference, direction)
            }
        }
        
        // Update occasion preference from context
        if let occasion = record.context.occasion {
            let current = preferences.occasionPreferences[occasion] ?? 0.5
            preferences.occasionPreferences[occasion] = ewmaUpdate(current, signalValue)
        }
    }
    
    private func updateWeightAdjustments(from record: InteractionRecord) {
        // Analyze what factors were salient in this interaction
        // If user repeatedly selects items with good color harmony but ignores occasion,
        // we should increase color weight and decrease occasion weight
        
        // This is a simplified version - full implementation would analyze
        // the characteristics of selected vs. dismissed items
        
        // For now, slightly boost relevance of factors in positive interactions
        if record.type == .selected || record.type == .favorited {
            // Small boost to color if outfit had good colors
            weightAdjustments.colorWeight = ewmaUpdate(weightAdjustments.colorWeight, 1.05, rate: 0.05)
        }
        
        // Normalize weights to prevent drift
        normalizeWeightAdjustments()
    }
    
    /// EWMA update formula
    private func ewmaUpdate(_ current: Double, _ signal: Double, rate: Double? = nil) -> Double {
        let α = rate ?? learningRate
        let updated = α * signal + (1 - α) * current
        return max(0, min(1, updated))
    }
    
    /// Apply decay to items not in current interaction
    private func applyDecayToItems(excluding: Set<String>) {
        for (itemId, score) in itemScores {
            if !excluding.contains(itemId) {
                // Decay toward neutral (0.5)
                let decayed = score * decayFactor + 0.5 * (1 - decayFactor)
                itemScores[itemId] = decayed
            }
        }
    }
    
    private func normalizeWeightAdjustments() {
        let weights = [
            weightAdjustments.colorWeight,
            weightAdjustments.occasionWeight,
            weightAdjustments.styleWeight,
            weightAdjustments.formalityWeight,
            weightAdjustments.seasonWeight,
            weightAdjustments.patternWeight
        ]
        
        let avg = weights.reduce(0, +) / Double(weights.count)
        
        if avg > 0 {
            weightAdjustments.colorWeight /= avg
            weightAdjustments.occasionWeight /= avg
            weightAdjustments.styleWeight /= avg
            weightAdjustments.formalityWeight /= avg
            weightAdjustments.seasonWeight /= avg
            weightAdjustments.patternWeight /= avg
        }
    }
    
    // MARK: - Signal Processing
    
    /// Get signal strength for interaction type
    private func signalStrength(for type: InteractionType) -> Double {
        switch type {
        case .selected: return 1.0
        case .worn: return 0.95
        case .favorited: return 0.9
        case .rated: return 0.85
        case .unfavorited: return 0.3
        case .dismissed: return 0.2
        case .viewed: return 0.55  // Weak positive signal
        }
    }
    
    /// Convert interaction type to learning signal value
    private func interactionSignalValue(_ type: InteractionType) -> Double {
        switch type {
        case .selected, .worn, .favorited: return 0.8
        case .rated: return 0.7
        case .viewed: return 0.55
        case .unfavorited: return 0.35
        case .dismissed: return 0.2
        }
    }
    
    // MARK: - Statistics
    
    /// Total number of recorded interactions
    public var totalInteractions: Int {
        return interactionCounts.values.reduce(0, +)
    }
    
    /// Check if we have enough data for personalization
    public var hasEnoughDataForPersonalization: Bool {
        return totalInteractions >= minimumInteractions
    }
    
    /// Get learning summary
    public func getLearningStatus() -> [String: Any] {
        return [
            "totalInteractions": totalInteractions,
            "hasPersonalization": hasEnoughDataForPersonalization,
            "topStylePreferences": getTopPreferences(preferences.stylePreferences, count: 3),
            "topColorPreferences": getTopPreferences(preferences.colorPreferences, count: 3),
            "formalityPreference": preferences.formalityPreference,
            "colorAdventurousness": preferences.colorAdventurousness,
            "learningRate": learningRate
        ]
    }
    
    private func getTopPreferences(_ prefs: [String: Double], count: Int) -> [(String, Double)] {
        return prefs
            .sorted { $0.value > $1.value }
            .prefix(count)
            .map { ($0.key, $0.value) }
    }
    
    // MARK: - Persistence
    
    private func savePreferences() {
        let encoder = JSONEncoder()
        
        if let data = try? encoder.encode(preferences) {
            UserDefaults.standard.set(data, forKey: preferencesKey)
        }
        
        if let data = try? encoder.encode(weightAdjustments) {
            UserDefaults.standard.set(data, forKey: weightsKey)
        }
        
        // Save recent history (last 100)
        let recentHistory = Array(interactionHistory.suffix(100))
        if let data = try? encoder.encode(recentHistory) {
            UserDefaults.standard.set(data, forKey: historyKey)
        }
        
        if let data = try? encoder.encode(itemScores) {
            UserDefaults.standard.set(data, forKey: scoresKey)
        }
    }
    
    private func loadPreferences() {
        let decoder = JSONDecoder()
        
        if let data = UserDefaults.standard.data(forKey: preferencesKey),
           let loaded = try? decoder.decode(PreferenceDimensions.self, from: data) {
            preferences = loaded
        }
        
        if let data = UserDefaults.standard.data(forKey: weightsKey),
           let loaded = try? decoder.decode(WeightAdjustments.self, from: data) {
            weightAdjustments = loaded
        }
        
        if let data = UserDefaults.standard.data(forKey: historyKey),
           let loaded = try? decoder.decode([InteractionRecord].self, from: data) {
            interactionHistory = loaded
            // Rebuild interaction counts
            for record in interactionHistory {
                interactionCounts[record.type, default: 0] += 1
            }
        }
        
        if let data = UserDefaults.standard.data(forKey: scoresKey),
           let loaded = try? decoder.decode([String: Double].self, from: data) {
            itemScores = loaded
        }
    }
    
    /// Reset all learned preferences
    public func resetLearning() {
        preferences = PreferenceDimensions()
        weightAdjustments = WeightAdjustments()
        interactionHistory = []
        interactionCounts = [:]
        itemScores = [:]
        
        UserDefaults.standard.removeObject(forKey: preferencesKey)
        UserDefaults.standard.removeObject(forKey: weightsKey)
        UserDefaults.standard.removeObject(forKey: historyKey)
        UserDefaults.standard.removeObject(forKey: scoresKey)
        
        print("📊 UserLearning: All preferences reset")
    }
    
    // MARK: - Helpers
    
    static func currentTimeOfDay() -> String {
        let hour = Calendar.current.component(.hour, from: Date())
        switch hour {
        case 5..<12: return "morning"
        case 12..<17: return "afternoon"
        default: return "evening"
        }
    }
}
