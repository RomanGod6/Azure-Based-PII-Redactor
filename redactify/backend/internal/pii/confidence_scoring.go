package pii

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// ConfidenceSignal represents an individual signal that affects confidence scoring
type ConfidenceSignal struct {
	SignalName string                 `json:"signal_name"`
	Value      float64                `json:"value"`
	Weight     float64                `json:"weight"`
	Source     string                 `json:"source"` // 'azure', 'gpt', 'ml', 'pattern', 'feedback'
	Timestamp  time.Time              `json:"timestamp"`
	Context    map[string]interface{} `json:"context"`
}

// ConfidenceScore represents a comprehensive confidence score for a detection
type ConfidenceScore struct {
	EntityText         string              `json:"entity_text"`
	EntityType         string              `json:"entity_type"`
	BaseConfidence     float64             `json:"base_confidence"`
	AdjustedConfidence float64             `json:"adjusted_confidence"`
	Signals            []ConfidenceSignal  `json:"signals"`
	Explanation        string              `json:"explanation"`
	Timestamp          time.Time           `json:"timestamp"`
}

// FeedbackRecord represents user feedback on detection accuracy
type FeedbackRecord struct {
	EntityText     string    `json:"entity_text"`
	EntityType     string    `json:"entity_type"`
	OriginalScore  float64   `json:"original_score"`
	UserDecision   string    `json:"user_decision"` // 'correct', 'incorrect', 'partial'
	UserConfidence float64   `json:"user_confidence"`
	Context        string    `json:"context"`
	Timestamp      time.Time `json:"timestamp"`
	SessionID      string    `json:"session_id"`
}

// ConfidenceEngine handles dynamic confidence scoring and learning
type ConfidenceEngine struct {
	feedbackHistory []FeedbackRecord
	signalWeights   map[string]float64
	entityTypeStats map[string]*EntityTypeStats
}

// EntityTypeStats tracks statistics for specific entity types
type EntityTypeStats struct {
	TotalDetections    int     `json:"total_detections"`
	TruePositives      int     `json:"true_positives"`
	FalsePositives     int     `json:"false_positives"`
	AverageConfidence  float64 `json:"average_confidence"`
	LastUpdated        time.Time `json:"last_updated"`
}

// NewConfidenceEngine creates a new confidence scoring engine
func NewConfidenceEngine() *ConfidenceEngine {
	return &ConfidenceEngine{
		feedbackHistory: make([]FeedbackRecord, 0),
		signalWeights: map[string]float64{
			"azure_confidence":    0.4,
			"gpt_validation":      0.3,
			"pattern_match":       0.15,
			"context_analysis":    0.1,
			"historical_accuracy": 0.05,
		},
		entityTypeStats: make(map[string]*EntityTypeStats),
	}
}

// NewConfidenceEngineWithDB creates a confidence engine and loads feedback from database
func NewConfidenceEngineWithDB(db *sql.DB) *ConfidenceEngine {
	ce := NewConfidenceEngine()
	
	// Load feedback history from database
	if db != nil {
		ce.loadFeedbackFromDB(db)
	}
	
	return ce
}

// CalculateConfidence calculates an adjusted confidence score for an entity
func (ce *ConfidenceEngine) CalculateConfidence(entity Entity, context string, gptValidation *ValidationResult) *ConfidenceScore {
	signals := []ConfidenceSignal{}
	now := time.Now()
	
	// Azure base confidence signal
	signals = append(signals, ConfidenceSignal{
		SignalName: "azure_confidence",
		Value:      entity.Confidence,
		Weight:     ce.signalWeights["azure_confidence"],
		Source:     "azure",
		Timestamp:  now,
		Context: map[string]interface{}{
			"entity_type": entity.Type,
			"text":        entity.Text,
		},
	})
	
	// GPT validation signal
	if gptValidation != nil {
		gptScore := 0.0
		if gptValidation.IsRealPII && gptValidation.ShouldRedact {
			gptScore = gptValidation.Confidence
		} else {
			gptScore = 1.0 - gptValidation.Confidence // Inverse if GPT says it's not PII
		}
		
		signals = append(signals, ConfidenceSignal{
			SignalName: "gpt_validation",
			Value:      gptScore,
			Weight:     ce.signalWeights["gpt_validation"],
			Source:     "gpt",
			Timestamp:  now,
			Context: map[string]interface{}{
				"gpt_explanation": gptValidation.Explanation,
				"should_redact":   gptValidation.ShouldRedact,
			},
		})
	}
	
	// Pattern matching signal
	patternScore := ce.calculatePatternMatchScore(entity)
	signals = append(signals, ConfidenceSignal{
		SignalName: "pattern_match",
		Value:      patternScore,
		Weight:     ce.signalWeights["pattern_match"],
		Source:     "pattern",
		Timestamp:  now,
		Context: map[string]interface{}{
			"pattern_type": entity.Type,
		},
	})
	
	// Context analysis signal
	contextScore := ce.analyzeContextualClues(entity, context)
	signals = append(signals, ConfidenceSignal{
		SignalName: "context_analysis",
		Value:      contextScore,
		Weight:     ce.signalWeights["context_analysis"],
		Source:     "context",
		Timestamp:  now,
		Context: map[string]interface{}{
			"context_length": len(context),
		},
	})
	
	// Specific training feedback signal - check for exact text matches
	trainingScore, trainingFound := ce.getTrainingFeedback(entity.Text, entity.Type)
	if trainingFound {
		signals = append(signals, ConfidenceSignal{
			SignalName: "training_feedback",
			Value:      trainingScore,
			Weight:     0.4, // High weight for specific training
			Source:     "user_training",
			Timestamp:  now,
			Context: map[string]interface{}{
				"text_trained": entity.Text,
				"trained_type": entity.Type,
			},
		})
	}
	
	// Historical accuracy signal
	historicalScore := ce.getHistoricalAccuracy(entity.Type)
	signals = append(signals, ConfidenceSignal{
		SignalName: "historical_accuracy",
		Value:      historicalScore,
		Weight:     ce.signalWeights["historical_accuracy"],
		Source:     "feedback",
		Timestamp:  now,
		Context: map[string]interface{}{
			"entity_type": entity.Type,
		},
	})
	
	// Calculate weighted average
	adjustedConfidence := ce.calculateWeightedAverage(signals)
	
	// Apply confidence bounds
	adjustedConfidence = math.Max(0.0, math.Min(1.0, adjustedConfidence))
	
	// Generate explanation
	explanation := ce.generateExplanation(signals, entity.Confidence, adjustedConfidence)
	
	return &ConfidenceScore{
		EntityText:         entity.Text,
		EntityType:         entity.Type,
		BaseConfidence:     entity.Confidence,
		AdjustedConfidence: adjustedConfidence,
		Signals:            signals,
		Explanation:        explanation,
		Timestamp:          now,
	}
}

// GetFeedbackHistory returns the feedback history for training entity creation
func (ce *ConfidenceEngine) GetFeedbackHistory() []FeedbackRecord {
	return ce.feedbackHistory
}

// getTrainingFeedback checks if this specific text was trained by the user
func (ce *ConfidenceEngine) getTrainingFeedback(entityText, entityType string) (float64, bool) {
	// Look for the most recent training feedback for this exact text and type
	for i := len(ce.feedbackHistory) - 1; i >= 0; i-- {
		feedback := ce.feedbackHistory[i]
		
		// Check for exact text match and entity type match
		if feedback.EntityText == entityText && feedback.EntityType == entityType {
			logrus.WithFields(logrus.Fields{
				"entity_text": entityText,
				"entity_type": entityType,
				"user_decision": feedback.UserDecision,
				"user_confidence": feedback.UserConfidence,
			}).Info("🎓 Found specific training feedback for entity")
			
			// If user marked this as correct for this type, return high confidence
			if feedback.UserDecision == "correct" {
				return 0.95, true
			} else {
				// If user marked this as incorrect for this type, return low confidence
				return 0.05, true
			}
		}
	}
	
	return 0.0, false
}

// calculatePatternMatchScore evaluates how well the entity matches known patterns
func (ce *ConfidenceEngine) calculatePatternMatchScore(entity Entity) float64 {
	switch entity.Type {
	case "email":
		// Strong pattern for emails
		return 0.95
	case "phone", "phone_us":
		// Good pattern for phones
		return 0.9
	case "ssn":
		// Very strong pattern for SSN
		return 0.98
	case "credit_card":
		// Strong pattern if it passed Luhn check
		return 0.92
	case "ip_address":
		// IP addresses have strong patterns
		return 0.85
	case "url":
		// URLs have decent patterns
		return 0.8
	default:
		// Default for other types
		return 0.7
	}
}

// analyzeContextualClues analyzes surrounding text for context clues
func (ce *ConfidenceEngine) analyzeContextualClues(entity Entity, context string) float64 {
	score := 0.5 // Base score
	
	// Look for business/professional context indicators
	businessKeywords := []string{
		"company", "business", "organization", "department",
		"contact", "support", "sales", "marketing", "hr",
		"example", "sample", "test", "demo", "placeholder",
	}
	
	// Look for sensitive context indicators
	sensitiveKeywords := []string{
		"customer", "client", "patient", "employee", "user",
		"personal", "private", "confidential", "ssn", "social",
	}
	
	contextLower := strings.ToLower(context)
	
	businessCount := 0
	for _, keyword := range businessKeywords {
		if strings.Contains(contextLower, keyword) {
			businessCount++
		}
	}
	
	sensitiveCount := 0
	for _, keyword := range sensitiveKeywords {
		if strings.Contains(contextLower, keyword) {
			sensitiveCount++
		}
	}
	
	// Adjust score based on context
	if businessCount > sensitiveCount {
		score -= 0.2 // Less likely to be real PII in business context
	} else if sensitiveCount > businessCount {
		score += 0.3 // More likely to be real PII in sensitive context
	}
	
	// Ensure bounds
	return math.Max(0.0, math.Min(1.0, score))
}

// getHistoricalAccuracy gets the historical accuracy for an entity type
func (ce *ConfidenceEngine) getHistoricalAccuracy(entityType string) float64 {
	stats, exists := ce.entityTypeStats[entityType]
	if !exists || stats.TotalDetections == 0 {
		return 0.5 // Default if no history
	}
	
	accuracy := float64(stats.TruePositives) / float64(stats.TotalDetections)
	return accuracy
}

// calculateWeightedAverage calculates the weighted average of all signals
func (ce *ConfidenceEngine) calculateWeightedAverage(signals []ConfidenceSignal) float64 {
	totalWeightedValue := 0.0
	totalWeight := 0.0
	
	for _, signal := range signals {
		totalWeightedValue += signal.Value * signal.Weight
		totalWeight += signal.Weight
	}
	
	if totalWeight == 0 {
		return 0.5 // Default
	}
	
	return totalWeightedValue / totalWeight
}

// generateExplanation generates a human-readable explanation for the confidence score
func (ce *ConfidenceEngine) generateExplanation(signals []ConfidenceSignal, baseConfidence, adjustedConfidence float64) string {
	diff := adjustedConfidence - baseConfidence
	
	if math.Abs(diff) < 0.05 {
		return fmt.Sprintf("Confidence unchanged (%.1f%%) - all signals align with base detection", adjustedConfidence*100)
	}
	
	if diff > 0 {
		return fmt.Sprintf("Confidence increased by %.1f%% due to supportive signals", diff*100)
	} else {
		return fmt.Sprintf("Confidence decreased by %.1f%% due to conflicting signals", math.Abs(diff)*100)
	}
}

// RecordFeedback records user feedback for learning
func (ce *ConfidenceEngine) RecordFeedback(feedback FeedbackRecord) {
	ce.feedbackHistory = append(ce.feedbackHistory, feedback)
	
	// Update entity type statistics
	stats, exists := ce.entityTypeStats[feedback.EntityType]
	if !exists {
		stats = &EntityTypeStats{
			TotalDetections: 0,
			TruePositives:   0,
			FalsePositives:  0,
		}
		ce.entityTypeStats[feedback.EntityType] = stats
	}
	
	stats.TotalDetections++
	stats.LastUpdated = feedback.Timestamp
	
	switch feedback.UserDecision {
	case "correct":
		stats.TruePositives++
	case "incorrect":
		stats.FalsePositives++
	}
	
	// Recalculate average confidence
	totalConfidence := 0.0
	for _, record := range ce.feedbackHistory {
		if record.EntityType == feedback.EntityType {
			totalConfidence += record.OriginalScore
		}
	}
	stats.AverageConfidence = totalConfidence / float64(stats.TotalDetections)
	
	// Optionally adjust signal weights based on feedback
	ce.adjustSignalWeights()
}

// adjustSignalWeights dynamically adjusts signal weights based on feedback
func (ce *ConfidenceEngine) adjustSignalWeights() {
	// Simple learning: if we have enough feedback, adjust weights
	if len(ce.feedbackHistory) < 10 {
		return // Not enough data
	}
	
	// This would require more sophisticated tracking of which signals
	// contributed to which decisions. For now, keep weights static.
	// In a production system, you'd implement more advanced ML here.
}

// GetStats returns current performance statistics
func (ce *ConfidenceEngine) GetStats() map[string]*EntityTypeStats {
	return ce.entityTypeStats
}

// SaveToJSON serializes the confidence engine state to JSON
func (ce *ConfidenceEngine) SaveToJSON() ([]byte, error) {
	data := map[string]interface{}{
		"feedback_history": ce.feedbackHistory,
		"signal_weights":   ce.signalWeights,
		"entity_stats":     ce.entityTypeStats,
	}
	
	return json.MarshalIndent(data, "", "  ")
}

// LoadFromJSON deserializes the confidence engine state from JSON
func (ce *ConfidenceEngine) LoadFromJSON(jsonData []byte) error {
	var data map[string]interface{}
	if err := json.Unmarshal(jsonData, &data); err != nil {
		return fmt.Errorf("failed to unmarshal confidence engine data: %w", err)
	}
	
	// Load feedback history
	if feedbackData, exists := data["feedback_history"]; exists {
		feedbackJSON, _ := json.Marshal(feedbackData)
		json.Unmarshal(feedbackJSON, &ce.feedbackHistory)
	}
	
	// Load signal weights
	if weightsData, exists := data["signal_weights"]; exists {
		weightsJSON, _ := json.Marshal(weightsData)
		json.Unmarshal(weightsJSON, &ce.signalWeights)
	}
	
	// Load entity stats
	if statsData, exists := data["entity_stats"]; exists {
		statsJSON, _ := json.Marshal(statsData)
		json.Unmarshal(statsJSON, &ce.entityTypeStats)
	}
	
	return nil
}

// loadFeedbackFromDB loads all feedback records from the database
func (ce *ConfidenceEngine) loadFeedbackFromDB(db *sql.DB) {
	query := `
		SELECT entity_text, entity_type, original_score, user_decision, user_confidence, context, timestamp, session_id
		FROM training_feedback 
		ORDER BY timestamp DESC
	`
	
	rows, err := db.Query(query)
	if err != nil {
		logrus.WithError(err).Warn("Failed to load feedback from database")
		return
	}
	defer rows.Close()
	
	loadedCount := 0
	for rows.Next() {
		var feedback FeedbackRecord
		err := rows.Scan(
			&feedback.EntityText,
			&feedback.EntityType, 
			&feedback.OriginalScore,
			&feedback.UserDecision,
			&feedback.UserConfidence,
			&feedback.Context,
			&feedback.Timestamp,
			&feedback.SessionID,
		)
		if err != nil {
			logrus.WithError(err).Warn("Failed to scan feedback row")
			continue
		}
		
		// Add to feedback history
		ce.feedbackHistory = append(ce.feedbackHistory, feedback)
		
		// Update entity type statistics
		stats, exists := ce.entityTypeStats[feedback.EntityType]
		if !exists {
			stats = &EntityTypeStats{
				TotalDetections: 0,
				TruePositives:   0,
				FalsePositives:  0,
			}
		}
		
		stats.TotalDetections++
		if feedback.UserDecision == "correct" {
			stats.TruePositives++
		} else {
			stats.FalsePositives++
		}
		stats.LastUpdated = feedback.Timestamp
		ce.entityTypeStats[feedback.EntityType] = stats
		
		loadedCount++
	}
	
	if loadedCount > 0 {
		logrus.WithField("feedback_count", loadedCount).Info("Loaded training feedback from database")
		// Recalculate signal weights based on loaded feedback
		ce.adjustSignalWeights()
	}
}