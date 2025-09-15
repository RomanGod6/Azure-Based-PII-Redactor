package pii

import (
	"database/sql"
	"fmt"
	"redactify/pkg/config"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

type Detector struct {
	config            *config.Config
	columnManager     *ColumnConfigManager
	confidenceEngine  *ConfidenceEngine
	gptValidator      *GPTValidator
}

type Entity struct {
	Type       string  `json:"type"`
	Text       string  `json:"text"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Confidence float64 `json:"confidence"`
	Category   string  `json:"category"`
}

type RedactOptions struct {
	RedactionMode string            `json:"redactionMode"` // "replace", "mask", "remove"
	CustomLabels  map[string]string `json:"customLabels"`
	PreserveCases bool              `json:"preserveCases"`
	UseTraining   bool              `json:"useTraining"`
}

type RedactResult struct {
	RedactedText string   `json:"redacted_text"`
	Entities     []Entity `json:"entities"`
}


func NewDetector(cfg *config.Config) *Detector {
	return &Detector{
		config:           cfg,
		columnManager:    NewColumnConfigManager(),
		confidenceEngine: NewConfidenceEngine(),
		gptValidator:     NewGPTValidator(cfg),
	}
}

func NewDetectorWithDB(cfg *config.Config, db *sql.DB) *Detector {
	return &Detector{
		config:           cfg,
		columnManager:    NewColumnConfigManager(),
		confidenceEngine: NewConfidenceEngineWithDB(db),
		gptValidator:     NewGPTValidator(cfg),
	}
}


func (d *Detector) Detect(text string) ([]Entity, error) {
	logrus.WithField("text_length", len(text)).Info("🔍 Detector.Detect() ENTRY")
	
	var entities []Entity
	
	// ULTRA DEBUG: Check Azure configuration in extreme detail
	logrus.WithFields(logrus.Fields{
		"azure_endpoint": d.config.Azure.Endpoint,
		"azure_endpoint_empty": d.config.Azure.Endpoint == "",
		"azure_apikey_length": len(d.config.Azure.APIKey),
		"azure_apikey_empty": d.config.Azure.APIKey == "",
		"config_nil": d.config == nil,
		"azure_region": d.config.Azure.Region,
		"azure_gpt_endpoint": d.config.Azure.GPTEndpoint,
	}).Info("🔧 Detector.Detect() - Azure configuration check")
	
	// Try Azure first if configured
	if d.config.Azure.Endpoint != "" && d.config.Azure.APIKey != "" {
		logrus.WithFields(logrus.Fields{
			"endpoint": d.config.Azure.Endpoint,
			"text_length": len(text),
			"api_key_prefix": func() string {
				if len(d.config.Azure.APIKey) > 10 {
					return d.config.Azure.APIKey[:10]
				}
				return d.config.Azure.APIKey
			}(),
		}).Info("☁️ Detector.Detect() - Calling Azure Text Analytics API")
		
		startTime := time.Now()
		azureEntities, err := d.detectWithAzure(text)
		duration := time.Since(startTime)
		
		if err != nil {
			logrus.WithError(err).WithField("duration", duration).Warn("❌ Azure detection failed, falling back to regex")
		} else {
			logrus.WithFields(logrus.Fields{
				"entities_count": len(azureEntities),
				"duration": duration,
			}).Info("✅ Azure detection completed successfully")
			entities = append(entities, azureEntities...)
		}
	} else {
		logrus.WithFields(logrus.Fields{
			"endpoint_empty": d.config.Azure.Endpoint == "",
			"apikey_empty": d.config.Azure.APIKey == "",
			"endpoint_value": d.config.Azure.Endpoint,
			"apikey_length": len(d.config.Azure.APIKey),
		}).Warn("⚠️ Azure not configured, using regex only")
	}
	
	// Always run regex detection as well (additional coverage)
	logrus.Info("🔤 Detector.Detect() - Running regex detection")
	regexEntities := d.detectWithRegex(text)
	logrus.WithField("regex_entities_count", len(regexEntities)).Info("✅ Regex detection completed")
	
	entities = append(entities, regexEntities...)
	
	// Deduplicate entities
	logrus.WithField("total_before_dedup", len(entities)).Info("🔄 Detector.Detect() - Deduplicating entities")
	entities = d.deduplicateEntities(entities)
	logrus.WithField("total_after_dedup", len(entities)).Info("✅ Detector.Detect() - Deduplication completed")
	
	// Apply training feedback to create new entities
	logrus.Info("🎓 Detector.Detect() - Applying training feedback")
	trainingEntities := d.applyTrainingFeedback(text)
	entities = append(entities, trainingEntities...)
	logrus.WithField("training_entities_added", len(trainingEntities)).Info("✅ Training feedback applied")
	
	// Final deduplication after adding training entities
	entities = d.deduplicateEntities(entities)
	logrus.WithField("final_entity_count", len(entities)).Info("✅ Final entity deduplication completed")
	
	return entities, nil
}

// DetectWithColumn performs advanced PII detection with column-specific configuration
func (d *Detector) DetectWithColumn(text string, columnName string) ([]Entity, []*ConfidenceScore, error) {
	// Get base entities from Azure + regex detection
	entities, err := d.Detect(text)
	if err != nil {
		return nil, nil, err
	}
	
	// Apply column-specific filtering
	entities = d.columnManager.ApplyColumnFiltering(columnName, entities, text)
	
	// Perform GPT validation if configured
	var gptValidations []ValidationResult
	if d.gptValidator.IsConfigured() && len(entities) > 0 {
		gptValidations, err = d.gptValidator.ValidateEntities(text, entities)
		if err != nil {
			logrus.WithError(err).Warn("GPT validation failed, proceeding without")
			// Create default validations
			gptValidations = make([]ValidationResult, len(entities))
			for i := range gptValidations {
				gptValidations[i] = ValidationResult{
					IsRealPII:    true,
					Confidence:   0.8,
					ShouldRedact: true,
				}
			}
		}
	}
	
	// Calculate enhanced confidence scores
	var confidenceScores []*ConfidenceScore
	var validatedEntities []Entity
	
	for i, entity := range entities {
		var gptValidation *ValidationResult
		if i < len(gptValidations) {
			gptValidation = &gptValidations[i]
		}
		
		// Calculate advanced confidence score
		score := d.confidenceEngine.CalculateConfidence(entity, text, gptValidation)
		confidenceScores = append(confidenceScores, score)
		
		// Only keep entities that pass GPT validation (if available)
		if gptValidation == nil || gptValidation.ShouldRedact {
			// Update entity with adjusted confidence
			entity.Confidence = score.AdjustedConfidence
			validatedEntities = append(validatedEntities, entity)
		}
	}
	
	return validatedEntities, confidenceScores, nil
}

func (d *Detector) Redact(text string, options *RedactOptions) (*RedactResult, error) {
	logrus.WithFields(logrus.Fields{
		"text_length": len(text),
		"options_nil": options == nil,
	}).Info("🎯 Detector.Redact() ENTRY")
	
	if options == nil {
		logrus.Info("⚙️ Detector.Redact() - Using default options")
		options = &RedactOptions{
			RedactionMode: "replace",
			CustomLabels:  make(map[string]string),
			PreserveCases: false,
			UseTraining:   true,
		}
	}
	
	logrus.WithFields(logrus.Fields{
		"use_training": options.UseTraining,
		"redaction_mode": options.RedactionMode,
	}).Info("⚙️ Detector.Redact() - Options configured")
	
	var entities []Entity
	var err error
	
	if options.UseTraining {
		logrus.Info("🎓 Detector.Redact() - Using training-enhanced detection path")
		entities, err = d.Detect(text)
	} else {
		logrus.Info("🚫 Detector.Redact() - Using base detection without training")
		entities, err = d.DetectWithoutTraining(text)
	}
	
	if err != nil {
		logrus.WithError(err).Error("❌ Detector.Redact() - Detection failed")
		return nil, err
	}
	
	logrus.WithField("entities_detected", len(entities)).Info("✅ Detector.Redact() - Detection completed, applying redaction")
	
	redactedText := d.applyRedaction(text, entities, options)
	
	logrus.WithFields(logrus.Fields{
		"original_length": len(text),
		"redacted_length": len(redactedText),
		"entities_count": len(entities),
	}).Info("✅ Detector.Redact() - Redaction completed successfully")
	
	return &RedactResult{
		RedactedText: redactedText,
		Entities:     entities,
	}, nil
}

// DetectWithoutTraining performs detection without applying training feedback
func (d *Detector) DetectWithoutTraining(text string) ([]Entity, error) {
	var entities []Entity
	
	// Try Azure first if configured
	if d.config.Azure.Endpoint != "" && d.config.Azure.APIKey != "" {
		azureEntities, err := d.detectWithAzure(text)
		if err != nil {
			logrus.WithError(err).Warn("Azure detection failed, falling back to regex")
		} else {
			entities = append(entities, azureEntities...)
		}
	}
	
	// Always run regex detection as well (additional coverage)
	regexEntities := d.detectWithRegex(text)
	entities = append(entities, regexEntities...)
	
	// Deduplicate entities (but don't apply training adjustments)
	entities = d.deduplicateEntities(entities)
	
	return entities, nil
}

// applyTrainingFeedback creates entities based on user training feedback
func (d *Detector) applyTrainingFeedback(text string) []Entity {
	var trainingEntities []Entity
	
	// Get all feedback records from the confidence engine
	feedbackHistory := d.confidenceEngine.GetFeedbackHistory()
	
	logrus.WithFields(logrus.Fields{
		"feedback_count": len(feedbackHistory),
		"text_preview": func() string {
			if len(text) > 100 {
				return text[:100] + "..."
			}
			return text
		}(),
		"text_length": len(text),
	}).Info("🎓 applyTrainingFeedback - Processing text with training data")
	
	for _, feedback := range feedbackHistory {
		// Only process feedback marked as "correct" (positive training)
		if feedback.UserDecision == "correct" {
			// Check if this trained text exists in the current text
			entityText := feedback.EntityText
			entityType := feedback.EntityType
			
			logrus.WithFields(logrus.Fields{
				"trained_text": entityText,
				"trained_type": entityType,
				"searching_in": text,
			}).Info("🔍 applyTrainingFeedback - Searching for trained text")
			
			// Find all occurrences of the trained text (case insensitive)
			lowerText := strings.ToLower(text)
			lowerEntityText := strings.ToLower(entityText)
			
			startIndex := 0
			for {
				index := strings.Index(lowerText[startIndex:], lowerEntityText)
				if index == -1 {
					break
				}
				
				actualIndex := startIndex + index
				logrus.WithFields(logrus.Fields{
					"trained_text": entityText,
					"trained_type": entityType,
					"position": actualIndex,
					"found_text": text[actualIndex:actualIndex+len(entityText)],
				}).Info("🎯 Creating entity from training feedback")
				
				trainingEntities = append(trainingEntities, Entity{
					Type:       entityType,
					Text:       text[actualIndex : actualIndex+len(entityText)], // Use original case
					Start:      actualIndex,
					End:        actualIndex + len(entityText),
					Confidence: 0.95, // High confidence for trained entities
					Category:   entityType,
				})
				
				startIndex = actualIndex + len(entityText)
			}
		} else {
			logrus.WithFields(logrus.Fields{
				"trained_text": feedback.EntityText,
				"user_decision": feedback.UserDecision,
			}).Info("🚫 applyTrainingFeedback - Skipping non-correct feedback")
		}
	}
	
	logrus.WithField("training_entities_created", len(trainingEntities)).Info("✅ applyTrainingFeedback - Completed")
	return trainingEntities
}

func (d *Detector) detectWithAzure(text string) ([]Entity, error) {
	logrus.WithFields(logrus.Fields{
		"endpoint": d.config.Azure.Endpoint,
		"text_length": len(text),
	}).Info("🌐 detectWithAzure() ENTRY - Creating Azure client")
	
	// Use native Go Azure client - no Python needed!
	azureClient := NewAzureClient(d.config.Azure.Endpoint, d.config.Azure.APIKey)
	
	logrus.Info("📡 detectWithAzure() - Calling azureClient.DetectPII()")
	entities, err := azureClient.DetectPII(text)
	
	if err != nil {
		logrus.WithError(err).Error("❌ detectWithAzure() - Azure client returned error")
		return nil, err
	}
	
	logrus.WithField("entities_returned", len(entities)).Info("✅ detectWithAzure() - Azure client returned successfully")
	return entities, nil
}

func (d *Detector) detectWithRegex(text string) []Entity {
	// Use the enhanced regex detector
	regexDetector := NewRegexDetector()
	return regexDetector.DetectPII(text)
}

func (d *Detector) deduplicateEntities(entities []Entity) []Entity {
	seen := make(map[string]bool)
	var unique []Entity
	
	for _, entity := range entities {
		key := fmt.Sprintf("%d-%d-%s", entity.Start, entity.End, entity.Type)
		if !seen[key] {
			seen[key] = true
			unique = append(unique, entity)
		}
	}
	
	return unique
}

func (d *Detector) applyRedaction(text string, entities []Entity, options *RedactOptions) string {
	if len(entities) == 0 {
		return text
	}
	
	// Create a copy of entities to avoid modifying the original
	entitiesCopy := make([]Entity, len(entities))
	copy(entitiesCopy, entities)
	
	// Sort entities by start position in reverse order to maintain indices
	for i := 0; i < len(entitiesCopy)-1; i++ {
		for j := i + 1; j < len(entitiesCopy); j++ {
			if entitiesCopy[i].Start < entitiesCopy[j].Start {
				entitiesCopy[i], entitiesCopy[j] = entitiesCopy[j], entitiesCopy[i]
			}
		}
	}
	
	redacted := text
	for _, entity := range entitiesCopy {
		// Validate entity bounds
		if entity.Start < 0 || entity.End > len(redacted) || entity.Start >= entity.End {
			logrus.WithFields(logrus.Fields{
				"start": entity.Start,
				"end":   entity.End,
				"text_len": len(redacted),
				"entity_text": entity.Text,
			}).Warn("Invalid entity bounds, skipping")
			continue
		}
		
		replacement := d.getReplacementText(entity, options)
		
		// Replace the entity text safely
		before := redacted[:entity.Start]
		after := redacted[entity.End:]
		redacted = before + replacement + after
	}
	
	return redacted
}

// GetConfidenceEngine exposes the confidence engine for feedback training
func (d *Detector) GetConfidenceEngine() *ConfidenceEngine {
	return d.confidenceEngine
}

func (d *Detector) getReplacementText(entity Entity, options *RedactOptions) string {
	// Check for custom labels first
	if label, exists := options.CustomLabels[entity.Type]; exists {
		return label
	}
	
	// Default labels based on entity type
	defaultLabels := map[string]string{
		"Person":      "[REDACTED_NAME]",
		"email":       "[REDACTED_EMAIL]",
		"phone":       "[REDACTED_PHONE]",
		"ssn":         "[REDACTED_SSN]",
		"credit_card": "[REDACTED_CARD]",
		"ip_address":  "[REDACTED_IP]",
	}
	
	if label, exists := defaultLabels[entity.Type]; exists {
		return label
	}
	
	// Fallback based on redaction mode
	switch options.RedactionMode {
	case "mask":
		return strings.Repeat("*", len(entity.Text))
	case "remove":
		return ""
	default: // "replace"
		return "[REDACTED]"
	}
}