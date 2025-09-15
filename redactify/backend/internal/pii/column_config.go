package pii

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// DetectionMode defines how aggressively to detect PII in a column
type DetectionMode string

const (
	DetectionModeAggressive    DetectionMode = "aggressive"
	DetectionModeBalanced      DetectionMode = "balanced"
	DetectionModeConservative  DetectionMode = "conservative"
	DetectionModeCustom        DetectionMode = "custom"
	DetectionModeDisabled      DetectionMode = "disabled"
)

// DataType defines the expected data type for a column
type DataType string

const (
	DataTypeText         DataType = "text"
	DataTypeEmail        DataType = "email"
	DataTypePhone        DataType = "phone"
	DataTypeName         DataType = "name"
	DataTypeAddress      DataType = "address"
	DataTypeIDNumber     DataType = "id_number"
	DataTypeFinancial    DataType = "financial"
	DataTypeDate         DataType = "date"
	DataTypeNumeric      DataType = "numeric"
	DataTypeCategorical  DataType = "categorical"
	DataTypeProductCode  DataType = "product_code"
	DataTypeReference    DataType = "reference"
	DataTypeDescription  DataType = "description"
	DataTypeComments     DataType = "comments"
	DataTypeURL          DataType = "url"
	DataTypeMedical      DataType = "medical"
	DataTypeLegal        DataType = "legal"
)

// WhitelistPattern represents a pattern that should NOT be redacted
type WhitelistPattern struct {
	Pattern       string `json:"pattern"`
	Description   string `json:"description"`
	Regex         bool   `json:"regex"`
	CaseSensitive bool   `json:"case_sensitive"`
	compiled      *regexp.Regexp
}

// BlacklistPattern represents a pattern that should ALWAYS be redacted
type BlacklistPattern struct {
	Pattern       string `json:"pattern"`
	Description   string `json:"description"`
	Replacement   string `json:"replacement"`
	Regex         bool   `json:"regex"`
	CaseSensitive bool   `json:"case_sensitive"`
	compiled      *regexp.Regexp
}

// EntityRule defines rules for specific PII entity types
type EntityRule struct {
	EntityType        string  `json:"entity_type"`
	Enabled           bool    `json:"enabled"`
	ConfidenceThreshold float64 `json:"confidence_threshold"`
	CustomReplacement *string `json:"custom_replacement,omitempty"`
	ContextAware      bool    `json:"context_aware"`
}

// ColumnConfig holds configuration for a single column
type ColumnConfig struct {
	ColumnName       string              `json:"column_name"`
	DetectionMode    DetectionMode       `json:"detection_mode"`
	ExpectedDataType DataType            `json:"expected_data_type"`
	
	// Entity-specific rules
	EntityRules map[string]*EntityRule `json:"entity_rules,omitempty"`
	
	// Custom patterns
	WhitelistPatterns []*WhitelistPattern `json:"whitelist_patterns,omitempty"`
	BlacklistPatterns []*BlacklistPattern `json:"blacklist_patterns,omitempty"`
	
	// Entity type exclusions
	ExcludedEntityTypes []string `json:"excluded_entity_types,omitempty"`
	
	// Validation rules
	MinConfidence        float64 `json:"min_confidence"`
	RequireHumanReview   bool    `json:"require_human_review"`
	
	// Context awareness
	BusinessContext *string  `json:"business_context,omitempty"`
	DomainKeywords  []string `json:"domain_keywords,omitempty"`
	
	// Advanced settings
	PreserveFormatting bool `json:"preserve_formatting"`
	PartialRedaction   bool `json:"partial_redaction"`
}

// ColumnConfigManager manages column-specific PII detection configurations
type ColumnConfigManager struct {
	configs map[string]*ColumnConfig
}

// NewColumnConfigManager creates a new column configuration manager
func NewColumnConfigManager() *ColumnConfigManager {
	return &ColumnConfigManager{
		configs: make(map[string]*ColumnConfig),
	}
}

// SetColumnConfig sets the configuration for a specific column
func (ccm *ColumnConfigManager) SetColumnConfig(config *ColumnConfig) error {
	// Compile regex patterns
	if err := ccm.compilePatterns(config); err != nil {
		return fmt.Errorf("failed to compile patterns for column %s: %w", config.ColumnName, err)
	}
	
	ccm.configs[config.ColumnName] = config
	return nil
}

// GetColumnConfig gets the configuration for a specific column
func (ccm *ColumnConfigManager) GetColumnConfig(columnName string) *ColumnConfig {
	if config, exists := ccm.configs[columnName]; exists {
		return config
	}
	
	// Return default configuration
	return &ColumnConfig{
		ColumnName:       columnName,
		DetectionMode:    DetectionModeBalanced,
		ExpectedDataType: DataTypeText,
		MinConfidence:    0.7,
		RequireHumanReview: false,
		PreserveFormatting: false,
		PartialRedaction:   false,
	}
}

// ApplyColumnFiltering applies column-specific filtering to detected entities
func (ccm *ColumnConfigManager) ApplyColumnFiltering(columnName string, entities []Entity, text string) []Entity {
	config := ccm.GetColumnConfig(columnName)
	
	// If detection is disabled, return empty
	if config.DetectionMode == DetectionModeDisabled {
		return []Entity{}
	}
	
	var filteredEntities []Entity
	
	for _, entity := range entities {
		// Apply confidence threshold
		if entity.Confidence < config.MinConfidence {
			continue
		}
		
		// Check if entity type is excluded
		if ccm.isEntityTypeExcluded(config, entity.Type) {
			continue
		}
		
		// Apply whitelist filtering
		if ccm.isWhitelisted(config, entity.Text) {
			continue
		}
		
		// Apply detection mode filtering
		if !ccm.shouldDetectByMode(config, entity) {
			continue
		}
		
		// Apply entity-specific rules
		if entityRule, exists := config.EntityRules[entity.Type]; exists {
			if !entityRule.Enabled {
				continue
			}
			if entity.Confidence < entityRule.ConfidenceThreshold {
				continue
			}
			// Apply custom replacement if specified
			if entityRule.CustomReplacement != nil {
				entity.Text = *entityRule.CustomReplacement
			}
		}
		
		filteredEntities = append(filteredEntities, entity)
	}
	
	// Apply blacklist patterns (always redact these)
	blacklistEntities := ccm.findBlacklistEntities(config, text)
	filteredEntities = append(filteredEntities, blacklistEntities...)
	
	return filteredEntities
}

// compilePatterns compiles regex patterns in the configuration
func (ccm *ColumnConfigManager) compilePatterns(config *ColumnConfig) error {
	// Compile whitelist patterns
	for _, pattern := range config.WhitelistPatterns {
		if pattern.Regex {
			regexPattern := pattern.Pattern
			if !pattern.CaseSensitive {
				regexPattern = "(?i)" + regexPattern
			}
			compiled, err := regexp.Compile(regexPattern)
			if err != nil {
				return fmt.Errorf("invalid whitelist regex pattern '%s': %w", pattern.Pattern, err)
			}
			pattern.compiled = compiled
		}
	}
	
	// Compile blacklist patterns
	for _, pattern := range config.BlacklistPatterns {
		if pattern.Regex {
			regexPattern := pattern.Pattern
			if !pattern.CaseSensitive {
				regexPattern = "(?i)" + regexPattern
			}
			compiled, err := regexp.Compile(regexPattern)
			if err != nil {
				return fmt.Errorf("invalid blacklist regex pattern '%s': %w", pattern.Pattern, err)
			}
			pattern.compiled = compiled
		}
	}
	
	return nil
}

// isEntityTypeExcluded checks if an entity type is excluded for this column
func (ccm *ColumnConfigManager) isEntityTypeExcluded(config *ColumnConfig, entityType string) bool {
	for _, excluded := range config.ExcludedEntityTypes {
		if excluded == entityType {
			return true
		}
	}
	return false
}

// isWhitelisted checks if text matches any whitelist patterns
func (ccm *ColumnConfigManager) isWhitelisted(config *ColumnConfig, text string) bool {
	for _, pattern := range config.WhitelistPatterns {
		if pattern.Regex {
			if pattern.compiled != nil && pattern.compiled.MatchString(text) {
				return true
			}
		} else {
			// Literal string matching
			if pattern.CaseSensitive {
				if strings.Contains(text, pattern.Pattern) {
					return true
				}
			} else {
				if strings.Contains(strings.ToLower(text), strings.ToLower(pattern.Pattern)) {
					return true
				}
			}
		}
	}
	return false
}

// shouldDetectByMode applies detection mode logic
func (ccm *ColumnConfigManager) shouldDetectByMode(config *ColumnConfig, entity Entity) bool {
	switch config.DetectionMode {
	case DetectionModeAggressive:
		return entity.Confidence > 0.5
	case DetectionModeConservative:
		return entity.Confidence > 0.9
	case DetectionModeBalanced:
		return entity.Confidence > 0.7
	case DetectionModeCustom:
		// Only use explicit rules, no default detection
		return false
	case DetectionModeDisabled:
		return false
	default:
		return entity.Confidence > 0.7
	}
}

// findBlacklistEntities finds entities that match blacklist patterns
func (ccm *ColumnConfigManager) findBlacklistEntities(config *ColumnConfig, text string) []Entity {
	var entities []Entity
	
	for _, pattern := range config.BlacklistPatterns {
		if pattern.Regex {
			if pattern.compiled != nil {
				matches := pattern.compiled.FindAllStringIndex(text, -1)
				for _, match := range matches {
					entities = append(entities, Entity{
						Type:       "blacklisted",
						Text:       text[match[0]:match[1]],
						Start:      match[0],
						End:        match[1],
						Confidence: 1.0, // Blacklisted items have max confidence
						Category:   "custom",
					})
				}
			}
		} else {
			// Literal string matching
			searchText := text
			searchPattern := pattern.Pattern
			
			if !pattern.CaseSensitive {
				searchText = strings.ToLower(text)
				searchPattern = strings.ToLower(pattern.Pattern)
			}
			
			index := strings.Index(searchText, searchPattern)
			if index != -1 {
				entities = append(entities, Entity{
					Type:       "blacklisted",
					Text:       text[index:index+len(pattern.Pattern)],
					Start:      index,
					End:        index + len(pattern.Pattern),
					Confidence: 1.0,
					Category:   "custom",
				})
			}
		}
	}
	
	return entities
}

// LoadFromJSON loads column configurations from JSON
func (ccm *ColumnConfigManager) LoadFromJSON(jsonData []byte) error {
	var configs []*ColumnConfig
	if err := json.Unmarshal(jsonData, &configs); err != nil {
		return fmt.Errorf("failed to unmarshal column configs: %w", err)
	}
	
	for _, config := range configs {
		if err := ccm.SetColumnConfig(config); err != nil {
			return err
		}
	}
	
	return nil
}

// SaveToJSON saves column configurations to JSON
func (ccm *ColumnConfigManager) SaveToJSON() ([]byte, error) {
	var configs []*ColumnConfig
	for _, config := range ccm.configs {
		configs = append(configs, config)
	}
	
	return json.MarshalIndent(configs, "", "  ")
}

// GetDefaultConfigForDataType returns a default configuration based on data type
func GetDefaultConfigForDataType(columnName string, dataType DataType) *ColumnConfig {
	config := &ColumnConfig{
		ColumnName:       columnName,
		ExpectedDataType: dataType,
		MinConfidence:    0.7,
		EntityRules:      make(map[string]*EntityRule),
	}
	
	switch dataType {
	case DataTypeEmail:
		config.DetectionMode = DetectionModeAggressive
		config.EntityRules["Email"] = &EntityRule{
			EntityType:          "Email",
			Enabled:             true,
			ConfidenceThreshold: 0.9,
			ContextAware:        true,
		}
	case DataTypePhone:
		config.DetectionMode = DetectionModeAggressive
		config.EntityRules["PhoneNumber"] = &EntityRule{
			EntityType:          "PhoneNumber",
			Enabled:             true,
			ConfidenceThreshold: 0.8,
			ContextAware:        true,
		}
	case DataTypeName:
		config.DetectionMode = DetectionModeBalanced
		config.EntityRules["Person"] = &EntityRule{
			EntityType:          "Person",
			Enabled:             true,
			ConfidenceThreshold: 0.8,
			ContextAware:        true,
		}
	case DataTypeFinancial:
		config.DetectionMode = DetectionModeAggressive
		config.MinConfidence = 0.9
	case DataTypeProductCode, DataTypeReference:
		config.DetectionMode = DetectionModeConservative
	default:
		config.DetectionMode = DetectionModeBalanced
	}
	
	return config
}