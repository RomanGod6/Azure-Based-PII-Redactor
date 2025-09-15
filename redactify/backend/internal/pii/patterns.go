package pii

import (
	"regexp"
	"strings"
)

// RegexDetector handles pattern-based PII detection
type RegexDetector struct {
	patterns map[string]*PatternConfig
}

// PatternConfig holds regex pattern and metadata
type PatternConfig struct {
	Pattern    *regexp.Regexp
	Confidence float64
	Category   string
	Label      string
}

// NewRegexDetector creates a new regex-based detector
func NewRegexDetector() *RegexDetector {
	detector := &RegexDetector{
		patterns: make(map[string]*PatternConfig),
	}
	detector.initPatterns()
	return detector
}

// initPatterns initializes all regex patterns for PII detection
func (d *RegexDetector) initPatterns() {
	patterns := map[string]struct {
		regex      string
		confidence float64
		category   string
		label      string
	}{
		// Email addresses
		"email": {
			regex:      `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`,
			confidence: 0.95,
			category:   "PersonalInfo",
			label:      "[EMAIL]",
		},
		
		// Phone numbers (multiple formats)
		"phone_us": {
			regex:      `\b(?:\+?1[-.\s]?)?\(?([2-9]\d{2})\)?[-.\s]?([2-9]\d{2})[-.\s]?(\d{4})\b`,
			confidence: 0.9,
			category:   "PersonalInfo",
			label:      "[PHONE]",
		},
		"phone_international": {
			regex:      `\b\+\d{1,3}[-.\s]?\d{1,14}\b`,
			confidence: 0.85,
			category:   "PersonalInfo",
			label:      "[PHONE]",
		},
		
		// Social Security Numbers
		"ssn": {
			regex:      `\b\d{3}-?\d{2}-?\d{4}\b`,
			confidence: 0.95,
			category:   "FinancialInfo",
			label:      "[SSN]",
		},
		
		// Credit Card Numbers
		"credit_card": {
			regex:      `\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b`,
			confidence: 0.9,
			category:   "FinancialInfo",
			label:      "[CREDIT_CARD]",
		},
		
		// IP Addresses
		"ip_address": {
			regex:      `\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b`,
			confidence: 0.8,
			category:   "TechnicalInfo",
			label:      "[IP_ADDRESS]",
		},
		
		// URLs
		"url": {
			regex:      `https?://[^\s/$.?#].[^\s]*`,
			confidence: 0.85,
			category:   "TechnicalInfo",
			label:      "[URL]",
		},
		
		// Bank Account Numbers (US)
		"bank_account": {
			regex:      `\b\d{8,17}\b`,
			confidence: 0.7, // Lower confidence as it could be other numbers
			category:   "FinancialInfo",
			label:      "[BANK_ACCOUNT]",
		},
		
		// Driver License Numbers (patterns vary by state)
		"driver_license": {
			regex:      `\b[A-Z]{1,2}\d{6,8}\b|\b\d{8,9}\b`,
			confidence: 0.75,
			category:   "PersonalInfo",
			label:      "[DRIVER_LICENSE]",
		},
		
		// Passport Numbers
		"passport": {
			regex:      `\b[A-Z]{1,2}\d{6,9}\b`,
			confidence: 0.8,
			category:   "PersonalInfo",
			label:      "[PASSPORT]",
		},
		
		// Dates (various formats)
		"date": {
			regex:      `\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b`,
			confidence: 0.7,
			category:   "DateTime",
			label:      "[DATE]",
		},
		
		// Canadian Social Insurance Number
		"canada_sin": {
			regex:      `\b\d{3}-?\d{3}-?\d{3}\b`,
			confidence: 0.9,
			category:   "PersonalInfo",
			label:      "[CA_SIN]",
		},
		
		// UK National Insurance Number
		"uk_nino": {
			regex:      `\b[A-CEGHJ-PR-TW-Z]{1}[A-CEGHJ-NPR-TW-Z]{1}\d{6}[A-D]{1}\b`,
			confidence: 0.95,
			category:   "PersonalInfo",
			label:      "[UK_NINO]",
		},
		
		// IBAN (International Bank Account Number)
		"iban": {
			regex:      `\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b`,
			confidence: 0.9,
			category:   "FinancialInfo",
			label:      "[IBAN]",
		},
		
		// Medical Record Numbers
		"medical_record": {
			regex:      `\bMR[N]?[-.\s]?\d{6,10}\b|\bMED[-.\s]?\d{6,10}\b`,
			confidence: 0.85,
			category:   "HealthInfo",
			label:      "[MEDICAL_RECORD]",
		},
		
		// Vehicle Identification Number (VIN)
		"vin": {
			regex:      `\b[A-HJ-NPR-Z0-9]{17}\b`,
			confidence: 0.8,
			category:   "PersonalInfo",
			label:      "[VIN]",
		},
		
		// MAC Addresses
		"mac_address": {
			regex:      `\b[0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}[:-][0-9a-fA-F]{2}\b`,
			confidence: 0.9,
			category:   "TechnicalInfo",
			label:      "[MAC_ADDRESS]",
		},
		
		// Bitcoin Addresses
		"bitcoin_address": {
			regex:      `\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|bc1[a-z0-9]{39,59}\b`,
			confidence: 0.85,
			category:   "FinancialInfo",
			label:      "[CRYPTO_ADDRESS]",
		},
	}
	
	// Compile all patterns
	for name, config := range patterns {
		compiled, err := regexp.Compile(config.regex)
		if err != nil {
			// Log error but continue - don't fail startup
			continue
		}
		
		d.patterns[name] = &PatternConfig{
			Pattern:    compiled,
			Confidence: config.confidence,
			Category:   config.category,
			Label:      config.label,
		}
	}
}

// DetectPII detects PII using regex patterns
func (d *RegexDetector) DetectPII(text string) []Entity {
	var entities []Entity
	
	for patternName, config := range d.patterns {
		matches := config.Pattern.FindAllStringIndex(text, -1)
		
		for _, match := range matches {
			start, end := match[0], match[1]
			matchedText := text[start:end]
			
			// Additional validation for some patterns
			if d.isValidMatch(patternName, matchedText) {
				entities = append(entities, Entity{
					Type:       patternName,
					Text:       matchedText,
					Start:      start,
					End:        end,
					Confidence: config.Confidence,
					Category:   config.Category,
				})
			}
		}
	}
	
	return entities
}

// isValidMatch performs additional validation for specific patterns
func (d *RegexDetector) isValidMatch(patternType, text string) bool {
	switch patternType {
	case "credit_card":
		return d.isValidCreditCard(text)
	case "ssn":
		return d.isValidSSN(text)
	case "phone_us":
		return d.isValidUSPhone(text)
	case "email":
		return d.isValidEmail(text)
	default:
		return true
	}
}

// isValidCreditCard validates credit card using Luhn algorithm
func (d *RegexDetector) isValidCreditCard(number string) bool {
	// Remove spaces and dashes
	cleaned := strings.ReplaceAll(strings.ReplaceAll(number, " ", ""), "-", "")
	
	if len(cleaned) < 13 || len(cleaned) > 19 {
		return false
	}
	
	// Luhn algorithm
	sum := 0
	alternate := false
	
	for i := len(cleaned) - 1; i >= 0; i-- {
		digit := int(cleaned[i] - '0')
		
		if alternate {
			digit *= 2
			if digit > 9 {
				digit = (digit % 10) + 1
			}
		}
		
		sum += digit
		alternate = !alternate
	}
	
	return sum%10 == 0
}

// isValidSSN performs basic SSN validation
func (d *RegexDetector) isValidSSN(ssn string) bool {
	// Remove dashes
	cleaned := strings.ReplaceAll(ssn, "-", "")
	
	if len(cleaned) != 9 {
		return false
	}
	
	// Check for invalid patterns
	invalidPatterns := []string{
		"000000000", "111111111", "222222222", "333333333",
		"444444444", "555555555", "666666666", "777777777",
		"888888888", "999999999", "123456789", "987654321",
	}
	
	for _, invalid := range invalidPatterns {
		if cleaned == invalid {
			return false
		}
	}
	
	// Area number cannot be 000 or 666
	area := cleaned[:3]
	if area == "000" || area == "666" {
		return false
	}
	
	// Group number cannot be 00
	if cleaned[3:5] == "00" {
		return false
	}
	
	// Serial number cannot be 0000
	if cleaned[5:] == "0000" {
		return false
	}
	
	return true
}

// isValidUSPhone validates US phone number format
func (d *RegexDetector) isValidUSPhone(phone string) bool {
	// Remove formatting
	cleaned := regexp.MustCompile(`[^\d]`).ReplaceAllString(phone, "")
	
	if len(cleaned) == 11 && cleaned[0] == '1' {
		cleaned = cleaned[1:] // Remove country code
	}
	
	if len(cleaned) != 10 {
		return false
	}
	
	// Area code cannot start with 0 or 1
	if cleaned[0] == '0' || cleaned[0] == '1' {
		return false
	}
	
	// Exchange code cannot start with 0 or 1
	if cleaned[3] == '0' || cleaned[3] == '1' {
		return false
	}
	
	return true
}

// isValidEmail performs basic email validation beyond regex
func (d *RegexDetector) isValidEmail(email string) bool {
	parts := strings.Split(email, "@")
	if len(parts) != 2 {
		return false
	}
	
	local, domain := parts[0], parts[1]
	
	// Basic validation
	if len(local) == 0 || len(domain) == 0 {
		return false
	}
	
	if len(local) > 64 || len(domain) > 253 {
		return false
	}
	
	// Domain should have at least one dot
	if !strings.Contains(domain, ".") {
		return false
	}
	
	return true
}