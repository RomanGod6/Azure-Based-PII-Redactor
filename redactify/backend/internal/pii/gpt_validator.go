package pii

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"redactify/pkg/config"
	"strings"
	"time"
)

// GPTValidator handles GPT-based PII validation
type GPTValidator struct {
	config *config.Config
	client *http.Client
}

// ValidationResult represents the result of GPT validation
type ValidationResult struct {
	IsRealPII       bool    `json:"is_real_pii"`
	Confidence      float64 `json:"confidence"`
	Explanation     string  `json:"explanation"`
	ShouldRedact    bool    `json:"should_redact"`
	SuggestedAction string  `json:"suggested_action"`
}

// GPTRequest represents the request to Azure OpenAI
type GPTRequest struct {
	Messages    []GPTMessage `json:"messages"`
	MaxTokens   int          `json:"max_tokens"`
	Temperature float64      `json:"temperature"`
	TopP        float64      `json:"top_p"`
}

// GPTMessage represents a message in the conversation
type GPTMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// GPTResponse represents the response from Azure OpenAI
type GPTResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

// NewGPTValidator creates a new GPT validator
func NewGPTValidator(cfg *config.Config) *GPTValidator {
	return &GPTValidator{
		config: cfg,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ValidateEntities validates a list of detected entities using GPT
func (v *GPTValidator) ValidateEntities(text string, entities []Entity) ([]ValidationResult, error) {
	if v.config.Azure.GPTEndpoint == "" || v.config.Azure.GPTAPIKey == "" {
		// Return default validation if GPT not configured
		results := make([]ValidationResult, len(entities))
		for i := range entities {
			results[i] = ValidationResult{
				IsRealPII:       true,
				Confidence:      0.8, // Default confidence
				Explanation:     "GPT validation not configured, using default approval",
				ShouldRedact:    true,
				SuggestedAction: "redact",
			}
		}
		return results, nil
	}

	var results []ValidationResult
	
	// Process entities in batches to avoid token limits
	batchSize := 5
	for i := 0; i < len(entities); i += batchSize {
		end := i + batchSize
		if end > len(entities) {
			end = len(entities)
		}
		
		batch := entities[i:end]
		batchResults, err := v.validateBatch(text, batch)
		if err != nil {
			// If GPT fails, default to allowing redaction
			for range batch {
				results = append(results, ValidationResult{
					IsRealPII:       true,
					Confidence:      0.7,
					Explanation:     fmt.Sprintf("GPT validation failed: %v", err),
					ShouldRedact:    true,
					SuggestedAction: "redact",
				})
			}
		} else {
			results = append(results, batchResults...)
		}
	}
	
	return results, nil
}

// validateBatch validates a batch of entities
func (v *GPTValidator) validateBatch(text string, entities []Entity) ([]ValidationResult, error) {
	// Create the validation prompt
	prompt := v.createValidationPrompt(text, entities)
	
	// Prepare GPT request
	request := GPTRequest{
		Messages: []GPTMessage{
			{
				Role:    "system",
				Content: v.getSystemPrompt(),
			},
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   1000,
		Temperature: 0.1, // Low temperature for consistent results
		TopP:        0.1,
	}
	
	// Make the API call
	response, err := v.callGPTAPI(request)
	if err != nil {
		return nil, err
	}
	
	// Parse the response
	return v.parseValidationResponse(response, len(entities))
}

// createValidationPrompt creates a prompt for validating the entities
func (v *GPTValidator) createValidationPrompt(text string, entities []Entity) string {
	var prompt strings.Builder
	
	prompt.WriteString("Please analyze the following text and validate whether the detected entities are actually sensitive PII that should be redacted:\n\n")
	prompt.WriteString("**Original Text:**\n")
	prompt.WriteString(text)
	prompt.WriteString("\n\n**Detected Entities:**\n")
	
	for i, entity := range entities {
		prompt.WriteString(fmt.Sprintf("%d. \"%s\" (Type: %s, Confidence: %.2f)\n", 
			i+1, entity.Text, entity.Type, entity.Confidence))
	}
	
	prompt.WriteString("\n**Instructions:**\n")
	prompt.WriteString("For each entity, respond with a JSON object containing:\n")
	prompt.WriteString("- is_real_pii: true/false\n")
	prompt.WriteString("- confidence: 0.0-1.0\n")
	prompt.WriteString("- explanation: brief reason\n")
	prompt.WriteString("- should_redact: true/false\n")
	prompt.WriteString("- suggested_action: \"redact\", \"keep\", or \"partial\"\n\n")
	prompt.WriteString("Return only a JSON array with one object per entity, in order.")
	
	return prompt.String()
}

// getSystemPrompt returns the system prompt for GPT
func (v *GPTValidator) getSystemPrompt() string {
	return `You are an expert PII (Personally Identifiable Information) validation system. Your job is to analyze detected entities and determine if they are actually sensitive PII that should be redacted.

Consider these factors:
1. Context: Is this information used in a business/professional context?
2. Sensitivity: Would this information pose a privacy risk if disclosed?
3. False Positives: Common business terms that look like PII but aren't sensitive
4. Placeholder Text: Example data that isn't real PII

Examples of what to KEEP (not redact):
- Business contact names in signatures
- Company names that happen to sound like person names
- Generic examples like "John Doe" or "example@company.com"
- Job titles and department names
- Product names or services

Examples of what to REDACT:
- Customer personal information
- Employee personal details in non-business context
- Real email addresses of individuals
- Phone numbers for personal contacts
- Addresses for individuals

Respond only with valid JSON arrays. Be conservative - when in doubt, choose to redact.`
}

// callGPTAPI makes the actual API call to Azure OpenAI
func (v *GPTValidator) callGPTAPI(request GPTRequest) (string, error) {
	// Build the URL
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		strings.TrimSuffix(v.config.Azure.GPTEndpoint, "/"),
		v.config.Azure.GPTDeployment,
		v.config.Azure.GPTAPIVersion)
	
	// Marshal request to JSON
	jsonData, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}
	
	// Create HTTP request
	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	
	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", v.config.Azure.GPTAPIKey)
	
	// Execute request
	resp, err := v.client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	
	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}
	
	// Check status code
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}
	
	// Parse response
	var gptResp GPTResponse
	if err := json.Unmarshal(body, &gptResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}
	
	// Check for errors
	if gptResp.Error != nil {
		return "", fmt.Errorf("GPT API error: %s", gptResp.Error.Message)
	}
	
	// Return content
	if len(gptResp.Choices) > 0 {
		return gptResp.Choices[0].Message.Content, nil
	}
	
	return "", fmt.Errorf("no response from GPT")
}

// parseValidationResponse parses the GPT response into ValidationResult objects
func (v *GPTValidator) parseValidationResponse(response string, expectedCount int) ([]ValidationResult, error) {
	// Clean up the response (remove markdown code blocks if present)
	response = strings.TrimSpace(response)
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimPrefix(response, "```")
	response = strings.TrimSuffix(response, "```")
	response = strings.TrimSpace(response)
	
	var results []ValidationResult
	if err := json.Unmarshal([]byte(response), &results); err != nil {
		return nil, fmt.Errorf("failed to parse validation response: %w", err)
	}
	
	// Ensure we have the expected number of results
	if len(results) != expectedCount {
		return nil, fmt.Errorf("expected %d validation results, got %d", expectedCount, len(results))
	}
	
	return results, nil
}

// IsConfigured returns true if GPT validation is properly configured
func (v *GPTValidator) IsConfigured() bool {
	return v.config.Azure.GPTEndpoint != "" && v.config.Azure.GPTAPIKey != ""
}