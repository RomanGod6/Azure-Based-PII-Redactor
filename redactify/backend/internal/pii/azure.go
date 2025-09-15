package pii

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/sirupsen/logrus"
)

// AzureClient handles Azure Text Analytics API calls
type AzureClient struct {
	endpoint string
	apiKey   string
	client   *http.Client
}

// AzureRequest represents the request payload for Azure Text Analytics
type AzureRequest struct {
	Documents []AzureDocument `json:"documents"`
}

// AzureDocument represents a single document in the Azure request
type AzureDocument struct {
	ID       string `json:"id"`
	Language string `json:"language"`
	Text     string `json:"text"`
}

// AzureResponse represents the response from Azure Text Analytics
type AzureResponse struct {
	Documents []AzureDocumentResult `json:"documents"`
	Errors    []AzureError          `json:"errors,omitempty"`
}

// AzureDocumentResult represents the result for a single document
type AzureDocumentResult struct {
	ID       string        `json:"id"`
	Entities []AzureEntity `json:"entities"`
}

// AzureEntity represents a detected entity from Azure
type AzureEntity struct {
	Text            string  `json:"text"`
	Category        string  `json:"category"`
	Subcategory     string  `json:"subcategory,omitempty"`
	ConfidenceScore float64 `json:"confidenceScore"`
	Offset          int     `json:"offset"`
	Length          int     `json:"length"`
}

// AzureError represents an error in the Azure response
type AzureError struct {
	ID    string `json:"id"`
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// NewAzureClient creates a new Azure Text Analytics client
func NewAzureClient(endpoint, apiKey string) *AzureClient {
	return &AzureClient{
		endpoint: endpoint,
		apiKey:   apiKey,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// DetectPII detects PII in text using Azure Text Analytics
func (c *AzureClient) DetectPII(text string) ([]Entity, error) {
	logrus.WithFields(logrus.Fields{
		"endpoint": c.endpoint,
		"api_key_length": len(c.apiKey),
		"text_length": len(text),
	}).Info("🚀 AzureClient.DetectPII() ENTRY")
	
	if c.endpoint == "" || c.apiKey == "" {
		logrus.WithFields(logrus.Fields{
			"endpoint_empty": c.endpoint == "",
			"api_key_empty": c.apiKey == "",
		}).Error("❌ AzureClient.DetectPII() - Azure endpoint and API key must be configured")
		return nil, fmt.Errorf("azure endpoint and API key must be configured")
	}

	logrus.Info("📋 AzureClient.DetectPII() - Preparing Azure request")
	// Prepare request
	req := AzureRequest{
		Documents: []AzureDocument{
			{
				ID:       "1",
				Language: "en",
				Text:     text,
			},
		},
	}
	
	logrus.WithField("document_count", len(req.Documents)).Info("📋 AzureClient.DetectPII() - Request prepared")

	// Convert to JSON
	logrus.Info("🔧 AzureClient.DetectPII() - Converting request to JSON")
	jsonData, err := json.Marshal(req)
	if err != nil {
		logrus.WithError(err).Error("❌ AzureClient.DetectPII() - Failed to marshal request")
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	logrus.WithField("json_length", len(jsonData)).Info("✅ AzureClient.DetectPII() - JSON marshaling completed")

	// Build URL - ensure it has the correct path
	url := c.endpoint
	if url[len(url)-1] == '/' {
		url = url[:len(url)-1]
	}
	if !contains(url, "/text/analytics/v3.1/entities/recognition/pii") {
		url += "/text/analytics/v3.1/entities/recognition/pii"
	}
	
	logrus.WithField("final_url", url).Info("🔗 AzureClient.DetectPII() - URL constructed")

	// Create HTTP request
	logrus.Info("📤 AzureClient.DetectPII() - Creating HTTP request")
	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		logrus.WithError(err).Error("❌ AzureClient.DetectPII() - Failed to create HTTP request")
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Ocp-Apim-Subscription-Key", c.apiKey)
	
	logrus.WithFields(logrus.Fields{
		"content_type": httpReq.Header.Get("Content-Type"),
		"has_auth_header": httpReq.Header.Get("Ocp-Apim-Subscription-Key") != "",
	}).Info("📋 AzureClient.DetectPII() - Headers set")

	// Execute request
	logrus.Info("🌐 AzureClient.DetectPII() - Executing HTTP request to Azure")
	requestStartTime := time.Now()
	resp, err := c.client.Do(httpReq)
	requestDuration := time.Since(requestStartTime)
	
	if err != nil {
		logrus.WithError(err).WithField("duration", requestDuration).Error("❌ AzureClient.DetectPII() - HTTP request failed")
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	
	logrus.WithFields(logrus.Fields{
		"status_code": resp.StatusCode,
		"duration": requestDuration,
		"content_length": resp.Header.Get("Content-Length"),
	}).Info("📥 AzureClient.DetectPII() - HTTP response received")

	// Read response
	logrus.Info("📖 AzureClient.DetectPII() - Reading response body")
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		logrus.WithError(err).Error("❌ AzureClient.DetectPII() - Failed to read response body")
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	
	logrus.WithField("body_length", len(body)).Info("✅ AzureClient.DetectPII() - Response body read successfully")

	// Check status code
	if resp.StatusCode != http.StatusOK {
		logrus.WithFields(logrus.Fields{
			"status_code": resp.StatusCode,
			"response_body": string(body),
		}).Error("❌ AzureClient.DetectPII() - Azure API returned non-200 status")
		return nil, fmt.Errorf("azure API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	logrus.Info("🔧 AzureClient.DetectPII() - Parsing JSON response")
	var azureResp AzureResponse
	if err := json.Unmarshal(body, &azureResp); err != nil {
		logrus.WithError(err).WithField("response_body", string(body)).Error("❌ AzureClient.DetectPII() - Failed to parse JSON response")
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	logrus.WithFields(logrus.Fields{
		"documents_count": len(azureResp.Documents),
		"errors_count": len(azureResp.Errors),
	}).Info("✅ AzureClient.DetectPII() - JSON response parsed successfully")

	// Check for errors
	if len(azureResp.Errors) > 0 {
		logrus.WithField("error_message", azureResp.Errors[0].Error.Message).Error("❌ AzureClient.DetectPII() - Azure API returned errors")
		return nil, fmt.Errorf("azure API error: %s", azureResp.Errors[0].Error.Message)
	}

	// Convert Azure entities to our format
	logrus.Info("🔄 AzureClient.DetectPII() - Converting Azure entities to internal format")
	var entities []Entity
	totalEntities := 0
	
	for _, doc := range azureResp.Documents {
		logrus.WithField("doc_id", doc.ID).WithField("entities_in_doc", len(doc.Entities)).Info("📋 Processing document entities")
		totalEntities += len(doc.Entities)
		
		for _, ent := range doc.Entities {
			entities = append(entities, Entity{
				Type:       ent.Category,
				Text:       ent.Text,
				Start:      ent.Offset,
				End:        ent.Offset + ent.Length,
				Confidence: ent.ConfidenceScore,
				Category:   ent.Category,
			})
		}
	}
	
	logrus.WithFields(logrus.Fields{
		"total_entities": totalEntities,
		"converted_entities": len(entities),
		"total_duration": time.Since(requestStartTime),
	}).Info("🎉 AzureClient.DetectPII() - Successfully completed Azure PII detection")

	return entities, nil
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		   (s == substr || 
		    (len(s) > len(substr) && 
		     (s[:len(substr)] == substr || 
		      s[len(s)-len(substr):] == substr || 
		      hasSubstring(s, substr))))
}

func hasSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}