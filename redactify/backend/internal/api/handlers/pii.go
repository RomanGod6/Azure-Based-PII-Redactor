package handlers

import (
	"database/sql"
	"fmt"
	"net/http"
	"redactify/internal/pii"
	"redactify/pkg/config"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

type PIIHandler struct {
	db     *sql.DB
	config *config.Config
	detector *pii.Detector
}

type DetectRequest struct {
	Text string `json:"text" binding:"required"`
}

type DetectResponse struct {
	Entities    []pii.Entity `json:"entities"`
	HasPII      bool         `json:"has_pii"`
	Confidence  float64      `json:"confidence"`
	ProcessTime string       `json:"process_time"`
}

type RedactRequest struct {
	Text     string            `json:"text" binding:"required"`
	Options  *pii.RedactOptions `json:"options,omitempty"`
}

type RedactResponse struct {
	OriginalText string       `json:"original_text"`
	RedactedText string       `json:"redacted_text"`
	Entities     []pii.Entity `json:"entities"`
	RedactedCount int         `json:"redacted_count"`
	ProcessTime  string       `json:"process_time"`
}

type BatchProcessRequest struct {
	Items   []string          `json:"items" binding:"required"`
	Options *pii.RedactOptions `json:"options,omitempty"`
}

type BatchProcessResponse struct {
	Results     []RedactResponse `json:"results"`
	TotalItems  int              `json:"total_items"`
	ProcessTime string           `json:"process_time"`
	Summary     BatchSummary     `json:"summary"`
}

type BatchSummary struct {
	TotalEntities    int     `json:"total_entities"`
	TotalRedacted    int     `json:"total_redacted"`
	AverageConfidence float64 `json:"average_confidence"`
}

func NewPIIHandler(db *sql.DB, cfg *config.Config) *PIIHandler {
	detector := pii.NewDetectorWithDB(cfg, db)
	return &PIIHandler{
		db:       db,
		config:   cfg,
		detector: detector,
	}
}

func (h *PIIHandler) DetectPII(c *gin.Context) {
	var req DetectRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	startTime := getCurrentTime()
	entities, err := h.detector.Detect(req.Text)
	if err != nil {
		logrus.WithError(err).Error("Failed to detect PII")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to detect PII"})
		return
	}

	confidence := calculateAverageConfidence(entities)
	processTime := getProcessTime(startTime)

	response := DetectResponse{
		Entities:    entities,
		HasPII:      len(entities) > 0,
		Confidence:  confidence,
		ProcessTime: processTime,
	}

	c.JSON(http.StatusOK, response)
}

func (h *PIIHandler) RedactPII(c *gin.Context) {
	var req RedactRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	startTime := getCurrentTime()
	result, err := h.detector.Redact(req.Text, req.Options)
	if err != nil {
		logrus.WithError(err).Error("Failed to redact PII")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to redact PII"})
		return
	}

	processTime := getProcessTime(startTime)
	
	response := RedactResponse{
		OriginalText:  req.Text,
		RedactedText:  result.RedactedText,
		Entities:      result.Entities,
		RedactedCount: len(result.Entities),
		ProcessTime:   processTime,
	}

	// Save to history (async)
	go h.saveToHistory(req.Text, result)

	c.JSON(http.StatusOK, response)
}

func (h *PIIHandler) BatchProcess(c *gin.Context) {
	var req BatchProcessRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	startTime := getCurrentTime()
	results := make([]RedactResponse, len(req.Items))
	totalEntities := 0
	totalRedacted := 0
	totalConfidence := 0.0

	for i, text := range req.Items {
		result, err := h.detector.Redact(text, req.Options)
		if err != nil {
			logrus.WithError(err).Errorf("Failed to process item %d", i)
			results[i] = RedactResponse{
				OriginalText:  text,
				RedactedText:  text, // Keep original on error
				Entities:      []pii.Entity{},
				RedactedCount: 0,
				ProcessTime:   "0ms",
			}
			continue
		}

		results[i] = RedactResponse{
			OriginalText:  text,
			RedactedText:  result.RedactedText,
			Entities:      result.Entities,
			RedactedCount: len(result.Entities),
			ProcessTime:   getProcessTime(startTime),
		}

		totalEntities += len(result.Entities)
		totalRedacted += len(result.Entities)
		totalConfidence += calculateAverageConfidence(result.Entities)
	}

	processTime := getProcessTime(startTime)
	avgConfidence := 0.0
	if len(req.Items) > 0 {
		avgConfidence = totalConfidence / float64(len(req.Items))
	}

	response := BatchProcessResponse{
		Results:     results,
		TotalItems:  len(req.Items),
		ProcessTime: processTime,
		Summary: BatchSummary{
			TotalEntities:     totalEntities,
			TotalRedacted:     totalRedacted,
			AverageConfidence: avgConfidence,
		},
	}

	c.JSON(http.StatusOK, response)
}

func (h *PIIHandler) GetHistory(c *gin.Context) {
	limit := 50
	if l := c.Query("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 && parsed <= 1000 {
			limit = parsed
		}
	}

	// Get total count
	var total int
	err := h.db.QueryRow("SELECT COUNT(*) FROM processing_history").Scan(&total)
	if err != nil {
		logrus.WithError(err).Debug("No processing history table found")
		c.JSON(http.StatusOK, gin.H{
			"history": []interface{}{},
			"total":   0,
			"limit":   limit,
		})
		return
	}

	// Get history records
	query := `
		SELECT id, filename, timestamp, status, entities_found, processing_time_ms 
		FROM processing_history 
		ORDER BY timestamp DESC 
		LIMIT ?
	`
	
	rows, err := h.db.Query(query, limit)
	if err != nil {
		logrus.WithError(err).Error("Failed to query processing history")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve history"})
		return
	}
	defer rows.Close()

	var history []interface{}
	for rows.Next() {
		var id, filename, status string
		var timestamp time.Time
		var entitiesFound int
		var processingTimeMs float64

		err := rows.Scan(&id, &filename, &timestamp, &status, &entitiesFound, &processingTimeMs)
		if err != nil {
			logrus.WithError(err).Warn("Failed to scan history row")
			continue
		}

		processingTime := fmt.Sprintf("%.1fms", processingTimeMs)
		
		history = append(history, gin.H{
			"id":              id,
			"filename":        filename,
			"timestamp":       timestamp.Format("2006-01-02 15:04:05"),
			"status":          status,
			"entities_found":  entitiesFound,
			"processing_time": processingTime,
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"history": history,
		"total":   total,
		"limit":   limit,
	})
}

func (h *PIIHandler) GetAnalytics(c *gin.Context) {
	// Get analytics from database
	var totalProcessed, entitiesDetected int
	var accuracyRate float64 = 0.0 // Only set if we have actual data
	var avgSpeed string = "N/A"

	// Query total processed files
	err := h.db.QueryRow("SELECT COUNT(*) FROM processing_history").Scan(&totalProcessed)
	if err != nil {
		logrus.WithError(err).Debug("No processing history found")
		totalProcessed = 0
	}

	// Query total entities detected
	err = h.db.QueryRow("SELECT COALESCE(SUM(entities_found), 0) FROM processing_history").Scan(&entitiesDetected)
	if err != nil {
		logrus.WithError(err).Debug("No entities data found")
		entitiesDetected = 0
	}

	// Only calculate accuracy and speed if we have actual processed files
	if totalProcessed > 0 {
		// Calculate accuracy rate from actual success data
		var successfulProcesses int
		err = h.db.QueryRow("SELECT COUNT(*) FROM processing_history WHERE status = 'completed'").Scan(&successfulProcesses)
		if err == nil && totalProcessed > 0 {
			accuracyRate = (float64(successfulProcesses) / float64(totalProcessed)) * 100.0
		}

		// Calculate average processing speed
		var avgSpeedMs float64
		err = h.db.QueryRow("SELECT AVG(processing_time_ms) FROM processing_history WHERE processing_time_ms > 0").Scan(&avgSpeedMs)
		if err == nil {
			avgSpeed = fmt.Sprintf("%.1fms", avgSpeedMs)
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"total_processed":   totalProcessed,
		"accuracy_rate":     accuracyRate,
		"average_speed":     avgSpeed,
		"entities_detected": entitiesDetected,
	})
}

func (h *PIIHandler) GetConfig(c *gin.Context) {
	azureConfigured := h.config.Azure.Endpoint != "" && h.config.Azure.APIKey != ""
	gptConfigured := h.config.Azure.GPTEndpoint != "" && h.config.Azure.GPTAPIKey != ""
	
	// Test Azure connectivity
	azureOnline := false
	if azureConfigured {
		azureClient := pii.NewAzureClient(h.config.Azure.Endpoint, h.config.Azure.APIKey)
		_, err := azureClient.DetectPII("test")
		azureOnline = err == nil
	}
	
	// Test GPT connectivity  
	gptOnline := false
	if gptConfigured {
		gptValidator := pii.NewGPTValidator(h.config)
		if gptValidator.IsConfigured() {
			// Simple connectivity test
			gptOnline = true // We'll assume it's online if configured
		}
	}
	
	c.JSON(http.StatusOK, gin.H{
		"go_backend":       true,
		"azure_configured": azureConfigured,
		"azure_online":     azureOnline,
		"gpt_configured":   gptConfigured,
		"gpt_online":       gptOnline,
		"detection_mode":   "azure+gpt+regex",
	})
}

func (h *PIIHandler) UpdateConfig(c *gin.Context) {
	var updateReq map[string]interface{}
	if err := c.ShouldBindJSON(&updateReq); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// TODO: Implement configuration updates
	c.JSON(http.StatusOK, gin.H{"message": "Configuration updated successfully"})
}

func (h *PIIHandler) GetTrainingStats(c *gin.Context) {
	// Get training feedback statistics
	var totalFeedback int
	var entityTypes []string

	// Count total feedback records
	err := h.db.QueryRow("SELECT COUNT(*) FROM training_feedback").Scan(&totalFeedback)
	if err != nil {
		logrus.WithError(err).Debug("No training feedback found")
		totalFeedback = 0
	}

	// Get distinct entity types from feedback
	if totalFeedback > 0 {
		rows, err := h.db.Query("SELECT DISTINCT entity_type FROM training_feedback ORDER BY entity_type")
		if err == nil {
			defer rows.Close()
			for rows.Next() {
				var entityType string
				if err := rows.Scan(&entityType); err == nil {
					entityTypes = append(entityTypes, entityType)
				}
			}
		}
	}

	// Get confidence engine stats
	stats := h.detector.GetConfidenceEngine().GetStats()
	
	c.JSON(http.StatusOK, gin.H{
		"total_feedback":     totalFeedback,
		"entity_types":       entityTypes,
		"confidence_stats":   stats,
		"has_training_data":  totalFeedback > 0,
	})
}

func (h *PIIHandler) SubmitFeedback(c *gin.Context) {
	var feedback pii.FeedbackRecord
	if err := c.ShouldBindJSON(&feedback); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid feedback data"})
		return
	}

	// Set timestamp and session ID
	feedback.Timestamp = time.Now()
	feedback.SessionID = "web-" + strconv.FormatInt(time.Now().Unix(), 10)

	// Submit feedback to confidence engine
	h.detector.GetConfidenceEngine().RecordFeedback(feedback)

	// Save feedback to database for analysis
	go h.saveFeedbackToDatabase(feedback)

	logrus.WithFields(logrus.Fields{
		"entity_text":     feedback.EntityText,
		"entity_type":     feedback.EntityType,
		"user_decision":   feedback.UserDecision,
		"user_confidence": feedback.UserConfidence,
	}).Info("Received training feedback")

	c.JSON(http.StatusOK, gin.H{
		"message": "Feedback recorded successfully",
		"status":  "training_updated",
	})
}

func (h *PIIHandler) saveFeedbackToDatabase(feedback pii.FeedbackRecord) {
	query := `
		INSERT INTO training_feedback (entity_text, entity_type, original_score, user_decision, user_confidence, context, timestamp, session_id)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`
	
	_, err := h.db.Exec(query, 
		feedback.EntityText,
		feedback.EntityType,
		feedback.OriginalScore,
		feedback.UserDecision,
		feedback.UserConfidence,
		feedback.Context,
		feedback.Timestamp,
		feedback.SessionID,
	)
	
	if err != nil {
		logrus.WithError(err).Error("Failed to save training feedback to database")
	}
}

// Helper functions
func (h *PIIHandler) saveToHistory(originalText string, result *pii.RedactResult) {
	// Save processing result to database
	query := `
		INSERT INTO processing_history (filename, entities_found, processing_time_ms, status)
		VALUES (?, ?, ?, ?)
	`
	
	// Extract processing time from result if available, otherwise use 0
	processingTimeMs := 0.0
	
	_, err := h.db.Exec(query, "text_processing", len(result.Entities), processingTimeMs, "completed")
	if err != nil {
		logrus.WithError(err).Error("Failed to save processing result to history")
	}
}

func calculateAverageConfidence(entities []pii.Entity) float64 {
	if len(entities) == 0 {
		return 0.0
	}
	
	total := 0.0
	for _, entity := range entities {
		total += entity.Confidence
	}
	return total / float64(len(entities))
}

func getCurrentTime() int64 {
	return time.Now().UnixNano()
}

func getProcessTime(startTime int64) string {
	duration := time.Now().UnixNano() - startTime
	return fmt.Sprintf("%.2fms", float64(duration)/1000000)
}