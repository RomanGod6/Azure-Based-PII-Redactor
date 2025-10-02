package handlers

import (
	"database/sql"
	"encoding/base64"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"redactify/internal/pii"
	"redactify/pkg/config"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gin-gonic/gin"
	"github.com/lib/pq"
	"github.com/sirupsen/logrus"
	"github.com/tealeg/xlsx/v3"
)

type FileHandler struct {
	db        *sql.DB
	config    *config.Config
	detector  *pii.Detector
	wsHandler *WebSocketHandler
}

type ProcessFileResponse struct {
	OriginalText  string       `json:"original_text"`
	RedactedText  string       `json:"redacted_text"`
	Entities      []pii.Entity `json:"entities,omitempty"`
	RedactedCount int          `json:"redacted_count"`
	ProcessTime   string       `json:"process_time"`
	RowsProcessed int          `json:"rows_processed"`
	FileName      string       `json:"file_name"`
	ResultID      string       `json:"result_id,omitempty"`
}

type ProgressUpdate struct {
	CurrentRow int                  `json:"current_row"`
	TotalRows  int                  `json:"total_rows"`
	Status     string               `json:"status"`
	Message    string               `json:"message"`
	IsComplete bool                 `json:"is_complete"`
	Entities   []pii.Entity         `json:"entities,omitempty"`
	Results    *ProcessFileResponse `json:"results,omitempty"`
}

type sessionEntity struct {
	ID         int64
	RowNumber  int
	Type       string
	Text       string
	Start      int
	End        int
	Confidence float64
	Category   string
	Approved   bool
}

type sessionRow struct {
	RowNumber      int
	OriginalText   string
	StoredRedacted string
	EntitiesCount  int
	ProcessingTime float64
	Status         string
	ErrorMessage   string
	CreatedAt      time.Time
}

type sessionMetadata struct {
	Filename         string
	Timestamp        time.Time
	ProcessingTimeMs float64
	EntitiesFound    int
	RedactionMode    string
	CustomLabels     map[string]string
}

func NewFileHandler(db *sql.DB, cfg *config.Config, wsHandler *WebSocketHandler) *FileHandler {
	return &FileHandler{
		db:        db,
		config:    cfg,
		detector:  pii.NewDetectorWithDB(cfg, db),
		wsHandler: wsHandler,
	}
}

func (h *FileHandler) UploadFile(c *gin.Context) {
	file, header, err := c.Request.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}
	defer file.Close()

	// Validate file type
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if ext != ".csv" && ext != ".xlsx" && ext != ".xls" && ext != ".txt" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Unsupported file type. Please upload CSV, Excel, or TXT files."})
		return
	}

	// Validate file size (50MB limit)
	if header.Size > 50*1024*1024 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "File too large. Maximum size is 50MB."})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":   "File uploaded successfully",
		"filename":  header.Filename,
		"size":      header.Size,
		"file_type": ext,
	})
}

// ProcessFileWebSocket processes file with WebSocket for real-time updates
func (h *FileHandler) ProcessFileWebSocket(c *gin.Context) {
	// Upgrade to WebSocket
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logrus.WithError(err).Error("Failed to upgrade to WebSocket for file processing")
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to establish WebSocket connection"})
		return
	}
	defer conn.Close()

	// Wait for file processing request
	var processRequest struct {
		FileName      string             `json:"filename"`
		FileContent   string             `json:"file_content"`
		FileType      string             `json:"file_type"`
		RedactOptions *pii.RedactOptions `json:"redact_options"`
	}

	logrus.Info("🔍 Waiting for file processing request via WebSocket...")

	err = conn.ReadJSON(&processRequest)
	if err != nil {
		logrus.WithError(err).Error("❌ Failed to read file processing request")
		return
	}

	logrus.WithFields(logrus.Fields{
		"filename":          processRequest.FileName,
		"file_type":         processRequest.FileType,
		"content_length":    len(processRequest.FileContent),
		"has_redact_options": processRequest.RedactOptions != nil,
	}).Info("✅ Received file processing request via WebSocket")

	// Create processing session
	sessionID := h.wsHandler.CreateSession(processRequest.FileName, conn)
	logrus.WithField("session_id", sessionID).Info("✅ CreateSession completed, sessionID created")

	// Start processing in goroutine
	logrus.WithField("session_id", sessionID).Info("🚀 Starting processing goroutine...")
	go func() {
		logrus.WithField("session_id", sessionID).Info("🎯 Goroutine started executing")
		defer func() {
			if r := recover(); r != nil {
				logrus.WithFields(logrus.Fields{
					"session_id": sessionID,
					"panic":      r,
					"stack":      string(debug.Stack()),
				}).Error("💥 PANIC in processFileWithWebSocket goroutine")
			}
		}()
		h.processFileWithWebSocket(sessionID, &processRequest)
	}()

	// Wait for processing to complete by monitoring the session status
	logrus.WithField("session_id", sessionID).Info("✅ Processing goroutine launched, waiting for completion...")

	// Keep the WebSocket connection alive by waiting for processing completion
	// Check session status periodically until it's completed
	for {
		time.Sleep(100 * time.Millisecond) // Check every 100ms

		h.wsHandler.mutex.RLock()
		session, exists := h.wsHandler.sessions[sessionID]
		if !exists {
			h.wsHandler.mutex.RUnlock()
			logrus.WithField("session_id", sessionID).Info("Session no longer exists, connection ending")
			break
		}

		status := session.Status
		h.wsHandler.mutex.RUnlock()

		if status == "completed" || status == "error" {
			logrus.WithField("session_id", sessionID).Info("Processing completed, connection can close")
			break
		}
	}
}

// processFileWithWebSocket handles file processing with WebSocket progress updates
func (h *FileHandler) processFileWithWebSocket(sessionID string, request *struct {
	FileName      string             `json:"filename"`
	FileContent   string             `json:"file_content"`
	FileType      string             `json:"file_type"`
	RedactOptions *pii.RedactOptions `json:"redact_options"`
}) {
	startTime := time.Now()
	logrus.Infof("🚀 Starting WebSocket file processing for session: %s", sessionID)

	// Debug: Log that this function is actually being called
	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"filename":   request.FileName,
		"file_type":  request.FileType,
		"content_length": len(request.FileContent),
	}).Info("🔍 processFileWithWebSocket called")

	// Decode base64 content
	decoded, err := base64.StdEncoding.DecodeString(request.FileContent)
	if err != nil {
		logrus.WithError(err).Error("❌ Failed to decode base64 content")
		h.wsHandler.NotifyError(sessionID, fmt.Errorf("failed to decode file content: %w", err))
		return
	}

	// Add debugging for content
	logrus.WithFields(logrus.Fields{
		"session_id":      sessionID,
		"decoded_size":    len(decoded),
		"first_100_chars": string(decoded[:minInt(100, len(decoded))]),
	}).Info("📋 CSV content decoded")

	// Parse CSV content from decoded string
	csvContent := strings.NewReader(string(decoded))

	// Process CSV with WebSocket updates
	h.processCSVWithWebSocket(sessionID, csvContent, request.FileName, request.RedactOptions, startTime)
}

// processCSVWithWebSocket processes CSV with WebSocket progress updates
func (h *FileHandler) processCSVWithWebSocket(sessionID string, file io.Reader, filename string, options *pii.RedactOptions, startTime time.Time) {
	// Read all content first
	content, err := io.ReadAll(file)
	if err != nil {
		h.wsHandler.NotifyError(sessionID, fmt.Errorf("failed to read file content: %w", err))
		return
	}

	contentStr := string(content)
	logrus.WithFields(logrus.Fields{
		"session_id":      sessionID,
		"content_length":  len(contentStr),
		"first_200_chars": contentStr[:minInt(200, len(contentStr))],
	}).Info("📋 Reading CSV content")

	// Try different delimiters if comma fails
	delimiters := []rune{',', ';', '\t', '|'}
	var records [][]string
	var headers []string
	var delimiter rune = ','  // default to comma

	logrus.WithField("session_id", sessionID).Info("🔍 Starting CSV delimiter detection")

	for _, testDelimiter := range delimiters {
		reader := csv.NewReader(strings.NewReader(contentStr))
		reader.Comma = testDelimiter
		reader.FieldsPerRecord = -1 // Allow variable number of fields

		tempRecords, err := reader.ReadAll()
		if err != nil {
			logrus.WithField("delimiter", string(testDelimiter)).Warn("⚠️ Failed to parse with delimiter")
			continue
		}

		if len(tempRecords) > 1 { // At least header + 1 data row
			records = tempRecords
			headers = records[0]
			delimiter = testDelimiter  // capture the successful delimiter
			logrus.WithFields(logrus.Fields{
				"session_id": sessionID,
				"delimiter":  string(delimiter),
				"rows":       len(records),
				"headers":    headers,
			}).Info("✅ Successfully parsed CSV")
			break
		}
	}

	// Store CSV metadata
	headersJSON, err := json.Marshal(headers)
	if err != nil {
		logrus.WithError(err).Error("Failed to marshal CSV headers")
		h.wsHandler.NotifyError(sessionID, fmt.Errorf("failed to process CSV headers"))
		return
	}

	// Initialize column PII settings (all enabled by default)
	columnPIISettings := make(map[string]bool)
	for _, header := range headers {
		columnPIISettings[header] = true
	}
	columnPIISettingsJSON, err := json.Marshal(columnPIISettings)
	if err != nil {
		logrus.WithError(err).Error("Failed to marshal column PII settings")
		h.wsHandler.NotifyError(sessionID, fmt.Errorf("failed to initialize column settings"))
		return
	}

	// Store CSV metadata in database
	_, err = h.db.Exec(`
		INSERT INTO csv_metadata (session_id, headers, column_pii_settings, delimiter, has_headers, total_columns)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (session_id) DO UPDATE SET
			headers = EXCLUDED.headers,
			column_pii_settings = EXCLUDED.column_pii_settings,
			delimiter = EXCLUDED.delimiter,
			has_headers = EXCLUDED.has_headers,
			total_columns = EXCLUDED.total_columns
	`, sessionID, string(headersJSON), string(columnPIISettingsJSON), string(delimiter), true, len(headers))

	if err != nil {
		logrus.WithError(err).Error("Failed to store CSV metadata")
		h.wsHandler.NotifyError(sessionID, fmt.Errorf("failed to store CSV metadata"))
		return
	}

	totalRows := len(records) - 1 // Subtract header
	logrus.WithFields(logrus.Fields{
		"session_id":    sessionID,
		"total_records": len(records),
		"total_rows":    totalRows,
		"headers":       headers,
	}).Info("🔍 CSV parsing completed successfully")

	if totalRows <= 0 {
		logrus.WithFields(logrus.Fields{
			"session_id":      sessionID,
			"records_found":   len(records),
			"content_preview": contentStr[:minInt(300, len(contentStr))],
		}).Error("❌ No data rows found in CSV")

		h.wsHandler.NotifyError(sessionID, fmt.Errorf("no data rows found in CSV. Found %d total records. Content preview: %s", len(records), contentStr[:minInt(200, len(contentStr))]))
		return
	}

	processedCount := 0

	// Send initial progress
	h.wsHandler.UpdateProgress(sessionID, &ProgressUpdate{
		CurrentRow: 0,
		TotalRows:  totalRows,
		Status:     "processing",
		Message:    "Starting file processing...",
		IsComplete: false,
	})

	// Process each row individually
	logrus.WithFields(logrus.Fields{
		"session_id":   sessionID,
		"total_rows":   totalRows,
		"total_records": len(records),
	}).Info("🔍 Starting row-by-row processing")

	for i, record := range records {
		if i == 0 { // Skip header
			continue
		}

		processedCount++
		rowStartTime := time.Now()

		// Store structured row data
		columnDataJSON, err := json.Marshal(record)
		if err != nil {
			logrus.WithError(err).Warnf("Failed to marshal row %d data", processedCount)
			continue
		}

		// Store row data in csv_row_data table
		_, err = h.db.Exec(`
			INSERT INTO csv_row_data (session_id, row_number, column_data)
			VALUES ($1, $2, $3)
			ON CONFLICT (session_id, row_number) DO UPDATE SET
				column_data = EXCLUDED.column_data
		`, sessionID, processedCount, string(columnDataJSON))

		if err != nil {
			logrus.WithError(err).Warnf("Failed to store row %d data", processedCount)
		}

		// Build text for PII detection from enabled columns only
		var piiEnabledText []string
		for j, cellValue := range record {
			if j < len(headers) && columnPIISettings[headers[j]] {
				piiEnabledText = append(piiEnabledText, cellValue)
			}
		}

		rowText := strings.Join(piiEnabledText, " ")

		// Debug log for troubleshooting - using Info level to ensure visibility
		logrus.WithFields(logrus.Fields{
			"session_id":         sessionID,
			"row_number":         processedCount,
			"pii_enabled_count":  len(piiEnabledText),
			"rowText_length":     len(rowText),
			"headers_count":      len(headers),
			"record_count":       len(record),
			"rowText_preview":    rowText[:minInt(50, len(rowText))],
		}).Info("🔍 Processing row with PII settings")

		// Process this row through PII detection with rate limiting handling
		result, err := h.detectWithRateLimit(sessionID, rowText, options)
		rowProcessingTime := time.Since(rowStartTime)

		if err != nil {
			logrus.WithError(err).Warnf("Failed to process row %d, skipping", processedCount)

			// Save failed row to database immediately
			if saveErr := h.saveRowResult(sessionID, processedCount, rowText, rowText, 0, rowProcessingTime, err.Error()); saveErr != nil {
				logrus.WithError(saveErr).Errorf("Failed to save failed row %d to database", processedCount)
			}

			h.wsHandler.UpdateProgress(sessionID, &ProgressUpdate{
				CurrentRow: processedCount,
				TotalRows:  totalRows,
				Status:     "processing",
				Message:    fmt.Sprintf("Processed row %d of %d (skipped due to error)", processedCount, totalRows),
				IsComplete: false,
			})
			continue
		}

		// Save successful row to database immediately
		if saveErr := h.saveRowResult(sessionID, processedCount, rowText, result.RedactedText, len(result.Entities), rowProcessingTime, ""); saveErr != nil {
			logrus.WithError(saveErr).Errorf("Failed to save row %d to database", processedCount)
		}

		if len(result.Entities) > 0 {
			if err := h.saveDetectedEntities(sessionID, result.Entities, processedCount); err != nil {
				logrus.WithError(err).Warnf("Failed to persist detected entities for row %d", processedCount)
			}
		}

		// Send progress update
		h.wsHandler.UpdateProgress(sessionID, &ProgressUpdate{
			CurrentRow: processedCount,
			TotalRows:  totalRows,
			Status:     "processing",
			Message:    fmt.Sprintf("Processed row %d of %d (saved to DB)", processedCount, totalRows),
			IsComplete: false,
			Entities:   result.Entities,
		})

		// Small delay to prevent overwhelming Azure API
		time.Sleep(10 * time.Millisecond)
	}

	// Get total entities count without loading all content into memory
	totalEntities, err := h.buildResultsFromRowsStreaming(sessionID)
	if err != nil {
		logrus.WithError(err).Error("Failed to get entities count from stored rows")
		h.wsHandler.NotifyError(sessionID, fmt.Errorf("failed to get entities count from database rows: %w", err))
		return
	}

	processingTime := time.Since(startTime)

	resultID, err := h.saveProcessingResultsMetadata(sessionID, filename, totalEntities, processingTime, totalRows)
	if err != nil {
		logrus.WithError(err).Error("Failed to save processing results")
		h.wsHandler.NotifyError(sessionID, fmt.Errorf("failed to save processing results: %w", err))
		return
	}

	// Create lightweight response (no large content)
	response := &ProcessFileResponse{
		OriginalText:  "",            // Don't send large content via WebSocket
		RedactedText:  "",            // Don't send large content via WebSocket
		RedactedCount: totalEntities, // Use entities count from database
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalRows,
		FileName:      filename,
		ResultID:      resultID, // Include result ID for retrieval
	}

	// Save to history
	redactionMode := options.RedactionMode
	if redactionMode == "" {
		redactionMode = "replace"
	}
	var labelsCopy map[string]string
	if len(options.CustomLabels) > 0 {
		labelsCopy = make(map[string]string, len(options.CustomLabels))
		for k, v := range options.CustomLabels {
			labelsCopy[k] = v
		}
	}
	go h.saveFileProcessingHistory(filename, totalEntities, processingTime, totalRows, resultID, sessionID, redactionMode, labelsCopy)

	// Send completion signal with lightweight response
	logrus.Infof("✅ File processing completed for session: %s, result ID: %s", sessionID, resultID)
	h.wsHandler.CompleteSession(sessionID, response)
}

// detectWithRateLimit performs PII detection with rate limiting handling
func (h *FileHandler) detectWithRateLimit(sessionID string, text string, options *pii.RedactOptions) (*pii.RedactResult, error) {
	maxRetries := 3
	baseDelay := time.Second

	for attempt := 0; attempt < maxRetries; attempt++ {
		result, err := h.detector.Redact(text, options)
		if err != nil {
			// Check if this is a rate limiting error
			if strings.Contains(err.Error(), "429") || strings.Contains(err.Error(), "rate limit") {
				retryDelay := baseDelay * time.Duration(1<<attempt) // Exponential backoff

				logrus.Warnf("⚠️ Rate limit detected for session %s, attempt %d/%d, waiting %v",
					sessionID, attempt+1, maxRetries, retryDelay)

				// Notify client about rate limiting
				h.wsHandler.NotifyRateLimit(sessionID, retryDelay)

				// Wait before retry
				time.Sleep(retryDelay)
				continue
			}
			// Non-rate-limit error, return immediately
			return nil, err
		}

		// Success
		return result, nil
	}

	// All retries exhausted
	return nil, fmt.Errorf("rate limit exceeded after %d attempts", maxRetries)
}

func (h *FileHandler) ProcessFile(c *gin.Context) {
	startTime := time.Now()
	logrus.Info("🚀 ProcessFile ENTRY - Starting file processing with progress updates")

	file, header, err := c.Request.FormFile("file")
	if err != nil {
		logrus.WithError(err).Error("❌ ProcessFile - No file uploaded")
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}
	defer file.Close()

	// Parse options
	optionsStr := c.PostForm("options")
	var options pii.RedactOptions
	if optionsStr != "" {
		if err := json.Unmarshal([]byte(optionsStr), &options); err != nil {
			logrus.WithError(err).Warn("Failed to parse options, using defaults")
		}
	}
	if options.RedactionMode == "" {
		options.RedactionMode = "replace"
	}
	if options.CustomLabels == nil {
		options.CustomLabels = make(map[string]string)
	}

	// Set response headers for SSE
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	// Send initial progress
	h.sendProgress(c, 0, 0, "starting", "Initializing file processing...", nil)

	// Process file based on type
	ext := strings.ToLower(filepath.Ext(header.Filename))

	switch ext {
	case ".csv":
		h.processCSVWithProgress(c, file, header.Filename, &options, startTime)
	case ".xlsx", ".xls":
		h.processExcelWithProgress(c, file, header.Filename, &options, startTime)
	case ".txt":
		h.processTextWithProgress(c, file, header.Filename, &options, startTime)
	default:
		h.sendProgress(c, 0, 0, "error", "Unsupported file type", nil)
		return
	}
}

func (h *FileHandler) DownloadFile(c *gin.Context) {
	fileID := c.Param("id")
	// TODO: Implement file download
	c.JSON(http.StatusOK, gin.H{
		"message": "File download endpoint - to be implemented",
		"file_id": fileID,
	})
}

func (h *FileHandler) GetFileStatus(c *gin.Context) {
	fileID := c.Param("id")
	// TODO: Implement file status check
	c.JSON(http.StatusOK, gin.H{
		"message": "File status endpoint - to be implemented",
		"file_id": fileID,
	})
}

func (h *FileHandler) saveFileProcessingHistory(filename string, entitiesFound int, processingTime time.Duration, rowsProcessed int, resultID, sessionID, redactionMode string, customLabels map[string]string) {
	query := `
		INSERT INTO processing_history (filename, entities_found, processing_time_ms, status, file_size, result_id, session_id, redaction_mode, custom_labels)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`

	processingTimeMs := float64(processingTime.Nanoseconds()) / 1000000

	labelsJSON := ""
	if len(customLabels) > 0 {
		if data, err := json.Marshal(customLabels); err == nil {
			labelsJSON = string(data)
		} else {
			logrus.WithError(err).Warn("Failed to marshal custom labels for history")
		}
	}

	_, err := h.db.Exec(query, filename, entitiesFound, processingTimeMs, "completed", rowsProcessed, resultID, sessionID, redactionMode, labelsJSON)
	if err != nil {
		logrus.WithError(err).Error("Failed to save file processing history")
	}
}

// sanitizeUTF8 removes invalid UTF-8 sequences from text
func sanitizeUTF8(text string) string {
	if utf8.ValidString(text) {
		return text
	}

	// Manual sanitization - remove invalid UTF-8 runes
	var builder strings.Builder
	for _, r := range text {
		if r == utf8.RuneError {
			builder.WriteString("�") // Replace with replacement character
		} else if utf8.ValidRune(r) {
			builder.WriteRune(r)
		} else {
			builder.WriteString("�") // Replace invalid runes
		}
	}
	return builder.String()
}

// saveProcessingResultsMetadata saves processing metadata without storing full text content
func (h *FileHandler) saveProcessingResultsMetadata(sessionID, filename string, entitiesFound int, processingTime time.Duration, rowsProcessed int) (string, error) {
	resultID := sessionID // Use session ID as result ID for simplicity
	sanitizedFilename := sanitizeUTF8(filename)

	// Only store metadata, not full text content to avoid memory issues
	query := `
		INSERT INTO processing_results (id, filename, original_text, redacted_text, entities_found, processing_time_ms, rows_processed, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
		ON CONFLICT (id) DO UPDATE SET
			filename = EXCLUDED.filename,
			entities_found = EXCLUDED.entities_found,
			processing_time_ms = EXCLUDED.processing_time_ms,
			rows_processed = EXCLUDED.rows_processed,
			created_at = CURRENT_TIMESTAMP
	`

	processingTimeMs := float64(processingTime.Nanoseconds()) / 1000000

	// Store empty strings for text content - actual content is stored in processing_rows
	_, err := h.db.Exec(query, resultID, sanitizedFilename, "", "", entitiesFound, processingTimeMs, rowsProcessed)
	if err != nil {
		return "", fmt.Errorf("failed to save processing results metadata: %w", err)
	}

	logrus.WithFields(logrus.Fields{
		"result_id":      resultID,
		"filename":       filename,
		"entities_found": entitiesFound,
		"rows_processed": rowsProcessed,
	}).Info("💾 Processing results metadata saved to database")

	return resultID, nil
}

// saveProcessingResults saves full processing results to database and returns result ID (LEGACY - use with caution)
func (h *FileHandler) saveProcessingResults(sessionID, filename, originalText, redactedText string, entitiesFound int, processingTime time.Duration, rowsProcessed int) (string, error) {
	resultID := sessionID // Use session ID as result ID for simplicity

	// Sanitize text content to ensure valid UTF-8
	sanitizedOriginal := sanitizeUTF8(originalText)
	sanitizedRedacted := sanitizeUTF8(redactedText)
	sanitizedFilename := sanitizeUTF8(filename)

	logrus.WithFields(logrus.Fields{
		"original_valid_utf8": utf8.ValidString(originalText),
		"redacted_valid_utf8": utf8.ValidString(redactedText),
		"original_size":       len(originalText),
		"sanitized_size":      len(sanitizedOriginal),
	}).Info("🧹 Sanitizing text for database storage")

	query := `
		INSERT INTO processing_results (id, filename, original_text, redacted_text, entities_found, processing_time_ms, rows_processed, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
		ON CONFLICT (id) DO UPDATE SET
			filename = EXCLUDED.filename,
			original_text = EXCLUDED.original_text,
			redacted_text = EXCLUDED.redacted_text,
			entities_found = EXCLUDED.entities_found,
			processing_time_ms = EXCLUDED.processing_time_ms,
			rows_processed = EXCLUDED.rows_processed,
			created_at = CURRENT_TIMESTAMP
	`

	processingTimeMs := float64(processingTime.Nanoseconds()) / 1000000

	_, err := h.db.Exec(query, resultID, sanitizedFilename, sanitizedOriginal, sanitizedRedacted, entitiesFound, processingTimeMs, rowsProcessed)
	if err != nil {
		return "", fmt.Errorf("failed to save processing results: %w", err)
	}

	logrus.WithFields(logrus.Fields{
		"result_id":      resultID,
		"filename":       filename,
		"content_size":   len(originalText),
		"entities_found": entitiesFound,
		"rows_processed": rowsProcessed,
	}).Info("💾 Processing results saved to database")

	return resultID, nil
}

// saveRowResult saves an individual row processing result
func (h *FileHandler) saveRowResult(sessionID string, rowNumber int, originalText, redactedText string, entitiesCount int, processingTime time.Duration, errorMsg string) error {
	status := "completed"
	if errorMsg != "" {
		status = "error"
	}

	// Sanitize text content
	sanitizedOriginal := sanitizeUTF8(originalText)
	sanitizedRedacted := sanitizeUTF8(redactedText)

	query := `
		INSERT INTO processing_rows (session_id, row_number, original_text, redacted_text, entities_count, processing_time_ms, status, error_message)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (session_id, row_number) DO UPDATE SET
			original_text = EXCLUDED.original_text,
			redacted_text = EXCLUDED.redacted_text,
			entities_count = EXCLUDED.entities_count,
			processing_time_ms = EXCLUDED.processing_time_ms,
			status = EXCLUDED.status,
			error_message = EXCLUDED.error_message,
			created_at = CURRENT_TIMESTAMP
	`

	processingTimeMs := float64(processingTime.Nanoseconds()) / 1000000

	_, err := h.db.Exec(query, sessionID, rowNumber, sanitizedOriginal, sanitizedRedacted, entitiesCount, processingTimeMs, status, errorMsg)
	if err != nil {
		return fmt.Errorf("failed to save row result: %w", err)
	}

	return nil
}

// getTotalEntitiesFromRows gets just the total entity count without loading all text into memory
func (h *FileHandler) getTotalEntitiesFromRows(sessionID string) (int, error) {
	query := `
		SELECT COALESCE(SUM(entities_count), 0) as total_entities
		FROM processing_rows
		WHERE session_id = $1
	`

	var totalEntities int
	err := h.db.QueryRow(query, sessionID).Scan(&totalEntities)
	if err != nil {
		return 0, fmt.Errorf("failed to query total entities: %w", err)
	}

	return totalEntities, nil
}

// buildResultsFromRowsStreaming retrieves summary statistics without loading all content into memory
func (h *FileHandler) buildResultsFromRowsStreaming(sessionID string) (int, error) {
	return h.getTotalEntitiesFromRows(sessionID)
}

// streamResultsToWriter streams processing results to a writer without loading all into memory
func (h *FileHandler) streamResultsToWriter(sessionID string, writer io.Writer) error {
	query := `
		SELECT original_text, redacted_text, entities_count
		FROM processing_rows
		WHERE session_id = $1
		ORDER BY row_number ASC
	`

	rows, err := h.db.Query(query, sessionID)
	if err != nil {
		return fmt.Errorf("failed to query row results: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var original, redacted string
		var entities int

		if err := rows.Scan(&original, &redacted, &entities); err != nil {
			return fmt.Errorf("failed to scan row result: %w", err)
		}

		// Write line with proper newline handling
		if _, err := writer.Write([]byte(redacted + "\n")); err != nil {
			return fmt.Errorf("failed to write row result: %w", err)
		}
	}

	return rows.Err()
}

// buildResultsFromRowsPaginated retrieves rows in chunks to avoid memory issues
func (h *FileHandler) buildResultsFromRowsPaginated(sessionID string, offset, limit int) ([]string, []string, int, bool, error) {
	query := `
		SELECT original_text, redacted_text, entities_count
		FROM processing_rows
		WHERE session_id = $1
		ORDER BY row_number ASC
		LIMIT $2 OFFSET $3
	`

	rows, err := h.db.Query(query, sessionID, limit, offset)
	if err != nil {
		return nil, nil, 0, false, fmt.Errorf("failed to query row results: %w", err)
	}
	defer rows.Close()

	var originalRows []string
	var redactedRows []string
	totalEntities := 0
	rowCount := 0

	for rows.Next() {
		var original, redacted string
		var entities int

		if err := rows.Scan(&original, &redacted, &entities); err != nil {
			return nil, nil, 0, false, fmt.Errorf("failed to scan row result: %w", err)
		}

		originalRows = append(originalRows, original)
		redactedRows = append(redactedRows, redacted)
		totalEntities += entities
		rowCount++
	}

	if err := rows.Err(); err != nil {
		return nil, nil, 0, false, fmt.Errorf("error iterating rows: %w", err)
	}

	hasMore := rowCount == limit
	return originalRows, redactedRows, totalEntities, hasMore, nil
}

// saveDetectedEntities saves individual detected entities to the database
func (h *FileHandler) saveDetectedEntities(sessionID string, entities []pii.Entity, rowNumber int) error {
	if len(entities) == 0 {
		return nil
	}

	query := `
		INSERT INTO detected_entities (session_id, entity_type, entity_text, start_position, end_position, confidence, category, row_number)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`

	for _, entity := range entities {
		_, err := h.db.Exec(query, sessionID, entity.Type, entity.Text, entity.Start, entity.End, entity.Confidence, entity.Category, rowNumber)
		if err != nil {
			return fmt.Errorf("failed to save entity %s: %w", entity.Text, err)
		}
	}

	return nil
}

func parseCustomLabels(raw string) map[string]string {
	if strings.TrimSpace(raw) == "" {
		return map[string]string{}
	}

	labels := make(map[string]string)
	if err := json.Unmarshal([]byte(raw), &labels); err != nil {
		logrus.WithError(err).Warn("Failed to parse custom labels from history")
		return map[string]string{}
	}
	return labels
}

func mergeCustomLabels(base, overrides map[string]string) map[string]string {
	if len(base) == 0 && len(overrides) == 0 {
		return map[string]string{}
	}

	merged := make(map[string]string, len(base)+len(overrides))
	for k, v := range base {
		merged[k] = v
	}
	for k, v := range overrides {
		if strings.TrimSpace(v) == "" {
			continue
		}
		merged[k] = v
	}

	return merged
}

func (h *FileHandler) getSessionMetadata(sessionID string) (*sessionMetadata, error) {
	query := `
		SELECT filename, timestamp, processing_time_ms, entities_found, redaction_mode, custom_labels
		FROM processing_history
		WHERE session_id = $1
		ORDER BY timestamp DESC
		LIMIT 1
	`

	var (
		filename        string
		timestamp       time.Time
		processingMs    sql.NullFloat64
		entitiesFound   sql.NullInt64
		redactionMode   sql.NullString
		customLabelsRaw sql.NullString
	)

	err := h.db.QueryRow(query, sessionID).Scan(&filename, &timestamp, &processingMs, &entitiesFound, &redactionMode, &customLabelsRaw)
	if err != nil {
		return nil, err
	}

	metadata := &sessionMetadata{
		Filename:         filename,
		Timestamp:        timestamp,
		ProcessingTimeMs: processingMs.Float64,
		EntitiesFound:    int(entitiesFound.Int64),
		RedactionMode:    redactionMode.String,
		CustomLabels:     parseCustomLabels(customLabelsRaw.String),
	}

	if metadata.RedactionMode == "" {
		metadata.RedactionMode = "replace"
	}

	return metadata, nil
}

func (h *FileHandler) fetchSessionRows(sessionID string) ([]sessionRow, error) {
	query := `
		SELECT row_number, original_text, redacted_text, entities_count, processing_time_ms, status, COALESCE(error_message, ''), created_at
		FROM processing_rows
		WHERE session_id = $1
		ORDER BY row_number ASC
	`

	rows, err := h.db.Query(query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to query processing rows: %w", err)
	}
	defer rows.Close()

	var results []sessionRow
	for rows.Next() {
		var (
			rowNumber      int
			originalText   sql.NullString
			storedRedacted sql.NullString
			entitiesCount  sql.NullInt64
			processingMs   sql.NullFloat64
			status         sql.NullString
			errorMessage   string
			createdAt      time.Time
		)

		if err := rows.Scan(&rowNumber, &originalText, &storedRedacted, &entitiesCount, &processingMs, &status, &errorMessage, &createdAt); err != nil {
			return nil, fmt.Errorf("failed to scan processing row: %w", err)
		}

		results = append(results, sessionRow{
			RowNumber:      rowNumber,
			OriginalText:   originalText.String,
			StoredRedacted: storedRedacted.String,
			EntitiesCount:  int(entitiesCount.Int64),
			ProcessingTime: processingMs.Float64,
			Status:         status.String,
			ErrorMessage:   errorMessage,
			CreatedAt:      createdAt,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating processing rows: %w", err)
	}

	return results, nil
}

func (h *FileHandler) fetchSessionEntities(sessionID string) ([]sessionEntity, error) {
	query := `
		SELECT id, row_number, entity_type, entity_text, start_position, end_position, confidence, category, approved
		FROM detected_entities
		WHERE session_id = $1
		ORDER BY row_number ASC, start_position ASC
	`

	rows, err := h.db.Query(query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to query detected entities: %w", err)
	}
	defer rows.Close()

	var entities []sessionEntity
	for rows.Next() {
		var (
			id         int64
			rowNumber  int
			entityType string
			entityText string
			startPos   sql.NullInt64
			endPos     sql.NullInt64
			confidence sql.NullFloat64
			category   sql.NullString
			approved   sql.NullBool
		)

		if err := rows.Scan(&id, &rowNumber, &entityType, &entityText, &startPos, &endPos, &confidence, &category, &approved); err != nil {
			return nil, fmt.Errorf("failed to scan detected entity: %w", err)
		}

		entity := sessionEntity{
			ID:         id,
			RowNumber:  rowNumber,
			Type:       entityType,
			Text:       entityText,
			Start:      int(startPos.Int64),
			End:        int(endPos.Int64),
			Confidence: confidence.Float64,
			Category:   category.String,
			Approved:   true,
		}

		if approved.Valid {
			entity.Approved = approved.Bool
		}

		entities = append(entities, entity)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating detected entities: %w", err)
	}

	return entities, nil
}

func buildRowRedaction(original string, entities []sessionEntity, redactionMode string, customLabels map[string]string, skip map[int64]bool) (string, int) {
	if len(entities) == 0 {
		return original, 0
	}

	replacements := make([]sessionEntity, 0, len(entities))
	for _, entity := range entities {
		approved := entity.Approved
		if skip != nil {
			_, shouldSkip := skip[entity.ID]
			approved = !shouldSkip
		}

		if !approved {
			continue
		}

		replacements = append(replacements, entity)
	}

	if len(replacements) == 0 {
		return original, 0
	}

	// Sort replacements in reverse order of start to maintain byte offsets
	sort.Slice(replacements, func(i, j int) bool {
		return replacements[i].Start > replacements[j].Start
	})

	redacted := original
	for _, entity := range replacements {
		// Validate bounds against the original text
		if entity.Start < 0 || entity.End > len(original) || entity.Start >= entity.End {
			logrus.WithFields(logrus.Fields{
				"start":        entity.Start,
				"end":          entity.End,
				"row_number":   entity.RowNumber,
				"entity_text":  entity.Text,
				"original_len": len(original),
			}).Warn("Skipping entity with invalid bounds during redaction rebuild")
			continue
		}

		// Since we're processing in reverse order, we can validate against current redacted length
		if entity.Start >= len(redacted) || entity.End > len(redacted) {
			logrus.WithFields(logrus.Fields{
				"start":        entity.Start,
				"end":          entity.End,
				"row_number":   entity.RowNumber,
				"entity_text":  entity.Text,
				"redacted_len": len(redacted),
			}).Warn("Skipping entity due to redacted text length mismatch")
			continue
		}

		replacement := resolveReplacement(entity, redactionMode, customLabels)
		before := redacted[:entity.Start]
		after := redacted[entity.End:]
		redacted = before + replacement + after
	}

	return redacted, len(replacements)
}

// cleanUTF8 ensures the string contains only valid UTF-8 sequences
func cleanUTF8(s string) string {
	if utf8.ValidString(s) {
		return s
	}

	// Replace invalid sequences with replacement character
	return strings.ToValidUTF8(s, "�")
}

func resolveReplacement(entity sessionEntity, redactionMode string, customLabels map[string]string) string {
	if label, exists := customLabels[entity.Type]; exists && label != "" {
		return label
	}

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

	switch redactionMode {
	case "mask":
		return strings.Repeat("*", len(entity.Text))
	case "remove":
		return ""
	default:
		if entity.Type != "" {
			upper := strings.ToUpper(entity.Type)
			return fmt.Sprintf("[REDACTED_%s]", upper)
		}
		return "[REDACTED]"
	}
}

// StreamProcessingResults streams processing results row by row without loading all into memory
func (h *FileHandler) StreamProcessingResults(c *gin.Context) {
	sessionID := c.Param("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	format := c.DefaultQuery("format", "csv") // csv or json

	query := `
		SELECT row_number, original_text, redacted_text, entities_count, COALESCE(error_message, '') as error_message
		FROM processing_rows
		WHERE session_id = $1
		ORDER BY row_number ASC
	`

	rows, err := h.db.Query(query, sessionID)
	if err != nil {
		logrus.WithError(err).Error("Failed to query processing rows")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve processing results"})
		return
	}
	defer rows.Close()

	if format == "csv" {
		c.Header("Content-Type", "text/csv")
		c.Header("Content-Disposition", "attachment; filename=processing_results.csv")

		// Use CSV writer for proper escaping and performance
		csvWriter := csv.NewWriter(c.Writer)
		csvWriter.Write([]string{"row_number", "original_text", "redacted_text", "entities_count", "error"})

		rowCount := 0
		for rows.Next() {
			var rowNum int
			var original, redacted, errorMsg string
			var entities int

			if err := rows.Scan(&rowNum, &original, &redacted, &entities, &errorMsg); err != nil {
				logrus.WithError(err).Error("Failed to scan processing row")
				continue
			}

			record := []string{
				strconv.Itoa(rowNum),
				original,
				redacted,
				strconv.Itoa(entities),
				errorMsg,
			}

			if err := csvWriter.Write(record); err != nil {
				logrus.WithError(err).Error("Failed to write CSV record")
				break
			}

			// Flush every 100 rows for better streaming performance
			rowCount++
			if rowCount%100 == 0 {
				csvWriter.Flush()
			}
		}

		csvWriter.Flush()
	} else {
		c.Header("Content-Type", "application/json")
		c.Writer.WriteString(`{"results":[`)

		first := true
		for rows.Next() {
			var rowNum int
			var original, redacted, errorMsg string
			var entities int

			if err := rows.Scan(&rowNum, &original, &redacted, &entities, &errorMsg); err != nil {
				logrus.WithError(err).Error("Failed to scan processing row")
				continue
			}

			if !first {
				c.Writer.WriteString(",")
			}

			// Use json.Marshal for proper escaping
			data := map[string]interface{}{
				"row_number":     rowNum,
				"original_text":  original,
				"redacted_text":  redacted,
				"entities_count": entities,
				"error":         errorMsg,
			}

			if jsonBytes, err := json.Marshal(data); err == nil {
				c.Writer.Write(jsonBytes)
			}
			first = false
		}

		c.Writer.WriteString("]}")
	}

	if err := rows.Err(); err != nil {
		logrus.WithError(err).Error("Error iterating processing rows")
	}
}

// GetProcessingRowsForViewer returns processing rows data formatted for the results viewer
func (h *FileHandler) GetProcessingRowsForViewer(c *gin.Context) {
	sessionID := c.Param("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	// Get processing metadata
	metadataQuery := `
		SELECT ph.filename, pr.created_at, COUNT(*) as total_rows, 
		       COALESCE(SUM(pr.entities_count), 0) as total_entities,
		       COALESCE(AVG(pr.processing_time_ms), 0) as avg_processing_time
		FROM processing_rows pr
		JOIN processing_history ph ON pr.session_id = ph.session_id
		WHERE pr.session_id = $1 
		GROUP BY ph.filename, pr.created_at
		LIMIT 1
	`

	var metadata struct {
		Filename          string    `json:"filename"`
		CreatedAt         time.Time `json:"created_at"`
		TotalRows         int       `json:"total_rows"`
		TotalEntities     int       `json:"total_entities"`
		AvgProcessingTime float64   `json:"avg_processing_time"`
	}

	err := h.db.QueryRow(metadataQuery, sessionID).Scan(
		&metadata.Filename,
		&metadata.CreatedAt,
		&metadata.TotalRows,
		&metadata.TotalEntities,
		&metadata.AvgProcessingTime,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			c.JSON(http.StatusNotFound, gin.H{"error": "Processing session not found"})
		} else {
			logrus.WithError(err).Error("Failed to get processing metadata")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve processing metadata"})
		}
		return
	}

	// Get all processing rows with pagination
	limit := 100 // Default limit
	if l := c.Query("limit"); l != "" {
		if parsed, parseErr := strconv.Atoi(l); parseErr == nil && parsed > 0 && parsed <= 1000 {
			limit = parsed
		}
	}

	offset := 0
	if o := c.Query("offset"); o != "" {
		if parsed, parseErr := strconv.Atoi(o); parseErr == nil && parsed >= 0 {
			offset = parsed
		}
	}

	rowsQuery := `
		SELECT row_number, original_text, redacted_text, entities_count, 
		       processing_time_ms, error_message, created_at
		FROM processing_rows
		WHERE session_id = $1
		ORDER BY row_number ASC
		LIMIT $2 OFFSET $3
	`

	rows, err := h.db.Query(rowsQuery, sessionID, limit, offset)
	if err != nil {
		logrus.WithError(err).Error("Failed to query processing rows")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve processing rows"})
		return
	}
	defer rows.Close()

	var processingRows []gin.H
	for rows.Next() {
		var rowNum, entitiesCount int
		var originalText, redactedText string
		var processingTime float64
		var errorMsg sql.NullString
		var createdAt time.Time

		if err := rows.Scan(&rowNum, &originalText, &redactedText, &entitiesCount, &processingTime, &errorMsg, &createdAt); err != nil {
			logrus.WithError(err).Error("Failed to scan processing row")
			continue
		}

		// Calculate what was redacted (simple diff for display)
		wasRedacted := originalText != redactedText

		row := gin.H{
			"row_number":         rowNum,
			"original_text":      originalText,
			"redacted_text":      redactedText,
			"entities_found":     entitiesCount,
			"processing_time_ms": processingTime,
			"was_redacted":       wasRedacted,
			"has_error":          errorMsg.Valid && errorMsg.String != "",
			"error_message":      errorMsg.String,
			"created_at":         createdAt.Format(time.RFC3339),
		}

		processingRows = append(processingRows, row)
	}

	if err := rows.Err(); err != nil {
		logrus.WithError(err).Error("Error iterating processing rows")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error reading processing rows"})
		return
	}

	// Return complete results data
	c.JSON(http.StatusOK, gin.H{
		"session_id": sessionID,
		"metadata":   metadata,
		"rows":       processingRows,
		"pagination": gin.H{
			"limit":      limit,
			"offset":     offset,
			"total_rows": metadata.TotalRows,
			"has_more":   offset+limit < metadata.TotalRows,
		},
	})
}

// GetSessionReviewData returns consolidated data needed for the interactive review workspace
func (h *FileHandler) GetSessionReviewData(c *gin.Context) {
	sessionID := c.Param("session_id")
	if strings.TrimSpace(sessionID) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	metadata, err := h.getSessionMetadata(sessionID)
	if err != nil {
		if err == sql.ErrNoRows {
			c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
			return
		}
		logrus.WithError(err).Error("Failed to fetch session metadata")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to load session metadata"})
		return
	}

	rows, err := h.fetchSessionRows(sessionID)
	if err != nil {
		logrus.WithError(err).Error("Failed to load session rows")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to load session rows"})
		return
	}

	entities, err := h.fetchSessionEntities(sessionID)
	if err != nil {
		logrus.WithError(err).Error("Failed to load session entities")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to load detected entities"})
		return
	}

	entitiesByRow := make(map[int][]sessionEntity)
	for _, entity := range entities {
		entitiesByRow[entity.RowNumber] = append(entitiesByRow[entity.RowNumber], entity)
	}

	customLabels := metadata.CustomLabels
	if customLabels == nil {
		customLabels = map[string]string{}
	}

	rowsPayload := make([]gin.H, 0, len(rows))
	finalRedactedLines := make([]string, 0, len(rows))
	originalLines := make([]string, 0, len(rows))
	approvedTotal := 0

	for _, row := range rows {
		rowEntities := entitiesByRow[row.RowNumber]
		finalRedacted, approvedCount := buildRowRedaction(row.OriginalText, rowEntities, metadata.RedactionMode, customLabels, nil)

		approvedTotal += approvedCount
		finalRedactedLines = append(finalRedactedLines, finalRedacted)
		originalLines = append(originalLines, row.OriginalText)

		rowsPayload = append(rowsPayload, gin.H{
			"row_number":           row.RowNumber,
			"original_text":        row.OriginalText,
			"stored_redacted_text": row.StoredRedacted,
			"review_redacted_text": finalRedacted,
			"detected_entities":    row.EntitiesCount,
			"approved_entities":    approvedCount,
			"processing_time_ms":   row.ProcessingTime,
			"status":               row.Status,
			"error_message":        row.ErrorMessage,
			"was_redacted":         finalRedacted != row.OriginalText,
			"created_at":           row.CreatedAt.Format(time.RFC3339),
		})
	}

	entityPayload := make([]gin.H, 0, len(entities))
	for _, entity := range entities {
		entityPayload = append(entityPayload, gin.H{
			"id":         entity.ID,
			"row_number": entity.RowNumber,
			"type":       entity.Type,
			"text":       entity.Text,
			"start":      entity.Start,
			"end":        entity.End,
			"confidence": entity.Confidence,
			"category":   entity.Category,
			"approved":   entity.Approved,
		})
	}

	response := gin.H{
		"session_id":         sessionID,
		"filename":           metadata.Filename,
		"created_at":         metadata.Timestamp.Format(time.RFC3339),
		"processing_time_ms": metadata.ProcessingTimeMs,
		"redaction_mode":     metadata.RedactionMode,
		"custom_labels":      customLabels,
		"rows":               rowsPayload,
		"entities":           entityPayload,
		"summary": gin.H{
			"total_rows":        len(rows),
			"total_entities":    len(entities),
			"approved_entities": approvedTotal,
		},
		"full_original_text": strings.Join(originalLines, "\n"),
		"full_redacted_text": strings.Join(finalRedactedLines, "\n"),
	}

	c.JSON(http.StatusOK, response)
}

// ExportSessionResults applies user approvals and streams a CSV for the specified session
func (h *FileHandler) ExportSessionResults(c *gin.Context) {
	sessionID := c.Param("session_id")
	if strings.TrimSpace(sessionID) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	var req struct {
		RedactionMode     string            `json:"redaction_mode"`
		CustomLabels      map[string]string `json:"custom_labels"`
		SkippedEntityIDs  []int64           `json:"skipped_entity_ids"`
		IncludeErrorField bool              `json:"include_error_field"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		logrus.WithError(err).Warn("Invalid export request payload")
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid export request"})
		return
	}

	metadata, err := h.getSessionMetadata(sessionID)
	if err != nil {
		if err == sql.ErrNoRows {
			c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
			return
		}
		logrus.WithError(err).Error("Failed to load session metadata for export")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to load session metadata"})
		return
	}

	rows, err := h.fetchSessionRows(sessionID)
	if err != nil {
		logrus.WithError(err).Error("Failed to load rows for export")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to load processing rows"})
		return
	}

	entities, err := h.fetchSessionEntities(sessionID)
	if err != nil {
		logrus.WithError(err).Error("Failed to load entities for export")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to load detected entities"})
		return
	}

	entitiesByRow := make(map[int][]sessionEntity)
	for _, entity := range entities {
		entitiesByRow[entity.RowNumber] = append(entitiesByRow[entity.RowNumber], entity)
	}

	redactionMode := metadata.RedactionMode
	if req.RedactionMode != "" {
		redactionMode = req.RedactionMode
	}

	customLabels := mergeCustomLabels(metadata.CustomLabels, req.CustomLabels)
	skipSet := make(map[int64]bool, len(req.SkippedEntityIDs))
	for _, id := range req.SkippedEntityIDs {
		skipSet[id] = true
	}

	totalApproved := 0
	approvedIDs := make([]int64, 0, len(entities))
	skippedIDs := make([]int64, 0, len(req.SkippedEntityIDs))

	// Pre-process entity approvals without storing all rows in memory
	for _, row := range rows {
		rowEntities := entitiesByRow[row.RowNumber]
		_, approvedCount := buildRowRedaction(row.OriginalText, rowEntities, redactionMode, customLabels, skipSet)

		for _, entity := range rowEntities {
			if skipSet[entity.ID] {
				skippedIDs = append(skippedIDs, entity.ID)
				continue
			}
			approvedIDs = append(approvedIDs, entity.ID)
		}

		totalApproved += approvedCount
	}

	tx, err := h.db.Begin()
	if err != nil {
		logrus.WithError(err).Error("Failed to start transaction for export updates")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to prepare export"})
		return
	}

	if len(approvedIDs) > 0 {
		if _, execErr := tx.Exec(`UPDATE detected_entities SET approved = TRUE WHERE id = ANY($1)`, pq.Array(approvedIDs)); execErr != nil {
			_ = tx.Rollback()
			logrus.WithError(execErr).Error("Failed to persist approved entities")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to persist approvals"})
			return
		}
	}

	if len(skippedIDs) > 0 {
		if _, execErr := tx.Exec(`UPDATE detected_entities SET approved = FALSE WHERE id = ANY($1)`, pq.Array(skippedIDs)); execErr != nil {
			_ = tx.Rollback()
			logrus.WithError(execErr).Error("Failed to persist skipped entities")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to persist approvals"})
			return
		}
	}

	// Update redacted text for each row (streaming approach - one row at a time)
	for _, row := range rows {
		rowEntities := entitiesByRow[row.RowNumber]
		finalRedacted, _ := buildRowRedaction(row.OriginalText, rowEntities, redactionMode, customLabels, skipSet)

		// Clean the redacted text to ensure valid UTF-8 before database insert
		cleanedRedacted := cleanUTF8(finalRedacted)

		if _, execErr := tx.Exec(`UPDATE processing_rows SET redacted_text = $1 WHERE session_id = $2 AND row_number = $3`, cleanedRedacted, sessionID, row.RowNumber); execErr != nil {
			_ = tx.Rollback()
			logrus.WithError(execErr).Error("Failed to update row redacted text")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to persist redacted rows"})
			return
		}
	}

	if commitErr := tx.Commit(); commitErr != nil {
		logrus.WithError(commitErr).Error("Failed to commit export updates")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to finalize export"})
		return
	}

	processingDuration := time.Duration(metadata.ProcessingTimeMs * float64(time.Millisecond))

	// Use metadata-only save to avoid memory issues with large files
	if _, err := h.saveProcessingResultsMetadata(sessionID, metadata.Filename, totalApproved, processingDuration, len(rows)); err != nil {
		logrus.WithError(err).Warn("Failed to persist aggregated processing results after export")
	}

	labelsJSON := ""
	if len(customLabels) > 0 {
		if data, marshalErr := json.Marshal(customLabels); marshalErr == nil {
			labelsJSON = string(data)
		} else {
			logrus.WithError(marshalErr).Warn("Failed to marshal custom labels for history update")
		}
	}

	if _, err := h.db.Exec(`UPDATE processing_history SET entities_found = $1, redaction_mode = $2, custom_labels = $3 WHERE session_id = $4`, totalApproved, redactionMode, labelsJSON, sessionID); err != nil {
		logrus.WithError(err).Warn("Failed to update processing history with export summary")
	}

	filename := strings.TrimSpace(metadata.Filename)
	if filename == "" {
		filename = fmt.Sprintf("session_%s", sessionID)
	}
	filename = strings.ReplaceAll(filename, " ", "_")
	if strings.HasSuffix(strings.ToLower(filename), ".csv") {
		filename = strings.TrimSuffix(filename, ".csv")
	}
	filename = filename + "_redacted.csv"

	c.Header("Content-Type", "text/csv")
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))

	// Get CSV metadata to restore original format
	var headersJSON, columnPIISettingsJSON, delimiter string
	var hasHeaders bool
	err = h.db.QueryRow(`
		SELECT headers, column_pii_settings, delimiter, has_headers
		FROM csv_metadata
		WHERE session_id = $1
	`, sessionID).Scan(&headersJSON, &columnPIISettingsJSON, &delimiter, &hasHeaders)

	if err != nil {
		// Fallback to old format if no CSV metadata found
		logrus.WithError(err).Warn("No CSV metadata found, falling back to legacy export format")
		writer := csv.NewWriter(c.Writer)
		headers := []string{"row_number", "original_text", "redacted_text", "approved_entities"}
		if req.IncludeErrorField {
			headers = append(headers, "error")
		}

		if err := writer.Write(headers); err != nil {
			logrus.WithError(err).Error("Failed to write CSV header")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to stream export"})
			return
		}

		// Use legacy export logic
		if streamErr := h.streamExportRows(writer, sessionID, entitiesByRow, redactionMode, customLabels, skipSet, req.IncludeErrorField); streamErr != nil {
			logrus.WithError(streamErr).Error("Failed to stream export rows")
			return
		}

		writer.Flush()
		return
	}

	// Parse CSV metadata
	var originalHeaders []string
	if err := json.Unmarshal([]byte(headersJSON), &originalHeaders); err != nil {
		logrus.WithError(err).Error("Failed to parse original headers")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse CSV headers"})
		return
	}

	var columnPIISettings map[string]bool
	if err := json.Unmarshal([]byte(columnPIISettingsJSON), &columnPIISettings); err != nil {
		logrus.WithError(err).Error("Failed to parse column PII settings")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse column settings"})
		return
	}

	// Set up CSV writer with original delimiter
	writer := csv.NewWriter(c.Writer)
	if delimiter != "" && len(delimiter) > 0 {
		writer.Comma = rune(delimiter[0])
	}

	// Write original headers
	if hasHeaders {
		if err := writer.Write(originalHeaders); err != nil {
			logrus.WithError(err).Error("Failed to write CSV headers")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to write CSV headers"})
			return
		}
	}

	// Stream CSV rows in original format
	if err := h.streamStructuredCSVRows(writer, sessionID, originalHeaders, columnPIISettings, entitiesByRow, redactionMode, customLabels, skipSet); err != nil {
		logrus.WithError(err).Error("Failed to stream structured CSV rows")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to stream export"})
		return
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		logrus.WithError(err).Error("CSV writer encountered an error")
	}
}

// streamExportRows streams CSV rows directly from database without loading all into memory
func (h *FileHandler) streamExportRows(writer *csv.Writer, sessionID string, entitiesByRow map[int][]sessionEntity, redactionMode string, customLabels map[string]string, skipSet map[int64]bool, includeErrorField bool) error {
	query := `
		SELECT row_number, original_text, COALESCE(error_message, '') as error_message
		FROM processing_rows
		WHERE session_id = $1
		ORDER BY row_number ASC
	`

	rows, err := h.db.Query(query, sessionID)
	if err != nil {
		return fmt.Errorf("failed to query processing rows for export: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var rowNumber int
		var originalText, errorMessage string

		if err := rows.Scan(&rowNumber, &originalText, &errorMessage); err != nil {
			return fmt.Errorf("failed to scan processing row for export: %w", err)
		}

		// Build redacted text for this row
		rowEntities := entitiesByRow[rowNumber]
		finalRedacted, approvedCount := buildRowRedaction(originalText, rowEntities, redactionMode, customLabels, skipSet)

		// Create CSV record
		record := []string{
			strconv.Itoa(rowNumber),
			originalText,
			finalRedacted,
			strconv.Itoa(approvedCount),
		}

		if includeErrorField {
			record = append(record, errorMessage)
		}

		// Write record immediately (streaming)
		if err := writer.Write(record); err != nil {
			return fmt.Errorf("failed to write CSV record for row %d: %w", rowNumber, err)
		}

		// Flush periodically to avoid buffering too much
		if rowNumber%100 == 0 {
			writer.Flush()
		}
	}

	return rows.Err()
}

// streamStructuredCSVRows streams CSV rows in original format with PII redaction applied per column
func (h *FileHandler) streamStructuredCSVRows(writer *csv.Writer, sessionID string, headers []string, columnPIISettings map[string]bool, entitiesByRow map[int][]sessionEntity, redactionMode string, customLabels map[string]string, skipSet map[int64]bool) error {
	query := `
		SELECT cr.row_number, cr.column_data, pr.original_text
		FROM csv_row_data cr
		JOIN processing_rows pr ON cr.session_id = pr.session_id AND cr.row_number = pr.row_number
		WHERE cr.session_id = $1
		ORDER BY cr.row_number ASC
	`

	rows, err := h.db.Query(query, sessionID)
	if err != nil {
		return fmt.Errorf("failed to query structured CSV data: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var rowNumber int
		var columnDataJSON, originalText string

		if err := rows.Scan(&rowNumber, &columnDataJSON, &originalText); err != nil {
			return fmt.Errorf("failed to scan structured CSV row %d: %w", rowNumber, err)
		}

		// Parse column data
		var columnData []string
		if err := json.Unmarshal([]byte(columnDataJSON), &columnData); err != nil {
			return fmt.Errorf("failed to parse column data for row %d: %w", rowNumber, err)
		}

		// Get entities for this row
		rowEntities := entitiesByRow[rowNumber]

		// Build the redacted row using the correct mapping approach
		redactedRow, err := h.buildRedactedCSVRow(columnData, headers, columnPIISettings, originalText, rowEntities, redactionMode, customLabels, skipSet)
		if err != nil {
			logrus.WithError(err).WithField("row_number", rowNumber).Warn("Failed to build redacted row, using original")
			redactedRow = columnData
		}

		// Write the redacted row
		if err := writer.Write(redactedRow); err != nil {
			return fmt.Errorf("failed to write CSV record for row %d: %w", rowNumber, err)
		}

		// Flush periodically
		if rowNumber%100 == 0 {
			writer.Flush()
		}
	}

	return rows.Err()
}

// buildRedactedCSVRow correctly applies redaction to individual CSV columns
func (h *FileHandler) buildRedactedCSVRow(columnData []string, headers []string, columnPIISettings map[string]bool, originalText string, entities []sessionEntity, redactionMode string, customLabels map[string]string, skipSet map[int64]bool) ([]string, error) {
	redactedRow := make([]string, len(columnData))

	// If no entities detected, return original columns
	if len(entities) == 0 {
		return columnData, nil
	}

	// Apply redaction per-column basis - this is the correct approach
	for i, cellValue := range columnData {
		if i >= len(headers) {
			// Handle case where row has more columns than headers
			redactedRow[i] = cellValue
			continue
		}

		headerName := headers[i]
		if !columnPIISettings[headerName] {
			// PII processing disabled for this column, keep original
			redactedRow[i] = cellValue
		} else {
			// PII processing enabled for this column
			// Find entities that are relevant to this specific cell content
			redactedRow[i] = h.redactCellValue(cellValue, entities, redactionMode, customLabels, skipSet)
		}
	}

	return redactedRow, nil
}


// redactCellValue applies redaction to a single cell value
func (h *FileHandler) redactCellValue(cellValue string, entities []sessionEntity, redactionMode string, customLabels map[string]string, skipSet map[int64]bool) string {
	// Find entities that are contained within this specific cell
	var relevantEntities []sessionEntity
	for _, entity := range entities {
		// Check if the entity text is actually found in this cell
		if strings.Contains(cellValue, entity.Text) {
			relevantEntities = append(relevantEntities, entity)
		}
	}

	if len(relevantEntities) == 0 {
		return cellValue
	}

	// Apply redaction using buildRowRedaction on just this cell
	redacted, _ := buildRowRedaction(cellValue, relevantEntities, redactionMode, customLabels, skipSet)
	return redacted
}

// determineColumnRedaction determines the appropriate redaction for a column value
func (h *FileHandler) determineColumnRedaction(cellValue string, entities []sessionEntity, redactionMode string, customLabels map[string]string) string {
	if len(entities) == 0 {
		return cellValue
	}

	// Find the most relevant entity for this cell
	var relevantEntity *sessionEntity
	for _, entity := range entities {
		if strings.Contains(cellValue, entity.Text) {
			relevantEntity = &entity
			break
		}
	}

	if relevantEntity == nil {
		// No direct match, use first entity as fallback
		relevantEntity = &entities[0]
	}

	return resolveReplacement(*relevantEntity, redactionMode, customLabels)
}

// DownloadLegacyResults serves results from the old processing_results table (before incremental system)
func (h *FileHandler) DownloadLegacyResults(c *gin.Context) {
	resultID := c.Param("result_id")
	if resultID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Result ID is required"})
		return
	}

	format := c.DefaultQuery("format", "csv")

	query := `
		SELECT filename, original_text, redacted_text, entities_found
		FROM processing_results
		WHERE id = $1
	`

	var filename, originalText, redactedText string
	var entitiesFound int

	err := h.db.QueryRow(query, resultID).Scan(&filename, &originalText, &redactedText, &entitiesFound)
	if err != nil {
		if err == sql.ErrNoRows {
			c.JSON(http.StatusNotFound, gin.H{"error": "Results not found"})
		} else {
			logrus.WithError(err).Error("Failed to query legacy results")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve results"})
		}
		return
	}

	if format == "csv" {
		c.Header("Content-Type", "text/csv")
		c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=%s_processed.csv", filename))

		// Convert the full text back to CSV format
		// Since we stored it as joined text, we need to split it back to rows
		redactedLines := strings.Split(redactedText, "\n")
		originalLines := strings.Split(originalText, "\n")

		c.Writer.WriteString("row_number,original_text,redacted_text,entities_found\n")

		maxLines := len(redactedLines)
		if len(originalLines) > maxLines {
			maxLines = len(originalLines)
		}

		for i := 0; i < maxLines; i++ {
			origLine := ""
			redLine := ""
			if i < len(originalLines) {
				origLine = strings.ReplaceAll(originalLines[i], `"`, `""`)
			}
			if i < len(redactedLines) {
				redLine = strings.ReplaceAll(redactedLines[i], `"`, `""`)
			}

			c.Writer.WriteString(fmt.Sprintf(`%d,"%s","%s","%d"`+"\n", i+1, origLine, redLine, entitiesFound))
		}
	} else {
		c.Header("Content-Type", "application/json")
		c.JSON(http.StatusOK, gin.H{
			"filename":       filename,
			"original_text":  originalText,
			"redacted_text":  redactedText,
			"entities_found": entitiesFound,
			"format":         "legacy_full_text",
		})
	}
}

// GetProcessingResults retrieves stored processing results by ID

// Helper method to send progress updates via SSE
func (h *FileHandler) sendProgress(c *gin.Context, currentRow, totalRows int, status, message string, entities []pii.Entity) {
	progress := ProgressUpdate{
		CurrentRow: currentRow,
		TotalRows:  totalRows,
		Status:     status,
		Message:    message,
		IsComplete: false,
		Entities:   entities,
	}

	jsonData, _ := json.Marshal(progress)
	c.Writer.WriteString(fmt.Sprintf("data: %s\n\n", jsonData))
	c.Writer.Flush()
}

// Helper method to send final results via SSE
func (h *FileHandler) sendFinalResults(c *gin.Context, results *ProcessFileResponse) {
	progress := ProgressUpdate{
		CurrentRow: results.RowsProcessed,
		TotalRows:  results.RowsProcessed,
		Status:     "completed",
		Message:    "File processing completed successfully",
		IsComplete: true,
		Results:    results,
	}

	jsonData, _ := json.Marshal(progress)
	c.Writer.WriteString(fmt.Sprintf("data: %s\n\n", jsonData))
	c.Writer.Flush()
}

// Process CSV file row by row with progress updates
func (h *FileHandler) processCSVWithProgress(c *gin.Context, file io.Reader, filename string, options *pii.RedactOptions, startTime time.Time) {
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		h.sendProgress(c, 0, 0, "error", "Failed to read CSV file", nil)
		return
	}

	totalRows := len(records) - 1 // Subtract header
	if totalRows <= 0 {
		h.sendProgress(c, 0, 0, "error", "No data rows found in CSV", nil)
		return
	}

	var redactedRows []string
	totalEntities := 0

	// Process each row individually
	processedCount := 0
	for i, record := range records {
		if i == 0 { // Skip header
			continue
		}

		rowText := strings.Join(record, " ")
		processedCount++

		// Process this row through PII detection
		result, err := h.detector.Redact(rowText, options)
		if err != nil {
			logrus.WithError(err).Warnf("Failed to process row %d, skipping", processedCount)
			redactedRows = append(redactedRows, rowText) // Keep original if processing fails
			h.sendProgress(c, processedCount, totalRows, "processing", fmt.Sprintf("Processed row %d of %d (skipped due to error)", processedCount, totalRows), nil)
			continue
		}

		redactedRows = append(redactedRows, result.RedactedText)
		totalEntities += len(result.Entities)

		h.sendProgress(c, processedCount, totalRows, "processing", fmt.Sprintf("Processed row %d of %d...", processedCount, totalRows), result.Entities)

		// Small delay to prevent overwhelming Azure API
		time.Sleep(10 * time.Millisecond)
	}

	// Don't combine all results to avoid memory issues - let frontend request data as needed
	processingTime := time.Since(startTime)

	response := &ProcessFileResponse{
		OriginalText:  "", // Don't send large content in response
		RedactedText:  "", // Don't send large content in response
		RedactedCount: totalEntities,
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalRows,
		FileName:      filename,
	}

	// Save to history
	redactionMode := "replace"
	if options != nil && options.RedactionMode != "" {
		redactionMode = options.RedactionMode
	}
	var customLabels map[string]string
	if options != nil && len(options.CustomLabels) > 0 {
		customLabels = make(map[string]string, len(options.CustomLabels))
		for k, v := range options.CustomLabels {
			customLabels[k] = v
		}
	}
	go h.saveFileProcessingHistory(filename, totalEntities, processingTime, totalRows, "", "", redactionMode, customLabels)

	// Send final completion signal
	logrus.Info("✅ Sending final completion signal")
	h.sendFinalResults(c, response)

	// Ensure the stream is properly closed
	c.Writer.WriteString("data: [DONE]\n\n")
	c.Writer.Flush()
}

// Process Excel file row by row with progress updates
func (h *FileHandler) processExcelWithProgress(c *gin.Context, file io.Reader, filename string, options *pii.RedactOptions, startTime time.Time) {
	data, err := io.ReadAll(file)
	if err != nil {
		h.sendProgress(c, 0, 0, "error", "Failed to read Excel file", nil)
		return
	}

	wb, err := xlsx.OpenBinary(data)
	if err != nil {
		h.sendProgress(c, 0, 0, "error", "Failed to open Excel file", nil)
		return
	}

	var redactedRows []string
	totalRows := 0
	processedCount := 0
	totalEntities := 0

	// Count total rows first
	for _, sheet := range wb.Sheets {
		sheet.ForEachRow(func(row *xlsx.Row) error {
			totalRows++
			return nil
		})
	}
	totalRows-- // Subtract header

	for _, sheet := range wb.Sheets {
		isHeader := true
		sheet.ForEachRow(func(row *xlsx.Row) error {
			if isHeader { // Skip header
				isHeader = false
				return nil
			}

			var rowData []string
			row.ForEachCell(func(cell *xlsx.Cell) error {
				rowData = append(rowData, cell.String())
				return nil
			})

			processedCount++
			rowText := strings.Join(rowData, " ")
			result, err := h.detector.Redact(rowText, options)
			if err != nil {
				logrus.WithError(err).Warnf("Failed to process row %d, skipping", processedCount)
				redactedRows = append(redactedRows, rowText)
				h.sendProgress(c, processedCount, totalRows, "processing", fmt.Sprintf("Processed row %d of %d (skipped due to error)", processedCount, totalRows), nil)
			} else {
				redactedRows = append(redactedRows, result.RedactedText)
				totalEntities += len(result.Entities)
				h.sendProgress(c, processedCount, totalRows, "processing", fmt.Sprintf("Processed row %d of %d...", processedCount, totalRows), result.Entities)
			}

			time.Sleep(10 * time.Millisecond)
			return nil
		})
	}

	processingTime := time.Since(startTime)

	response := &ProcessFileResponse{
		OriginalText:  "", // Don't send large content in response
		RedactedText:  "", // Don't send large content in response
		RedactedCount: totalEntities,
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalRows,
		FileName:      filename,
	}

	redactionMode := "replace"
	if options != nil && options.RedactionMode != "" {
		redactionMode = options.RedactionMode
	}
	var customLabels map[string]string
	if options != nil && len(options.CustomLabels) > 0 {
		customLabels = make(map[string]string, len(options.CustomLabels))
		for k, v := range options.CustomLabels {
			customLabels[k] = v
		}
	}
	go h.saveFileProcessingHistory(filename, totalEntities, processingTime, totalRows, "", "", redactionMode, customLabels)

	// Send final completion signal
	logrus.Info("✅ Sending final completion signal for Excel")
	h.sendFinalResults(c, response)

	// Ensure the stream is properly closed
	c.Writer.WriteString("data: [DONE]\n\n")
	c.Writer.Flush()
}

// Process text file with progress updates (split by lines)
func (h *FileHandler) processTextWithProgress(c *gin.Context, file io.Reader, filename string, options *pii.RedactOptions, startTime time.Time) {
	data, err := io.ReadAll(file)
	if err != nil {
		h.sendProgress(c, 0, 0, "error", "Failed to read text file", nil)
		return
	}

	lines := strings.Split(string(data), "\n")
	totalLines := len(lines)
	var redactedLines []string
	totalEntities := 0

	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			redactedLines = append(redactedLines, line)
			h.sendProgress(c, i+1, totalLines, "processing", fmt.Sprintf("Processed line %d of %d...", i+1, totalLines), nil)
			continue
		}

		result, err := h.detector.Redact(line, options)
		if err != nil {
			logrus.WithError(err).Warnf("Failed to process line %d, skipping", i+1)
			redactedLines = append(redactedLines, line)
			h.sendProgress(c, i+1, totalLines, "processing", fmt.Sprintf("Processed line %d of %d (skipped due to error)", i+1, totalLines), nil)
		} else {
			redactedLines = append(redactedLines, result.RedactedText)
			totalEntities += len(result.Entities)
			h.sendProgress(c, i+1, totalLines, "processing", fmt.Sprintf("Processed line %d of %d...", i+1, totalLines), result.Entities)
		}

		time.Sleep(10 * time.Millisecond)
	}

	processingTime := time.Since(startTime)

	response := &ProcessFileResponse{
		OriginalText:  "", // Don't send large content in response
		RedactedText:  "", // Don't send large content in response
		RedactedCount: totalEntities,
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalLines,
		FileName:      filename,
	}

	redactionMode := "replace"
	if options != nil && options.RedactionMode != "" {
		redactionMode = options.RedactionMode
	}
	var customLabels map[string]string
	if options != nil && len(options.CustomLabels) > 0 {
		customLabels = make(map[string]string, len(options.CustomLabels))
		for k, v := range options.CustomLabels {
			customLabels[k] = v
		}
	}
	go h.saveFileProcessingHistory(filename, totalEntities, processingTime, totalLines, "", "", redactionMode, customLabels)

	// Send final completion signal
	logrus.Info("✅ Sending final completion signal for text file")
	h.sendFinalResults(c, response)

	// Ensure the stream is properly closed
	c.Writer.WriteString("data: [DONE]\n\n")
	c.Writer.Flush()
}

// GetProcessingResults retrieves stored processing results by ID
func (h *FileHandler) GetProcessingResults(c *gin.Context) {
	resultID := c.Param("id")
	if resultID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Result ID is required"})
		return
	}

	query := `
		SELECT id, filename, original_text, redacted_text, entities_found, processing_time_ms, rows_processed, created_at
		FROM processing_results
		WHERE id = $1
	`

	var result struct {
		ID               string  `json:"id"`
		Filename         string  `json:"filename"`
		OriginalText     string  `json:"original_text"`
		RedactedText     string  `json:"redacted_text"`
		EntitiesFound    int     `json:"entities_found"`
		ProcessingTimeMs float64 `json:"processing_time_ms"`
		RowsProcessed    int     `json:"rows_processed"`
		CreatedAt        string  `json:"created_at"`
	}

	err := h.db.QueryRow(query, resultID).Scan(
		&result.ID,
		&result.Filename,
		&result.OriginalText,
		&result.RedactedText,
		&result.EntitiesFound,
		&result.ProcessingTimeMs,
		&result.RowsProcessed,
		&result.CreatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			c.JSON(http.StatusNotFound, gin.H{"error": "Processing results not found"})
		} else {
			logrus.WithError(err).Error("Failed to retrieve processing results")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve processing results"})
		}
		return
	}

	logrus.WithFields(logrus.Fields{
		"result_id":      resultID,
		"content_size":   len(result.OriginalText),
		"entities_found": result.EntitiesFound,
	}).Info("📤 Retrieved processing results from database")

	c.JSON(http.StatusOK, result)
}

// GetCSVMetadata returns CSV headers and column PII settings for a session
func (h *FileHandler) GetCSVMetadata(c *gin.Context) {
	sessionID := c.Param("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	var headersJSON, columnPIISettingsJSON, delimiter string
	var hasHeaders bool
	var totalColumns int

	err := h.db.QueryRow(`
		SELECT headers, column_pii_settings, delimiter, has_headers, total_columns
		FROM csv_metadata
		WHERE session_id = $1
	`, sessionID).Scan(&headersJSON, &columnPIISettingsJSON, &delimiter, &hasHeaders, &totalColumns)

	if err != nil {
		if err == sql.ErrNoRows {
			c.JSON(http.StatusNotFound, gin.H{"error": "CSV metadata not found for this session"})
		} else {
			logrus.WithError(err).Error("Failed to query CSV metadata")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve CSV metadata"})
		}
		return
	}

	// Parse JSON fields
	var headers []string
	var columnPIISettings map[string]bool

	if err := json.Unmarshal([]byte(headersJSON), &headers); err != nil {
		logrus.WithError(err).Error("Failed to parse CSV headers")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse CSV headers"})
		return
	}

	if err := json.Unmarshal([]byte(columnPIISettingsJSON), &columnPIISettings); err != nil {
		logrus.WithError(err).Error("Failed to parse column PII settings")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse column settings"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"session_id":           sessionID,
		"headers":              headers,
		"column_pii_settings":  columnPIISettings,
		"delimiter":            delimiter,
		"has_headers":          hasHeaders,
		"total_columns":        totalColumns,
	})
}

// UpdateColumnPIISettings updates which columns should have PII detection enabled/disabled
func (h *FileHandler) UpdateColumnPIISettings(c *gin.Context) {
	sessionID := c.Param("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	var req struct {
		ColumnPIISettings map[string]bool `json:"column_pii_settings"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}

	// Validate that the session exists
	var exists bool
	err := h.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM csv_metadata WHERE session_id = $1)`, sessionID).Scan(&exists)
	if err != nil {
		logrus.WithError(err).Error("Failed to check session existence")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to validate session"})
		return
	}

	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	// Update column PII settings
	columnPIISettingsJSON, err := json.Marshal(req.ColumnPIISettings)
	if err != nil {
		logrus.WithError(err).Error("Failed to marshal column PII settings")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process settings"})
		return
	}

	_, err = h.db.Exec(`
		UPDATE csv_metadata
		SET column_pii_settings = $1
		WHERE session_id = $2
	`, string(columnPIISettingsJSON), sessionID)

	if err != nil {
		logrus.WithError(err).Error("Failed to update column PII settings")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update column settings"})
		return
	}

	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"settings":   req.ColumnPIISettings,
	}).Info("✅ Updated column PII settings")

	c.JSON(http.StatusOK, gin.H{
		"message":              "Column PII settings updated successfully",
		"session_id":           sessionID,
		"column_pii_settings":  req.ColumnPIISettings,
	})
}

// minInt returns the minimum of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
