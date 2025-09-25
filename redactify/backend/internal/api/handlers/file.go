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
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
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

	err = conn.ReadJSON(&processRequest)
	if err != nil {
		logrus.WithError(err).Error("Failed to read file processing request")
		return
	}

	// Create processing session
	sessionID := h.wsHandler.CreateSession(processRequest.FileName, conn)

	// Start processing in goroutine
	go h.processFileWithWebSocket(sessionID, &processRequest)

	// Keep connection alive and handle client messages
	for {
		var msg WebSocketMessage
		err := conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				logrus.WithError(err).Error("WebSocket error during file processing")
			}
			break
		}

		// Handle client messages (like cancel requests)
		if msg.Type == "cancel" {
			logrus.Infof("Processing cancelled for session: %s", sessionID)
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

	for _, delimiter := range delimiters {
		reader := csv.NewReader(strings.NewReader(contentStr))
		reader.Comma = delimiter
		reader.FieldsPerRecord = -1 // Allow variable number of fields

		tempRecords, err := reader.ReadAll()
		if err != nil {
			logrus.WithField("delimiter", string(delimiter)).Warn("⚠️ Failed to parse with delimiter")
			continue
		}

		if len(tempRecords) > 1 { // At least header + 1 data row
			records = tempRecords
			headers = records[0]
			logrus.WithFields(logrus.Fields{
				"session_id": sessionID,
				"delimiter":  string(delimiter),
				"rows":       len(records),
				"headers":    headers,
			}).Info("✅ Successfully parsed CSV")
			break
		}
	}

	totalRows := len(records) - 1 // Subtract header
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
	for i, record := range records {
		if i == 0 { // Skip header
			continue
		}

		rowText := strings.Join(record, " ")
		processedCount++
		rowStartTime := time.Now()

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

	// Get total entities count from saved rows (without loading all text into memory)
	dbTotalEntities, err := h.getTotalEntitiesFromRows(sessionID)
	if err != nil {
		logrus.WithError(err).Error("Failed to get entities count from saved rows")
		h.wsHandler.NotifyError(sessionID, fmt.Errorf("failed to get results summary from database: %w", err))
		return
	}

	processingTime := time.Since(startTime)

	// Save metadata only to processing_results (no large text content)
	resultID, err := h.saveProcessingResults(sessionID, filename, "", "", dbTotalEntities, processingTime, totalRows)
	if err != nil {
		logrus.WithError(err).Error("Failed to save processing results")
		h.wsHandler.NotifyError(sessionID, fmt.Errorf("failed to save processing results: %w", err))
		return
	}

	// Create lightweight response (no large content)
	response := &ProcessFileResponse{
		OriginalText:  "",              // Don't send large content via WebSocket
		RedactedText:  "",              // Don't send large content via WebSocket
		RedactedCount: dbTotalEntities, // Use entities count from database
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalRows,
		FileName:      filename,
		ResultID:      resultID, // Include result ID for retrieval
	}

	// Save to history
	go h.saveFileProcessingHistory(filename, dbTotalEntities, processingTime, totalRows, resultID, sessionID)

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

func (h *FileHandler) saveFileProcessingHistory(filename string, entitiesFound int, processingTime time.Duration, rowsProcessed int, resultID, sessionID string) {
	query := `
		INSERT INTO processing_history (filename, entities_found, processing_time_ms, status, file_size, result_id, session_id)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`

	processingTimeMs := float64(processingTime.Nanoseconds()) / 1000000

	_, err := h.db.Exec(query, filename, entitiesFound, processingTimeMs, "completed", rowsProcessed, resultID, sessionID)
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

// saveProcessingResults saves full processing results to database and returns result ID
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

// buildResultsFromRows retrieves and combines all processed rows for a session
// WARNING: This loads all results into memory - only use when specifically requested by user
func (h *FileHandler) buildResultsFromRows(sessionID string) (string, string, int, error) {
	query := `
		SELECT original_text, redacted_text, entities_count
		FROM processing_rows
		WHERE session_id = $1
		ORDER BY row_number ASC
	`

	rows, err := h.db.Query(query, sessionID)
	if err != nil {
		return "", "", 0, fmt.Errorf("failed to query row results: %w", err)
	}
	defer rows.Close()

	var originalRows []string
	var redactedRows []string
	totalEntities := 0

	for rows.Next() {
		var original, redacted string
		var entities int

		if err := rows.Scan(&original, &redacted, &entities); err != nil {
			return "", "", 0, fmt.Errorf("failed to scan row result: %w", err)
		}

		originalRows = append(originalRows, original)
		redactedRows = append(redactedRows, redacted)
		totalEntities += entities
	}

	if err := rows.Err(); err != nil {
		return "", "", 0, fmt.Errorf("error iterating rows: %w", err)
	}

	originalText := strings.Join(originalRows, "\n")
	redactedText := strings.Join(redactedRows, "\n")

	return originalText, redactedText, totalEntities, nil
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

// StreamProcessingResults streams processing results row by row without loading all into memory
func (h *FileHandler) StreamProcessingResults(c *gin.Context) {
	sessionID := c.Param("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	format := c.DefaultQuery("format", "csv") // csv or json

	query := `
		SELECT row_number, original_text, redacted_text, entities_count, error_message
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
		c.Writer.WriteString("row_number,original_text,redacted_text,entities_count,error\n")
	} else {
		c.Header("Content-Type", "application/json")
		c.Writer.WriteString(`{"results":[`)
	}

	first := true
	for rows.Next() {
		var rowNum int
		var original, redacted, errorMsg sql.NullString
		var entities int

		if err := rows.Scan(&rowNum, &original, &redacted, &entities, &errorMsg); err != nil {
			logrus.WithError(err).Error("Failed to scan processing row")
			continue
		}

		if format == "csv" {
			// CSV format - escape quotes and commas
			origText := strings.ReplaceAll(original.String, `"`, `""`)
			redText := strings.ReplaceAll(redacted.String, `"`, `""`)
			errText := strings.ReplaceAll(errorMsg.String, `"`, `""`)

			c.Writer.WriteString(fmt.Sprintf(`%d,"%s","%s",%d,"%s"`+"\n",
				rowNum, origText, redText, entities, errText))
		} else {
			// JSON format
			if !first {
				c.Writer.WriteString(",")
			}
			c.Writer.WriteString(fmt.Sprintf(`{"row_number":%d,"original_text":%q,"redacted_text":%q,"entities_count":%d,"error":%q}`,
				rowNum, original.String, redacted.String, entities, errorMsg.String))
			first = false
		}

		c.Writer.Flush() // Stream immediately
	}

	if format == "json" {
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

	// Combine all results
	finalText := strings.Join(redactedRows, "\n")
	processingTime := time.Since(startTime)

	response := &ProcessFileResponse{
		OriginalText:  finalText, // Send redacted as "original" for frontend
		RedactedText:  finalText,
		RedactedCount: totalEntities,
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalRows,
		FileName:      filename,
	}

	// Save to history
	go h.saveFileProcessingHistory(filename, totalEntities, processingTime, totalRows, "", "")

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

	finalText := strings.Join(redactedRows, "\n")
	processingTime := time.Since(startTime)

	response := &ProcessFileResponse{
		OriginalText:  finalText,
		RedactedText:  finalText,
		RedactedCount: totalEntities,
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalRows,
		FileName:      filename,
	}

	go h.saveFileProcessingHistory(filename, totalEntities, processingTime, totalRows, "", "")

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

	finalText := strings.Join(redactedLines, "\n")
	processingTime := time.Since(startTime)

	response := &ProcessFileResponse{
		OriginalText:  finalText,
		RedactedText:  finalText,
		RedactedCount: totalEntities,
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalLines,
		FileName:      filename,
	}

	go h.saveFileProcessingHistory(filename, totalEntities, processingTime, totalLines, "", "")

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

// minInt returns the minimum of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
