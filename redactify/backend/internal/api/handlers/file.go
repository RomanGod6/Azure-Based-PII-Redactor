package handlers

import (
	"database/sql"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"redactify/internal/pii"
	"redactify/pkg/config"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
	"github.com/tealeg/xlsx/v3"
)

type FileHandler struct {
	db       *sql.DB
	config   *config.Config
	detector *pii.Detector
}

type ProcessFileResponse struct {
	OriginalText  string       `json:"original_text"`
	RedactedText  string       `json:"redacted_text"`
	Entities      []pii.Entity `json:"entities"`
	RedactedCount int          `json:"redacted_count"`
	ProcessTime   string       `json:"process_time"`
	RowsProcessed int          `json:"rows_processed"`
	FileName      string       `json:"file_name"`
}

type ProgressUpdate struct {
	CurrentRow    int    `json:"current_row"`
	TotalRows     int    `json:"total_rows"`
	Status        string `json:"status"`
	Message       string `json:"message"`
	IsComplete    bool   `json:"is_complete"`
	Results       *ProcessFileResponse `json:"results,omitempty"`
}

func NewFileHandler(db *sql.DB, cfg *config.Config) *FileHandler {
	return &FileHandler{
		db:       db,
		config:   cfg,
		detector: pii.NewDetectorWithDB(cfg, db),
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
	h.sendProgress(c, 0, 0, "starting", "Initializing file processing...")
	
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
		h.sendProgress(c, 0, 0, "error", "Unsupported file type")
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

// Helper functions for processing different file types

func (h *FileHandler) processCSV(file io.Reader) (string, int, error) {
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return "", 0, fmt.Errorf("failed to read CSV: %w", err)
	}

	var content strings.Builder
	for i, record := range records {
		if i > 0 { // Skip header if exists
			content.WriteString(strings.Join(record, " "))
			content.WriteString("\n")
		}
	}

	return content.String(), len(records), nil
}

func (h *FileHandler) processExcel(file io.Reader) (string, int, error) {
	// Read all content into memory
	data, err := io.ReadAll(file)
	if err != nil {
		return "", 0, fmt.Errorf("failed to read Excel file: %w", err)
	}

	// Open Excel file
	wb, err := xlsx.OpenBinary(data)
	if err != nil {
		return "", 0, fmt.Errorf("failed to open Excel file: %w", err)
	}

	var content strings.Builder
	totalRows := 0

	for _, sheet := range wb.Sheets {
		sheet.ForEachRow(func(row *xlsx.Row) error {
			if totalRows > 0 { // Skip header
				var rowData []string
				row.ForEachCell(func(cell *xlsx.Cell) error {
					rowData = append(rowData, cell.String())
					return nil
				})
				content.WriteString(strings.Join(rowData, " "))
				content.WriteString("\n")
			}
			totalRows++
			return nil
		})
	}

	return content.String(), totalRows, nil
}

func (h *FileHandler) processText(file io.Reader) (string, int, error) {
	data, err := io.ReadAll(file)
	if err != nil {
		return "", 0, fmt.Errorf("failed to read text file: %w", err)
	}

	content := string(data)
	lines := strings.Split(content, "\n")
	return content, len(lines), nil
}

func (h *FileHandler) saveFileProcessingHistory(filename string, entitiesFound int, processingTime time.Duration, rowsProcessed int) {
	query := `
		INSERT INTO processing_history (filename, entities_found, processing_time_ms, status, file_size)
		VALUES (?, ?, ?, ?, ?)
	`
	
	processingTimeMs := float64(processingTime.Nanoseconds()) / 1000000
	
	_, err := h.db.Exec(query, filename, entitiesFound, processingTimeMs, "completed", rowsProcessed)
	if err != nil {
		logrus.WithError(err).Error("Failed to save file processing history")
	}
}

// Helper method to send progress updates via SSE
func (h *FileHandler) sendProgress(c *gin.Context, currentRow, totalRows int, status, message string) {
	progress := ProgressUpdate{
		CurrentRow: currentRow,
		TotalRows:  totalRows,
		Status:     status,
		Message:    message,
		IsComplete: false,
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
		h.sendProgress(c, 0, 0, "error", "Failed to read CSV file")
		return
	}
	
	totalRows := len(records) - 1 // Subtract header
	if totalRows <= 0 {
		h.sendProgress(c, 0, 0, "error", "No data rows found in CSV")
		return
	}
	
	var allEntities []pii.Entity
	var redactedRows []string
	
	// Process each row individually
	for i, record := range records {
		if i == 0 { // Skip header
			continue
		}
		
		rowText := strings.Join(record, " ")
		currentRowNum := i // 1-based counting
		
		h.sendProgress(c, currentRowNum, totalRows, "processing", fmt.Sprintf("Processing row %d of %d...", currentRowNum, totalRows))
		
		// Process this row through PII detection
		result, err := h.detector.Redact(rowText, options)
		if err != nil {
			logrus.WithError(err).Warn("Failed to process row %d, skipping", currentRowNum)
			redactedRows = append(redactedRows, rowText) // Keep original if processing fails
			continue
		}
		
		allEntities = append(allEntities, result.Entities...)
		redactedRows = append(redactedRows, result.RedactedText)
		
		// Small delay to prevent overwhelming Azure API
		time.Sleep(10 * time.Millisecond)
	}
	
	// Combine all results
	finalText := strings.Join(redactedRows, "\n")
	processingTime := time.Since(startTime)
	
	response := &ProcessFileResponse{
		OriginalText:  finalText, // Send redacted as "original" for frontend
		RedactedText:  finalText,
		Entities:      allEntities,
		RedactedCount: len(allEntities),
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalRows,
		FileName:      filename,
	}
	
	// Save to history
	go h.saveFileProcessingHistory(filename, len(allEntities), processingTime, totalRows)
	
	h.sendFinalResults(c, response)
}

// Process Excel file row by row with progress updates
func (h *FileHandler) processExcelWithProgress(c *gin.Context, file io.Reader, filename string, options *pii.RedactOptions, startTime time.Time) {
	data, err := io.ReadAll(file)
	if err != nil {
		h.sendProgress(c, 0, 0, "error", "Failed to read Excel file")
		return
	}
	
	wb, err := xlsx.OpenBinary(data)
	if err != nil {
		h.sendProgress(c, 0, 0, "error", "Failed to open Excel file")
		return
	}
	
	var allEntities []pii.Entity
	var redactedRows []string
	totalRows := 0
	currentRow := 0
	
	// Count total rows first
	for _, sheet := range wb.Sheets {
		sheet.ForEachRow(func(row *xlsx.Row) error {
			totalRows++
			return nil
		})
	}
	totalRows-- // Subtract header
	
	for _, sheet := range wb.Sheets {
		sheet.ForEachRow(func(row *xlsx.Row) error {
			if currentRow == 0 { // Skip header
				currentRow++
				return nil
			}
			
			var rowData []string
			row.ForEachCell(func(cell *xlsx.Cell) error {
				rowData = append(rowData, cell.String())
				return nil
			})
			
			rowText := strings.Join(rowData, " ")
			h.sendProgress(c, currentRow, totalRows, "processing", fmt.Sprintf("Processing row %d of %d...", currentRow, totalRows))
			
			result, err := h.detector.Redact(rowText, options)
			if err != nil {
				logrus.WithError(err).Warn("Failed to process row %d, skipping", currentRow)
				redactedRows = append(redactedRows, rowText)
			} else {
				allEntities = append(allEntities, result.Entities...)
				redactedRows = append(redactedRows, result.RedactedText)
			}
			
			currentRow++
			time.Sleep(10 * time.Millisecond)
			return nil
		})
	}
	
	finalText := strings.Join(redactedRows, "\n")
	processingTime := time.Since(startTime)
	
	response := &ProcessFileResponse{
		OriginalText:  finalText,
		RedactedText:  finalText,
		Entities:      allEntities,
		RedactedCount: len(allEntities),
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalRows,
		FileName:      filename,
	}
	
	go h.saveFileProcessingHistory(filename, len(allEntities), processingTime, totalRows)
	h.sendFinalResults(c, response)
}

// Process text file with progress updates (split by lines)
func (h *FileHandler) processTextWithProgress(c *gin.Context, file io.Reader, filename string, options *pii.RedactOptions, startTime time.Time) {
	data, err := io.ReadAll(file)
	if err != nil {
		h.sendProgress(c, 0, 0, "error", "Failed to read text file")
		return
	}
	
	lines := strings.Split(string(data), "\n")
	totalLines := len(lines)
	var allEntities []pii.Entity
	var redactedLines []string
	
	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			redactedLines = append(redactedLines, line)
			continue
		}
		
		h.sendProgress(c, i+1, totalLines, "processing", fmt.Sprintf("Processing line %d of %d...", i+1, totalLines))
		
		result, err := h.detector.Redact(line, options)
		if err != nil {
			logrus.WithError(err).Warn("Failed to process line %d, skipping", i+1)
			redactedLines = append(redactedLines, line)
		} else {
			allEntities = append(allEntities, result.Entities...)
			redactedLines = append(redactedLines, result.RedactedText)
		}
		
		time.Sleep(10 * time.Millisecond)
	}
	
	finalText := strings.Join(redactedLines, "\n")
	processingTime := time.Since(startTime)
	
	response := &ProcessFileResponse{
		OriginalText:  finalText,
		RedactedText:  finalText,
		Entities:      allEntities,
		RedactedCount: len(allEntities),
		ProcessTime:   fmt.Sprintf("%.2fms", float64(processingTime.Nanoseconds())/1000000),
		RowsProcessed: totalLines,
		FileName:      filename,
	}
	
	go h.saveFileProcessingHistory(filename, len(allEntities), processingTime, totalLines)
	h.sendFinalResults(c, response)
}