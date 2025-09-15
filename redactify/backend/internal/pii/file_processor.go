package pii

import (
	"encoding/csv"
	"fmt"
	"io"
	"mime/multipart"
	"path/filepath"
	"strings"

	"github.com/sirupsen/logrus"
)

// FileProcessor handles file-based PII processing
type FileProcessor struct {
	detector *Detector
}

// ProcessResult represents the result of processing a file
type ProcessResult struct {
	OriginalFileName string              `json:"original_filename"`
	ProcessedRows    int                 `json:"processed_rows"`
	TotalEntities    int                 `json:"total_entities"`
	ProcessedData    [][]string          `json:"processed_data"`
	EntitySummary    map[string]int      `json:"entity_summary"`
	Errors           []string            `json:"errors"`
}

// NewFileProcessor creates a new file processor
func NewFileProcessor(detector *Detector) *FileProcessor {
	return &FileProcessor{
		detector: detector,
	}
}

// ProcessCSV processes a CSV file for PII detection and redaction
func (fp *FileProcessor) ProcessCSV(file multipart.File, header *multipart.FileHeader, options *RedactOptions) (*ProcessResult, error) {
	defer file.Close()
	
	// Validate file type
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if ext != ".csv" {
		return nil, fmt.Errorf("unsupported file type: %s", ext)
	}
	
	// Create CSV reader
	reader := csv.NewReader(file)
	reader.LazyQuotes = true
	reader.TrimLeadingSpace = true
	
	var processedData [][]string
	var allErrors []string
	entitySummary := make(map[string]int)
	totalEntities := 0
	rowCount := 0
	
	// Read and process CSV
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			allErrors = append(allErrors, fmt.Sprintf("Row %d: %v", rowCount+1, err))
			continue
		}
		
		// Process each cell in the row
		processedRow := make([]string, len(row))
		for i, cell := range row {
			if strings.TrimSpace(cell) == "" {
				processedRow[i] = cell
				continue
			}
			
			// Detect and redact PII in this cell
			result, err := fp.detector.Redact(cell, options)
			if err != nil {
				logrus.WithError(err).Warnf("Failed to process cell at row %d, column %d", rowCount+1, i+1)
				processedRow[i] = cell // Keep original on error
				continue
			}
			
			processedRow[i] = result.RedactedText
			
			// Update statistics
			for _, entity := range result.Entities {
				entitySummary[entity.Type]++
				totalEntities++
			}
		}
		
		processedData = append(processedData, processedRow)
		rowCount++
		
		// Log progress for large files
		if rowCount%1000 == 0 {
			logrus.Infof("Processed %d rows", rowCount)
		}
	}
	
	return &ProcessResult{
		OriginalFileName: header.Filename,
		ProcessedRows:    rowCount,
		TotalEntities:    totalEntities,
		ProcessedData:    processedData,
		EntitySummary:    entitySummary,
		Errors:           allErrors,
	}, nil
}

// ProcessText processes plain text for PII detection and redaction
func (fp *FileProcessor) ProcessText(content string, options *RedactOptions) (*ProcessResult, error) {
	result, err := fp.detector.Redact(content, options)
	if err != nil {
		return nil, fmt.Errorf("failed to process text: %w", err)
	}
	
	entitySummary := make(map[string]int)
	for _, entity := range result.Entities {
		entitySummary[entity.Type]++
	}
	
	// Return as single "row" for consistency
	processedData := [][]string{{result.RedactedText}}
	
	return &ProcessResult{
		OriginalFileName: "text_input",
		ProcessedRows:    1,
		TotalEntities:    len(result.Entities),
		ProcessedData:    processedData,
		EntitySummary:    entitySummary,
		Errors:           []string{},
	}, nil
}

// ProcessBatch processes multiple text strings in batch
func (fp *FileProcessor) ProcessBatch(texts []string, options *RedactOptions) (*ProcessResult, error) {
	var processedData [][]string
	var allErrors []string
	entitySummary := make(map[string]int)
	totalEntities := 0
	
	for i, text := range texts {
		if strings.TrimSpace(text) == "" {
			processedData = append(processedData, []string{text})
			continue
		}
		
		result, err := fp.detector.Redact(text, options)
		if err != nil {
			allErrors = append(allErrors, fmt.Sprintf("Item %d: %v", i+1, err))
			processedData = append(processedData, []string{text}) // Keep original on error
			continue
		}
		
		processedData = append(processedData, []string{result.RedactedText})
		
		// Update statistics
		for _, entity := range result.Entities {
			entitySummary[entity.Type]++
			totalEntities++
		}
	}
	
	return &ProcessResult{
		OriginalFileName: "batch_input",
		ProcessedRows:    len(texts),
		TotalEntities:    totalEntities,
		ProcessedData:    processedData,
		EntitySummary:    entitySummary,
		Errors:           allErrors,
	}, nil
}

// ExportCSV exports processed data back to CSV format
func (fp *FileProcessor) ExportCSV(result *ProcessResult, writer io.Writer) error {
	csvWriter := csv.NewWriter(writer)
	defer csvWriter.Flush()
	
	// Write all processed rows
	for _, row := range result.ProcessedData {
		if err := csvWriter.Write(row); err != nil {
			return fmt.Errorf("failed to write CSV row: %w", err)
		}
	}
	
	return nil
}

// GetSupportedFormats returns list of supported file formats
func (fp *FileProcessor) GetSupportedFormats() []string {
	return []string{".csv", ".txt"}
}

// ValidateFile checks if file can be processed
func (fp *FileProcessor) ValidateFile(header *multipart.FileHeader) error {
	// Check file size (limit to 100MB)
	const maxSize = 100 * 1024 * 1024
	if header.Size > maxSize {
		return fmt.Errorf("file too large: %d bytes (max: %d bytes)", header.Size, maxSize)
	}
	
	// Check file extension
	ext := strings.ToLower(filepath.Ext(header.Filename))
	supported := fp.GetSupportedFormats()
	
	for _, supportedExt := range supported {
		if ext == supportedExt {
			return nil
		}
	}
	
	return fmt.Errorf("unsupported file type: %s (supported: %v)", ext, supported)
}