package db

import (
	"database/sql"
	"fmt"
	
	_ "github.com/mattn/go-sqlite3"
)

func Init(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	if err := createTables(db); err != nil {
		return nil, fmt.Errorf("failed to create tables: %w", err)
	}

	return db, nil
}

func createTables(db *sql.DB) error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS redaction_history (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
			filename TEXT,
			original_text TEXT,
			redacted_text TEXT,
			entities_detected INTEGER,
			confidence_score REAL,
			processing_time_ms INTEGER,
			status TEXT DEFAULT 'completed'
		)`,
		
		`CREATE TABLE IF NOT EXISTS file_processing (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			filename TEXT NOT NULL,
			file_size INTEGER,
			status TEXT DEFAULT 'pending',
			progress REAL DEFAULT 0.0,
			rows_processed INTEGER DEFAULT 0,
			entities_detected INTEGER DEFAULT 0,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			completed_at DATETIME
		)`,
		
		`CREATE TABLE IF NOT EXISTS processing_history (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			filename TEXT NOT NULL,
			timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
			status TEXT DEFAULT 'completed',
			entities_found INTEGER DEFAULT 0,
			processing_time_ms REAL DEFAULT 0.0,
			file_size INTEGER DEFAULT 0,
			success_rate REAL DEFAULT 100.0
		)`,
		
		`CREATE TABLE IF NOT EXISTS app_config (
			key TEXT PRIMARY KEY,
			value TEXT,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)`,
		
		`CREATE TABLE IF NOT EXISTS training_feedback (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			entity_text TEXT NOT NULL,
			entity_type TEXT NOT NULL,
			original_score REAL DEFAULT 0.0,
			user_decision TEXT NOT NULL,
			user_confidence REAL DEFAULT 1.0,
			context TEXT,
			timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
			session_id TEXT
		)`,
	}

	for _, query := range queries {
		if _, err := db.Exec(query); err != nil {
			return fmt.Errorf("failed to execute query: %w", err)
		}
	}

	return nil
}