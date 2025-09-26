package db

import (
	"database/sql"
	"fmt"
	"os"
	"strconv"
	"time"

	_ "github.com/lib/pq"
	"github.com/sirupsen/logrus"
)

func Init(dbURL string) (*sql.DB, error) {
	// If dbURL is empty, use environment variables or defaults
	if dbURL == "" {
		dbURL = getDefaultPostgresURL()
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pooling for optimal performance
	if err := configureConnectionPool(db); err != nil {
		return nil, fmt.Errorf("failed to configure connection pool: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	if err := createTables(db); err != nil {
		return nil, fmt.Errorf("failed to create tables: %w", err)
	}

	logrus.Info("📊 Database initialized with connection pooling")
	return db, nil
}

// configureConnectionPool sets up optimal connection pool settings
func configureConnectionPool(db *sql.DB) error {
	// Maximum number of open connections to the database
	maxOpenConns := getEnvAsInt("DB_MAX_OPEN_CONNS", 25)
	db.SetMaxOpenConns(maxOpenConns)

	// Maximum number of idle connections in the pool
	maxIdleConns := getEnvAsInt("DB_MAX_IDLE_CONNS", 5)
	db.SetMaxIdleConns(maxIdleConns)

	// Maximum amount of time a connection may be reused (in minutes)
	maxLifetimeMinutes := getEnvAsInt("DB_MAX_LIFETIME_MINUTES", 30)
	db.SetConnMaxLifetime(time.Duration(maxLifetimeMinutes) * time.Minute)

	// Maximum amount of time a connection may be idle (in minutes)
	maxIdleTimeMinutes := getEnvAsInt("DB_MAX_IDLE_TIME_MINUTES", 15)
	db.SetConnMaxIdleTime(time.Duration(maxIdleTimeMinutes) * time.Minute)

	logrus.WithFields(logrus.Fields{
		"max_open_conns":        maxOpenConns,
		"max_idle_conns":        maxIdleConns,
		"max_lifetime_minutes":  maxLifetimeMinutes,
		"max_idle_time_minutes": maxIdleTimeMinutes,
	}).Info("📊 Database connection pool configured")

	return nil
}

func getDefaultPostgresURL() string {
	host := getEnvOrDefault("DB_HOST", "localhost")
	port := getEnvOrDefault("DB_PORT", "5432")
	user := getEnvOrDefault("DB_USER", "redactify")
	password := getEnvOrDefault("DB_PASSWORD", "redactify_dev_password")
	dbname := getEnvOrDefault("DB_NAME", "redactify")
	sslmode := getEnvOrDefault("DB_SSLMODE", "disable")

	return fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=%s",
		host, port, user, password, dbname, sslmode)
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvAsInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func createTables(db *sql.DB) error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS redaction_history (
			id SERIAL PRIMARY KEY,
			timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			filename TEXT,
			original_text TEXT,
			redacted_text TEXT,
			entities_detected INTEGER,
			confidence_score REAL,
			processing_time_ms INTEGER,
			status TEXT DEFAULT 'completed'
		)`,

		`CREATE TABLE IF NOT EXISTS file_processing (
			id SERIAL PRIMARY KEY,
			filename TEXT NOT NULL,
			file_size INTEGER,
			status TEXT DEFAULT 'pending',
			progress REAL DEFAULT 0.0,
			rows_processed INTEGER DEFAULT 0,
			entities_detected INTEGER DEFAULT 0,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			completed_at TIMESTAMP
		)`,

		`CREATE TABLE IF NOT EXISTS processing_history (
			id SERIAL PRIMARY KEY,
			filename TEXT NOT NULL,
			timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			status TEXT DEFAULT 'completed',
			entities_found INTEGER DEFAULT 0,
			processing_time_ms REAL DEFAULT 0.0,
			file_size INTEGER DEFAULT 0,
			success_rate REAL DEFAULT 100.0,
			result_id TEXT,
			session_id TEXT,
			redaction_mode TEXT,
			custom_labels TEXT
		)`,

		`CREATE TABLE IF NOT EXISTS app_config (
			key TEXT PRIMARY KEY,
			value TEXT,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,

		`CREATE TABLE IF NOT EXISTS training_feedback (
			id SERIAL PRIMARY KEY,
			entity_text TEXT NOT NULL,
			entity_type TEXT NOT NULL,
			original_score REAL DEFAULT 0.0,
			user_decision TEXT NOT NULL,
			user_confidence REAL DEFAULT 1.0,
			context TEXT,
			timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			session_id TEXT
		)`,

		`CREATE TABLE IF NOT EXISTS processing_results (
			id TEXT PRIMARY KEY,
			filename TEXT NOT NULL,
			original_text TEXT,
			redacted_text TEXT,
			entities_found INTEGER DEFAULT 0,
			processing_time_ms REAL DEFAULT 0.0,
			rows_processed INTEGER DEFAULT 0,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,

		`CREATE TABLE IF NOT EXISTS processing_rows (
			id SERIAL PRIMARY KEY,
			session_id TEXT NOT NULL,
			row_number INTEGER NOT NULL,
			original_text TEXT,
			redacted_text TEXT,
			entities_count INTEGER DEFAULT 0,
			processing_time_ms REAL DEFAULT 0.0,
			status TEXT DEFAULT 'completed',
			error_message TEXT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			UNIQUE(session_id, row_number)
		)`,

		`CREATE TABLE IF NOT EXISTS detected_entities (
			id SERIAL PRIMARY KEY,
			session_id TEXT,
			result_id TEXT,
			entity_type TEXT NOT NULL,
			entity_text TEXT NOT NULL,
			start_position INTEGER,
			end_position INTEGER,
			confidence REAL,
			category TEXT,
			row_number INTEGER,
			approved BOOLEAN DEFAULT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
	}

	for _, query := range queries {
		if _, err := db.Exec(query); err != nil {
			return fmt.Errorf("failed to execute query: %w", err)
		}
	}

	// Add migrations for existing tables
	migrations := []string{
		// Add result_id and session_id columns to processing_history if they don't exist
		`ALTER TABLE processing_history ADD COLUMN IF NOT EXISTS result_id TEXT`,
		`ALTER TABLE processing_history ADD COLUMN IF NOT EXISTS session_id TEXT`,
		`ALTER TABLE processing_history ADD COLUMN IF NOT EXISTS redaction_mode TEXT`,
		`ALTER TABLE processing_history ADD COLUMN IF NOT EXISTS custom_labels TEXT`,
	}

	// Performance indexes for faster queries
	performanceIndexes := []string{
		// Primary lookup indexes
		`CREATE INDEX IF NOT EXISTS idx_processing_rows_session_id ON processing_rows(session_id)`,
		`CREATE INDEX IF NOT EXISTS idx_processing_rows_session_row ON processing_rows(session_id, row_number)`,
		`CREATE INDEX IF NOT EXISTS idx_detected_entities_session_id ON detected_entities(session_id)`,
		`CREATE INDEX IF NOT EXISTS idx_detected_entities_session_row ON detected_entities(session_id, row_number)`,

		// History and results indexes
		`CREATE INDEX IF NOT EXISTS idx_processing_history_session_id ON processing_history(session_id)`,
		`CREATE INDEX IF NOT EXISTS idx_processing_history_timestamp ON processing_history(timestamp DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_processing_results_created_at ON processing_results(created_at DESC)`,

		// Query optimization indexes
		`CREATE INDEX IF NOT EXISTS idx_processing_rows_status ON processing_rows(status)`,
		`CREATE INDEX IF NOT EXISTS idx_detected_entities_type ON detected_entities(entity_type)`,
		`CREATE INDEX IF NOT EXISTS idx_detected_entities_approved ON detected_entities(approved)`,
		`CREATE INDEX IF NOT EXISTS idx_processing_history_filename ON processing_history(filename)`,

		// Composite indexes for complex queries
		`CREATE INDEX IF NOT EXISTS idx_processing_rows_session_status ON processing_rows(session_id, status)`,
		`CREATE INDEX IF NOT EXISTS idx_detected_entities_session_type ON detected_entities(session_id, entity_type)`,
		`CREATE INDEX IF NOT EXISTS idx_detected_entities_session_approved ON detected_entities(session_id, approved)`,
	}

	for _, migration := range migrations {
		_, err := db.Exec(migration)
		if err != nil {
			logrus.WithError(err).Warnf("Migration failed (may already exist): %s", migration)
			// Don't return error as columns might already exist
		}
	}

	// Create performance indexes
	for _, index := range performanceIndexes {
		_, err := db.Exec(index)
		if err != nil {
			logrus.WithError(err).Warnf("Index creation failed (may already exist): %s", index)
			// Don't return error as indexes might already exist
		}
	}

	logrus.Info("📊 Database performance indexes created successfully")
	return nil
}
