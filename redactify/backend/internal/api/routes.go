package api

import (
	"database/sql"
	"net/http"
	"redactify/internal/api/handlers"
	"redactify/pkg/config"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func SetupRoutes(router *gin.Engine, db *sql.DB, cfg *config.Config) {
	// CORS middleware
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowOrigins = []string{"http://localhost:3000", "file://*"}
	corsConfig.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
	corsConfig.AllowHeaders = []string{"Origin", "Content-Length", "Content-Type", "Authorization"}
	corsConfig.AllowCredentials = true
	router.Use(cors.New(corsConfig))

	// Initialize handlers
	wsHandler := handlers.NewWebSocketHandler()
	piiHandler := handlers.NewPIIHandler(db, cfg)
	fileHandler := handlers.NewFileHandler(db, cfg, wsHandler)
	healthHandler := handlers.NewHealthHandler()

	// WebSocket endpoints
	router.GET("/ws", wsHandler.HandleWebSocket)
	router.GET("/ws/process", fileHandler.ProcessFileWebSocket)

	// Health check (outside of versioned API) - support both GET and HEAD
	router.GET("/health", healthHandler.Health)
	router.HEAD("/health", healthHandler.Health)

	// API version 1
	v1 := router.Group("/api/v1")
	{

		// PII detection and redaction
		v1.POST("/pii/detect", piiHandler.DetectPII)
		v1.POST("/pii/redact", piiHandler.RedactPII)
		v1.POST("/pii/batch", piiHandler.BatchProcess)
		v1.POST("/pii/feedback", piiHandler.SubmitFeedback)

		// File processing
		v1.POST("/files/upload", fileHandler.UploadFile)
		v1.POST("/files/process", fileHandler.ProcessFile)
		v1.GET("/files/download/:id", fileHandler.DownloadFile)
		v1.GET("/files/status/:id", fileHandler.GetFileStatus)
		v1.GET("/files/results/:id", fileHandler.GetProcessingResults)
		v1.GET("/files/stream/:session_id", fileHandler.StreamProcessingResults)
		v1.GET("/files/results/view/:session_id", fileHandler.GetProcessingRowsForViewer)
		v1.GET("/files/sessions/:session_id/detail", fileHandler.GetSessionReviewData)
		v1.POST("/files/sessions/:session_id/export", fileHandler.ExportSessionResults)
		v1.GET("/files/legacy/:result_id", fileHandler.DownloadLegacyResults)

		// History and analytics
		v1.GET("/history", piiHandler.GetHistory)
		v1.GET("/analytics", piiHandler.GetAnalytics)

		// Configuration
		v1.GET("/config", piiHandler.GetConfig)
		v1.PUT("/config", piiHandler.UpdateConfig)

		// Training
		v1.GET("/training/stats", piiHandler.GetTrainingStats)
	}

	// Static files (for development)
	router.Static("/static", "./static")

	// Fallback for frontend routing
	router.NoRoute(func(c *gin.Context) {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "Route not found",
			"path":  c.Request.URL.Path,
		})
	})
}
