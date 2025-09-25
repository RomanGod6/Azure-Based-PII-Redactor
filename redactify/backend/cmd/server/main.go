package main

import (
	"fmt"
	"log"
	"redactify/internal/api"
	"redactify/internal/db"
	"redactify/pkg/config"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/sirupsen/logrus"
)

func main() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		logrus.Warn("No .env file found")
	}

	// Initialize configuration
	cfg := config.New()

	// Initialize database
	database, err := db.Init(cfg.Database.URL)
	if err != nil {
		log.Fatal("Failed to initialize database:", err)
	}
	defer database.Close()

	// Set Gin mode
	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	// Initialize router
	router := gin.Default()

	// Setup API routes
	api.SetupRoutes(router, database, cfg)

	// Start server
	port := cfg.Server.Port
	if port == "" {
		port = "8080"
	}

	fmt.Printf("🚀 Redactify Backend Server starting on port %s\n", port)
	fmt.Printf("📊 Environment: %s\n", cfg.Environment)
	fmt.Printf("🔗 Health check: http://localhost:%s/health\n", port)

	if err := router.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}
