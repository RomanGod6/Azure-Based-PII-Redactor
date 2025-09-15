package handlers

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type HealthHandler struct {
	startTime time.Time
}

func NewHealthHandler() *HealthHandler {
	return &HealthHandler{
		startTime: time.Now(),
	}
}

func (h *HealthHandler) Health(c *gin.Context) {
	uptime := time.Since(h.startTime)
	
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"uptime":    uptime.String(),
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"version":   "1.0.0",
		"service":   "redactify-backend",
	})
}