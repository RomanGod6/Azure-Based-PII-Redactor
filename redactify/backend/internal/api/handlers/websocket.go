package handlers

import (
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow connections from any origin in development
	},
}

// ProcessingSession represents an active file processing session
type ProcessingSession struct {
	ID         string                 `json:"id"`
	Status     string                 `json:"status"`
	Progress   *ProgressUpdate        `json:"progress"`
	StartTime  time.Time              `json:"start_time"`
	LastUpdate time.Time              `json:"last_update"`
	FileName   string                 `json:"filename"`
	Connection *websocket.Conn        `json:"-"`
	Context    map[string]interface{} `json:"context"`
}

// WebSocketMessage represents messages sent over WebSocket
type WebSocketMessage struct {
	Type      string      `json:"type"`
	SessionID string      `json:"session_id,omitempty"`
	Data      interface{} `json:"data,omitempty"`
	Timestamp time.Time   `json:"timestamp"`
}

// WebSocketHandler manages WebSocket connections and processing sessions
type WebSocketHandler struct {
	sessions map[string]*ProcessingSession
	mutex    sync.RWMutex
}

// NewWebSocketHandler creates a new WebSocket handler
func NewWebSocketHandler() *WebSocketHandler {
	return &WebSocketHandler{
		sessions: make(map[string]*ProcessingSession),
		mutex:    sync.RWMutex{},
	}
}

// HandleWebSocket upgrades HTTP connection to WebSocket
func (wsh *WebSocketHandler) HandleWebSocket(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logrus.WithError(err).Error("Failed to upgrade to WebSocket")
		return
	}
	defer conn.Close()

	logrus.Info("🔌 New WebSocket connection established")

	// Handle incoming messages
	for {
		var msg WebSocketMessage
		err := conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				logrus.WithError(err).Error("WebSocket error")
			}
			break
		}

		// Process different message types
		switch msg.Type {
		case "subscribe":
			wsh.handleSubscribe(conn, &msg)
		case "get_active_sessions":
			wsh.handleGetActiveSessions(conn)
		case "get_session_details":
			wsh.handleGetSessionDetails(conn, &msg)
		case "ping":
			wsh.sendMessage(conn, WebSocketMessage{
				Type:      "pong",
				Timestamp: time.Now(),
			})
		default:
			logrus.Warnf("Unknown WebSocket message type: %s", msg.Type)
		}
	}

	logrus.Info("🔌 WebSocket connection closed")
}

// CreateSession creates a new processing session
func (wsh *WebSocketHandler) CreateSession(filename string, conn *websocket.Conn) string {
	wsh.mutex.Lock()
	defer wsh.mutex.Unlock()

	sessionID := generateSessionID()
	session := &ProcessingSession{
		ID:         sessionID,
		Status:     "starting",
		StartTime:  time.Now(),
		LastUpdate: time.Now(),
		FileName:   filename,
		Connection: conn,
		Context:    make(map[string]interface{}),
	}

	wsh.sessions[sessionID] = session
	logrus.Infof("📝 Created processing session: %s for file: %s", sessionID, filename)

	// Notify client of session creation
	wsh.sendMessage(conn, WebSocketMessage{
		Type:      "session_created",
		SessionID: sessionID,
		Data:      session,
		Timestamp: time.Now(),
	})

	return sessionID
}

// UpdateProgress updates the progress of a processing session
func (wsh *WebSocketHandler) UpdateProgress(sessionID string, progress *ProgressUpdate) {
	wsh.mutex.Lock()
	session, exists := wsh.sessions[sessionID]
	if !exists {
		wsh.mutex.Unlock()
		logrus.Warnf("Attempt to update non-existent session: %s", sessionID)
		return
	}

	session.Progress = progress
	session.LastUpdate = time.Now()
	session.Status = progress.Status
	wsh.mutex.Unlock()

	// Send progress update to client
	if session.Connection != nil {
		wsh.sendMessage(session.Connection, WebSocketMessage{
			Type:      "progress_update",
			SessionID: sessionID,
			Data:      progress,
			Timestamp: time.Now(),
		})
	}
}

// CompleteSession marks a session as completed
func (wsh *WebSocketHandler) CompleteSession(sessionID string, results interface{}) {
	wsh.mutex.Lock()
	session, exists := wsh.sessions[sessionID]
	if !exists {
		wsh.mutex.Unlock()
		logrus.Warnf("Attempt to complete non-existent session: %s", sessionID)
		return
	}

	session.Status = "completed"
	session.LastUpdate = time.Now()
	wsh.mutex.Unlock()

	// Send completion notification
	if session.Connection != nil {
		wsh.sendMessage(session.Connection, WebSocketMessage{
			Type:      "session_completed",
			SessionID: sessionID,
			Data:      results,
			Timestamp: time.Now(),
		})
	}

	logrus.Infof("✅ Session completed: %s", sessionID)
}

// NotifyRateLimit notifies about rate limiting
func (wsh *WebSocketHandler) NotifyRateLimit(sessionID string, retryAfter time.Duration) {
	wsh.mutex.RLock()
	session, exists := wsh.sessions[sessionID]
	wsh.mutex.RUnlock()

	if !exists || session.Connection == nil {
		return
	}

	rateLimitInfo := map[string]interface{}{
		"message":     "Rate limit reached. Waiting before retry...",
		"retry_after": retryAfter.Seconds(),
		"status":      "rate_limited",
	}

	wsh.sendMessage(session.Connection, WebSocketMessage{
		Type:      "rate_limit",
		SessionID: sessionID,
		Data:      rateLimitInfo,
		Timestamp: time.Now(),
	})

	logrus.Warnf("⚠️ Rate limit notification sent for session: %s", sessionID)
}

// NotifyError sends error notification
func (wsh *WebSocketHandler) NotifyError(sessionID string, err error) {
	wsh.mutex.Lock()
	session, exists := wsh.sessions[sessionID]
	if exists {
		session.Status = "error"
		session.LastUpdate = time.Now()
	}
	wsh.mutex.Unlock()

	if !exists || session.Connection == nil {
		return
	}

	errorInfo := map[string]interface{}{
		"message": err.Error(),
		"status":  "error",
	}

	wsh.sendMessage(session.Connection, WebSocketMessage{
		Type:      "error",
		SessionID: sessionID,
		Data:      errorInfo,
		Timestamp: time.Now(),
	})

	logrus.Errorf("❌ Error notification sent for session %s: %v", sessionID, err)
}

// handleSubscribe handles subscription to a specific session
func (wsh *WebSocketHandler) handleSubscribe(conn *websocket.Conn, msg *WebSocketMessage) {
	sessionID, ok := msg.Data.(string)
	if !ok {
		logrus.Error("Invalid session ID in subscribe message")
		return
	}

	wsh.mutex.Lock()
	session, exists := wsh.sessions[sessionID]
	if exists {
		session.Connection = conn // Update connection
	}
	wsh.mutex.Unlock()

	if exists {
		// Send current session state
		wsh.sendMessage(conn, WebSocketMessage{
			Type:      "session_state",
			SessionID: sessionID,
			Data:      session,
			Timestamp: time.Now(),
		})
		logrus.Infof("🔗 Client subscribed to session: %s", sessionID)
	} else {
		wsh.sendMessage(conn, WebSocketMessage{
			Type:      "error",
			Data:      map[string]string{"message": "Session not found"},
			Timestamp: time.Now(),
		})
	}
}

// handleGetActiveSessions returns all active sessions
func (wsh *WebSocketHandler) handleGetActiveSessions(conn *websocket.Conn) {
	wsh.mutex.RLock()
	activeSessions := make([]*ProcessingSession, 0)
	for _, session := range wsh.sessions {
		if session.Status == "processing" || session.Status == "starting" {
			activeSessions = append(activeSessions, session)
		}
	}
	wsh.mutex.RUnlock()

	wsh.sendMessage(conn, WebSocketMessage{
		Type:      "active_sessions",
		Data:      activeSessions,
		Timestamp: time.Now(),
	})
}

// handleGetSessionDetails returns details for a specific session
func (wsh *WebSocketHandler) handleGetSessionDetails(conn *websocket.Conn, msg *WebSocketMessage) {
	sessionID, ok := msg.Data.(string)
	if !ok {
		wsh.sendMessage(conn, WebSocketMessage{
			Type:      "error",
			Data:      map[string]string{"message": "Invalid session ID"},
			Timestamp: time.Now(),
		})
		return
	}

	wsh.mutex.RLock()
	session, exists := wsh.sessions[sessionID]
	wsh.mutex.RUnlock()

	if exists {
		wsh.sendMessage(conn, WebSocketMessage{
			Type:      "session_details",
			SessionID: sessionID,
			Data:      session,
			Timestamp: time.Now(),
		})
	} else {
		wsh.sendMessage(conn, WebSocketMessage{
			Type:      "error",
			Data:      map[string]string{"message": "Session not found"},
			Timestamp: time.Now(),
		})
	}
}

// sendMessage sends a WebSocket message to the client
func (wsh *WebSocketHandler) sendMessage(conn *websocket.Conn, msg WebSocketMessage) {
	if conn == nil {
		return
	}

	if err := conn.WriteJSON(msg); err != nil {
		logrus.WithError(err).Error("Failed to send WebSocket message")
	}
}

// CleanupOldSessions removes sessions older than 24 hours
func (wsh *WebSocketHandler) CleanupOldSessions() {
	wsh.mutex.Lock()
	defer wsh.mutex.Unlock()

	cutoff := time.Now().Add(-24 * time.Hour)
	for sessionID, session := range wsh.sessions {
		if session.LastUpdate.Before(cutoff) {
			delete(wsh.sessions, sessionID)
			logrus.Infof("🧹 Cleaned up old session: %s", sessionID)
		}
	}
}

// generateSessionID generates a unique session ID
func generateSessionID() string {
	return time.Now().Format("20060102150405") + "-" + randomString(8)
}

func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(b)
}
