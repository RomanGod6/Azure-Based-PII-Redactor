package handlers

import (
	"context"
	"crypto/rand"
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
	ReadBufferSize:  1024 * 4, // 4KB read buffer
	WriteBufferSize: 1024 * 4, // 4KB write buffer
}

// ProcessingSession represents an active file processing session with enhanced buffering
type ProcessingSession struct {
	ID         string                 `json:"id"`
	Status     string                 `json:"status"`
	Progress   *ProgressUpdate        `json:"progress"`
	StartTime  time.Time              `json:"start_time"`
	LastUpdate time.Time              `json:"last_update"`
	FileName   string                 `json:"filename"`
	Connection *websocket.Conn        `json:"-"`
	Context    map[string]interface{} `json:"context"`

	// Enhanced buffering and connection management
	MessageBuffer   chan WebSocketMessage `json:"-"`
	IsConnected     bool                  `json:"-"`
	LastPong        time.Time             `json:"-"`
	BufferSize      int                   `json:"-"`
	DroppedMessages int64                 `json:"dropped_messages"`
	ctx             context.Context       `json:"-"`
	cancel          context.CancelFunc    `json:"-"`
	mu              sync.RWMutex          `json:"-"`
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

// CreateSession creates a new processing session with enhanced buffering
func (wsh *WebSocketHandler) CreateSession(filename string, conn *websocket.Conn) string {
	sessionID := generateSessionID()
	ctx, cancel := context.WithCancel(context.Background())

	session := &ProcessingSession{
		ID:              sessionID,
		Status:          "starting",
		StartTime:       time.Now(),
		LastUpdate:      time.Now(),
		FileName:        filename,
		Connection:      conn,
		Context:         make(map[string]interface{}),
		MessageBuffer:   make(chan WebSocketMessage, 1000), // Buffer up to 1000 messages
		IsConnected:     true,
		LastPong:        time.Now(),
		BufferSize:      1000,
		DroppedMessages: 0,
		ctx:             ctx,
		cancel:          cancel,
	}

	// Store session with lock
	wsh.mutex.Lock()
	wsh.sessions[sessionID] = session
	wsh.mutex.Unlock()

	// Start buffered message sender goroutine
	go wsh.messageBufferHandler(session)

	// Start connection health monitor
	go wsh.connectionHealthMonitor(session)

	logrus.Infof("📝 Created processing session: %s for file: %s with enhanced buffering", sessionID, filename)

	// Send session creation notification through buffer (outside of lock to prevent deadlock)
	wsh.sendBufferedMessage(sessionID, WebSocketMessage{
		Type:      "session_created",
		SessionID: sessionID,
		Data: map[string]interface{}{
			"id":         sessionID,
			"filename":   filename,
			"status":     "starting",
			"start_time": session.StartTime,
		},
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

	// Send progress update through buffer
	wsh.sendBufferedMessage(sessionID, WebSocketMessage{
		Type:      "progress_update",
		SessionID: sessionID,
		Data:      progress,
		Timestamp: time.Now(),
	})
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

	// Send completion notification through buffer
	wsh.sendBufferedMessage(sessionID, WebSocketMessage{
		Type:      "session_completed",
		SessionID: sessionID,
		Data:      results,
		Timestamp: time.Now(),
	})

	// Clean up session resources after a delay
	go func() {
		time.Sleep(5 * time.Minute) // Keep session alive for 5 minutes after completion
		wsh.cleanupSession(sessionID)
	}()

	logrus.Infof("✅ Session completed: %s", sessionID)
}

// NotifyRateLimit notifies about rate limiting
func (wsh *WebSocketHandler) NotifyRateLimit(sessionID string, retryAfter time.Duration) {
	wsh.mutex.RLock()
	_, exists := wsh.sessions[sessionID]
	wsh.mutex.RUnlock()

	if !exists {
		return
	}

	rateLimitInfo := map[string]interface{}{
		"message":     "Rate limit reached. Waiting before retry...",
		"retry_after": retryAfter.Seconds(),
		"status":      "rate_limited",
	}

	wsh.sendBufferedMessage(sessionID, WebSocketMessage{
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

	if !exists {
		return
	}

	errorInfo := map[string]interface{}{
		"message": err.Error(),
		"status":  "error",
	}

	wsh.sendBufferedMessage(sessionID, WebSocketMessage{
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

// cleanupSession properly cleans up a session and its resources
func (wsh *WebSocketHandler) cleanupSession(sessionID string) {
	wsh.mutex.Lock()
	session, exists := wsh.sessions[sessionID]
	if !exists {
		wsh.mutex.Unlock()
		return
	}

	// Cancel the session context to stop background goroutines
	if session.cancel != nil {
		session.cancel()
	}

	// Close the message buffer
	if session.MessageBuffer != nil {
		close(session.MessageBuffer)
	}

	// Close the WebSocket connection if still open
	if session.Connection != nil {
		session.Connection.Close()
	}

	// Remove from sessions map
	delete(wsh.sessions, sessionID)
	wsh.mutex.Unlock()

	logrus.Infof("🧹 Session cleaned up: %s", sessionID)
}

// generateSessionID generates a unique session ID
func generateSessionID() string {
	return time.Now().Format("20060102150405") + "-" + randomString(8)
}

func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	_, err := rand.Read(b)
	if err != nil {
		// Fallback to time-based generation
		for i := range b {
			b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
		}
	} else {
		for i := range b {
			b[i] = charset[b[i]%byte(len(charset))]
		}
	}
	return string(b)
}

// messageBufferHandler handles buffered message sending for a session
func (wsh *WebSocketHandler) messageBufferHandler(session *ProcessingSession) {
	defer func() {
		if r := recover(); r != nil {
			logrus.Errorf("Message buffer handler panic for session %s: %v", session.ID, r)
		}
	}()

	writeTimeout := 10 * time.Second
	ticker := time.NewTicker(30 * time.Second) // Heartbeat every 30 seconds
	defer ticker.Stop()

	for {
		select {
		case <-session.ctx.Done():
			logrus.Infof("🔌 Message buffer handler stopped for session: %s", session.ID)
			return

		case <-ticker.C:
			// Send heartbeat to check connection health
			if session.Connection != nil && session.IsConnected {
				session.Connection.SetWriteDeadline(time.Now().Add(writeTimeout))
				if err := session.Connection.WriteMessage(websocket.PingMessage, []byte{}); err != nil {
					session.mu.Lock()
					session.IsConnected = false
					session.mu.Unlock()
					logrus.Warnf("⚠️ Heartbeat failed for session %s: %v", session.ID, err)
				}
			}

		case msg, ok := <-session.MessageBuffer:
			if !ok {
				logrus.Infof("🔌 Message buffer closed for session: %s", session.ID)
				return
			}

			session.mu.RLock()
			conn := session.Connection
			isConnected := session.IsConnected
			session.mu.RUnlock()

			if !isConnected || conn == nil {
				// Connection lost, drop message but count it
				session.mu.Lock()
				session.DroppedMessages++
				session.mu.Unlock()
				continue
			}

			// Set write deadline
			conn.SetWriteDeadline(time.Now().Add(writeTimeout))

			// Send message
			if err := conn.WriteJSON(msg); err != nil {
				session.mu.Lock()
				session.IsConnected = false
				session.DroppedMessages++
				session.mu.Unlock()
				logrus.Warnf("⚠️ Failed to send buffered message for session %s: %v", session.ID, err)
			}
		}
	}
}

// connectionHealthMonitor monitors connection health and handles reconnection
func (wsh *WebSocketHandler) connectionHealthMonitor(session *ProcessingSession) {
	defer func() {
		if r := recover(); r != nil {
			logrus.Errorf("Connection health monitor panic for session %s: %v", session.ID, r)
		}
	}()

	ticker := time.NewTicker(60 * time.Second) // Check every minute
	defer ticker.Stop()

	pongTimeout := 90 * time.Second

	for {
		select {
		case <-session.ctx.Done():
			logrus.Infof("🔌 Connection health monitor stopped for session: %s", session.ID)
			return

		case <-ticker.C:
			session.mu.RLock()
			lastPong := session.LastPong
			isConnected := session.IsConnected
			session.mu.RUnlock()

			if isConnected && time.Since(lastPong) > pongTimeout {
				session.mu.Lock()
				session.IsConnected = false
				session.mu.Unlock()
				logrus.Warnf("⚠️ Connection timeout for session %s (no pong received)", session.ID)

				// Try to clean up connection
				if session.Connection != nil {
					session.Connection.Close()
				}
			}

			// Log connection stats periodically
			session.mu.RLock()
			droppedCount := session.DroppedMessages
			bufferLen := len(session.MessageBuffer)
			session.mu.RUnlock()

			if droppedCount > 0 || bufferLen > session.BufferSize/2 {
				logrus.WithFields(logrus.Fields{
					"session_id":       session.ID,
					"dropped_messages": droppedCount,
					"buffer_length":    bufferLen,
					"buffer_capacity":  session.BufferSize,
					"is_connected":     isConnected,
				}).Info("📊 WebSocket session stats")
			}
		}
	}
}

// sendBufferedMessage sends a message through the buffer with backpressure handling
func (wsh *WebSocketHandler) sendBufferedMessage(sessionID string, msg WebSocketMessage) {
	wsh.mutex.RLock()
	session, exists := wsh.sessions[sessionID]
	wsh.mutex.RUnlock()

	if !exists {
		logrus.Warnf("Attempt to send message to non-existent session: %s", sessionID)
		return
	}
	select {
	case session.MessageBuffer <- msg:
		// Message successfully buffered
	default:
		// Buffer is full, implement backpressure
		session.mu.Lock()
		session.DroppedMessages++
		droppedCount := session.DroppedMessages
		session.mu.Unlock()

		logrus.Warnf("⚠️ Message buffer full for session %s, dropped message (total dropped: %d)", sessionID, droppedCount)

		// Try to remove oldest message and add new one
		select {
		case <-session.MessageBuffer:
			// Successfully removed old message, try again
			select {
			case session.MessageBuffer <- msg:
				logrus.Infof("📤 Recovered from buffer overflow for session %s", sessionID)
			default:
				// Still full, give up
				session.mu.Lock()
				session.DroppedMessages++
				session.mu.Unlock()
			}
		default:
			// Buffer is completely stuck, connection likely dead
			session.mu.Lock()
			session.IsConnected = false
			session.mu.Unlock()
		}
	}
}
