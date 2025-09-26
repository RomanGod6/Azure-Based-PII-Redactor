/**
 * Session Persistence Service
 * Provides comprehensive session management and crash recovery capabilities
 */

export interface SessionState {
  id: string;
  type: 'file_processing' | 'text_processing' | 'results_viewing';
  status: 'active' | 'completed' | 'error' | 'cancelled';
  progress: number;
  timestamp: number;
  lastUpdate: number;

  // Processing data
  fileName?: string;
  fileSize?: number;
  fileType?: string;
  processingOptions?: any;

  // Results data
  sessionId?: string;
  resultId?: string;
  entitiesCount?: number;

  // Error recovery data
  errorMessage?: string;
  retryCount?: number;

  // Additional metadata
  metadata?: Record<string, any>;
}

export interface CrashReport {
  id: string;
  timestamp: number;
  sessionStates: SessionState[];
  userAgent: string;
  url: string;
  error?: string;
}

class SessionPersistenceService {
  private static instance: SessionPersistenceService;
  private readonly storageKey = 'pii_redactor_sessions';
  private readonly crashReportKey = 'pii_redactor_crash_reports';
  private readonly maxSessions = 10;
  private readonly maxCrashReports = 5;
  private readonly sessionTimeout = 24 * 60 * 60 * 1000; // 24 hours

  public static getInstance(): SessionPersistenceService {
    if (!SessionPersistenceService.instance) {
      SessionPersistenceService.instance = new SessionPersistenceService();
    }
    return SessionPersistenceService.instance;
  }

  private constructor() {
    // Set up crash detection
    this.setupCrashDetection();

    // Clean up old sessions on init
    this.cleanupExpiredSessions();
  }

  private setupCrashDetection(): void {
    // Detect page unload
    window.addEventListener('beforeunload', () => {
      this.recordGracefulExit();
    });

    // Detect crashes on page load
    window.addEventListener('load', () => {
      this.checkForCrashes();
    });

    // Periodic session updates
    setInterval(() => {
      this.updateActiveSessionTimestamps();
    }, 30000); // Every 30 seconds
  }

  private recordGracefulExit(): void {
    try {
      localStorage.setItem('pii_graceful_exit', Date.now().toString());
    } catch (error) {
      console.warn('Failed to record graceful exit:', error);
    }
  }

  private checkForCrashes(): void {
    try {
      const gracefulExit = localStorage.getItem('pii_graceful_exit');
      const sessions = this.getAllSessions();
      const activeSessions = sessions.filter(s => s.status === 'active');

      if (activeSessions.length > 0 && (!gracefulExit || Date.now() - parseInt(gracefulExit) > 60000)) {
        // Potential crash detected
        this.recordCrashReport(activeSessions, 'Potential crash detected - active sessions found without graceful exit');
        this.offerSessionRecovery(activeSessions);
      }

      // Clear graceful exit flag
      localStorage.removeItem('pii_graceful_exit');
    } catch (error) {
      console.warn('Failed to check for crashes:', error);
    }
  }

  private offerSessionRecovery(sessions: SessionState[]): void {
    if (sessions.length === 0) return;

    const sessionDescriptions = sessions.map(s =>
      `• ${s.fileName || 'Text processing'} (${s.progress.toFixed(1)}% complete)`
    ).join('\n');

    const shouldRecover = window.confirm(
      `It looks like your previous session was interrupted. Would you like to recover the following sessions?\n\n${sessionDescriptions}\n\nClick OK to recover or Cancel to start fresh.`
    );

    if (shouldRecover) {
      // Store recovery flag for components to check
      localStorage.setItem('pii_recovery_sessions', JSON.stringify(sessions));

      // Redirect to appropriate page based on session type
      const primarySession = sessions[0];
      if (primarySession.type === 'results_viewing' && primarySession.sessionId) {
        window.location.href = `/results/${primarySession.sessionId}?mode=review&recovered=true`;
      } else {
        // Go to main processor page
        window.location.href = '/?recovered=true';
      }
    } else {
      // Mark sessions as cancelled
      sessions.forEach(session => {
        this.updateSession(session.id, { status: 'cancelled' });
      });
    }
  }

  private recordCrashReport(sessions: SessionState[], error?: string): void {
    try {
      const crashReport: CrashReport = {
        id: this.generateId(),
        timestamp: Date.now(),
        sessionStates: sessions,
        userAgent: navigator.userAgent,
        url: window.location.href,
        error
      };

      const reports = this.getCrashReports();
      reports.unshift(crashReport);

      // Keep only the most recent crash reports
      const trimmedReports = reports.slice(0, this.maxCrashReports);

      localStorage.setItem(this.crashReportKey, JSON.stringify(trimmedReports));
    } catch (error) {
      console.warn('Failed to record crash report:', error);
    }
  }

  public saveSession(session: SessionState): void {
    try {
      const sessions = this.getAllSessions();
      const existingIndex = sessions.findIndex(s => s.id === session.id);

      const updatedSession = {
        ...session,
        lastUpdate: Date.now()
      };

      if (existingIndex >= 0) {
        sessions[existingIndex] = updatedSession;
      } else {
        sessions.unshift(updatedSession);
      }

      // Keep only the most recent sessions
      const trimmedSessions = sessions.slice(0, this.maxSessions);

      localStorage.setItem(this.storageKey, JSON.stringify(trimmedSessions));
    } catch (error) {
      console.warn('Failed to save session:', error);
    }
  }

  public getSession(sessionId: string): SessionState | null {
    try {
      const sessions = this.getAllSessions();
      return sessions.find(s => s.id === sessionId) || null;
    } catch (error) {
      console.warn('Failed to get session:', error);
      return null;
    }
  }

  public getAllSessions(): SessionState[] {
    try {
      const stored = localStorage.getItem(this.storageKey);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.warn('Failed to load sessions:', error);
      return [];
    }
  }

  public getActiveSessions(): SessionState[] {
    return this.getAllSessions().filter(s => s.status === 'active');
  }

  public updateSession(sessionId: string, updates: Partial<SessionState>): void {
    const session = this.getSession(sessionId);
    if (session) {
      const updatedSession = { ...session, ...updates, lastUpdate: Date.now() };
      this.saveSession(updatedSession);
    }
  }

  public deleteSession(sessionId: string): void {
    try {
      const sessions = this.getAllSessions().filter(s => s.id !== sessionId);
      localStorage.setItem(this.storageKey, JSON.stringify(sessions));
    } catch (error) {
      console.warn('Failed to delete session:', error);
    }
  }

  public getCrashReports(): CrashReport[] {
    try {
      const stored = localStorage.getItem(this.crashReportKey);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.warn('Failed to load crash reports:', error);
      return [];
    }
  }

  public clearCrashReports(): void {
    try {
      localStorage.removeItem(this.crashReportKey);
    } catch (error) {
      console.warn('Failed to clear crash reports:', error);
    }
  }

  public getRecoverySessions(): SessionState[] {
    try {
      const stored = localStorage.getItem('pii_recovery_sessions');
      if (stored) {
        localStorage.removeItem('pii_recovery_sessions');
        return JSON.parse(stored);
      }
      return [];
    } catch (error) {
      console.warn('Failed to get recovery sessions:', error);
      return [];
    }
  }

  private updateActiveSessionTimestamps(): void {
    const activeSessions = this.getActiveSessions();
    activeSessions.forEach(session => {
      this.updateSession(session.id, { lastUpdate: Date.now() });
    });
  }

  private cleanupExpiredSessions(): void {
    try {
      const sessions = this.getAllSessions();
      const cutoff = Date.now() - this.sessionTimeout;

      const validSessions = sessions.filter(s => s.lastUpdate > cutoff);

      if (validSessions.length !== sessions.length) {
        localStorage.setItem(this.storageKey, JSON.stringify(validSessions));
        console.log(`Cleaned up ${sessions.length - validSessions.length} expired sessions`);
      }
    } catch (error) {
      console.warn('Failed to cleanup expired sessions:', error);
    }
  }

  private generateId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Utility methods for components
  public createProcessingSession(fileName: string, fileSize: number, fileType: string, options: any): string {
    const sessionId = this.generateId();
    const session: SessionState = {
      id: sessionId,
      type: 'file_processing',
      status: 'active',
      progress: 0,
      timestamp: Date.now(),
      lastUpdate: Date.now(),
      fileName,
      fileSize,
      fileType,
      processingOptions: options
    };

    this.saveSession(session);
    return sessionId;
  }

  public createTextProcessingSession(options: any): string {
    const sessionId = this.generateId();
    const session: SessionState = {
      id: sessionId,
      type: 'text_processing',
      status: 'active',
      progress: 0,
      timestamp: Date.now(),
      lastUpdate: Date.now(),
      processingOptions: options
    };

    this.saveSession(session);
    return sessionId;
  }

  public createResultsViewingSession(sessionId: string, resultId?: string): string {
    const viewingSessionId = this.generateId();
    const session: SessionState = {
      id: viewingSessionId,
      type: 'results_viewing',
      status: 'active',
      progress: 100,
      timestamp: Date.now(),
      lastUpdate: Date.now(),
      sessionId,
      resultId
    };

    this.saveSession(session);
    return viewingSessionId;
  }
}

// Export singleton instance
export const sessionPersistence = SessionPersistenceService.getInstance();