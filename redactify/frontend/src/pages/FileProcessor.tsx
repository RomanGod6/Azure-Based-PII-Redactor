import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { FeedbackTrainer } from '../components/FeedbackTrainer';
import { PerformanceDashboard } from '../components/PerformanceDashboard';
import { fileProcessingWebSocket } from '../services/websocket';
import { sessionPersistence, SessionState } from '../services/sessionPersistence';
import { performanceMonitor } from '../services/performanceMonitor';

interface ProcessingOptions {
  redactionMode: 'replace' | 'mask' | 'remove';
  detectionMode: 'aggressive' | 'balanced' | 'conservative';
  customLabels: Record<string, string>;
  useTraining: boolean;
}

interface DetectedEntity {
  type: string;
  text: string;
  start: number;
  end: number;
  confidence: number;
  category: string;
  approved?: boolean;
}

interface ProcessingResult {
  original_text: string;
  redacted_text: string;
  entities?: DetectedEntity[];
  redacted_count: number;
  process_time: string;
  rows_processed?: number;
  file_name?: string;
  result_id?: string;
}

export const FileProcessor: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [results, setResults] = useState<ProcessingResult | null>(null);
  const [entities, setEntities] = useState<DetectedEntity[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [step, setStep] = useState<'upload' | 'options' | 'processing' | 'review' | 'complete'>('upload');
  const [options, setOptions] = useState<ProcessingOptions>({
    redactionMode: 'replace',
    detectionMode: 'balanced',
    customLabels: {},
    useTraining: true
  });
  const [trainingStats, setTrainingStats] = useState<{count: number, entityTypes: string[]} | null>(null);
  const [textInput, setTextInput] = useState('');
  const [processingMode, setProcessingMode] = useState<'file' | 'text'>('file');
  const [showTrainer, setShowTrainer] = useState(false);
  const [originalContent, setOriginalContent] = useState('');

  // Filtering and pagination state
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedEntityTypes, setSelectedEntityTypes] = useState<Set<string>>(new Set<string>());
  const [confidenceThreshold, setConfidenceThreshold] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(50);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const entitiesBufferRef = useRef<DetectedEntity[]>([]);

  // Error recovery and progressive loading state
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [isRetrying, setIsRetrying] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [showProgressDetails, setShowProgressDetails] = useState(false);
  const [recoveredSession, setRecoveredSession] = useState<SessionState | null>(null);
  const [showPerformanceDashboard, setShowPerformanceDashboard] = useState(false);

  // Performance monitoring state
  const [processingStartTime, setProcessingStartTime] = useState<number | null>(null);

  // Fetch training statistics when component mounts or step changes to options
  useEffect(() => {
    if (step === 'options') {
      fetchTrainingStats();
    }
  }, [step]);

  const fetchTrainingStats = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/v1/training/stats');
      if (response.ok) {
        const data = await response.json();
        setTrainingStats({
          count: data.total_feedback || 0,
          entityTypes: data.entity_types || []
        });
      }
    } catch (error) {
      console.error('Failed to fetch training stats:', error);
      setTrainingStats({ count: 0, entityTypes: [] });
    }
  };

  // Enhanced session management with crash recovery
  const createProcessingSession = useCallback((file: File | null, textContent: string) => {
    try {
      let sessionId: string;

      if (file) {
        sessionId = sessionPersistence.createProcessingSession(
          file.name,
          file.size,
          file.type,
          options
        );
      } else {
        sessionId = sessionPersistence.createTextProcessingSession(options);
      }

      setCurrentSessionId(sessionId);
      return sessionId;
    } catch (error) {
      console.warn('Failed to create processing session:', error);
      return null;
    }
  }, [options]);

  const updateSessionProgress = useCallback((progress: number, entitiesCount?: number, processingSessionId?: string) => {
    if (currentSessionId) {
      try {
        sessionPersistence.updateSession(currentSessionId, {
          progress,
          entitiesCount,
          sessionId: processingSessionId,
          status: progress >= 100 ? 'completed' : 'active',
          lastUpdate: Date.now()
        });
      } catch (error) {
        console.warn('Failed to update session progress:', error);
      }
    }
  }, [currentSessionId]);

  const handleSessionError = useCallback((errorMessage: string) => {
    if (currentSessionId) {
      try {
        sessionPersistence.updateSession(currentSessionId, {
          status: 'error',
          errorMessage,
          retryCount,
          lastUpdate: Date.now()
        });
      } catch (error) {
        console.warn('Failed to update session error:', error);
      }
    }
  }, [currentSessionId, retryCount]);

  const clearCurrentSession = useCallback(() => {
    if (currentSessionId) {
      try {
        sessionPersistence.updateSession(currentSessionId, {
          status: 'completed',
          lastUpdate: Date.now()
        });
        setCurrentSessionId(null);
      } catch (error) {
        console.warn('Failed to clear current session:', error);
      }
    }
  }, [currentSessionId]);

  const handleError = useCallback((error: string, canRetry: boolean = true) => {
    console.error('Processing error:', error);
    setError(error);
    setStep('processing'); // Stay on processing step to show error

    // Update session with error
    handleSessionError(error);

    // Record error in performance monitoring
    performanceMonitor.recordError(currentSessionId || undefined);

    if (canRetry && retryCount < 3) {
      // Auto-retry after a delay for network errors
      setTimeout(() => {
        if (error.includes('network') || error.includes('connection') || error.includes('timeout')) {
          handleRetry();
        }
      }, 5000 * (retryCount + 1)); // Exponential backoff
    }
  }, [retryCount, handleSessionError, currentSessionId]);

  const handleRetry = useCallback(async () => {
    if (isRetrying) return;

    setIsRetrying(true);
    setError(null);
    setRetryCount(prev => prev + 1);

    try {
      // Wait a bit before retrying
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Retry the processing
      if (file && processingMode === 'file') {
        await processFile();
      } else if (textInput && processingMode === 'text') {
        await processFile();
      }
    } catch (error) {
      handleError(error instanceof Error ? error.message : 'Retry failed');
    } finally {
      setIsRetrying(false);
    }
  }, [isRetrying, file, processingMode, textInput]);

  // Check for recovery sessions on component mount
  useEffect(() => {
    const recoverySessions = sessionPersistence.getRecoverySessions();
    const processingSession = recoverySessions.find(s =>
      s.type === 'file_processing' || s.type === 'text_processing'
    );

    if (processingSession && processingSession.progress > 0 && processingSession.progress < 100) {
      setRecoveredSession(processingSession);
      setCurrentSessionId(processingSession.id);

      // Try to restore basic state
      if (processingSession.progress > 0) {
        setProgress(processingSession.progress);
        setProgressMessage(`Recovered session - ${processingSession.progress.toFixed(1)}% complete`);
      }

      // If session was viewing results, redirect
      if (processingSession.sessionId && processingSession.progress >= 100) {
        window.location.href = `/results/${processingSession.sessionId}?mode=review&recovered=true`;
      } else {
        // Show recovery UI
        setStep('processing');
      }
    }

    // Also check for URL recovery parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('recovered') === 'true') {
      // User was redirected here for recovery
      const activeSessions = sessionPersistence.getActiveSessions();
      const activeProcessing = activeSessions.find(s =>
        s.type === 'file_processing' || s.type === 'text_processing'
      );
      if (activeProcessing) {
        setRecoveredSession(activeProcessing);
        setCurrentSessionId(activeProcessing.id);
      }
    }
  }, []);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setStep('options');
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setStep('options');
    }
  };

  const processFile = async () => {
    if (!file && !textInput) return;

    // Clear any previous errors
    setError(null);
    setRetryCount(0);

    setStep('processing');
    setProgress(0);
    setProgressMessage('Initializing...');
    entitiesBufferRef.current = [];
    setEntities([]);

    // Create new processing session
    const sessionId = createProcessingSession(file, textInput);
    if (!sessionId) {
      handleError('Failed to create processing session', false);
      return;
    }

    // Start performance monitoring
    setProcessingStartTime(Date.now());
    performanceMonitor.startMonitoring();

    try {
      if (processingMode === 'text') {
        setProgressMessage('Processing text...');

        // Process text directly using existing API
        const response = await fetch('http://localhost:8080/api/v1/pii/redact', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: textInput,
            options: {
              redaction_mode: options.redactionMode,
              custom_labels: options.customLabels,
              preserve_cases: false,
              use_training: options.useTraining
            }
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.error || `Text processing failed: ${response.status}`);
        }

        const data = await response.json();
        setResults(data);
        setOriginalContent(data.original_text || textInput);

        if (data.entities) {
          data.entities.forEach((entity: DetectedEntity) => {
            entity.approved = true;
          });
          setEntities(data.entities);
          // Update session with completion
          updateSessionProgress(100, data.entities.length);
        }

        setStep('review');

        // Record performance metrics
        if (processingStartTime) {
          const processingTime = Date.now() - processingStartTime;
          performanceMonitor.recordProcessingMetrics(
            processingTime,
            1, // Single text processing counts as 1 row
            data.entities?.length || 0,
            textInput.length,
            sessionId
          );
        }

        clearCurrentSession(); // Clear session on successful completion
      } else {
        // Process file using WebSocket
        console.log('🚀 Frontend: Processing file with WebSocket', {
          fileName: file!.name,
          fileSize: file!.size,
          options: options
        });

        // Convert file to base64 for WebSocket transmission
        const fileContent = await fileToBase64(file!);
        
        // Set up WebSocket handlers with progressive state saving
        fileProcessingWebSocket.setProgressHandler((progressData: any) => {
          console.log('📊 Progress update:', progressData);

          // Update entities buffer
          if (Array.isArray(progressData.entities) && progressData.entities.length > 0) {
            const chunk = progressData.entities as DetectedEntity[];
            chunk.forEach((entity) => {
              entity.approved = true;
            });
            entitiesBufferRef.current.push(...chunk);
          }

          // Update progress
          let currentProgress = progress;
          if (progressData.total_rows > 0) {
            currentProgress = Math.min(100, (progressData.current_row / progressData.total_rows) * 100);
            setProgress(currentProgress);
          }

          // Save progress state periodically (every 5% or 100 rows)
          if (progressData.current_row % 100 === 0 || currentProgress % 5 < 1) {
            updateSessionProgress(currentProgress, entitiesBufferRef.current.length, progressData.session_id || null);
          }

          // Update status message
          if (progressData.message) {
            console.log(`🔄 ${progressData.message}`);
            setProgressMessage(progressData.message);
          }
        });

        fileProcessingWebSocket.setCompleteHandler(async (resultData: any) => {
          console.log('✅ Processing completed!', resultData);
          setProgress(100);

          const parsedResults = resultData as ProcessingResult;
          
          // Check if we have a result_id (lightweight response)
          if (parsedResults.result_id && (!parsedResults.original_text || !parsedResults.redacted_text)) {
            console.log('📥 Fetching full results from backend...', parsedResults.result_id);
            
            try {
              // Fetch full results from backend
              const response = await fetch(`http://localhost:8080/api/v1/files/results/${parsedResults.result_id}`);
              if (!response.ok) {
                throw new Error(`Failed to fetch results: ${response.statusText}`);
              }
              
              const fullResults = await response.json();
              console.log('📋 Full results retrieved', { 
                id: fullResults.id, 
                filename: fullResults.filename,
                content_size: fullResults.original_text?.length || 0,
                entities_found: fullResults.entities_found 
              });
              
              // Use full results but keep the lightweight metadata
              const cleanedResults: ProcessingResult = {
                ...parsedResults,
                original_text: fullResults.original_text,
                redacted_text: fullResults.redacted_text,
                entities: undefined,
              };
              
              setResults(cleanedResults);
              setOriginalContent(fullResults.original_text || '');
              
            } catch (error) {
              console.error('❌ Failed to fetch full results:', error);
              alert('Processing completed but failed to retrieve full results. Please try again.');
              setStep('options');
              return;
            }
          } else {
            // Legacy response with full content
            const cleanedResults: ProcessingResult = {
              ...parsedResults,
              entities: undefined,
            };
            setResults(cleanedResults);
            setOriginalContent(cleanedResults.original_text || '');
          }

          // Set final entities
          const finalEntities = [...entitiesBufferRef.current];
          setEntities(finalEntities);
          entitiesBufferRef.current = [];
          setSearchQuery('');
          setSelectedEntityTypes(new Set<string>());
          setConfidenceThreshold(0);
          setCurrentPage(1);
          setStep('review');

          // Record performance metrics for file processing
          if (processingStartTime && parsedResults.rows_processed) {
            const processingTime = Date.now() - processingStartTime;
            performanceMonitor.recordProcessingMetrics(
              processingTime,
              parsedResults.rows_processed,
              finalEntities.length,
              file?.size,
              sessionId
            );
          }

          // Clear session on successful completion
          clearCurrentSession();
        });

        fileProcessingWebSocket.setErrorHandler((error: any) => {
          console.error('❌ Processing error:', error);
          const errorMessage = error.message || 'Processing failed due to unknown error';
          handleError(errorMessage, true); // Enable retry for WebSocket errors
        });

        fileProcessingWebSocket.setRateLimitHandler((rateLimitInfo: any) => {
          console.warn('⚠️ Rate limit hit:', rateLimitInfo);
          setProgressMessage(`Rate limit reached. Retrying in ${rateLimitInfo.retry_after}s...`);
        });

        // Start WebSocket processing
        await fileProcessingWebSocket.processFile(
          file!.name,
          fileContent,
          file!.type,
          {
            redaction_mode: options.redactionMode,
            custom_labels: options.customLabels,
            preserve_cases: false,
            use_training: options.useTraining,
            detection_mode: options.detectionMode
          }
        );
      }
    } catch (error) {
      console.error('Processing error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Processing failed';
      handleError(errorMessage, true);
    }
  };

  // Helper function to convert file to base64
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          // Remove data URL prefix (data:mime/type;base64,)
          const base64 = reader.result.split(',')[1];
          resolve(base64);
        } else {
          reject(new Error('Failed to convert file to base64'));
        }
      };
      reader.onerror = error => reject(error);
    });
  };

  // Cancel processing function
  const cancelProcessing = () => {
    fileProcessingWebSocket.cancel();
    setProgress(0);
    setProgressMessage('Processing cancelled');
    setStep('options');
  };

  const toggleEntityApproval = (index: number) => {
    setEntities(prev => prev.map((entity, i) => 
      i === index ? { ...entity, approved: !entity.approved } : entity
    ));
  };

  const downloadResults = () => {
    if (!results) return;

    // Create approved entities list
    const approvedEntities = entities.filter(e => e.approved);
    
    // Apply final redaction based on user approval
    let finalText = results.original_text;
    approvedEntities
      .sort((a, b) => b.start - a.start) // Sort in reverse order to maintain indices
      .forEach(entity => {
        const before = finalText.substring(0, entity.start);
        const after = finalText.substring(entity.end);
        let replacement = '';
        
        switch (options.redactionMode) {
          case 'mask':
            replacement = '*'.repeat(entity.text.length);
            break;
          case 'remove':
            replacement = '';
            break;
          default:
            replacement = options.customLabels[entity.type] || `[REDACTED_${entity.type.toUpperCase()}]`;
        }
        
        finalText = before + replacement + after;
      });

    // Download redacted content
    const blob = new Blob([finalText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `redacted_${file?.name || 'text.txt'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // Download report
    const report = {
      original_filename: file?.name || 'text_input',
      processing_time: results.process_time,
      total_entities_found: entities.length,
      entities_redacted: approvedEntities.length,
      redaction_mode: options.redactionMode,
      detection_mode: options.detectionMode,
      entities: approvedEntities
    };
    
    const reportBlob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const reportUrl = URL.createObjectURL(reportBlob);
    const reportLink = document.createElement('a');
    reportLink.href = reportUrl;
    reportLink.download = `report_${file?.name || 'text'}.json`;
    document.body.appendChild(reportLink);
    reportLink.click();
    document.body.removeChild(reportLink);
    URL.revokeObjectURL(reportUrl);

    setStep('complete');
  };

  // Filter and pagination logic
  const filteredEntities = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase();

    return entities.filter(entity => {
      if (normalizedQuery && !entity.text.toLowerCase().includes(normalizedQuery)) {
        return false;
      }

      if (selectedEntityTypes.size > 0 && !selectedEntityTypes.has(entity.type)) {
        return false;
      }

      if (entity.confidence < confidenceThreshold / 100) {
        return false;
      }

      return true;
    });
  }, [entities, searchQuery, selectedEntityTypes, confidenceThreshold]);

  const totalPages = Math.ceil(filteredEntities.length / itemsPerPage);

  const paginatedEntities = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredEntities.slice(startIndex, startIndex + itemsPerPage);
  }, [filteredEntities, currentPage, itemsPerPage]);

  const entityTypes = useMemo(() => {
    const stats = new Map<string, { count: number; totalConfidence: number }>();

    entities.forEach(entity => {
      const record = stats.get(entity.type);
      if (record) {
        record.count += 1;
        record.totalConfidence += entity.confidence;
      } else {
        stats.set(entity.type, { count: 1, totalConfidence: entity.confidence });
      }
    });

    return Array.from(stats.entries())
      .map(([type, data]) => ({
        type,
        count: data.count,
        avgConfidence: Math.round((data.totalConfidence / data.count) * 100)
      }))
      .sort((a, b) => b.count - a.count);
  }, [entities]);

  const visiblePages = useMemo(() => {
    if (totalPages <= 7) {
      return Array.from({ length: totalPages }, (_, index) => index + 1);
    }

    const pages = new Set<number>([1, totalPages]);
    for (let offset = -2; offset <= 2; offset++) {
      const page = currentPage + offset;
      if (page > 1 && page < totalPages) {
        pages.add(page);
      }
    }

    return Array.from(pages).sort((a, b) => a - b);
  }, [totalPages, currentPage]);

  const entityIndexMap = useMemo(() => {
    const map = new Map<DetectedEntity, number>();
    entities.forEach((entity, index) => {
      map.set(entity, index);
    });
    return map;
  }, [entities]);

  const safeTotalPages = Math.max(totalPages, 1);
  const safeCurrentPage = Math.min(currentPage, safeTotalPages);
  const pageStart = filteredEntities.length === 0
    ? 0
    : (safeCurrentPage - 1) * itemsPerPage + 1;
  const pageEnd = filteredEntities.length === 0
    ? 0
    : Math.min(safeCurrentPage * itemsPerPage, filteredEntities.length);
  const isOnFirstPage = filteredEntities.length === 0 || safeCurrentPage <= 1;
  const isOnLastPage = filteredEntities.length === 0 || safeCurrentPage >= safeTotalPages;

  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, confidenceThreshold, selectedEntityTypes, itemsPerPage]);

  useEffect(() => {
    if (totalPages === 0) {
      if (currentPage !== 1) {
        setCurrentPage(1);
      }
      return;
    }

    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [currentPage, totalPages]);

  // Bulk actions
  const toggleEntityType = (type: string) => {
    setSelectedEntityTypes(prev => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  };

  const bulkSkipEntityType = (type: string) => {
    setEntities(prev =>
      prev.map(entity =>
        entity.type === type ? { ...entity, approved: false } : entity
      )
    );
  };

  const bulkIncludeEntityType = (type: string) => {
    setEntities(prev =>
      prev.map(entity =>
        entity.type === type ? { ...entity, approved: true } : entity
      )
    );
  };

  const resetProcessor = () => {
    setFile(null);
    setTextInput('');
    setResults(null);
    setEntities([]);
    entitiesBufferRef.current = [];
    setStep('upload');
    setProgress(0);
    setProgressMessage('');
    // Reset filters
    setSearchQuery('');
    setSelectedEntityTypes(new Set<string>());
    setConfidenceThreshold(0);
    setCurrentPage(1);
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Process Files</h1>
            <p className="text-gray-600 mt-2">
              Upload files or enter text for PII detection and redaction
            </p>
          </div>
          <button
            onClick={() => setShowPerformanceDashboard(true)}
            className="flex items-center space-x-2 px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            title="Open Performance Dashboard"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span>Performance</span>
          </button>
        </div>
      </div>

      {/* Session Recovery Information */}
      {recoveredSession && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <div className="flex items-start">
            <div className="text-yellow-600 mr-3">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="flex-1">
              <h4 className="text-yellow-800 font-semibold mb-1">Session Recovered</h4>
              <p className="text-yellow-700 text-sm mb-3">
                Your previous {recoveredSession.type.replace('_', ' ')} session has been restored.
                {recoveredSession.fileName && ` File: ${recoveredSession.fileName}`}
                {recoveredSession.progress > 0 && ` (${recoveredSession.progress.toFixed(1)}% complete)`}
              </p>
              <div className="flex space-x-2">
                <button
                  onClick={() => {
                    if (recoveredSession.progress >= 100 && recoveredSession.sessionId) {
                      window.location.href = `/results/${recoveredSession.sessionId}?mode=review`;
                    } else {
                      setStep('processing');
                    }
                  }}
                  className="text-sm bg-yellow-200 text-yellow-800 px-3 py-1 rounded hover:bg-yellow-300 transition-colors"
                >
                  {recoveredSession.progress >= 100 ? 'View Results' : 'Continue Session'}
                </button>
                <button
                  onClick={() => {
                    clearCurrentSession();
                    setRecoveredSession(null);
                  }}
                  className="text-sm bg-gray-200 text-gray-700 px-3 py-1 rounded hover:bg-gray-300 transition-colors"
                >
                  Start Fresh
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Step Indicator */}
      <div className="flex justify-center">
        <div className="flex items-center space-x-4">
          {[
            { key: 'upload', label: 'Upload', icon: '📁' },
            { key: 'options', label: 'Configure', icon: '⚙️' },
            { key: 'processing', label: 'Process', icon: '🔄' },
            { key: 'review', label: 'Review', icon: '👀' },
            { key: 'complete', label: 'Complete', icon: '✅' }
          ].map((stepItem, index) => (
            <div key={stepItem.key} className="flex items-center">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
                step === stepItem.key 
                  ? 'bg-blue-600 text-white' 
                  : ['upload', 'options', 'processing', 'review', 'complete'].indexOf(step) > index
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-200 text-gray-600'
              }`}>
                {stepItem.icon}
              </div>
              <span className={`ml-2 text-sm ${
                step === stepItem.key ? 'text-blue-600 font-medium' : 'text-gray-500'
              }`}>
                {stepItem.label}
              </span>
              {index < 4 && <div className="w-8 h-px bg-gray-300 ml-4"></div>}
            </div>
          ))}
        </div>
      </div>

      {/* Upload Step */}
      {step === 'upload' && (
        <div className="space-y-6">
          {/* Processing Mode Toggle */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Choose Input Method</h3>
            <div className="flex space-x-4">
              <button
                className={`flex-1 p-4 rounded-lg border-2 ${
                  processingMode === 'file' 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setProcessingMode('file')}
              >
                <div className="text-center">
                  <div className="text-2xl mb-2">📁</div>
                  <div className="font-medium">Upload File</div>
                  <div className="text-sm text-gray-500">CSV, Excel, TXT files</div>
                </div>
              </button>
              <button
                className={`flex-1 p-4 rounded-lg border-2 ${
                  processingMode === 'text' 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setProcessingMode('text')}
              >
                <div className="text-center">
                  <div className="text-2xl mb-2">📝</div>
                  <div className="font-medium">Enter Text</div>
                  <div className="text-sm text-gray-500">Type or paste text</div>
                </div>
              </button>
            </div>
          </div>

          {processingMode === 'file' ? (
            <div className="card">
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls,.txt"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <div className="text-6xl mb-4">📁</div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Drop your file here or click to browse
                </h3>
                <p className="text-gray-500 mb-4">
                  Supports CSV, Excel (XLSX/XLS), and TXT files up to 50MB
                </p>
                <button
                  className="btn-primary"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Choose File
                </button>
              </div>
            </div>
          ) : (
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Enter Text to Process</h3>
              <textarea
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Paste or type the text you want to scan for PII..."
                className="w-full h-40 border border-gray-300 rounded-lg p-4 resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              <div className="mt-4 flex justify-between items-center">
                <span className="text-sm text-gray-500">{textInput.length} characters</span>
                <button
                  className="btn-primary"
                  onClick={() => setStep('options')}
                  disabled={!textInput.trim()}
                >
                  Continue
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Options Step */}
      {step === 'options' && (
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">File Information</h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center">
                <div className="text-2xl mr-3">
                  {processingMode === 'file' ? '📄' : '📝'}
                </div>
                <div>
                  <p className="font-medium">
                    {processingMode === 'file' ? file?.name : 'Text Input'}
                  </p>
                  <p className="text-sm text-gray-500">
                    {processingMode === 'file' 
                      ? `${(file!.size / 1024).toFixed(1)} KB` 
                      : `${textInput.length} characters`
                    }
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Detection Settings</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Detection Sensitivity
                  </label>
                  <select
                    value={options.detectionMode}
                    onChange={(e) => setOptions(prev => ({ 
                      ...prev, 
                      detectionMode: e.target.value as 'aggressive' | 'balanced' | 'conservative'
                    }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2"
                  >
                    <option value="conservative">Conservative (High Confidence)</option>
                    <option value="balanced">Balanced (Recommended)</option>
                    <option value="aggressive">Aggressive (Catch More)</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    {options.detectionMode === 'conservative' && 'Only detects very confident PII matches'}
                    {options.detectionMode === 'balanced' && 'Good balance between accuracy and coverage'}
                    {options.detectionMode === 'aggressive' && 'Detects more potential PII, may include false positives'}
                  </p>
                </div>

                {/* Training Toggle */}
                <div className="border-t border-gray-200 pt-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm font-medium text-gray-700">
                        🎓 Apply Custom Training
                      </label>
                      <p className="text-xs text-gray-500 mt-1">
                        Use your coaching feedback to improve detection accuracy
                      </p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        className="sr-only peer"
                        checked={options.useTraining}
                        onChange={(e) => setOptions(prev => ({
                          ...prev,
                          useTraining: e.target.checked
                        }))}
                      />
                      <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    </label>
                  </div>

                  {/* Training Statistics */}
                  {trainingStats && (
                    <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                      <div className="flex items-center justify-between text-sm">
                        <div>
                          <span className="font-medium text-blue-800">
                            {trainingStats.count} training corrections
                          </span>
                          {trainingStats.count > 0 && (
                            <p className="text-blue-600 text-xs mt-1">
                              Trained on: {trainingStats.entityTypes.slice(0, 3).join(', ')}
                              {trainingStats.entityTypes.length > 3 && ` +${trainingStats.entityTypes.length - 3} more`}
                            </p>
                          )}
                        </div>
                        {trainingStats.count === 0 && (
                          <span className="text-blue-600 text-xs">
                            No training data yet
                          </span>
                        )}
                      </div>
                      
                      {!options.useTraining && trainingStats.count > 0 && (
                        <div className="mt-2 text-xs text-orange-600 bg-orange-50 border border-orange-200 rounded px-2 py-1">
                          ⚠️ Your training improvements are disabled
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Redaction Settings</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Redaction Mode
                  </label>
                  <select
                    value={options.redactionMode}
                    onChange={(e) => setOptions(prev => ({ 
                      ...prev, 
                      redactionMode: e.target.value as 'replace' | 'mask' | 'remove'
                    }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2"
                  >
                    <option value="replace">Replace with Labels</option>
                    <option value="mask">Mask with Asterisks</option>
                    <option value="remove">Remove Completely</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    {options.redactionMode === 'replace' && 'Replace PII with descriptive labels like [REDACTED_EMAIL]'}
                    {options.redactionMode === 'mask' && 'Replace PII with asterisks (****)'}
                    {options.redactionMode === 'remove' && 'Remove PII completely from text'}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-between">
            <button
              className="btn-outline"
              onClick={() => setStep('upload')}
            >
              ← Back
            </button>
            <button
              className="btn-primary"
              onClick={processFile}
            >
              Start Processing →
            </button>
          </div>
        </div>
      )}

      {/* Processing Step */}
      {step === 'processing' && (
        <div className="card">
          <div className="text-center py-8">
            {error ? (
              // Error state
              <>
                <div className="text-6xl mb-4">❌</div>
                <h3 className="text-xl font-semibold text-red-700 mb-2">Processing Error</h3>
                <p className="text-red-600 mb-6">{error}</p>

                {retryCount < 3 && (
                  <div className="mb-6">
                    <button
                      onClick={handleRetry}
                      disabled={isRetrying}
                      className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors mr-4"
                    >
                      {isRetrying ? 'Retrying...' : `Retry (${3 - retryCount} attempts left)`}
                    </button>
                    <button
                      onClick={() => {
                        clearCurrentSession();
                        setError(null);
                        setStep('options');
                      }}
                      className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                    >
                      Start Over
                    </button>
                  </div>
                )}

                {retryCount >= 3 && (
                  <div className="mb-6">
                    <p className="text-gray-600 mb-4">Maximum retry attempts reached.</p>
                    <button
                      onClick={() => {
                        clearCurrentSession();
                        setError(null);
                        setRetryCount(0);
                        setStep('options');
                      }}
                      className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                    >
                      Start Over
                    </button>
                  </div>
                )}

                {progress > 0 && (
                  <div className="mt-6 p-4 bg-yellow-50 rounded-lg">
                    <p className="text-sm text-yellow-800 mb-2">
                      Processing was {progress.toFixed(1)}% complete when the error occurred.
                    </p>
                    <p className="text-xs text-yellow-600">
                      Your progress has been saved and will be restored if you retry.
                    </p>
                  </div>
                )}
              </>
            ) : (
              // Normal processing state
              <>
                <div className="text-6xl mb-4">🔄</div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Processing Your File</h3>
                <p className="text-gray-500 mb-6">
                  {progressMessage || 'Scanning for PII using advanced AI detection...'}
                </p>
                <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                  <div
                    className="bg-blue-600 h-3 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <p className="text-sm text-gray-500 mb-4">{Math.round(progress)}% complete</p>

                {/* Progress details toggle */}
                <button
                  onClick={() => setShowProgressDetails(!showProgressDetails)}
                  className="text-sm text-blue-600 hover:text-blue-800 mb-4"
                >
                  {showProgressDetails ? 'Hide' : 'Show'} Details
                </button>

                {showProgressDetails && (
                  <div className="mb-6 p-4 bg-gray-50 rounded-lg text-sm text-gray-600 text-left">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <strong>Entities Found:</strong> {entities.length}
                      </div>
                      <div>
                        <strong>Retry Count:</strong> {retryCount}
                      </div>
                      <div>
                        <strong>File:</strong> {file?.name || 'Text Input'}
                      </div>
                      <div>
                        <strong>Mode:</strong> {processingMode}
                      </div>
                    </div>
                  </div>
                )}

                {/* Cancel Button */}
                <button
                  onClick={cancelProcessing}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  Cancel Processing
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* Review Step */}
      {step === 'review' && results && (
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Processing Results</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{entities.length}</div>
                <div className="text-sm text-gray-600">PII Entities Found</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{entities.filter(e => e.approved).length}</div>
                <div className="text-sm text-gray-600">Approved for Redaction</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">{results.process_time}</div>
                <div className="text-sm text-gray-600">Processing Time</div>
              </div>
            </div>
          </div>

          {entities.length > 0 && (
            <>
              {/* Search and Filters */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">🔍 Search & Filters</h3>
                
                {/* Search Bar */}
                <div className="mb-4">
                  <input
                    type="text"
                    placeholder="Search for specific text..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {/* Entity Type Filters */}
                <div className="mb-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">📊 Entity Types:</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                    {entityTypes.map(({ type, count, avgConfidence }) => (
                      <div key={type} className="flex items-center justify-between p-2 border border-gray-200 rounded">
                        <label className="flex items-center cursor-pointer flex-1">
                          <input
                            type="checkbox"
                            checked={selectedEntityTypes.has(type)}
                            onChange={() => toggleEntityType(type)}
                            className="h-4 w-4 text-blue-600 mr-2"
                          />
                          <div className="flex-1">
                            <span className="text-sm font-medium">{type}</span>
                            <div className="text-xs text-gray-500">
                              {count} items · {avgConfidence}% avg confidence
                            </div>
                          </div>
                        </label>
                        <div className="flex space-x-1 ml-2">
                          <button
                            onClick={() => bulkSkipEntityType(type)}
                            className="px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200"
                            title="Skip all"
                          >
                            Skip All
                          </button>
                          <button
                            onClick={() => bulkIncludeEntityType(type)}
                            className="px-2 py-1 text-xs bg-green-100 text-green-700 rounded hover:bg-green-200"
                            title="Include all"
                          >
                            Include All
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Confidence Threshold */}
                <div className="mb-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">🎚️ Minimum Confidence: {confidenceThreshold}%</h4>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Filter Summary */}
                <div className="bg-gray-50 p-3 rounded-lg">
                  <div className="text-sm text-gray-600">
                    Showing <span className="font-medium">{filteredEntities.length}</span> of <span className="font-medium">{entities.length}</span> detected entities
                    {selectedEntityTypes.size > 0 && (
                      <span> · Filtered by: {Array.from(selectedEntityTypes).join(', ')}</span>
                    )}
                    {searchQuery && (
                      <span> · Search: "{searchQuery}"</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Results List */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">📋 Review Results</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Review each detected PII entity below. Uncheck items you want to keep in the final output.
                </p>

                {filteredEntities.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <div className="text-4xl mb-2">🔍</div>
                    <p>No entities match your current filters</p>
                    <button
                      onClick={() => {
                        setSearchQuery('');
                        setSelectedEntityTypes(new Set<string>());
                        setConfidenceThreshold(0);
                      }}
                      className="mt-2 text-blue-600 hover:underline"
                    >
                      Clear all filters
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="space-y-2 mb-4">
                      {paginatedEntities.map((entity) => {
                        const globalIndex = entityIndexMap.get(entity);
                        if (globalIndex === undefined) {
                          return null;
                        }

                        return (
                          <div key={globalIndex} className="flex items-center justify-between p-3 border rounded-lg">
                            <div className="flex items-center space-x-3">
                              <input
                                type="checkbox"
                                checked={entity.approved}
                                onChange={() => toggleEntityApproval(globalIndex)}
                                className="h-4 w-4 text-blue-600"
                              />
                              <div>
                                <span className="font-medium text-gray-900">"{entity.text}"</span>
                                <div className="flex items-center space-x-2 text-sm text-gray-500">
                                  <span className="px-2 py-1 bg-gray-100 rounded text-xs">{entity.type}</span>
                                  <span>{Math.round(entity.confidence * 100)}% confidence</span>
                                </div>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-sm font-medium text-gray-900">
                                {entity.approved ? 'Will Redact' : 'Will Keep'}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Pagination */}
                    {filteredEntities.length > 0 && (
                      <div className="flex flex-col gap-4 border-t pt-4 md:flex-row md:items-center md:justify-between">
                        <div className="text-sm text-gray-600">
                          Showing {pageStart}-{pageEnd} of {filteredEntities.length} results · Page {safeCurrentPage} of {safeTotalPages}
                        </div>
                        <div className="flex flex-col gap-3 md:flex-row md:items-center md:gap-4">
                          <label className="flex items-center space-x-2 text-sm text-gray-600">
                            <span>Rows per page:</span>
                            <select
                              value={itemsPerPage}
                              onChange={(e) => setItemsPerPage(Number(e.target.value))}
                              className="rounded border border-gray-300 px-2 py-1 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                            >
                              {[25, 50, 100, 250, 500].map(size => (
                                <option key={size} value={size}>{size}</option>
                              ))}
                            </select>
                          </label>
                          <div className="flex items-center space-x-1">
                            <button
                              onClick={() => setCurrentPage(1)}
                              disabled={isOnFirstPage}
                              className="px-3 py-1 border rounded text-sm disabled:cursor-not-allowed disabled:opacity-50 hover:bg-gray-50"
                            >
                              First
                            </button>
                            <button
                              onClick={() => setCurrentPage(Math.max(1, safeCurrentPage - 1))}
                              disabled={isOnFirstPage}
                              className="px-3 py-1 border rounded text-sm disabled:cursor-not-allowed disabled:opacity-50 hover:bg-gray-50"
                            >
                              Previous
                            </button>
                            {visiblePages.map((page, index) => {
                              const showEllipsis = index > 0 && page - visiblePages[index - 1] > 1;
                              return (
                                <React.Fragment key={page}>
                                  {showEllipsis && <span className="px-1 text-sm text-gray-500">...</span>}
                                  <button
                                    onClick={() => setCurrentPage(page)}
                                    className={`px-3 py-1 border rounded text-sm ${safeCurrentPage === page ? 'bg-blue-600 text-white border-blue-600' : 'hover:bg-gray-50'}`}
                                  >
                                    {page}
                                  </button>
                                </React.Fragment>
                              );
                            })}
                            <button
                              onClick={() => setCurrentPage(Math.min(safeTotalPages, safeCurrentPage + 1))}
                              disabled={isOnLastPage}
                              className="px-3 py-1 border rounded text-sm disabled:cursor-not-allowed disabled:opacity-50 hover:bg-gray-50"
                            >
                              Next
                            </button>
                            <button
                              onClick={() => setCurrentPage(safeTotalPages)}
                              disabled={isOnLastPage}
                              className="px-3 py-1 border rounded text-sm disabled:cursor-not-allowed disabled:opacity-50 hover:bg-gray-50"
                            >
                              Last
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </>
          )}

          <div className="flex justify-between">
            <div className="flex space-x-3">
              <button
                className="btn-outline"
                onClick={() => setStep('options')}
              >
                ← Back to Options
              </button>
              <button
                className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors"
                onClick={() => setShowTrainer(true)}
              >
                🎓 Train AI Model
              </button>
            </div>
            <button
              className="btn-primary"
              onClick={downloadResults}
            >
              Download Results →
            </button>
          </div>
        </div>
      )}

      {/* Complete Step */}
      {step === 'complete' && (
        <div className="card">
          <div className="text-center py-8">
            <div className="text-6xl mb-4">✅</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Processing Complete!</h3>
            <p className="text-gray-500 mb-6">
              Your files have been downloaded successfully.
            </p>
            <div className="space-x-4">
              <button
                className="btn-primary"
                onClick={resetProcessor}
              >
                Process Another File
              </button>
              <button
                className="btn-outline"
                onClick={() => window.location.href = '/'}
              >
                Back to Dashboard
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Feedback Trainer Modal */}
      <FeedbackTrainer
        isOpen={showTrainer}
        onClose={() => setShowTrainer(false)}
        trainingData={originalContent ? {
          file_content: originalContent.split('\n'),
          detected_entities: entities
        } : null}
      />

      {/* Performance Dashboard Modal */}
      <PerformanceDashboard
        isOpen={showPerformanceDashboard}
        onClose={() => setShowPerformanceDashboard(false)}
      />
    </div>
  );
};
