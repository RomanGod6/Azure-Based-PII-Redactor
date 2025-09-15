import React, { useState, useRef, useCallback, useEffect } from 'react';
import { FeedbackTrainer } from '../components/FeedbackTrainer';

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
  entities: DetectedEntity[];
  redacted_count: number;
  process_time: string;
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
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    
    setStep('processing');
    setProgress(0);
    setProgressMessage('Initializing...');

    try {
      let response;
      
      if (processingMode === 'text') {
        // Process text directly
        response = await fetch('http://localhost:8080/api/v1/pii/redact', {
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
      } else {
        // Process file with SSE progress updates
        console.log('🚀 Frontend: Processing file with progress updates', {
          fileName: file!.name,
          fileSize: file!.size,
          options: options,
          optionsJson: JSON.stringify(options)
        });
        
        const formData = new FormData();
        formData.append('file', file!);
        formData.append('options', JSON.stringify(options));

        console.log('📤 Frontend: Starting SSE connection to /api/v1/files/process');
        
        // Use fetch for SSE connection
        response = await fetch('http://localhost:8080/api/v1/files/process', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Processing failed');
        }

        // Handle SSE stream
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        if (!reader) {
          throw new Error('No response stream available');
        }

        while (true) {
          const { done, value } = await reader.read();
          
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          
          // Process complete SSE messages
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6)); // Remove 'data: ' prefix
                console.log('📊 Progress update:', data);
                
                // Update progress based on SSE data
                if (data.total_rows > 0) {
                  const progressPercent = (data.current_row / data.total_rows) * 100;
                  setProgress(progressPercent);
                }
                
                // Update status message
                if (data.message) {
                  console.log(`🔄 ${data.message}`);
                  setProgressMessage(data.message);
                }
                
                // Handle completion
                if (data.is_complete && data.results) {
                  console.log('✅ Processing completed!', data.results);
                  setProgress(100);
                  
                  setResults(data.results);
                  setOriginalContent(data.results.original_text || '');
                  setEntities(data.results.entities.map((entity: DetectedEntity) => ({
                    ...entity,
                    approved: true // Default to approved, user can uncheck
                  })));
                  setStep('review');
                  return; // Exit the function
                }
                
                // Handle errors
                if (data.status === 'error') {
                  throw new Error(data.message || 'Processing failed');
                }
              } catch (parseError) {
                console.warn('Failed to parse SSE data:', line, parseError);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Processing error:', error);
      alert('Processing failed. Please try again.');
      setStep('options');
    }
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
      total_entities_found: results.entities.length,
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

  const resetProcessor = () => {
    setFile(null);
    setTextInput('');
    setResults(null);
    setEntities([]);
    setStep('upload');
    setProgress(0);
    setProgressMessage('');
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h1 className="text-3xl font-bold text-gray-900">Process Files</h1>
        <p className="text-gray-600 mt-2">
          Upload files or enter text for PII detection and redaction
        </p>
      </div>

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
            <p className="text-sm text-gray-500">{Math.round(progress)}% complete</p>
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
                <div className="text-2xl font-bold text-blue-600">{results.entities.length}</div>
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
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Review Detected PII</h3>
              <p className="text-sm text-gray-600 mb-4">
                Review each detected PII entity below. Uncheck items you want to keep in the final output.
              </p>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {entities.map((entity, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <input
                        type="checkbox"
                        checked={entity.approved}
                        onChange={() => toggleEntityApproval(index)}
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
                ))}
              </div>
            </div>
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
    </div>
  );
};