import React, { useState, useEffect } from 'react';

interface DetectedEntity {
  type: string;
  text: string;
  start: number;
  end: number;
  confidence: number;
  category: string;
}

interface FeedbackItem {
  entity_text: string;
  entity_type: string;
  original_score: number;
  user_decision: string; // 'correct', 'incorrect', 'partial'
  user_confidence: number;
  context: string;
}

interface TrainingData {
  file_content: string[];
  detected_entities: DetectedEntity[];
}

interface FeedbackTrainerProps {
  isOpen: boolean;
  onClose: () => void;
  trainingData: TrainingData | null;
}

interface DocumentRow {
  original: string;
  redacted: string;
  entities: DetectedEntity[];
  rowIndex: number;
}

export const FeedbackTrainer: React.FC<FeedbackTrainerProps> = ({ 
  isOpen, 
  onClose, 
  trainingData 
}) => {
  const [currentRowIndex, setCurrentRowIndex] = useState(0);
  const [documentRows, setDocumentRows] = useState<DocumentRow[]>([]);
  const [selectedText, setSelectedText] = useState('');
  const [feedbackType, setFeedbackType] = useState<'false_positive' | 'false_negative' | 'wrong_type'>('false_positive');
  const [entityType, setEntityType] = useState('');
  const [correctType, setCorrectType] = useState('');
  const [submittedFeedback, setSubmittedFeedback] = useState<FeedbackItem[]>([]);
  const [showFeedbackPanel, setShowFeedbackPanel] = useState(false);

  const entityTypes = [
    'Person', 'Email', 'PhoneNumber', 'SSN', 'CreditCard', 'IPAddress',
    'Address', 'Organization', 'Date', 'URL', 'EmployeeID', 'AccountNumber'
  ];

  // Process training data into document rows
  useEffect(() => {
    if (!trainingData) {
      setDocumentRows([]);
      return;
    }

    const rows: DocumentRow[] = trainingData.file_content.map((content, index) => {
      // Mock redacted version and entities for each row
      // In a real implementation, this would call the redaction API for each row
      const entities = trainingData.detected_entities.filter(entity => 
        content.includes(entity.text)
      );
      
      let redacted = content;
      entities.forEach(entity => {
        const replacement = `[REDACTED_${entity.type.toUpperCase()}]`;
        redacted = redacted.replace(entity.text, replacement);
      });

      return {
        original: content,
        redacted,
        entities,
        rowIndex: index
      };
    });

    setDocumentRows(rows);
  }, [trainingData]);

  const handleTextSelection = () => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;
    
    const selectedText = selection.toString().trim();
    if (selectedText.length === 0) return;

    // Check if selected text was detected as PII
    const isDetected = currentRow?.entities.some(entity => 
      entity.text.toLowerCase().includes(selectedText.toLowerCase()) ||
      selectedText.toLowerCase().includes(entity.text.toLowerCase())
    ) || false;

    setSelectedText(selectedText);
    setFeedbackType(isDetected ? 'false_positive' : 'false_negative');
    setShowFeedbackPanel(true);
    
    // Clear the selection
    selection.removeAllRanges();
  };

  const renderHighlightedText = (text: string, entities: DetectedEntity[]) => {
    if (entities.length === 0) {
      return (
        <div 
          className="cursor-text select-text user-select-text"
          onMouseUp={handleTextSelection}
        >
          {text}
        </div>
      );
    }

    let result = [];
    let lastIndex = 0;

    entities
      .sort((a, b) => a.start - b.start)
      .forEach((entity, index) => {
        // Add text before entity
        if (entity.start > lastIndex) {
          const beforeText = text.slice(lastIndex, entity.start);
          result.push(
            <span key={`before-${index}`}>
              {beforeText}
            </span>
          );
        }

        // Add highlighted entity
        result.push(
          <span
            key={`entity-${index}`}
            className="bg-red-200 border border-red-300 rounded px-1 hover:bg-red-300 transition-colors"
            title={`${entity.type} (${(entity.confidence * 100).toFixed(1)}% confidence)`}
          >
            {entity.text}
          </span>
        );

        lastIndex = entity.end;
      });

    // Add remaining text
    if (lastIndex < text.length) {
      const remainingText = text.slice(lastIndex);
      result.push(
        <span key="remaining">
          {remainingText}
        </span>
      );
    }

    return (
      <div 
        className="cursor-text select-text user-select-text"
        onMouseUp={handleTextSelection}
      >
        {result}
      </div>
    );
  };

  const currentRow = documentRows[currentRowIndex];
  const totalRows = documentRows.length;

  const submitFeedback = async () => {
    if (!selectedText || !entityType) return;

    const feedback = {
      entity_text: selectedText,
      entity_type: entityType,
      original_score: 1.0,
      user_decision: feedbackType === 'false_positive' ? 'incorrect' : 'correct',
      user_confidence: 0.9,
      context: currentRow?.original || ''
    };

    try {
      const response = await fetch('http://localhost:8080/api/v1/pii/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedback),
      });

      if (response.ok) {
        setSubmittedFeedback(prev => [...prev, feedback]);
        // Reset feedback panel
        setSelectedText('');
        setEntityType('');
        setCorrectType('');
        setShowFeedbackPanel(false);
        alert('Feedback submitted! This will improve future detection.');
      }
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      alert('Failed to submit feedback. Please try again.');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-2xl w-full max-w-7xl max-h-[95vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-3xl font-bold">📋 Document Trainer</h2>
              <p className="text-blue-100 mt-1">Review and train the AI on your document content</p>
            </div>
            <button
              onClick={onClose}
              className="text-white hover:text-gray-200 text-2xl"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="flex h-[calc(95vh-120px)]">
          {/* Main Document View */}
          <div className="flex-1 flex flex-col">
            {/* Navigation */}
            <div className="bg-gray-50 border-b px-6 py-4 flex justify-between items-center">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setCurrentRowIndex(Math.max(0, currentRowIndex - 1))}
                  disabled={currentRowIndex === 0}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors"
                >
                  ← Previous Row
                </button>
                <span className="text-sm font-medium text-gray-600">
                  Row {currentRowIndex + 1} of {totalRows}
                </span>
                <button
                  onClick={() => setCurrentRowIndex(Math.min(totalRows - 1, currentRowIndex + 1))}
                  disabled={currentRowIndex === totalRows - 1}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors"
                >
                  Next Row →
                </button>
              </div>
              <div className="text-sm text-gray-500">
                Select any text with your mouse to provide training feedback
              </div>
            </div>

            {/* Document Content */}
            <div className="flex-1 overflow-auto p-6">
              {currentRow ? (
                <div className="space-y-6">
                  {/* Original Text */}
                  <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
                    <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                      <h3 className="text-lg font-semibold text-gray-800">📄 Original Text</h3>
                    </div>
                    <div className="p-4">
                      <div className="prose max-w-none text-gray-700 leading-relaxed font-mono text-sm whitespace-pre-wrap" style={{userSelect: 'text', WebkitUserSelect: 'text', MozUserSelect: 'text'}}>
                        {renderHighlightedText(currentRow.original, currentRow.entities)}
                      </div>
                    </div>
                  </div>

                  {/* Redacted Preview */}
                  <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
                    <div className="bg-red-50 px-4 py-2 border-b border-red-200">
                      <h3 className="text-lg font-semibold text-red-800">🔒 Redacted Version</h3>
                    </div>
                    <div className="p-4">
                      <div className="prose max-w-none text-gray-700 leading-relaxed font-mono text-sm whitespace-pre-wrap">
                        {currentRow.redacted}
                      </div>
                    </div>
                  </div>

                  {/* Entity Summary */}
                  <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
                    <div className="bg-green-50 px-4 py-2 border-b border-green-200">
                      <h3 className="text-lg font-semibold text-green-800">🏷️ Detected Entities</h3>
                    </div>
                    <div className="p-4">
                      {currentRow.entities.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {currentRow.entities.map((entity, index) => (
                            <div key={index} className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                              <div className="flex justify-between items-start">
                                <div>
                                  <p className="font-medium text-gray-800">"{entity.text}"</p>
                                  <p className="text-sm text-gray-600">{entity.type}</p>
                                </div>
                                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                                  {(entity.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-500 italic">No PII entities detected in this row</p>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <div className="text-6xl mb-4">📄</div>
                  <p className="text-xl">No document data available</p>
                  <p>Please process a file first to start training</p>
                </div>
              )}
            </div>
          </div>

          {/* Feedback Sidebar */}
          <div className="w-80 bg-gray-50 border-l border-gray-200 flex flex-col">
            <div className="p-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-800">🎓 Training Panel</h3>
            </div>
            
            <div className="flex-1 overflow-auto p-4 space-y-4">
              {showFeedbackPanel && selectedText ? (
                <>
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <p className="text-sm font-medium text-gray-700 mb-2">Selected Text:</p>
                    <p className="text-sm bg-yellow-100 border border-yellow-300 rounded px-2 py-1 font-mono">
                      "{selectedText}"
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Feedback Type:
                    </label>
                    <select
                      value={feedbackType}
                      onChange={(e) => setFeedbackType(e.target.value as any)}
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    >
                      <option value="false_positive">❌ Incorrectly Detected</option>
                      <option value="false_negative">⚠️ Missed Detection</option>
                      <option value="wrong_type">🔄 Wrong Category</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Entity Type:
                    </label>
                    <select
                      value={entityType}
                      onChange={(e) => setEntityType(e.target.value)}
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    >
                      <option value="">Select type...</option>
                      {entityTypes.map(type => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                  </div>

                  {feedbackType === 'wrong_type' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Correct Type:
                      </label>
                      <select
                        value={correctType}
                        onChange={(e) => setCorrectType(e.target.value)}
                        className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      >
                        <option value="">Select correct type...</option>
                        {entityTypes.map(type => (
                          <option key={type} value={type}>{type}</option>
                        ))}
                      </select>
                    </div>
                  )}

                  <button
                    onClick={submitFeedback}
                    disabled={!selectedText || !entityType}
                    className="w-full bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    Submit Feedback
                  </button>

                  <button
                    onClick={() => setShowFeedbackPanel(false)}
                    className="w-full bg-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-400 transition-colors"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <div className="text-4xl mb-3">🖱️</div>
                  <p className="text-sm">Select text in the document with your mouse to provide feedback</p>
                  <div className="mt-4 space-y-2 text-xs">
                    <div className="flex items-center">
                      <div className="w-4 h-4 bg-red-200 border border-red-300 rounded mr-2"></div>
                      <span>Highlighted PII (select to mark as incorrect)</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-4 h-4 bg-gray-100 border border-gray-300 rounded mr-2"></div>
                      <span>Regular text (select to mark as missed PII)</span>
                    </div>
                  </div>
                  <div className="mt-3 text-xs text-gray-400">
                    💡 You can select single words, phrases, or any text portion
                  </div>
                </div>
              )}

              {/* Training History */}
              {submittedFeedback.length > 0 && (
                <div className="border-t border-gray-200 pt-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Recent Feedback:</h4>
                  <div className="space-y-2 max-h-32 overflow-y-auto">
                    {submittedFeedback.slice(-3).map((item, index) => (
                      <div key={index} className="text-xs bg-white border border-gray-200 rounded p-2">
                        <p className="font-medium">"{item.entity_text}"</p>
                        <p className="text-gray-500">{item.entity_type} - {item.user_decision}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};