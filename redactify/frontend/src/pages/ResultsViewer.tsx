import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

interface ProcessingRowData {
  id: number;
  session_id: string;
  row_number: number;
  original_text: string;
  redacted_text: string;
  entities_found: number;
  processing_time_ms: number;
  was_redacted: boolean;
  has_error: boolean;
  error_message?: string;
  created_at: string;
}

interface ResultsData {
  filename: string;
  total_rows: number;
  session_id: string;
  created_at: string;
  rows: ProcessingRowData[];
}

export const ResultsViewer: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [results, setResults] = useState<ResultsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setError('No session ID provided');
      setLoading(false);
      return;
    }

    fetchResults();
  }, [sessionId]);

  const fetchResults = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8080/api/v1/files/results/view/${sessionId}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch results');
      }

      const data: ResultsData = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load results');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-4xl mb-4">���</div>
          <div className="text-xl font-semibold text-gray-700">Loading results...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-4xl mb-4">❌</div>
          <div className="text-xl font-semibold text-gray-700 mb-2">Failed to load results</div>
          <div className="text-gray-500 mb-4">{error}</div>
          <button
            onClick={() => navigate('/history')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Back to History
          </button>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-4xl mb-4">���</div>
          <div className="text-xl font-semibold text-gray-700">No results found</div>
        </div>
      </div>
    );
  }

  const totalEntitiesFound = results.rows.reduce((sum, row) => sum + row.entities_found, 0);
  const redactedRows = results.rows.filter(row => row.was_redacted).length;
  const avgProcessingTime = results.rows.length > 0 
    ? results.rows.reduce((sum, row) => sum + row.processing_time_ms, 0) / results.rows.length 
    : 0;

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h1 className="text-3xl font-bold text-gray-900">Processing Results</h1>
        <p className="text-gray-600 mt-2">
          Review detected PII entities from file: {results.filename}
        </p>
      </div>

      {/* Step Indicator */}
      <div className="flex justify-center">
        <div className="flex items-center space-x-4">
          {[
            { key: 'upload', label: 'Upload', icon: '���', active: true },
            { key: 'options', label: 'Configure', icon: '⚙️', active: true },
            { key: 'processing', label: 'Process', icon: '���', active: true },
            { key: 'review', label: 'Review', icon: '���', active: true },
            { key: 'complete', label: 'Complete', icon: '✅', active: false }
          ].map((stepItem, index) => (
            <div key={stepItem.key} className="flex items-center">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
                stepItem.key === 'review'
                  ? 'bg-blue-600 text-white' 
                  : stepItem.active
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-200 text-gray-600'
              }`}>
                {stepItem.icon}
              </div>
              <span className={`ml-2 text-sm ${
                stepItem.key === 'review' ? 'text-blue-600 font-medium' : 'text-gray-500'
              }`}>
                {stepItem.label}
              </span>
              {index < 4 && <div className="w-8 h-px bg-gray-300 ml-4"></div>}
            </div>
          ))}
        </div>
      </div>

      {/* Processing Results Statistics */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Processing Results</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{totalEntitiesFound}</div>
            <div className="text-sm text-gray-600">PII Entities Found</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{redactedRows}</div>
            <div className="text-sm text-gray-600">Rows Redacted</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">{avgProcessingTime.toFixed(1)}ms</div>
            <div className="text-sm text-gray-600">Avg Processing Time</div>
          </div>
        </div>
      </div>

      {/* Results Table */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">��� Processing Details</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Row
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Original Text (Preview)
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Entities Found
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Processing Time
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {results.rows.map((row) => (
                <tr key={row.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {row.row_number}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-900 max-w-md">
                    <div className="truncate">
                      {row.original_text.length > 100 
                        ? `${row.original_text.substring(0, 100)}...` 
                        : row.original_text}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {row.has_error ? (
                      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-red-100 text-red-800">
                        Error
                      </span>
                    ) : row.was_redacted ? (
                      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-yellow-100 text-yellow-800">
                        Redacted
                      </span>
                    ) : (
                      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">
                        Clean
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {row.entities_found}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {row.processing_time_ms}ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="flex justify-between">
        <button
          className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
          onClick={() => navigate('/history')}
        >
          ← Back to History
        </button>
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          onClick={() => {
            // TODO: Implement download functionality
            alert('Download functionality coming soon!');
          }}
        >
          Download Results →
        </button>
      </div>
    </div>
  );
};
