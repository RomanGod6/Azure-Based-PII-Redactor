import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface HistoryItem {
  id?: string;
  filename?: string;
  timestamp?: string;
  status?: string;
  entities_found?: number;
  processing_time?: string;
  result_id?: string;
  session_id?: string;
  fallback_result_id?: string;
  fallback_session_id?: string;
  has_results?: boolean;
}

interface HistoryData {
  history: HistoryItem[];
  total: number;
  limit: number;
}

export const History: React.FC = () => {
  const navigate = useNavigate();
  const [history, setHistory] = useState<HistoryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(20);
  const [downloading, setDownloading] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch(`http://localhost:8080/api/v1/history?limit=${limit}`);
        if (!response.ok) {
          throw new Error('Failed to fetch history data');
        }
        const data = await response.json();
        setHistory(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [limit]);

  const handleDownloadResults = async (item: HistoryItem, format: 'csv' | 'json' = 'csv') => {
    try {
      setDownloading(item.id || '');
      
      let response: Response;
      
      // Check if we have session_id (new incremental system) or need to use legacy result_id
      const sessionId = item.session_id || item.fallback_session_id;
      const resultId = item.result_id || item.fallback_result_id;
      
      if (sessionId) {
        // New system: stream from processing_rows table
        response = await fetch(`http://localhost:8080/api/v1/files/stream/${sessionId}?format=${format}`);
      } else if (resultId) {
        // Legacy system: download from processing_results table
        response = await fetch(`http://localhost:8080/api/v1/files/legacy/${resultId}?format=${format}`);
      } else {
        alert('No results available for this item');
        return;
      }
      
      if (!response.ok) {
        throw new Error('Failed to download results');
      }

      // Download as file
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${item.filename || 'results'}_processed.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Download error:', err);
      alert('Failed to download results: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setDownloading(null);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="border-b border-gray-200 pb-4">
          <h1 className="text-3xl font-bold text-gray-900">History</h1>
          <p className="text-gray-600 mt-2">View past processing jobs and results</p>
        </div>
        <div className="card">
          <div className="animate-pulse space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="flex items-center space-x-4">
                <div className="w-8 h-8 bg-gray-200 rounded"></div>
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div className="border-b border-gray-200 pb-4">
          <h1 className="text-3xl font-bold text-gray-900">History</h1>
          <p className="text-gray-600 mt-2">View past processing jobs and results</p>
        </div>
        <div className="card bg-red-50 border-red-200">
          <div className="text-red-800">
            <h3 className="font-semibold">Error loading history</h3>
            <p className="text-sm mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  const hasHistory = history && history.history && history.history.length > 0;

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">History</h1>
          <p className="text-gray-600 mt-2">
            View past processing jobs and results
          </p>
        </div>
        {hasHistory && (
          <div className="flex items-center space-x-2">
            <label htmlFor="limit" className="text-sm text-gray-600">Show:</label>
            <select
              id="limit"
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="border border-gray-300 rounded-md text-sm px-2 py-1"
            >
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
        )}
      </div>

      {!hasHistory ? (
        <div className="card">
          <div className="text-center py-12">
            <div className="text-gray-400 text-6xl mb-4">📁</div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Processing History</h3>
            <p className="text-gray-500 mb-4">
              You haven't processed any files yet. Start by uploading and processing your first file.
            </p>
            <button 
              className="btn-primary"
              onClick={() => navigate('/process')}
            >
              🚀 Process Your First File
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="card">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Processing History</h3>
              <p className="text-sm text-gray-500">
                Showing {history.history.length} of {history.total} records
              </p>
            </div>

            <div className="space-y-3">
              {history.history.map((item, index) => (
                <div key={item.id || index} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50">
                  <div className="flex items-center space-x-4">
                    <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                      <span className="text-blue-600 font-semibold">📄</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900">
                        {item.filename || `Processing Job #${index + 1}`}
                      </h4>
                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <span>{item.timestamp || 'Recently processed'}</span>
                        {item.processing_time && (
                          <span>⚡ {item.processing_time}</span>
                        )}
                        {item.entities_found !== undefined && (
                          <span>🛡️ {item.entities_found} PII found</span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className={`px-2 py-1 text-xs rounded-full font-medium ${
                      item.status === 'completed' || item.status === 'complete' 
                        ? 'bg-green-100 text-green-800' 
                        : item.status === 'processing'
                        ? 'bg-yellow-100 text-yellow-800'
                        : item.status === 'failed'
                        ? 'bg-red-100 text-red-800'
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {item.status || 'Complete'}
                    </span>
                    {item.has_results ? (
                      <div className="flex space-x-2">
                        <button 
                          onClick={() => {
                            if (item.session_id) {
                              navigate(`/results/session/${item.session_id}`);
                            } else if (item.result_id) {
                              navigate(`/results/${item.result_id}`);
                            }
                          }}
                          className="text-purple-600 hover:text-purple-800 text-sm font-medium"
                        >
                          👁️ View Details
                        </button>
                        <button 
                          onClick={() => handleDownloadResults(item, 'csv')}
                          disabled={downloading === item.id}
                          className="text-blue-600 hover:text-blue-800 text-sm font-medium disabled:text-gray-400"
                        >
                          {downloading === item.id ? '⬇️ Downloading...' : '📥 Download CSV'}
                        </button>
                        <button 
                          onClick={() => handleDownloadResults(item, 'json')}
                          disabled={downloading === item.id}
                          className="text-green-600 hover:text-green-800 text-sm font-medium disabled:text-gray-400"
                        >
                          📋 JSON
                        </button>
                      </div>
                    ) : (
                      <span className="text-gray-400 text-sm">No results available</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Pagination placeholder */}
          {history.total > history.limit && (
            <div className="card">
              <div className="text-center py-4">
                <p className="text-sm text-gray-500">
                  Showing {history.history.length} of {history.total} records
                </p>
                <button className="btn-outline mt-2">
                  Load More
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};