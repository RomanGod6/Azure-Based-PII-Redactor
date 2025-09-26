import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import {
  fetchSessionReview,
  exportSessionResults,
  SessionReviewResponse,
  SessionReviewEntity,
  SessionReviewRow
} from '../services/resultsApi';

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

interface LegacyResult {
  id: string;
  filename: string;
  original_text: string;
  redacted_text: string;
  entities_found: number;
  processing_time_ms: number;
  rows_processed: number;
  created_at: string;
}

type TableRow = SessionReviewRow | ProcessingRowData;

const isReviewRow = (row: TableRow): row is SessionReviewRow => {
  return (row as SessionReviewRow).review_redacted_text !== undefined;
};

export const ResultsViewer: React.FC = () => {
  const { sessionId, resultId } = useParams<{ sessionId?: string; resultId?: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const [results, setResults] = useState<ResultsData | null>(null);
  const [legacyResults, setLegacyResults] = useState<LegacyResult | null>(null);
  const [sessionReview, setSessionReview] = useState<SessionReviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reviewMode, setReviewMode] = useState(false);

  const [approvedMap, setApprovedMap] = useState<Map<number, boolean>>(new Map());
  const [redactionMode, setRedactionMode] = useState('replace');
  const [customLabels, setCustomLabels] = useState<Record<string, string>>({});
  const [includeErrors, setIncludeErrors] = useState(false);
  const [exporting, setExporting] = useState(false);

  // Virtual scrolling and pagination state
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 50 });
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(600);
  const ROW_HEIGHT = 80; // Approximate height of each table row
  const BUFFER_SIZE = 10; // Extra rows to render for smooth scrolling

  useEffect(() => {
    const modeParam = searchParams.get('mode');
    setReviewMode(modeParam === 'review');
  }, [searchParams]);

  useEffect(() => {
    if (!sessionId && !resultId) {
      setError('No results selected');
      setLoading(false);
      return;
    }

    if (sessionId) {
      if (reviewMode) {
        loadSessionReview(sessionId);
      } else {
        loadSessionResults(sessionId);
      }
    } else if (resultId) {
      loadLegacyResult(resultId);
    }
  }, [sessionId, resultId, reviewMode]);

  const loadSessionResults = async (id: string) => {
    try {
      setLoading(true);
      setError(null);
      setSessionReview(null);
      setApprovedMap(new Map());
      const response = await fetch(`http://localhost:8080/api/v1/files/results/view/${id}`);
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Failed to fetch results');
      }

      const data: ResultsData = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load results');
    } finally {
      setLoading(false);
    }
  };

  const loadSessionReview = async (id: string) => {
    try {
      setLoading(true);
      setError(null);
      setResults(null);
      setLegacyResults(null);

      const data = await fetchSessionReview(id);
      setSessionReview(data);
      setRedactionMode(data.redaction_mode || 'replace');
      setCustomLabels(data.custom_labels || {});

      const map = new Map<number, boolean>();
      data.entities.forEach(entity => {
        map.set(entity.id, entity.approved !== false);
      });
      setApprovedMap(map);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load session review data');
    } finally {
      setLoading(false);
    }
  };

  const loadLegacyResult = async (id: string) => {
    try {
      setLoading(true);
      setError(null);
      setResults(null);
      setSessionReview(null);

      const response = await fetch(`http://localhost:8080/api/v1/files/legacy/${id}`);
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Failed to fetch legacy results');
      }

      const data: LegacyResult = await response.json();
      setLegacyResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load legacy results');
    } finally {
      setLoading(false);
    }
  };

  const inReview = Boolean(sessionId && reviewMode && sessionReview);
  const tableRows: TableRow[] = useMemo(() => {
    if (inReview && sessionReview) {
      return sessionReview.rows;
    }
    if (results) {
      return results.rows;
    }
    return [];
  }, [inReview, sessionReview, results]);

  const reviewEntitiesByRow = useMemo(() => {
    if (!inReview || !sessionReview) {
      return new Map<number, SessionReviewEntity[]>();
    }

    const map = new Map<number, SessionReviewEntity[]>();
    sessionReview.entities.forEach(entity => {
      const approved = approvedMap.get(entity.id);
      const shaped: SessionReviewEntity = { ...entity, approved: approved !== false };
      map.set(entity.row_number, [...(map.get(entity.row_number) || []), shaped]);
    });
    return map;
  }, [inReview, sessionReview, approvedMap]);

  const entityTypeStats = useMemo(() => {
    if (!inReview || !sessionReview) {
      return [] as Array<{ type: string; total: number; approved: number }>;
    }

    const stats = new Map<string, { total: number; approved: number }>();
    sessionReview.entities.forEach(entity => {
      const approved = approvedMap.get(entity.id) !== false;
      const current = stats.get(entity.type) || { total: 0, approved: 0 };
      current.total += 1;
      if (approved) {
        current.approved += 1;
      }
      stats.set(entity.type, current);
    });

    return Array.from(stats.entries())
      .map(([type, value]) => ({ type, ...value }))
      .sort((a, b) => b.total - a.total);
  }, [inReview, sessionReview, approvedMap]);

  const totalEntitiesFound = useMemo(() => {
    if (inReview && sessionReview) {
      return sessionReview.summary.total_entities;
    }
    if (results) {
      return results.rows.reduce((sum, row) => sum + row.entities_found, 0);
    }
    if (legacyResults) {
      return legacyResults.entities_found;
    }
    return 0;
  }, [inReview, sessionReview, results, legacyResults]);

  const redactedRows = useMemo(() => {
    if (inReview && sessionReview) {
      return sessionReview.rows.filter(row => row.review_redacted_text !== row.original_text).length;
    }
    if (results) {
      return results.rows.filter(row => row.was_redacted).length;
    }
    return 0;
  }, [inReview, sessionReview, results]);

  const avgProcessingTime = useMemo(() => {
    if (tableRows.length === 0) {
      return 0;
    }

    const total = tableRows.reduce((sum, row) => {
      return sum + row.processing_time_ms;
    }, 0);

    return total / tableRows.length;
  }, [tableRows]);

  // Virtual scrolling calculations
  const visibleRows = useMemo(() => {
    return tableRows.slice(visibleRange.start, visibleRange.end);
  }, [tableRows, visibleRange]);

  const totalHeight = tableRows.length * ROW_HEIGHT;
  const offsetY = visibleRange.start * ROW_HEIGHT;

  const toggleEntityApproval = (entityId: number, approved: boolean) => {
    setApprovedMap(prev => new Map(prev).set(entityId, approved));
  };

  const handleBulkToggle = (entityType: string, approve: boolean) => {
    if (!sessionReview) {
      return;
    }

    const updated = new Map(approvedMap);
    sessionReview.entities
      .filter(entity => entity.type === entityType)
      .forEach(entity => {
        updated.set(entity.id, approve);
      });
    setApprovedMap(updated);
  };

  // Virtual scrolling handlers
  const updateVisibleRange = useCallback((scrollTop: number, containerHeight: number) => {
    const totalRows = tableRows.length;
    const startIndex = Math.floor(scrollTop / ROW_HEIGHT);
    const endIndex = Math.min(
      startIndex + Math.ceil(containerHeight / ROW_HEIGHT) + BUFFER_SIZE,
      totalRows
    );

    const bufferedStart = Math.max(0, startIndex - BUFFER_SIZE);
    setVisibleRange({ start: bufferedStart, end: endIndex });
  }, [tableRows.length, ROW_HEIGHT, BUFFER_SIZE]);

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const scrollTop = e.currentTarget.scrollTop;
    setScrollTop(scrollTop);
    updateVisibleRange(scrollTop, containerHeight);
  }, [updateVisibleRange, containerHeight]);

  // Update visible range when data changes
  useEffect(() => {
    updateVisibleRange(scrollTop, containerHeight);
  }, [tableRows.length, updateVisibleRange, scrollTop, containerHeight]);

  const handleExport = async () => {
    if (!sessionReview || !sessionId) {
      return;
    }

    try {
      setExporting(true);

      const skippedIds: number[] = [];
      sessionReview.entities.forEach(entity => {
        const approved = approvedMap.get(entity.id);
        if (approved === false) {
          skippedIds.push(entity.id);
        }
      });

      const blob = await exportSessionResults(sessionReview.session_id, {
        redaction_mode: redactionMode,
        custom_labels: customLabels,
        skipped_entity_ids: skippedIds,
        include_error_field: includeErrors,
      });

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${sessionReview.filename || 'results'}_redacted.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      await loadSessionReview(sessionReview.session_id); // Refresh state with persisted decisions
    } catch (err) {
      console.error('Export failed', err);
      alert(err instanceof Error ? err.message : 'Failed to export results');
    } finally {
      setExporting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-4xl mb-4">🔄</div>
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

  if (!sessionReview && !results && !legacyResults) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-4xl mb-4">📭</div>
          <div className="text-xl font-semibold text-gray-700">No results found</div>
        </div>
      </div>
    );
  }

  const title = (() => {
    if (inReview && sessionReview) {
      return `Review and finalize redaction for: ${sessionReview.filename}`;
    }
    if (results) {
      return `Review detected PII entities from file: ${results.filename}`;
    }
    if (legacyResults) {
      return `Legacy processing result: ${legacyResults.filename}`;
    }
    return 'Processing Results';
  })();

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h1 className="text-3xl font-bold text-gray-900">Processing Results</h1>
        <p className="text-gray-600 mt-2">{title}</p>
      </div>

      {legacyResults && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Legacy Result Overview</h3>
          <p className="text-sm text-gray-600 mb-4">
            This result was processed before per-row history was available. You can still download the
            original export from the history page.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-4 rounded">
              <div className="text-2xl font-bold text-blue-600">{legacyResults.entities_found}</div>
              <div className="text-sm text-gray-600">Entities Found</div>
            </div>
            <div className="bg-green-50 p-4 rounded">
              <div className="text-2xl font-bold text-green-600">{legacyResults.rows_processed}</div>
              <div className="text-sm text-gray-600">Rows Processed</div>
            </div>
            <div className="bg-purple-50 p-4 rounded">
              <div className="text-2xl font-bold text-purple-600">{legacyResults.processing_time_ms.toFixed(1)}ms</div>
              <div className="text-sm text-gray-600">Processing Time</div>
            </div>
          </div>
        </div>
      )}

      {!legacyResults && (
        <>
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Processing Results</h3>
              {inReview && (
                <div className="flex items-center space-x-2">
                  <label className="text-sm text-gray-600">Redaction Mode</label>
                  <select
                    value={redactionMode}
                    onChange={(e) => setRedactionMode(e.target.value)}
                    className="border border-gray-300 rounded px-2 py-1 text-sm"
                  >
                    <option value="replace">Replace with labels</option>
                    <option value="mask">Mask with asterisks</option>
                    <option value="remove">Remove completely</option>
                  </select>
                  <label className="flex items-center space-x-1 text-sm text-gray-600">
                    <input
                      type="checkbox"
                      checked={includeErrors}
                      onChange={(e) => setIncludeErrors(e.target.checked)}
                    />
                    <span>Include error column</span>
                  </label>
                  <button
                    onClick={handleExport}
                    disabled={exporting}
                    className="bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 text-sm disabled:bg-gray-400"
                  >
                    {exporting ? 'Exporting…' : 'Export CSV'}
                  </button>
                </div>
              )}
            </div>
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

          {inReview && (
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Entity Overview</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left font-medium text-gray-500 uppercase tracking-wide">Type</th>
                      <th className="px-4 py-2 text-left font-medium text-gray-500 uppercase tracking-wide">Approved</th>
                      <th className="px-4 py-2 text-left font-medium text-gray-500 uppercase tracking-wide">Actions</th>
                      <th className="px-4 py-2 text-left font-medium text-gray-500 uppercase tracking-wide">Label Override</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {entityTypeStats.map(stat => (
                      <tr key={stat.type}>
                        <td className="px-4 py-2 font-medium text-gray-800">{stat.type}</td>
                        <td className="px-4 py-2 text-gray-700">{stat.approved} / {stat.total}</td>
                        <td className="px-4 py-2 space-x-2">
                          <button
                            className="text-blue-600 hover:text-blue-800"
                            onClick={() => handleBulkToggle(stat.type, true)}
                          >
                            Redact All
                          </button>
                          <button
                            className="text-red-600 hover:text-red-800"
                            onClick={() => handleBulkToggle(stat.type, false)}
                          >
                            Keep All
                          </button>
                        </td>
                        <td className="px-4 py-2">
                          <input
                            type="text"
                            className="border border-gray-300 rounded px-2 py-1 text-sm w-full"
                            value={customLabels[stat.type] || ''}
                            placeholder={`e.g. [REDACTED_${stat.type.toUpperCase()}]`}
                            onChange={(e) => setCustomLabels(prev => ({ ...prev, [stat.type]: e.target.value }))}
                          />
                        </td>
                      </tr>
                    ))}
                    {entityTypeStats.length === 0 && (
                      <tr>
                        <td className="px-4 py-2 text-gray-500" colSpan={4}>
                          No detected entities in this session.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-900">Processing Details</h3>
                <div className="flex items-center space-x-4">
                  <span className="text-sm text-gray-600">
                    Showing {visibleRange.start + 1}-{Math.min(visibleRange.end, tableRows.length)} of {tableRows.length} rows
                  </span>
                  {inReview && (
                    <details className="text-sm text-gray-600">
                      <summary className="cursor-pointer select-none">Custom Labels (JSON)</summary>
                      <pre className="mt-2 bg-gray-50 border border-gray-200 rounded p-2 text-xs text-gray-700 whitespace-pre-wrap">
{JSON.stringify(customLabels, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            </div>
            <div
              className="overflow-auto"
              style={{ height: `${containerHeight}px` }}
              onScroll={handleScroll}
            >
              <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
                <div style={{ transform: `translateY(${offsetY}px)` }}>
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50 sticky top-0 z-10">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Row</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Original Text (Preview)</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {inReview ? 'Entities (Approved / Detected)' : 'Entities Found'}
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Processing Time</th>
                        {inReview && (
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Entity Decisions</th>
                        )}
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {visibleRows.map((row, index) => {
                        const rowNumber = row.row_number;
                        const originalText = row.original_text;
                        const hasError = isReviewRow(row) ? row.status === 'error' : row.has_error;
                        const wasRedacted = isReviewRow(row)
                          ? row.review_redacted_text !== row.original_text
                          : row.was_redacted;
                        const detectedCount = isReviewRow(row) ? row.detected_entities : row.entities_found;
                        const approvedCount = isReviewRow(row) ? row.approved_entities : row.entities_found;
                        const processingTime = row.processing_time_ms;
                        const entitiesForRow = inReview ? (reviewEntitiesByRow.get(rowNumber) || []) : [];

                        return (
                          <tr
                            key={`row-${rowNumber}`}
                            style={{ height: `${ROW_HEIGHT}px` }}
                            className="border-b border-gray-200"
                          >
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{rowNumber}</td>
                            <td className="px-6 py-4 text-sm text-gray-900 max-w-md">
                              <div className="truncate">
                                {originalText.length > 100
                                  ? `${originalText.substring(0, 100)}...`
                                  : originalText}
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                              {hasError ? (
                                <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-red-100 text-red-800">Error</span>
                              ) : wasRedacted ? (
                                <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-yellow-100 text-yellow-800">Redacted</span>
                              ) : (
                                <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">Clean</span>
                              )}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {inReview ? `${approvedCount}/${detectedCount}` : detectedCount}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{processingTime}ms</td>
                            {inReview && (
                              <td className="px-6 py-4 text-sm text-gray-900">
                                <div className="space-y-2 max-h-16 overflow-y-auto">
                                  {entitiesForRow.length === 0 && (
                                    <span className="text-xs text-gray-400">No detected entities</span>
                                  )}
                                  {entitiesForRow.slice(0, 3).map(entity => {
                                    const approved = approvedMap.get(entity.id) !== false;
                                    return (
                                      <div key={entity.id} className="border border-gray-200 rounded p-2">
                                        <div className="flex items-center justify-between">
                                          <span className="text-xs font-semibold text-gray-700">{entity.type}</span>
                                          <label className="flex items-center space-x-1 text-xs">
                                            <input
                                              type="checkbox"
                                              checked={approved}
                                              onChange={(e) => toggleEntityApproval(entity.id, e.target.checked)}
                                            />
                                            <span>{approved ? 'Redact' : 'Keep'}</span>
                                          </label>
                                        </div>
                                        <div className="text-xs text-gray-500 truncate">{entity.text}</div>
                                        <div className="text-xs text-gray-400">Confidence: {(entity.confidence * 100).toFixed(1)}%</div>
                                      </div>
                                    );
                                  })}
                                  {entitiesForRow.length > 3 && (
                                    <div className="text-xs text-gray-500">
                                      + {entitiesForRow.length - 3} more entities
                                    </div>
                                  )}
                                </div>
                              </td>
                            )}
                          </tr>
                        );
                      })}
                      {tableRows.length === 0 && (
                        <tr>
                          <td className="px-6 py-4 text-sm text-gray-500" colSpan={inReview ? 6 : 5}>
                            No row level results available for this session yet.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      <div className="flex justify-between">
        <button
          className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
          onClick={() => navigate('/history')}
        >
          ← Back to History
        </button>
        {!inReview && !legacyResults && (
          <button
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            onClick={() => alert('Download functionality coming soon!')}
          >
            Download Results →
          </button>
        )}
      </div>
    </div>
  );
};

