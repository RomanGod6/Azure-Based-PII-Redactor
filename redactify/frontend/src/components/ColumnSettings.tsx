import React, { useState, useEffect } from 'react';

interface ColumnSettingsProps {
  sessionId: string;
  isOpen: boolean;
  onClose: () => void;
  onSettingsUpdated?: () => void;
}

interface CSVMetadata {
  session_id: string;
  headers: string[];
  column_pii_settings: Record<string, boolean>;
  delimiter: string;
  has_headers: boolean;
  total_columns: number;
}

export const ColumnSettings: React.FC<ColumnSettingsProps> = ({
  sessionId,
  isOpen,
  onClose,
  onSettingsUpdated
}) => {
  const [metadata, setMetadata] = useState<CSVMetadata | null>(null);
  const [settings, setSettings] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadMetadata = async () => {
    if (!sessionId) return;

    setLoading(true);
    setError(null);

    try {
      console.log('🔍 Fetching CSV metadata for session:', sessionId);
      const response = await fetch(`/api/v1/files/sessions/${sessionId}/csv-metadata`);

      if (!response.ok) {
        if (response.status === 404) {
          setError('No CSV metadata found. This feature is only available for CSV files processed after the column settings feature was added.');
          return;
        }
        const errorText = await response.text();
        console.error('❌ CSV metadata request failed:', response.status, errorText);
        throw new Error(`Failed to load CSV metadata: ${response.status} ${errorText}`);
      }

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const responseText = await response.text();
        console.error('❌ Expected JSON but got:', contentType, responseText.substring(0, 200));
        throw new Error('Server returned HTML instead of JSON. Please try again.');
      }

      const data: CSVMetadata = await response.json();
      setMetadata(data);
      setSettings({ ...data.column_pii_settings });
    } catch (err) {
      console.error('Failed to load CSV metadata:', err);
      setError(err instanceof Error ? err.message : 'Failed to load column settings');
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async () => {
    if (!sessionId) return;

    setSaving(true);
    setError(null);

    try {
      const response = await fetch(`/api/v1/files/sessions/${sessionId}/column-settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          column_pii_settings: settings
        })
      });

      if (!response.ok) {
        throw new Error('Failed to save column settings');
      }

      // Update metadata with new settings
      if (metadata) {
        setMetadata({ ...metadata, column_pii_settings: settings });
      }

      // Notify parent component
      onSettingsUpdated?.();

      // Show success message briefly
      setError(null);
    } catch (err) {
      console.error('Failed to save column settings:', err);
      setError(err instanceof Error ? err.message : 'Failed to save column settings');
    } finally {
      setSaving(false);
    }
  };

  const toggleColumn = (columnName: string) => {
    setSettings(prev => ({
      ...prev,
      [columnName]: !prev[columnName]
    }));
  };

  const toggleAll = (enabled: boolean) => {
    if (!metadata) return;

    const newSettings: Record<string, boolean> = {};
    metadata.headers.forEach(header => {
      newSettings[header] = enabled;
    });
    setSettings(newSettings);
  };

  useEffect(() => {
    if (isOpen) {
      loadMetadata();
    }
  }, [isOpen, sessionId]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-2">
            <span className="text-blue-600 text-lg">⚙️</span>
            <h2 className="text-lg font-semibold text-gray-900">Column PII Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors text-lg font-bold"
          >
            ✕
          </button>
        </div>

        <div className="p-6 overflow-y-auto max-h-[60vh]">
          {loading && (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <p className="mt-2 text-sm text-gray-600">Loading column settings...</p>
            </div>
          )}

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-start space-x-2">
                <span className="text-red-500 mt-0.5">⚠️</span>
                <div>
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <p className="mt-1 text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          {metadata && !loading && (
            <>
              <div className="mb-6">
                <h3 className="text-sm font-medium text-gray-900 mb-2">File Information</h3>
                <div className="bg-gray-50 p-3 rounded-lg text-sm">
                  <p><strong>Delimiter:</strong> "{metadata.delimiter}"</p>
                  <p><strong>Columns:</strong> {metadata.total_columns}</p>
                  <p><strong>Has Headers:</strong> {metadata.has_headers ? 'Yes' : 'No'}</p>
                </div>
              </div>

              <div className="mb-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-gray-900">
                    PII Detection Settings by Column
                  </h3>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => toggleAll(true)}
                      className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors"
                    >
                      Enable All
                    </button>
                    <button
                      onClick={() => toggleAll(false)}
                      className="text-xs px-2 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors"
                    >
                      Disable All
                    </button>
                  </div>
                </div>

                <p className="text-xs text-gray-600 mb-4">
                  Toggle PII detection for each column. Disabled columns will be exported as-is without redaction.
                </p>

                <div className="space-y-3">
                  {metadata.headers.map((header, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center space-x-3">
                        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded font-mono">
                          #{index + 1}
                        </span>
                        <span className="font-medium text-gray-900">{header}</span>
                      </div>

                      <button
                        onClick={() => toggleColumn(header)}
                        className={`flex items-center space-x-2 px-3 py-1 rounded-lg transition-all ${
                          settings[header]
                            ? 'bg-green-100 text-green-700 hover:bg-green-200'
                            : 'bg-red-100 text-red-700 hover:bg-red-200'
                        }`}
                      >
                        {settings[header] ? (
                          <>
                            <span>👁️</span>
                            <span className="text-xs font-medium">PII Enabled</span>
                          </>
                        ) : (
                          <>
                            <span>🚫</span>
                            <span className="text-xs font-medium">PII Disabled</span>
                          </>
                        )}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>

        {metadata && !loading && (
          <div className="flex items-center justify-between p-6 border-t border-gray-200 bg-gray-50">
            <div className="text-sm text-gray-600">
              {Object.values(settings).filter(Boolean).length} of {metadata.headers.length} columns have PII detection enabled
            </div>
            <div className="flex space-x-3">
              <button
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={saveSettings}
                disabled={saving}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                {saving && (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                )}
                <span>💾</span>
                <span>{saving ? 'Saving...' : 'Save Settings'}</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};