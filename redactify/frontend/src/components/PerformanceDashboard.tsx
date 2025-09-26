import React, { useState, useEffect } from 'react';
import { performanceMonitor, PerformanceConfig, SystemHealth, PerformanceMetrics } from '../services/performanceMonitor';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export const PerformanceDashboard: React.FC<Props> = ({ isOpen, onClose }) => {
  const [config, setConfig] = useState<PerformanceConfig>(performanceMonitor.getConfig());
  const [systemHealth, setSystemHealth] = useState<SystemHealth>(performanceMonitor.getSystemHealth());
  const [recentMetrics, setRecentMetrics] = useState<PerformanceMetrics[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'config'>('overview');
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (isOpen) {
      // Start monitoring when dashboard opens
      performanceMonitor.startMonitoring();

      // Set up refresh interval
      const interval = setInterval(() => {
        setSystemHealth(performanceMonitor.getSystemHealth());
        setRecentMetrics(performanceMonitor.getRecentMetrics(300000)); // Last 5 minutes
      }, 2000);

      setRefreshInterval(interval);

      // Initial load
      setRecentMetrics(performanceMonitor.getRecentMetrics(300000));
    } else {
      // Clear refresh interval when closed
      if (refreshInterval) {
        clearInterval(refreshInterval);
        setRefreshInterval(null);
      }
    }

    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, [isOpen]);

  const handleConfigChange = (key: keyof PerformanceConfig, value: any) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    performanceMonitor.updateConfig({ [key]: value });
  };

  const handleThresholdChange = (threshold: keyof PerformanceConfig['alertThresholds'], value: number) => {
    const newThresholds = { ...config.alertThresholds, [threshold]: value };
    const newConfig = { ...config, alertThresholds: newThresholds };
    setConfig(newConfig);
    performanceMonitor.updateConfig({ alertThresholds: newThresholds });
  };

  const resetConfig = () => {
    performanceMonitor.resetConfig();
    setConfig(performanceMonitor.getConfig());
  };

  const exportMetrics = () => {
    const data = performanceMonitor.exportMetrics();
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `performance-metrics-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const clearMetrics = () => {
    if (window.confirm('Are you sure you want to clear all performance metrics?')) {
      performanceMonitor.clearMetrics();
      setRecentMetrics([]);
    }
  };

  // Calculate averages for display
  const avgProcessingTime = recentMetrics.filter(m => m.processingTime).reduce((sum, m) => sum + (m.processingTime || 0), 0) / Math.max(1, recentMetrics.filter(m => m.processingTime).length);
  const avgMemoryUsage = recentMetrics.filter(m => m.memoryUsage).reduce((sum, m) => sum + (m.memoryUsage?.percentage || 0), 0) / Math.max(1, recentMetrics.filter(m => m.memoryUsage).length);
  const totalErrors = recentMetrics.filter(m => m.errorCount).length;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-2xl font-bold text-gray-900">Performance Dashboard</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl"
          >
            ×
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b">
          {['overview', 'metrics', 'config'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab as any)}
              className={`px-6 py-3 font-medium capitalize ${
                activeTab === tab
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="p-6 overflow-y-auto max-h-[70vh]">
          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* System Health */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold">System Health</h3>
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                    systemHealth.status === 'healthy' ? 'bg-green-100 text-green-800' :
                    systemHealth.status === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {systemHealth.status.charAt(0).toUpperCase() + systemHealth.status.slice(1)}
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Health Score</span>
                    <span>{systemHealth.score}/100</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        systemHealth.score >= 80 ? 'bg-green-500' :
                        systemHealth.score >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${systemHealth.score}%` }}
                    />
                  </div>
                </div>

                {systemHealth.issues.length > 0 && (
                  <div className="mb-4">
                    <h4 className="font-medium text-red-700 mb-2">Issues:</h4>
                    <ul className="list-disc list-inside text-sm text-red-600 space-y-1">
                      {systemHealth.issues.map((issue, idx) => (
                        <li key={idx}>{issue}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {systemHealth.recommendations.length > 0 && (
                  <div>
                    <h4 className="font-medium text-blue-700 mb-2">Recommendations:</h4>
                    <ul className="list-disc list-inside text-sm text-blue-600 space-y-1">
                      {systemHealth.recommendations.map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Key Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">Avg Processing Time</h4>
                  <div className="text-2xl font-bold text-blue-700">
                    {avgProcessingTime > 0 ? `${avgProcessingTime.toFixed(0)}ms` : 'N/A'}
                  </div>
                </div>

                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-medium text-green-900 mb-2">Memory Usage</h4>
                  <div className="text-2xl font-bold text-green-700">
                    {avgMemoryUsage > 0 ? `${avgMemoryUsage.toFixed(1)}%` : 'N/A'}
                  </div>
                </div>

                <div className="bg-red-50 p-4 rounded-lg">
                  <h4 className="font-medium text-red-900 mb-2">Errors (5min)</h4>
                  <div className="text-2xl font-bold text-red-700">{totalErrors}</div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex space-x-4">
                <button
                  onClick={exportMetrics}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Export Metrics
                </button>
                <button
                  onClick={clearMetrics}
                  className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
                >
                  Clear Metrics
                </button>
              </div>
            </div>
          )}

          {/* Metrics Tab */}
          {activeTab === 'metrics' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">Recent Metrics (Last 5 minutes)</h3>
                <div className="text-sm text-gray-600">
                  {recentMetrics.length} data points
                </div>
              </div>

              {recentMetrics.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No metrics data available
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Processing Time</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Memory %</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Rows/sec</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Errors</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {recentMetrics.slice(-20).reverse().map((metric, idx) => (
                        <tr key={idx} className="hover:bg-gray-50">
                          <td className="px-4 py-2 text-sm text-gray-600">
                            {new Date(metric.timestamp).toLocaleTimeString()}
                          </td>
                          <td className="px-4 py-2 text-sm">
                            {metric.processingTime ? `${metric.processingTime}ms` : '-'}
                          </td>
                          <td className="px-4 py-2 text-sm">
                            {metric.memoryUsage ? `${metric.memoryUsage.percentage}%` : '-'}
                          </td>
                          <td className="px-4 py-2 text-sm">
                            {metric.rowsPerSecond ? metric.rowsPerSecond.toFixed(1) : '-'}
                          </td>
                          <td className="px-4 py-2 text-sm">
                            {metric.errorCount || 0}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* Config Tab */}
          {activeTab === 'config' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">Performance Configuration</h3>
                <button
                  onClick={resetConfig}
                  className="px-3 py-1 text-sm bg-gray-600 text-white rounded hover:bg-gray-700"
                >
                  Reset to Defaults
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="font-medium text-gray-900">Processing Settings</h4>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Max Concurrent Requests
                    </label>
                    <input
                      type="number"
                      value={config.maxConcurrentRequests}
                      onChange={(e) => handleConfigChange('maxConcurrentRequests', parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded px-3 py-2"
                      min="1"
                      max="20"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Batch Size
                    </label>
                    <input
                      type="number"
                      value={config.batchSize}
                      onChange={(e) => handleConfigChange('batchSize', parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded px-3 py-2"
                      min="10"
                      max="1000"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Request Timeout (ms)
                    </label>
                    <input
                      type="number"
                      value={config.requestTimeout}
                      onChange={(e) => handleConfigChange('requestTimeout', parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded px-3 py-2"
                      min="5000"
                      max="120000"
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-medium text-gray-900">Alert Thresholds</h4>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Processing Time Alert (ms)
                    </label>
                    <input
                      type="number"
                      value={config.alertThresholds.processingTime}
                      onChange={(e) => handleThresholdChange('processingTime', parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded px-3 py-2"
                      min="1000"
                      max="30000"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Memory Usage Alert (%)
                    </label>
                    <input
                      type="number"
                      value={config.alertThresholds.memoryUsage}
                      onChange={(e) => handleThresholdChange('memoryUsage', parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded px-3 py-2"
                      min="50"
                      max="95"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Error Rate Alert (%)
                    </label>
                    <input
                      type="number"
                      value={config.alertThresholds.errorRate * 100}
                      onChange={(e) => handleThresholdChange('errorRate', parseInt(e.target.value) / 100)}
                      className="w-full border border-gray-300 rounded px-3 py-2"
                      min="1"
                      max="50"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};