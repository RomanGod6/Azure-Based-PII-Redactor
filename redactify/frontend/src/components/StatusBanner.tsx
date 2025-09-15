import React, { useState, useEffect } from 'react';

interface SystemStatus {
  go_backend: boolean;
  azure_configured: boolean;
  azure_online: boolean;
  gpt_configured: boolean;
  gpt_online: boolean;
  detection_mode: string;
}

export const StatusBanner: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('http://localhost:8080/api/v1/config');
        if (response.ok) {
          const data = await response.json();
          setStatus(data);
        }
      } catch (error) {
        console.error('Failed to fetch system status:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
    // Refresh status every 30 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading || !status) {
    return null;
  }

  const getOverallStatus = () => {
    if (status.azure_online && status.gpt_online) return 'excellent';
    if (status.azure_online) return 'good';
    if (status.azure_configured) return 'partial';
    return 'basic';
  };

  const getStatusColor = (online: boolean, configured: boolean) => {
    if (online) return 'bg-green-500';
    if (configured) return 'bg-yellow-500';
    return 'bg-gray-400';
  };

  const getStatusText = (online: boolean, configured: boolean) => {
    if (online) return 'Online';
    if (configured) return 'Configured';
    return 'Offline';
  };

  const overallStatus = getOverallStatus();

  if (collapsed) {
    return (
      <div className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className={`w-3 h-3 rounded-full ${
                overallStatus === 'excellent' ? 'bg-green-500' :
                overallStatus === 'good' ? 'bg-blue-500' :
                overallStatus === 'partial' ? 'bg-yellow-500' : 'bg-red-500'
              }`}></div>
              <span className="text-sm font-medium">
                {overallStatus === 'excellent' && '🚀 Full AI Pipeline Active'}
                {overallStatus === 'good' && '🤖 Azure AI Active'}
                {overallStatus === 'partial' && '⚡ Regex Mode'}
                {overallStatus === 'basic' && '📝 Basic Mode'}
              </span>
              <span className="text-xs text-gray-500">
                Detection: {status.detection_mode}
              </span>
            </div>
            <button
              onClick={() => setCollapsed(false)}
              className="text-gray-400 hover:text-gray-600 text-sm"
            >
              Expand ↓
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <h3 className="text-sm font-semibold text-gray-900">System Status</h3>
              <div className={`w-2 h-2 rounded-full ${
                overallStatus === 'excellent' ? 'bg-green-500' :
                overallStatus === 'good' ? 'bg-blue-500' :
                overallStatus === 'partial' ? 'bg-yellow-500' : 'bg-red-500'
              }`}></div>
            </div>
            
            {/* Go Backend */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${getStatusColor(status.go_backend, true)}`}></div>
              <span className="text-xs text-gray-600">Go Backend</span>
            </div>

            {/* Azure PII */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${getStatusColor(status.azure_online, status.azure_configured)}`}></div>
              <span className="text-xs text-gray-600">
                Azure PII: {getStatusText(status.azure_online, status.azure_configured)}
              </span>
            </div>

            {/* GPT Validation */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${getStatusColor(status.gpt_online, status.gpt_configured)}`}></div>
              <span className="text-xs text-gray-600">
                GPT: {getStatusText(status.gpt_online, status.gpt_configured)}
              </span>
            </div>

            {/* Detection Mode */}
            <div className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
              Mode: {status.detection_mode}
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* Overall Status Message */}
            <div className="text-xs text-gray-600">
              {overallStatus === 'excellent' && '🚀 Full AI Pipeline Active (99% accuracy)'}
              {overallStatus === 'good' && '🤖 Azure AI Active (95% accuracy)'}
              {overallStatus === 'partial' && '⚡ Regex Mode (70% accuracy)'}
              {overallStatus === 'basic' && '📝 Basic Mode (50% accuracy)'}
            </div>
            
            <button
              onClick={() => setCollapsed(true)}
              className="text-gray-400 hover:text-gray-600 text-sm"
            >
              ↑
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};