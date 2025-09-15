import React, { useState, useEffect } from 'react';

interface ConfigData {
  azure_configured: boolean;
  gpt_configured: boolean;
  go_backend: boolean;
}

export const Settings: React.FC = () => {
  const [config, setConfig] = useState<ConfigData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [azureEndpoint, setAzureEndpoint] = useState('');
  const [azureApiKey, setAzureApiKey] = useState('');
  const [gptEndpoint, setGptEndpoint] = useState('');
  const [gptApiKey, setGptApiKey] = useState('');
  const [showApiKeys, setShowApiKeys] = useState(false);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/v1/config');
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error('Failed to fetch config:', error);
    } finally {
      setLoading(false);
    }
  };

  const saveConfig = async () => {
    setSaving(true);
    try {
      const response = await fetch('http://localhost:8080/api/v1/config', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          azure_endpoint: azureEndpoint,
          azure_api_key: azureApiKey,
          gpt_endpoint: gptEndpoint,
          gpt_api_key: gptApiKey,
        }),
      });

      if (response.ok) {
        alert('Configuration saved successfully!');
        fetchConfig(); // Refresh config status
      } else {
        alert('Failed to save configuration');
      }
    } catch (error) {
      console.error('Failed to save config:', error);
      alert('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="border-b border-gray-200 pb-4">
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <p className="text-gray-600 mt-2">
            Configure Azure credentials and application preferences
          </p>
        </div>
        <div className="card">
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600 mt-2">
          Configure Azure credentials and application preferences
        </p>
      </div>

      {/* System Status */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${config?.go_backend ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm">Go Backend</span>
            <span className={`text-xs px-2 py-1 rounded ${config?.go_backend ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
              {config?.go_backend ? 'Running' : 'Offline'}
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${config?.azure_configured ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
            <span className="text-sm">Azure Text Analytics</span>
            <span className={`text-xs px-2 py-1 rounded ${config?.azure_configured ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
              {config?.azure_configured ? 'Configured' : 'Not Configured'}
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${config?.gpt_configured ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
            <span className="text-sm">GPT Validation</span>
            <span className={`text-xs px-2 py-1 rounded ${config?.gpt_configured ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
              {config?.gpt_configured ? 'Configured' : 'Not Configured'}
            </span>
          </div>
        </div>
      </div>

      {/* Azure Configuration */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Azure Text Analytics</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Azure Endpoint URL
            </label>
            <input
              type="url"
              value={azureEndpoint}
              onChange={(e) => setAzureEndpoint(e.target.value)}
              placeholder="https://your-resource.cognitiveservices.azure.com"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              API Key
            </label>
            <div className="relative">
              <input
                type={showApiKeys ? "text" : "password"}
                value={azureApiKey}
                onChange={(e) => setAzureApiKey(e.target.value)}
                placeholder="Enter your Azure API key"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 pr-10 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              <button
                type="button"
                onClick={() => setShowApiKeys(!showApiKeys)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center"
              >
                <span className="text-gray-400 text-sm">{showApiKeys ? '🙈' : '👁️'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* GPT Configuration */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">GPT Validation (Optional)</h3>
        <p className="text-sm text-gray-600 mb-4">
          Configure GPT-5 to validate PII detections and reduce false positives.
        </p>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              GPT Endpoint URL
            </label>
            <input
              type="url"
              value={gptEndpoint}
              onChange={(e) => setGptEndpoint(e.target.value)}
              placeholder="https://your-openai-resource.openai.azure.com"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              GPT API Key
            </label>
            <input
              type={showApiKeys ? "text" : "password"}
              value={gptApiKey}
              onChange={(e) => setGptApiKey(e.target.value)}
              placeholder="Enter your GPT API key"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        </div>
      </div>

      {/* Default Settings */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Default Processing Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Default Detection Mode
            </label>
            <select className="w-full border border-gray-300 rounded-lg px-3 py-2">
              <option value="balanced">Balanced (Recommended)</option>
              <option value="conservative">Conservative</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Default Redaction Mode
            </label>
            <select className="w-full border border-gray-300 rounded-lg px-3 py-2">
              <option value="replace">Replace with Labels</option>
              <option value="mask">Mask with Asterisks</option>
              <option value="remove">Remove Completely</option>
            </select>
          </div>
        </div>
      </div>

      {/* Performance Settings */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Settings</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-900">Enable GPT Validation</h4>
              <p className="text-sm text-gray-500">Use GPT to validate PII detections (slower but more accurate)</p>
            </div>
            <input type="checkbox" className="h-4 w-4 text-blue-600" defaultChecked />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-900">Parallel Processing</h4>
              <p className="text-sm text-gray-500">Process multiple files simultaneously</p>
            </div>
            <input type="checkbox" className="h-4 w-4 text-blue-600" defaultChecked />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-900">Save Processing History</h4>
              <p className="text-sm text-gray-500">Keep records of all processed files</p>
            </div>
            <input type="checkbox" className="h-4 w-4 text-blue-600" defaultChecked />
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <button
          onClick={saveConfig}
          disabled={saving}
          className={`btn-primary ${saving ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {saving ? 'Saving...' : 'Save Configuration'}
        </button>
      </div>
    </div>
  );
};