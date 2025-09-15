import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface AnalyticsData {
  total_processed: number;
  accuracy_rate: number;
  average_speed: string;
  entities_detected: number;
}

interface HistoryData {
  history: any[];
  total: number;
}

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [history, setHistory] = useState<HistoryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [analyticsRes, historyRes] = await Promise.all([
          fetch('http://localhost:8080/api/v1/analytics'),
          fetch('http://localhost:8080/api/v1/history?limit=3')
        ]);

        if (!analyticsRes.ok || !historyRes.ok) {
          throw new Error('Failed to fetch data');
        }

        const analyticsData = await analyticsRes.json();
        const historyData = await historyRes.json();

        setAnalytics(analyticsData);
        setHistory(historyData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="border-b border-gray-200 pb-4">
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-2">
            Welcome to Redactify - Your AI-powered PII redaction tool
          </p>
        </div>
        <div className="card">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div className="border-b border-gray-200 pb-4">
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-2">
            Welcome to Redactify - Your AI-powered PII redaction tool
          </p>
        </div>
        <div className="card bg-red-50 border-red-200">
          <div className="text-red-800">
            <h3 className="font-semibold">Error loading dashboard data</h3>
            <p className="text-sm mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Welcome to Redactify - Your AI-powered PII redaction tool
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Stats Cards */}
        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center">
                <span className="text-primary-600 font-semibold">📁</span>
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Files Processed</p>
              <p className="text-2xl font-bold text-gray-900">
                {analytics?.total_processed ?? 0}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-success-100 rounded-lg flex items-center justify-center">
                <span className="text-success-600 font-semibold">🎯</span>
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Accuracy Rate</p>
              <p className="text-2xl font-bold text-gray-900">
                {analytics?.accuracy_rate && analytics.accuracy_rate > 0 ? `${analytics.accuracy_rate}%` : 'N/A'}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-warning-100 rounded-lg flex items-center justify-center">
                <span className="text-warning-600 font-semibold">⚡</span>
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg Speed</p>
              <p className="text-2xl font-bold text-gray-900">
                {analytics?.average_speed ?? 'N/A'}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-danger-100 rounded-lg flex items-center justify-center">
                <span className="text-danger-600 font-semibold">🛡️</span>
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">PII Detected</p>
              <p className="text-2xl font-bold text-gray-900">
                {analytics?.entities_detected ?? 0}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Quick Actions */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
          <div className="space-y-3">
            <button 
              className="btn-primary w-full"
              onClick={() => navigate('/process')}
            >
              🚀 Process New File
            </button>
            <button 
              className="btn-outline w-full"
              onClick={() => navigate('/analytics')}
            >
              📊 View Analytics
            </button>
            <button 
              className="btn-outline w-full"
              onClick={() => navigate('/settings')}
            >
              ⚙️ Configure Settings
            </button>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
          <div className="space-y-3">
            {history?.history && history.history.length > 0 ? (
              history.history.map((item: any, index: number) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium text-gray-900">{item.filename || `Processing Job #${index + 1}`}</p>
                    <p className="text-sm text-gray-500">{item.timestamp || 'Recently processed'}</p>
                  </div>
                  <span className={`badge-${item.status === 'complete' ? 'success' : 'info'}`}>
                    {item.status || 'Complete'}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-center py-8">
                <div className="text-gray-400 text-4xl mb-2">📄</div>
                <p className="text-gray-500 text-sm">No processing history yet</p>
                <p className="text-gray-400 text-xs mt-1">Process your first file to see activity here</p>
                <button 
                  className="btn-primary mt-3"
                  onClick={() => navigate('/process')}
                >
                  🚀 Process Your First File
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};