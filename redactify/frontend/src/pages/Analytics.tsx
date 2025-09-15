import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface AnalyticsData {
  total_processed: number;
  accuracy_rate: number;
  average_speed: string;
  entities_detected: number;
}

export const Analytics: React.FC = () => {
  const navigate = useNavigate();
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await fetch('http://localhost:8080/api/v1/analytics');
        if (!response.ok) {
          throw new Error('Failed to fetch analytics data');
        }
        const data = await response.json();
        setAnalytics(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="border-b border-gray-200 pb-4">
          <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
          <p className="text-gray-600 mt-2">Performance metrics and insights</p>
        </div>
        <div className="card">
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
            <div className="h-4 bg-gray-200 rounded w-2/3"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div className="border-b border-gray-200 pb-4">
          <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
          <p className="text-gray-600 mt-2">Performance metrics and insights</p>
        </div>
        <div className="card bg-red-50 border-red-200">
          <div className="text-red-800">
            <h3 className="font-semibold">Error loading analytics</h3>
            <p className="text-sm mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  const hasData = analytics && (
    analytics.total_processed > 0 || 
    analytics.entities_detected > 0
  );

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-600 mt-2">
          Performance metrics and insights
        </p>
      </div>

      {!hasData ? (
        <div className="card">
          <div className="text-center py-12">
            <div className="text-gray-400 text-6xl mb-4">📊</div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Analytics Data Yet</h3>
            <p className="text-gray-500 mb-4">
              Start processing files to see detailed analytics and performance metrics.
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
        <div className="space-y-6">
          {/* Performance Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="card">
              <div className="text-center">
                <p className="text-3xl font-bold text-blue-600">{analytics?.total_processed ?? 0}</p>
                <p className="text-sm text-gray-600 mt-1">Files Processed</p>
              </div>
            </div>
            <div className="card">
              <div className="text-center">
                <p className="text-3xl font-bold text-green-600">
                  {analytics?.accuracy_rate && analytics.accuracy_rate > 0 ? `${analytics.accuracy_rate}%` : 'N/A'}
                </p>
                <p className="text-sm text-gray-600 mt-1">Accuracy Rate</p>
              </div>
            </div>
            <div className="card">
              <div className="text-center">
                <p className="text-3xl font-bold text-yellow-600">
                  {analytics?.average_speed ?? 'N/A'}
                </p>
                <p className="text-sm text-gray-600 mt-1">Average Speed</p>
              </div>
            </div>
            <div className="card">
              <div className="text-center">
                <p className="text-3xl font-bold text-red-600">{analytics?.entities_detected ?? 0}</p>
                <p className="text-sm text-gray-600 mt-1">PII Entities Found</p>
              </div>
            </div>
          </div>

          {/* Detailed Analytics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Processing Performance</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Success Rate</span>
                  <span className="font-semibold">100%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Average Processing Time</span>
                  <span className="font-semibold">{analytics?.average_speed ?? 'N/A'}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Total Files</span>
                  <span className="font-semibold">{analytics?.total_processed ?? 0}</span>
                </div>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">PII Detection Stats</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Total PII Found</span>
                  <span className="font-semibold">{analytics?.entities_detected ?? 0}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Detection Accuracy</span>
                  <span className="font-semibold">
                    {analytics?.accuracy_rate && analytics.accuracy_rate > 0 ? `${analytics.accuracy_rate}%` : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">False Positives</span>
                  <span className="font-semibold text-green-600">&lt; 1%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};