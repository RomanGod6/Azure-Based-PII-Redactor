/**
 * Performance Monitoring Service
 * Provides comprehensive performance tracking, metrics collection, and system monitoring
 */

export interface PerformanceMetrics {
  timestamp: number;
  sessionId?: string;

  // Processing metrics
  processingTime?: number;
  rowsPerSecond?: number;
  entitiesPerSecond?: number;
  fileSize?: number;

  // Memory metrics
  memoryUsage?: {
    used: number;
    total: number;
    percentage: number;
  };

  // Network metrics
  networkLatency?: number;
  networkThroughput?: number;

  // User experience metrics
  pageLoadTime?: number;
  timeToInteractive?: number;

  // Error metrics
  errorCount?: number;
  errorRate?: number;
}

export interface SystemHealth {
  status: 'healthy' | 'warning' | 'critical';
  score: number; // 0-100
  issues: string[];
  recommendations: string[];
}

export interface PerformanceConfig {
  // Processing configuration
  maxConcurrentRequests: number;
  batchSize: number;
  requestTimeout: number;
  retryAttempts: number;

  // Memory configuration
  maxMemoryUsage: number; // MB
  memoryWarningThreshold: number; // percentage

  // UI configuration
  virtualScrollThreshold: number;
  progressUpdateInterval: number;

  // Monitoring configuration
  metricsCollectionInterval: number;
  alertThresholds: {
    processingTime: number;
    memoryUsage: number;
    errorRate: number;
  };
}

class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: PerformanceMetrics[] = [];
  private config: PerformanceConfig;
  private isMonitoring = false;
  private monitoringInterval?: number;

  private readonly defaultConfig: PerformanceConfig = {
    maxConcurrentRequests: 5,
    batchSize: 100,
    requestTimeout: 30000,
    retryAttempts: 3,
    maxMemoryUsage: 512,
    memoryWarningThreshold: 80,
    virtualScrollThreshold: 1000,
    progressUpdateInterval: 1000,
    metricsCollectionInterval: 10000,
    alertThresholds: {
      processingTime: 5000,
      memoryUsage: 90,
      errorRate: 0.05
    }
  };

  public static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  private constructor() {
    this.config = this.loadConfig();
    this.initializePerformanceObserver();
  }

  private loadConfig(): PerformanceConfig {
    try {
      const saved = localStorage.getItem('pii_performance_config');
      if (saved) {
        return { ...this.defaultConfig, ...JSON.parse(saved) };
      }
    } catch (error) {
      console.warn('Failed to load performance config:', error);
    }
    return this.defaultConfig;
  }

  private saveConfig(): void {
    try {
      localStorage.setItem('pii_performance_config', JSON.stringify(this.config));
    } catch (error) {
      console.warn('Failed to save performance config:', error);
    }
  }

  private initializePerformanceObserver(): void {
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          entries.forEach((entry) => {
            this.recordWebVitals(entry);
          });
        });

        observer.observe({ entryTypes: ['navigation', 'paint', 'largest-contentful-paint', 'first-input', 'layout-shift'] });
      } catch (error) {
        console.warn('Performance Observer not supported:', error);
      }
    }
  }

  private recordWebVitals(entry: PerformanceEntry): void {
    const metrics: Partial<PerformanceMetrics> = {
      timestamp: Date.now()
    };

    switch (entry.entryType) {
      case 'navigation':
        const navEntry = entry as PerformanceNavigationTiming;
        metrics.pageLoadTime = navEntry.loadEventEnd - navEntry.fetchStart;
        metrics.timeToInteractive = navEntry.domInteractive - navEntry.fetchStart;
        break;

      case 'paint':
        if (entry.name === 'first-contentful-paint') {
          metrics.timeToInteractive = entry.startTime;
        }
        break;
    }

    if (Object.keys(metrics).length > 1) {
      this.recordMetrics(metrics);
    }
  }

  public startMonitoring(): void {
    if (this.isMonitoring) return;

    this.isMonitoring = true;
    this.monitoringInterval = window.setInterval(() => {
      this.collectSystemMetrics();
    }, this.config.metricsCollectionInterval);

    console.log('Performance monitoring started');
  }

  public stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = undefined;
    }
    this.isMonitoring = false;
    console.log('Performance monitoring stopped');
  }

  private collectSystemMetrics(): void {
    const metrics: PerformanceMetrics = {
      timestamp: Date.now(),
      memoryUsage: this.getMemoryUsage()
    };

    this.recordMetrics(metrics);
    this.checkAlerts(metrics);
  }

  private getMemoryUsage(): PerformanceMetrics['memoryUsage'] {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        used: Math.round(memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round(memory.totalJSHeapSize / 1024 / 1024),
        percentage: Math.round((memory.usedJSHeapSize / memory.totalJSHeapSize) * 100)
      };
    }
    return undefined;
  }

  public recordMetrics(metrics: Partial<PerformanceMetrics>): void {
    const fullMetrics: PerformanceMetrics = {
      timestamp: Date.now(),
      ...metrics
    };

    this.metrics.push(fullMetrics);

    // Keep only recent metrics (last 1000)
    if (this.metrics.length > 1000) {
      this.metrics = this.metrics.slice(-1000);
    }
  }

  public recordProcessingMetrics(
    processingTime: number,
    rowsProcessed: number,
    entitiesFound: number,
    fileSize?: number,
    sessionId?: string
  ): void {
    const rowsPerSecond = rowsProcessed / (processingTime / 1000);
    const entitiesPerSecond = entitiesFound / (processingTime / 1000);

    this.recordMetrics({
      sessionId,
      processingTime,
      rowsPerSecond,
      entitiesPerSecond,
      fileSize
    });
  }

  public recordNetworkMetrics(latency: number, throughput: number): void {
    this.recordMetrics({
      networkLatency: latency,
      networkThroughput: throughput
    });
  }

  public recordError(sessionId?: string): void {
    this.recordMetrics({
      sessionId,
      errorCount: 1
    });
  }

  private checkAlerts(metrics: PerformanceMetrics): void {
    const issues: string[] = [];

    if (metrics.processingTime && metrics.processingTime > this.config.alertThresholds.processingTime) {
      issues.push(`High processing time: ${metrics.processingTime}ms`);
    }

    if (metrics.memoryUsage && metrics.memoryUsage.percentage > this.config.alertThresholds.memoryUsage) {
      issues.push(`High memory usage: ${metrics.memoryUsage.percentage}%`);
    }

    const recentErrors = this.getRecentMetrics(60000).filter(m => m.errorCount).length;
    const totalRecentMetrics = this.getRecentMetrics(60000).length;
    const errorRate = totalRecentMetrics > 0 ? recentErrors / totalRecentMetrics : 0;

    if (errorRate > this.config.alertThresholds.errorRate) {
      issues.push(`High error rate: ${(errorRate * 100).toFixed(1)}%`);
    }

    if (issues.length > 0) {
      this.triggerAlert(issues);
    }
  }

  private triggerAlert(issues: string[]): void {
    console.warn('Performance Alert:', issues);

    // You could emit events or show notifications here
    window.dispatchEvent(new CustomEvent('performance-alert', {
      detail: { issues }
    }));
  }

  public getMetrics(timeRange?: number): PerformanceMetrics[] {
    if (timeRange) {
      return this.getRecentMetrics(timeRange);
    }
    return this.metrics;
  }

  public getRecentMetrics(timeRange: number): PerformanceMetrics[] {
    const cutoff = Date.now() - timeRange;
    return this.metrics.filter(m => m.timestamp > cutoff);
  }

  public getSystemHealth(): SystemHealth {
    const recentMetrics = this.getRecentMetrics(300000); // Last 5 minutes
    const issues: string[] = [];
    const recommendations: string[] = [];
    let score = 100;

    if (recentMetrics.length === 0) {
      return {
        status: 'healthy',
        score: 100,
        issues: [],
        recommendations: []
      };
    }

    // Check memory usage
    const memoryMetrics = recentMetrics.filter(m => m.memoryUsage);
    if (memoryMetrics.length > 0) {
      const avgMemory = memoryMetrics.reduce((sum, m) => sum + (m.memoryUsage?.percentage || 0), 0) / memoryMetrics.length;
      if (avgMemory > this.config.memoryWarningThreshold) {
        issues.push(`High memory usage: ${avgMemory.toFixed(1)}%`);
        recommendations.push('Consider reducing batch size or closing unused tabs');
        score -= 20;
      }
    }

    // Check processing performance
    const processingMetrics = recentMetrics.filter(m => m.processingTime);
    if (processingMetrics.length > 0) {
      const avgProcessingTime = processingMetrics.reduce((sum, m) => sum + (m.processingTime || 0), 0) / processingMetrics.length;
      if (avgProcessingTime > this.config.alertThresholds.processingTime) {
        issues.push(`Slow processing: ${avgProcessingTime.toFixed(0)}ms average`);
        recommendations.push('Consider reducing concurrent requests or batch size');
        score -= 15;
      }
    }

    // Check error rate
    const errorMetrics = recentMetrics.filter(m => m.errorCount);
    const errorRate = recentMetrics.length > 0 ? errorMetrics.length / recentMetrics.length : 0;
    if (errorRate > this.config.alertThresholds.errorRate) {
      issues.push(`High error rate: ${(errorRate * 100).toFixed(1)}%`);
      recommendations.push('Check network connectivity and API availability');
      score -= 25;
    }

    // Determine status
    let status: SystemHealth['status'] = 'healthy';
    if (score < 80) status = 'warning';
    if (score < 60) status = 'critical';

    return {
      status,
      score: Math.max(0, score),
      issues,
      recommendations
    };
  }

  public getConfig(): PerformanceConfig {
    return { ...this.config };
  }

  public updateConfig(updates: Partial<PerformanceConfig>): void {
    this.config = { ...this.config, ...updates };
    this.saveConfig();
    console.log('Performance config updated:', updates);
  }

  public resetConfig(): void {
    this.config = { ...this.defaultConfig };
    this.saveConfig();
    console.log('Performance config reset to defaults');
  }

  public exportMetrics(): string {
    return JSON.stringify({
      timestamp: Date.now(),
      config: this.config,
      metrics: this.metrics,
      systemHealth: this.getSystemHealth()
    }, null, 2);
  }

  public clearMetrics(): void {
    this.metrics = [];
    console.log('Performance metrics cleared');
  }
}

// Export singleton instance
export const performanceMonitor = PerformanceMonitor.getInstance();