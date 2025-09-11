#!/usr/bin/env python3
"""
Advanced Performance Monitoring System
Real-time tracking and analysis of PII detection accuracy and performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
import time
from collections import defaultdict, deque
import statistics

# Data visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    timestamp: datetime
    metric_name: str
    value: float
    context: Dict
    session_id: str


@dataclass
class ProcessingSession:
    """Processing session information"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_records: int
    processed_records: int
    total_detections: int
    false_positives: int
    false_negatives: int
    processing_layers: Dict[str, int]
    cost_breakdown: Dict[str, float]
    accuracy_estimate: float
    confidence_distribution: Dict[str, int]


@dataclass
class AccuracyFeedback:
    """User feedback for accuracy measurement"""
    session_id: str
    timestamp: datetime
    original_text: str
    detected_entities: List[Dict]
    user_marked_correct: List[bool]
    user_added_entities: List[Dict]
    confidence_before: float
    confidence_after: float
    feedback_type: str  # 'false_positive', 'false_negative', 'correct'


class PerformanceMonitor:
    """
    Advanced performance monitoring system for PII detection
    Provides real-time metrics, accuracy tracking, and performance analytics
    """
    
    def __init__(self, db_path: str = "pii_performance.db"):
        """Initialize performance monitor"""
        self.db_path = db_path
        self.current_sessions: Dict[str, ProcessingSession] = {}
        self.metrics_buffer: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.feedback_buffer: deque = deque(maxlen=1000)   # Keep last 1k feedback items
        
        # Real-time tracking
        self.real_time_metrics = {
            'current_accuracy': 0.0,
            'current_precision': 0.0,
            'current_recall': 0.0,
            'current_f1': 0.0,
            'false_positive_rate': 0.0,
            'false_negative_rate': 0.0,
            'processing_speed': 0.0,  # records per second
            'cost_per_record': 0.0,
            'confidence_avg': 0.0
        }
        
        # Performance targets
        self.performance_targets = {
            'target_accuracy': 0.99,
            'max_false_positive_rate': 0.01,
            'max_false_negative_rate': 0.01,
            'min_processing_speed': 10.0,  # records per second
            'max_cost_per_record': 0.001   # $0.001 per record
        }
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Initialize database
        self._initialize_database()
        
        print("✅ Performance Monitor initialized")
        print(f"   🗃️ Database: {db_path}")
        print(f"   🎯 Target accuracy: {self.performance_targets['target_accuracy']:.1%}")
    
    def _initialize_database(self):
        """Initialize SQLite database for performance tracking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    context TEXT,
                    session_id TEXT NOT NULL
                )
            """)
            
            # Processing sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_records INTEGER NOT NULL,
                    processed_records INTEGER NOT NULL,
                    total_detections INTEGER NOT NULL,
                    false_positives INTEGER NOT NULL,
                    false_negatives INTEGER NOT NULL,
                    processing_layers TEXT,
                    cost_breakdown TEXT,
                    accuracy_estimate REAL NOT NULL,
                    confidence_distribution TEXT
                )
            """)
            
            # Accuracy feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    detected_entities TEXT NOT NULL,
                    user_marked_correct TEXT NOT NULL,
                    user_added_entities TEXT NOT NULL,
                    confidence_before REAL NOT NULL,
                    confidence_after REAL NOT NULL,
                    feedback_type TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_session ON performance_metrics(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON accuracy_feedback(timestamp)")
            
            conn.commit()
    
    def start_session(self, session_id: str, total_records: int) -> str:
        """Start a new processing session"""
        session = ProcessingSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            total_records=total_records,
            processed_records=0,
            total_detections=0,
            false_positives=0,
            false_negatives=0,
            processing_layers={},
            cost_breakdown={},
            accuracy_estimate=0.0,
            confidence_distribution={}
        )
        
        self.current_sessions[session_id] = session
        
        print(f"📊 Started monitoring session: {session_id}")
        print(f"   📋 Total records: {total_records}")
        
        return session_id
    
    def update_session_progress(self, session_id: str, processed_records: int, 
                              detections: int = 0, layer_stats: Dict = None):
        """Update session progress"""
        if session_id not in self.current_sessions:
            return
        
        session = self.current_sessions[session_id]
        session.processed_records = processed_records
        session.total_detections += detections
        
        if layer_stats:
            for layer, count in layer_stats.items():
                session.processing_layers[layer] = session.processing_layers.get(layer, 0) + count
        
        # Calculate progress percentage
        progress = (processed_records / session.total_records) * 100 if session.total_records > 0 else 0
        
        # Record progress metric
        self.record_metric(session_id, "processing_progress", progress, {"processed": processed_records})
        
        # Update real-time processing speed
        elapsed_time = (datetime.now() - session.start_time).total_seconds()
        if elapsed_time > 0:
            speed = processed_records / elapsed_time
            self.real_time_metrics['processing_speed'] = speed
            self.record_metric(session_id, "processing_speed", speed)
    
    def end_session(self, session_id: str, final_stats: Dict = None):
        """End a processing session"""
        if session_id not in self.current_sessions:
            return
        
        session = self.current_sessions[session_id]
        session.end_time = datetime.now()
        
        # Update final statistics
        if final_stats:
            session.accuracy_estimate = final_stats.get('accuracy_estimate', 0.0)
            session.cost_breakdown = final_stats.get('cost_breakdown', {})
            session.confidence_distribution = final_stats.get('confidence_distribution', {})
        
        # Save to database
        self._save_session_to_db(session)
        
        # Remove from active sessions
        del self.current_sessions[session_id]
        
        print(f"✅ Session {session_id} completed")
        print(f"   📊 Processed: {session.processed_records}/{session.total_records}")
        print(f"   🎯 Accuracy: {session.accuracy_estimate:.1%}")
    
    def record_metric(self, session_id: str, metric_name: str, value: float, context: Dict = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            context=context or {},
            session_id=session_id
        )
        
        # Add to buffer
        self.metrics_buffer.append(metric)
        
        # Update real-time metrics
        if metric_name in self.real_time_metrics:
            self.real_time_metrics[metric_name] = value
        
        # Save to database (batch save for performance)
        if len(self.metrics_buffer) % 100 == 0:
            self._save_metrics_batch()
    
    def add_accuracy_feedback(self, session_id: str, original_text: str, 
                            detected_entities: List[Dict], user_corrections: Dict):
        """Add user feedback for accuracy measurement"""
        feedback = AccuracyFeedback(
            session_id=session_id,
            timestamp=datetime.now(),
            original_text=original_text,
            detected_entities=detected_entities,
            user_marked_correct=user_corrections.get('marked_correct', []),
            user_added_entities=user_corrections.get('added_entities', []),
            confidence_before=user_corrections.get('confidence_before', 0.0),
            confidence_after=user_corrections.get('confidence_after', 0.0),
            feedback_type=user_corrections.get('feedback_type', 'unknown')
        )
        
        self.feedback_buffer.append(feedback)
        
        # Update session false positive/negative counts
        if session_id in self.current_sessions:
            session = self.current_sessions[session_id]
            
            if feedback.feedback_type == 'false_positive':
                session.false_positives += 1
            elif feedback.feedback_type == 'false_negative':
                session.false_negatives += 1
        
        # Recalculate real-time accuracy
        self._update_real_time_accuracy()
        
        # Save to database
        self._save_feedback_to_db(feedback)
        
        print(f"📝 Feedback recorded: {feedback.feedback_type}")
    
    def get_real_time_metrics(self) -> Dict:
        """Get current real-time metrics"""
        # Add session information
        active_sessions = len(self.current_sessions)
        total_processing = sum(s.processed_records for s in self.current_sessions.values())
        
        metrics = self.real_time_metrics.copy()
        metrics.update({
            'active_sessions': active_sessions,
            'total_processing': total_processing,
            'timestamp': datetime.now().isoformat()
        })
        
        return metrics
    
    def get_performance_report(self, days: int = 7) -> Dict:
        """Generate comprehensive performance report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get session statistics
            sessions_df = pd.read_sql_query("""
                SELECT * FROM processing_sessions 
                WHERE start_time >= ? AND start_time <= ?
                ORDER BY start_time DESC
            """, conn, params=[start_date.isoformat(), end_date.isoformat()])
            
            # Get feedback statistics
            feedback_df = pd.read_sql_query("""
                SELECT * FROM accuracy_feedback
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """, conn, params=[start_date.isoformat(), end_date.isoformat()])
        
        # Calculate aggregate statistics
        total_sessions = len(sessions_df)
        total_records = sessions_df['total_records'].sum() if not sessions_df.empty else 0
        total_detections = sessions_df['total_detections'].sum() if not sessions_df.empty else 0
        
        # Accuracy metrics
        avg_accuracy = sessions_df['accuracy_estimate'].mean() if not sessions_df.empty else 0.0
        total_false_positives = sessions_df['false_positives'].sum() if not sessions_df.empty else 0
        total_false_negatives = sessions_df['false_negatives'].sum() if not sessions_df.empty else 0
        
        false_positive_rate = (total_false_positives / max(total_detections, 1)) * 100
        false_negative_rate = (total_false_negatives / max(total_records, 1)) * 100
        
        # Performance trends
        if not sessions_df.empty:
            sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'])
            accuracy_trend = sessions_df.set_index('start_time')['accuracy_estimate'].rolling('1D').mean()
        else:
            accuracy_trend = pd.Series()
        
        report = {
            'reporting_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': days
            },
            'summary_statistics': {
                'total_sessions': total_sessions,
                'total_records_processed': int(total_records),
                'total_detections': int(total_detections),
                'average_accuracy': avg_accuracy,
                'false_positive_rate': false_positive_rate,
                'false_negative_rate': false_negative_rate
            },
            'performance_vs_targets': {
                'accuracy_target_met': avg_accuracy >= self.performance_targets['target_accuracy'],
                'fp_rate_target_met': false_positive_rate <= self.performance_targets['max_false_positive_rate'] * 100,
                'fn_rate_target_met': false_negative_rate <= self.performance_targets['max_false_negative_rate'] * 100
            },
            'trends': {
                'accuracy_improving': self._calculate_trend(accuracy_trend),
                'processing_speed_trend': self._get_processing_speed_trend(sessions_df)
            },
            'feedback_analysis': self._analyze_feedback(feedback_df),
            'recommendations': self._generate_recommendations(sessions_df, feedback_df)
        }
        
        return report
    
    def generate_performance_dashboard(self, days: int = 7) -> Optional[str]:
        """Generate interactive performance dashboard"""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - cannot generate dashboard")
            return None
        
        # Get data for dashboard
        report = self.get_performance_report(days)
        
        # Create dashboard with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Accuracy Over Time', 'Detection Volume',
                'False Positive/Negative Rates', 'Processing Speed',
                'Cost Analysis', 'Confidence Distribution'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "pie"}]
            ]
        )
        
        # Get detailed data from database
        with sqlite3.connect(self.db_path) as conn:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            metrics_df = pd.read_sql_query("""
                SELECT timestamp, metric_name, value FROM performance_metrics
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """, conn, params=[start_date.isoformat(), end_date.isoformat()])
            
            sessions_df = pd.read_sql_query("""
                SELECT * FROM processing_sessions
                WHERE start_time >= ? AND start_time <= ?
                ORDER BY start_time
            """, conn, params=[start_date.isoformat(), end_date.isoformat()])
        
        if not metrics_df.empty:
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        
        if not sessions_df.empty:
            sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'])
        
        # 1. Accuracy Over Time
        if not sessions_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sessions_df['start_time'],
                    y=sessions_df['accuracy_estimate'] * 100,
                    mode='lines+markers',
                    name='Accuracy %',
                    line=dict(color='green', width=2)
                ),
                row=1, col=1
            )
            
            # Add target line
            fig.add_hline(
                y=self.performance_targets['target_accuracy'] * 100,
                line_dash="dash", 
                line_color="red",
                row=1, col=1
            )
        
        # 2. Detection Volume
        if not sessions_df.empty:
            fig.add_trace(
                go.Bar(
                    x=sessions_df['start_time'].dt.date,
                    y=sessions_df['total_detections'],
                    name='Detections',
                    marker_color='blue'
                ),
                row=1, col=2
            )
        
        # 3. False Positive/Negative Rates
        if not sessions_df.empty:
            fp_rates = (sessions_df['false_positives'] / sessions_df['total_detections'].replace(0, 1)) * 100
            fn_rates = (sessions_df['false_negatives'] / sessions_df['total_records']) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=sessions_df['start_time'],
                    y=fp_rates,
                    mode='lines+markers',
                    name='False Positive %',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sessions_df['start_time'],
                    y=fn_rates,
                    mode='lines+markers',
                    name='False Negative %',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
        
        # 4. Processing Speed
        if not metrics_df.empty:
            speed_data = metrics_df[metrics_df['metric_name'] == 'processing_speed']
            if not speed_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=speed_data['timestamp'],
                        y=speed_data['value'],
                        mode='lines',
                        name='Records/sec',
                        line=dict(color='purple')
                    ),
                    row=2, col=2
                )
        
        # 5. Cost Analysis
        if not sessions_df.empty and 'cost_breakdown' in sessions_df.columns:
            # This would need cost data parsing - simplified for now
            avg_cost = 0.001  # Placeholder
            fig.add_trace(
                go.Bar(
                    x=['Azure AI', 'GPT Validation', 'ML Processing'],
                    y=[avg_cost * 0.6, avg_cost * 0.3, avg_cost * 0.1],
                    name='Cost Breakdown',
                    marker_color=['blue', 'green', 'orange']
                ),
                row=3, col=1
            )
        
        # 6. Confidence Distribution (placeholder)
        confidence_labels = ['High (90-100%)', 'Medium (70-89%)', 'Low (50-69%)']
        confidence_values = [60, 30, 10]  # Placeholder values
        
        fig.add_trace(
            go.Pie(
                labels=confidence_labels,
                values=confidence_values,
                name="Confidence"
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text=f"PII Detection Performance Dashboard - Last {days} Days",
            showlegend=True
        )
        
        # Save dashboard
        dashboard_path = f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(dashboard_path)
        
        print(f"📊 Performance dashboard generated: {dashboard_path}")
        
        return dashboard_path
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            print("⚠️ Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, args=(interval_seconds,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        print(f"🔄 Started continuous monitoring (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Flush remaining metrics
        if self.metrics_buffer:
            self._save_metrics_batch()
        
        print("⏹️ Monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                # Update real-time metrics
                self._update_real_time_accuracy()
                
                # Check performance alerts
                self._check_performance_alerts()
                
                # Flush metrics buffer if needed
                if len(self.metrics_buffer) > 50:
                    self._save_metrics_batch()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"⚠️ Monitoring loop error: {e}")
                time.sleep(interval_seconds)
    
    def _update_real_time_accuracy(self):
        """Update real-time accuracy calculations"""
        if not self.feedback_buffer:
            return
        
        recent_feedback = list(self.feedback_buffer)[-100:]  # Last 100 feedback items
        
        if recent_feedback:
            false_positives = sum(1 for f in recent_feedback if f.feedback_type == 'false_positive')
            false_negatives = sum(1 for f in recent_feedback if f.feedback_type == 'false_negative')
            correct = len(recent_feedback) - false_positives - false_negatives
            
            accuracy = correct / len(recent_feedback)
            precision = correct / max(correct + false_positives, 1)
            recall = correct / max(correct + false_negatives, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 0.001)
            
            self.real_time_metrics.update({
                'current_accuracy': accuracy,
                'current_precision': precision,
                'current_recall': recall,
                'current_f1': f1,
                'false_positive_rate': false_positives / len(recent_feedback),
                'false_negative_rate': false_negatives / len(recent_feedback)
            })
    
    def _check_performance_alerts(self):
        """Check for performance threshold violations"""
        alerts = []
        
        # Check accuracy
        if self.real_time_metrics['current_accuracy'] < self.performance_targets['target_accuracy']:
            alerts.append(f"⚠️ Accuracy below target: {self.real_time_metrics['current_accuracy']:.1%} < {self.performance_targets['target_accuracy']:.1%}")
        
        # Check false positive rate
        if self.real_time_metrics['false_positive_rate'] > self.performance_targets['max_false_positive_rate']:
            alerts.append(f"⚠️ False positive rate too high: {self.real_time_metrics['false_positive_rate']:.1%} > {self.performance_targets['max_false_positive_rate']:.1%}")
        
        # Check processing speed
        if self.real_time_metrics['processing_speed'] < self.performance_targets['min_processing_speed']:
            alerts.append(f"⚠️ Processing speed too slow: {self.real_time_metrics['processing_speed']:.1f} < {self.performance_targets['min_processing_speed']:.1f} records/sec")
        
        # Log alerts
        for alert in alerts:
            print(alert)
    
    def _save_session_to_db(self, session: ProcessingSession):
        """Save session to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO processing_sessions 
                (session_id, start_time, end_time, total_records, processed_records, 
                 total_detections, false_positives, false_negatives, processing_layers,
                 cost_breakdown, accuracy_estimate, confidence_distribution)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.start_time.isoformat(),
                session.end_time.isoformat() if session.end_time else None,
                session.total_records,
                session.processed_records,
                session.total_detections,
                session.false_positives,
                session.false_negatives,
                json.dumps(session.processing_layers),
                json.dumps(session.cost_breakdown),
                session.accuracy_estimate,
                json.dumps(session.confidence_distribution)
            ))
            conn.commit()
    
    def _save_metrics_batch(self):
        """Save metrics buffer to database"""
        if not self.metrics_buffer:
            return
        
        metrics_to_save = list(self.metrics_buffer)
        self.metrics_buffer.clear()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for metric in metrics_to_save:
                cursor.execute("""
                    INSERT INTO performance_metrics 
                    (timestamp, metric_name, value, context, session_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric.timestamp.isoformat(),
                    metric.metric_name,
                    metric.value,
                    json.dumps(metric.context),
                    metric.session_id
                ))
            
            conn.commit()
    
    def _save_feedback_to_db(self, feedback: AccuracyFeedback):
        """Save feedback to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO accuracy_feedback
                (session_id, timestamp, original_text, detected_entities,
                 user_marked_correct, user_added_entities, confidence_before,
                 confidence_after, feedback_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.session_id,
                feedback.timestamp.isoformat(),
                feedback.original_text,
                json.dumps(feedback.detected_entities),
                json.dumps(feedback.user_marked_correct),
                json.dumps(feedback.user_added_entities),
                feedback.confidence_before,
                feedback.confidence_after,
                feedback.feedback_type
            ))
            conn.commit()
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        if len(series) < 2:
            return "insufficient_data"
        
        recent_values = series.dropna().tail(5)
        if len(recent_values) < 2:
            return "insufficient_data"
        
        slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _get_processing_speed_trend(self, sessions_df: pd.DataFrame) -> str:
        """Calculate processing speed trend"""
        if sessions_df.empty or len(sessions_df) < 2:
            return "insufficient_data"
        
        # Calculate processing speed for each session
        sessions_df = sessions_df.copy()
        sessions_df['processing_time'] = pd.to_datetime(sessions_df['start_time'])
        
        speeds = []
        for _, row in sessions_df.iterrows():
            if row['processed_records'] > 0:
                # Estimate processing time (would need actual end times)
                estimated_time = row['processed_records'] / 10  # Placeholder
                speed = row['processed_records'] / max(estimated_time, 1)
                speeds.append(speed)
        
        if len(speeds) < 2:
            return "insufficient_data"
        
        recent_speed = statistics.mean(speeds[-3:]) if len(speeds) >= 3 else speeds[-1]
        early_speed = statistics.mean(speeds[:3]) if len(speeds) >= 3 else speeds[0]
        
        if recent_speed > early_speed * 1.1:
            return "improving"
        elif recent_speed < early_speed * 0.9:
            return "declining"
        else:
            return "stable"
    
    def _analyze_feedback(self, feedback_df: pd.DataFrame) -> Dict:
        """Analyze user feedback patterns"""
        if feedback_df.empty:
            return {"error": "No feedback data available"}
        
        total_feedback = len(feedback_df)
        false_positives = len(feedback_df[feedback_df['feedback_type'] == 'false_positive'])
        false_negatives = len(feedback_df[feedback_df['feedback_type'] == 'false_negative'])
        correct_detections = total_feedback - false_positives - false_negatives
        
        return {
            'total_feedback_items': total_feedback,
            'correct_detections': correct_detections,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'user_accuracy_rating': correct_detections / max(total_feedback, 1),
            'false_positive_rate': false_positives / max(total_feedback, 1),
            'false_negative_rate': false_negatives / max(total_feedback, 1)
        }
    
    def _generate_recommendations(self, sessions_df: pd.DataFrame, feedback_df: pd.DataFrame) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if not sessions_df.empty:
            avg_accuracy = sessions_df['accuracy_estimate'].mean()
            if avg_accuracy < self.performance_targets['target_accuracy']:
                recommendations.append("Consider increasing confidence thresholds to reduce false positives")
                recommendations.append("Add more business-specific terms to whitelist patterns")
                recommendations.append("Enable GPT validation if not already active")
        
        if not feedback_df.empty:
            fp_rate = len(feedback_df[feedback_df['feedback_type'] == 'false_positive']) / len(feedback_df)
            if fp_rate > 0.05:  # >5% false positive rate
                recommendations.append("High false positive rate detected - review and update whitelist patterns")
                recommendations.append("Consider implementing more aggressive ML-based filtering")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Performance Monitor - Example Usage")
    
    # Initialize monitor
    monitor = PerformanceMonitor("test_performance.db")
    
    # Simulate a processing session
    session_id = monitor.start_session("test_session_1", 1000)
    
    # Simulate processing progress
    for i in range(0, 1000, 100):
        monitor.update_session_progress(
            session_id, 
            i + 100, 
            detections=10,
            layer_stats={'azure': 10, 'gpt': 2, 'ml': 1}
        )
        time.sleep(0.1)  # Simulate processing time
    
    # Simulate some feedback
    monitor.add_accuracy_feedback(
        session_id,
        "Test text with user information",
        [{"text": "user", "category": "PersonType", "confidence": 0.8}],
        {
            "marked_correct": [False],
            "feedback_type": "false_positive",
            "confidence_before": 0.8,
            "confidence_after": 0.0
        }
    )
    
    # End session
    monitor.end_session(session_id, {
        'accuracy_estimate': 0.95,
        'cost_breakdown': {'azure': 0.50, 'gpt': 0.30},
        'confidence_distribution': {'high': 70, 'medium': 25, 'low': 5}
    })
    
    # Get real-time metrics
    metrics = monitor.get_real_time_metrics()
    print(f"\n📊 Real-time metrics: {json.dumps(metrics, indent=2, default=str)}")
    
    # Generate performance report
    report = monitor.get_performance_report(days=1)
    print(f"\n📋 Performance report: {json.dumps(report, indent=2, default=str)}")
    
    # Generate dashboard if plotly is available
    if PLOTLY_AVAILABLE:
        dashboard_path = monitor.generate_performance_dashboard()
        if dashboard_path:
            print(f"📊 Dashboard saved to: {dashboard_path}")
    
    print("✅ Example completed")