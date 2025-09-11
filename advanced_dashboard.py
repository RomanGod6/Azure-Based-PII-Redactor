#!/usr/bin/env python3
"""
Advanced Performance Dashboard for PII Detection System
Interactive web-based dashboard with real-time monitoring and user experience optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
from dataclasses import asdict

# Import our components
from performance_monitor import PerformanceMonitor
from enhanced_ml_detector import EnhancedMLPIIDetector, ValidationFeedback


class AdvancedPIIDashboard:
    """
    Advanced dashboard for PII detection performance monitoring and user experience
    Provides comprehensive analytics, real-time monitoring, and user-friendly interfaces
    """
    
    def __init__(self):
        """Initialize the advanced dashboard"""
        self.performance_monitor = PerformanceMonitor()
        self.setup_session_state()
        
        # Dashboard configuration
        self.refresh_interval = 30  # seconds
        self.chart_colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c', 
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17becf'
        }
    
    def setup_session_state(self):
        """Initialize Streamlit session state"""
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = '7 days'
        
        if 'dashboard_mode' not in st.session_state:
            st.session_state.dashboard_mode = 'overview'
    
    def run_dashboard(self):
        """Main dashboard application"""
        st.set_page_config(
            page_title="PII Detection Performance Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-danger { color: #dc3545; }
        .big-metric { font-size: 2rem; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("🛡️ PII Detection Performance Dashboard")
        st.markdown("**Real-time monitoring and analytics for your PII redaction system**")
        
        # Sidebar configuration
        with st.sidebar:
            self.render_sidebar()
        
        # Main dashboard content
        if st.session_state.dashboard_mode == 'overview':
            self.render_overview_dashboard()
        elif st.session_state.dashboard_mode == 'detailed':
            self.render_detailed_analytics()
        elif st.session_state.dashboard_mode == 'feedback':
            self.render_feedback_interface()
        elif st.session_state.dashboard_mode == 'settings':
            self.render_settings_panel()
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(self.refresh_interval)
            st.rerun()
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.header("⚙️ Dashboard Controls")
        
        # Dashboard mode selection
        st.session_state.dashboard_mode = st.selectbox(
            "Dashboard Mode",
            ['overview', 'detailed', 'feedback', 'settings'],
            format_func=lambda x: {
                'overview': '📊 Overview',
                'detailed': '🔍 Detailed Analytics', 
                'feedback': '💬 User Feedback',
                'settings': '⚙️ Settings'
            }[x]
        )
        
        # Time frame selection
        st.session_state.selected_timeframe = st.selectbox(
            "Time Frame",
            ['1 hour', '6 hours', '24 hours', '7 days', '30 days'],
            index=3
        )
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.checkbox(
            f"Auto-refresh ({self.refresh_interval}s)",
            value=st.session_state.auto_refresh
        )
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        # Status indicators
        st.subheader("🚦 System Status")
        self.render_system_status()
        
        # Quick stats
        st.subheader("📈 Quick Stats")
        self.render_quick_stats()
    
    def render_system_status(self):
        """Render system status indicators"""
        try:
            metrics = self.performance_monitor.get_real_time_metrics()
            
            # Overall system health
            accuracy = metrics.get('current_accuracy', 0)
            if accuracy >= 0.99:
                status_color = "status-good"
                status_icon = "✅"
                status_text = "Excellent"
            elif accuracy >= 0.95:
                status_color = "status-warning"
                status_icon = "⚠️"
                status_text = "Good"
            else:
                status_color = "status-danger"
                status_icon = "❌"
                status_text = "Needs Attention"
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>System Health</strong><br>
                <span class="{status_color}">{status_icon} {status_text}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Active sessions
            active_sessions = metrics.get('active_sessions', 0)
            st.markdown(f"""
            <div class="metric-card">
                <strong>Active Sessions</strong><br>
                <span class="big-metric">{active_sessions}</span>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading system status: {str(e)}")
    
    def render_quick_stats(self):
        """Render quick statistics"""
        try:
            # Get performance report for quick stats
            days = self._parse_timeframe(st.session_state.selected_timeframe)
            report = self.performance_monitor.get_performance_report(days)
            
            summary = report.get('summary_statistics', {})
            
            # Total records processed
            total_records = summary.get('total_records_processed', 0)
            st.metric("Records Processed", f"{total_records:,}")
            
            # Average accuracy
            avg_accuracy = summary.get('average_accuracy', 0)
            st.metric("Accuracy", f"{avg_accuracy:.1%}")
            
            # False positive rate
            fp_rate = summary.get('false_positive_rate', 0)
            st.metric("False Positive Rate", f"{fp_rate:.2f}%")
            
        except Exception as e:
            st.error(f"Error loading quick stats: {str(e)}")
    
    def render_overview_dashboard(self):
        """Render main overview dashboard"""
        st.header("📊 Performance Overview")
        
        # Real-time metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            metrics = self.performance_monitor.get_real_time_metrics()
            
            with col1:
                accuracy = metrics.get('current_accuracy', 0)
                target_met = accuracy >= 0.99
                delta_color = "normal" if target_met else "inverse"
                st.metric(
                    "Current Accuracy", 
                    f"{accuracy:.1%}",
                    delta=f"Target: 99%",
                    delta_color=delta_color
                )
            
            with col2:
                fp_rate = metrics.get('false_positive_rate', 0)
                st.metric(
                    "False Positive Rate",
                    f"{fp_rate:.2%}",
                    delta=f"Target: <1%",
                    delta_color="inverse" if fp_rate > 0.01 else "normal"
                )
            
            with col3:
                processing_speed = metrics.get('processing_speed', 0)
                st.metric(
                    "Processing Speed",
                    f"{processing_speed:.1f} rec/sec",
                    delta=f"Target: >10 rec/sec",
                    delta_color="normal" if processing_speed > 10 else "inverse"
                )
            
            with col4:
                active_sessions = metrics.get('active_sessions', 0)
                st.metric("Active Sessions", active_sessions)
        
        except Exception as e:
            st.error(f"Error loading real-time metrics: {str(e)}")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_accuracy_trend_chart()
        
        with col2:
            self.render_detection_volume_chart()
        
        # Second charts row
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_false_positive_trend()
        
        with col2:
            self.render_cost_analysis_chart()
        
        # Performance targets status
        st.subheader("🎯 Performance Targets")
        self.render_performance_targets()
    
    def render_accuracy_trend_chart(self):
        """Render accuracy trend chart"""
        st.subheader("📈 Accuracy Trend")
        
        try:
            days = self._parse_timeframe(st.session_state.selected_timeframe)
            
            # Get data from database
            with sqlite3.connect(self.performance_monitor.db_path) as conn:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                df = pd.read_sql_query("""
                    SELECT start_time, accuracy_estimate
                    FROM processing_sessions
                    WHERE start_time >= ? AND start_time <= ?
                    ORDER BY start_time
                """, conn, params=[start_date.isoformat(), end_date.isoformat()])
            
            if not df.empty:
                df['start_time'] = pd.to_datetime(df['start_time'])
                
                fig = go.Figure()
                
                # Accuracy line
                fig.add_trace(go.Scatter(
                    x=df['start_time'],
                    y=df['accuracy_estimate'] * 100,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color=self.chart_colors['primary'], width=2),
                    marker=dict(size=6)
                ))
                
                # Target line
                fig.add_hline(
                    y=99,
                    line_dash="dash",
                    line_color=self.chart_colors['success'],
                    annotation_text="99% Target"
                )
                
                fig.update_layout(
                    title="Accuracy Over Time",
                    xaxis_title="Time",
                    yaxis_title="Accuracy (%)",
                    yaxis=dict(range=[90, 100]),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected timeframe")
        
        except Exception as e:
            st.error(f"Error rendering accuracy chart: {str(e)}")
    
    def render_detection_volume_chart(self):
        """Render detection volume chart"""
        st.subheader("📊 Detection Volume")
        
        try:
            days = self._parse_timeframe(st.session_state.selected_timeframe)
            
            with sqlite3.connect(self.performance_monitor.db_path) as conn:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                df = pd.read_sql_query("""
                    SELECT start_time, total_detections, total_records
                    FROM processing_sessions
                    WHERE start_time >= ? AND start_time <= ?
                    ORDER BY start_time
                """, conn, params=[start_date.isoformat(), end_date.isoformat()])
            
            if not df.empty:
                df['start_time'] = pd.to_datetime(df['start_time'])
                df['detection_rate'] = (df['total_detections'] / df['total_records'] * 100).fillna(0)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Detection volume bars
                fig.add_trace(
                    go.Bar(
                        x=df['start_time'],
                        y=df['total_detections'],
                        name='Detections',
                        marker_color=self.chart_colors['info']
                    ),
                    secondary_y=False
                )
                
                # Detection rate line
                fig.add_trace(
                    go.Scatter(
                        x=df['start_time'],
                        y=df['detection_rate'],
                        mode='lines+markers',
                        name='Detection Rate (%)',
                        line=dict(color=self.chart_colors['warning'], width=2)
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title="Detection Volume and Rate",
                    height=300
                )
                
                fig.update_yaxes(title_text="Number of Detections", secondary_y=False)
                fig.update_yaxes(title_text="Detection Rate (%)", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected timeframe")
        
        except Exception as e:
            st.error(f"Error rendering detection volume chart: {str(e)}")
    
    def render_false_positive_trend(self):
        """Render false positive trend chart"""
        st.subheader("⚠️ False Positive Trend")
        
        try:
            days = self._parse_timeframe(st.session_state.selected_timeframe)
            
            with sqlite3.connect(self.performance_monitor.db_path) as conn:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                df = pd.read_sql_query("""
                    SELECT start_time, false_positives, false_negatives, total_detections
                    FROM processing_sessions
                    WHERE start_time >= ? AND start_time <= ?
                    ORDER BY start_time
                """, conn, params=[start_date.isoformat(), end_date.isoformat()])
            
            if not df.empty:
                df['start_time'] = pd.to_datetime(df['start_time'])
                df['fp_rate'] = (df['false_positives'] / df['total_detections'].replace(0, 1) * 100).fillna(0)
                df['fn_rate'] = (df['false_negatives'] / df['total_detections'].replace(0, 1) * 100).fillna(0)
                
                fig = go.Figure()
                
                # False positive rate
                fig.add_trace(go.Scatter(
                    x=df['start_time'],
                    y=df['fp_rate'],
                    mode='lines+markers',
                    name='False Positive Rate',
                    line=dict(color=self.chart_colors['danger'], width=2)
                ))
                
                # False negative rate
                fig.add_trace(go.Scatter(
                    x=df['start_time'],
                    y=df['fn_rate'],
                    mode='lines+markers',
                    name='False Negative Rate',
                    line=dict(color=self.chart_colors['warning'], width=2)
                ))
                
                # Target line
                fig.add_hline(
                    y=1,
                    line_dash="dash",
                    line_color=self.chart_colors['success'],
                    annotation_text="1% Target"
                )
                
                fig.update_layout(
                    title="False Positive/Negative Rates",
                    xaxis_title="Time",
                    yaxis_title="Rate (%)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected timeframe")
        
        except Exception as e:
            st.error(f"Error rendering false positive chart: {str(e)}")
    
    def render_cost_analysis_chart(self):
        """Render cost analysis chart"""
        st.subheader("💰 Cost Analysis")
        
        try:
            # For now, show a placeholder cost breakdown
            # In a real implementation, this would pull actual cost data
            
            categories = ['Azure AI', 'GPT Validation', 'ML Processing', 'Storage']
            costs = [0.60, 0.25, 0.10, 0.05]  # Relative cost distribution
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=categories,
                    values=costs,
                    hole=0.4,
                    marker_colors=[
                        self.chart_colors['primary'],
                        self.chart_colors['success'],
                        self.chart_colors['info'],
                        self.chart_colors['warning']
                    ]
                )
            ])
            
            fig.update_layout(
                title="Cost Breakdown",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error rendering cost chart: {str(e)}")
    
    def render_performance_targets(self):
        """Render performance targets status"""
        try:
            days = self._parse_timeframe(st.session_state.selected_timeframe)
            report = self.performance_monitor.get_performance_report(days)
            
            targets = report.get('performance_vs_targets', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy_met = targets.get('accuracy_target_met', False)
                icon = "✅" if accuracy_met else "❌"
                color = "status-good" if accuracy_met else "status-danger"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{icon} Accuracy Target</strong><br>
                    <span class="{color}">{'Met' if accuracy_met else 'Not Met'}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fp_met = targets.get('fp_rate_target_met', False)
                icon = "✅" if fp_met else "❌"
                color = "status-good" if fp_met else "status-danger"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{icon} False Positive Target</strong><br>
                    <span class="{color}">{'Met' if fp_met else 'Not Met'}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                fn_met = targets.get('fn_rate_target_met', False)
                icon = "✅" if fn_met else "❌"
                color = "status-good" if fn_met else "status-danger"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{icon} False Negative Target</strong><br>
                    <span class="{color}">{'Met' if fn_met else 'Not Met'}</span>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error rendering performance targets: {str(e)}")
    
    def render_detailed_analytics(self):
        """Render detailed analytics dashboard"""
        st.header("🔍 Detailed Analytics")
        
        # Time series analysis
        st.subheader("📈 Time Series Analysis")
        
        try:
            days = self._parse_timeframe(st.session_state.selected_timeframe)
            
            with sqlite3.connect(self.performance_monitor.db_path) as conn:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Get detailed metrics
                metrics_df = pd.read_sql_query("""
                    SELECT timestamp, metric_name, value
                    FROM performance_metrics
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                """, conn, params=[start_date.isoformat(), end_date.isoformat()])
                
                if not metrics_df.empty:
                    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
                    
                    # Create multi-metric chart
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            'Processing Speed', 'Processing Progress',
                            'Accuracy Metrics', 'Cost Metrics'
                        )
                    )
                    
                    # Processing speed
                    speed_data = metrics_df[metrics_df['metric_name'] == 'processing_speed']
                    if not speed_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=speed_data['timestamp'],
                                y=speed_data['value'],
                                mode='lines',
                                name='Speed (rec/sec)',
                                line=dict(color=self.chart_colors['primary'])
                            ),
                            row=1, col=1
                        )
                    
                    # Processing progress
                    progress_data = metrics_df[metrics_df['metric_name'] == 'processing_progress']
                    if not progress_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=progress_data['timestamp'],
                                y=progress_data['value'],
                                mode='lines',
                                name='Progress (%)',
                                line=dict(color=self.chart_colors['success'])
                            ),
                            row=1, col=2
                        )
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No detailed metrics available for the selected timeframe")
        
        except Exception as e:
            st.error(f"Error rendering detailed analytics: {str(e)}")
        
        # Entity type analysis
        st.subheader("🏷️ Entity Type Analysis")
        self.render_entity_type_analysis()
        
        # Column performance analysis
        st.subheader("📋 Column Performance Analysis")
        self.render_column_performance()
    
    def render_entity_type_analysis(self):
        """Render entity type analysis"""
        try:
            # This would analyze which entity types are most commonly detected
            # and their accuracy rates
            
            # Placeholder data for demonstration
            entity_types = ['Person', 'Email', 'PhoneNumber', 'PersonType', 'Organization']
            detection_counts = [150, 89, 45, 234, 67]
            accuracy_rates = [0.95, 0.98, 0.92, 0.78, 0.88]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(
                        x=entity_types,
                        y=detection_counts,
                        marker_color=self.chart_colors['info']
                    )
                ])
                fig.update_layout(
                    title="Detection Counts by Entity Type",
                    xaxis_title="Entity Type",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[
                    go.Bar(
                        x=entity_types,
                        y=[rate * 100 for rate in accuracy_rates],
                        marker_color=self.chart_colors['warning']
                    )
                ])
                fig.update_layout(
                    title="Accuracy by Entity Type",
                    xaxis_title="Entity Type",
                    yaxis_title="Accuracy (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error rendering entity analysis: {str(e)}")
    
    def render_column_performance(self):
        """Render column performance analysis"""
        try:
            # Placeholder data for column performance
            columns = ['subject', 'description', 'comments', 'requester', 'tags']
            detection_rates = [15.2, 8.7, 12.1, 45.6, 2.3]
            accuracy_rates = [92.1, 88.5, 95.2, 78.9, 96.8]
            
            df = pd.DataFrame({
                'Column': columns,
                'Detection Rate (%)': detection_rates,
                'Accuracy (%)': accuracy_rates
            })
            
            st.dataframe(
                df.style.background_gradient(subset=['Detection Rate (%)', 'Accuracy (%)']),
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error rendering column performance: {str(e)}")
    
    def render_feedback_interface(self):
        """Render user feedback interface"""
        st.header("💬 User Feedback & Learning")
        
        # Feedback submission form
        st.subheader("📝 Submit Feedback")
        
        with st.form("feedback_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                original_text = st.text_area(
                    "Original Text",
                    placeholder="Enter the original text that was processed..."
                )
                
                feedback_type = st.selectbox(
                    "Feedback Type",
                    ["false_positive", "false_negative", "correct_detection", "suggestion"]
                )
            
            with col2:
                detected_entities = st.text_area(
                    "Detected Entities (JSON)",
                    placeholder='[{"text": "example", "category": "Person", "confidence": 0.8}]'
                )
                
                user_comments = st.text_area(
                    "Comments",
                    placeholder="Additional comments or suggestions..."
                )
            
            if st.form_submit_button("Submit Feedback"):
                try:
                    # Parse detected entities
                    entities = json.loads(detected_entities) if detected_entities else []
                    
                    # Create feedback object
                    feedback = ValidationFeedback(
                        original_text=original_text,
                        detected_entities=entities,
                        user_corrections={
                            'feedback_type': feedback_type,
                            'comments': user_comments
                        },
                        is_false_positive=feedback_type == 'false_positive',
                        is_false_negative=feedback_type == 'false_negative',
                        correct_entities=[],
                        timestamp=datetime.now(),
                        confidence_before=0.0,
                        confidence_after=0.0,
                        context='user_feedback'
                    )
                    
                    # Add to performance monitor
                    self.performance_monitor.add_accuracy_feedback(
                        session_id="user_feedback",
                        original_text=original_text,
                        detected_entities=entities,
                        user_corrections={
                            'feedback_type': feedback_type,
                            'comments': user_comments
                        }
                    )
                    
                    st.success("✅ Feedback submitted successfully!")
                    
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format for detected entities")
                except Exception as e:
                    st.error(f"❌ Error submitting feedback: {str(e)}")
        
        # Recent feedback display
        st.subheader("📋 Recent Feedback")
        self.display_recent_feedback()
        
        # Learning progress
        st.subheader("🎓 Learning Progress")
        self.display_learning_progress()
    
    def display_recent_feedback(self):
        """Display recent user feedback"""
        try:
            with sqlite3.connect(self.performance_monitor.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT timestamp, feedback_type, original_text, confidence_before, confidence_after
                    FROM accuracy_feedback
                    ORDER BY timestamp DESC
                    LIMIT 10
                """, conn)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                df['original_text'] = df['original_text'].str[:50] + '...'
                
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No feedback data available")
        
        except Exception as e:
            st.error(f"Error loading feedback data: {str(e)}")
    
    def display_learning_progress(self):
        """Display learning progress metrics"""
        try:
            # Get feedback statistics
            with sqlite3.connect(self.performance_monitor.db_path) as conn:
                feedback_stats = pd.read_sql_query("""
                    SELECT 
                        feedback_type,
                        COUNT(*) as count,
                        AVG(confidence_after - confidence_before) as avg_confidence_change
                    FROM accuracy_feedback
                    GROUP BY feedback_type
                """, conn)
            
            if not feedback_stats.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    false_positives = feedback_stats[
                        feedback_stats['feedback_type'] == 'false_positive'
                    ]['count'].sum()
                    st.metric("False Positives Reported", false_positives)
                
                with col2:
                    false_negatives = feedback_stats[
                        feedback_stats['feedback_type'] == 'false_negative'
                    ]['count'].sum()
                    st.metric("False Negatives Reported", false_negatives)
                
                with col3:
                    correct_detections = feedback_stats[
                        feedback_stats['feedback_type'] == 'correct_detection'
                    ]['count'].sum()
                    st.metric("Correct Detections Confirmed", correct_detections)
            else:
                st.info("No learning progress data available")
        
        except Exception as e:
            st.error(f"Error loading learning progress: {str(e)}")
    
    def render_settings_panel(self):
        """Render settings and configuration panel"""
        st.header("⚙️ Settings & Configuration")
        
        # Performance targets
        st.subheader("🎯 Performance Targets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_accuracy = st.slider(
                "Target Accuracy (%)",
                min_value=90.0,
                max_value=99.9,
                value=99.0,
                step=0.1
            )
            
            max_fp_rate = st.slider(
                "Max False Positive Rate (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            )
        
        with col2:
            max_fn_rate = st.slider(
                "Max False Negative Rate (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            )
            
            min_processing_speed = st.slider(
                "Min Processing Speed (records/sec)",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=1.0
            )
        
        if st.button("💾 Save Performance Targets"):
            # Update performance targets
            self.performance_monitor.performance_targets.update({
                'target_accuracy': target_accuracy / 100,
                'max_false_positive_rate': max_fp_rate / 100,
                'max_false_negative_rate': max_fn_rate / 100,
                'min_processing_speed': min_processing_speed
            })
            st.success("✅ Performance targets updated!")
        
        # Dashboard settings
        st.subheader("📊 Dashboard Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            refresh_interval = st.slider(
                "Auto-refresh Interval (seconds)",
                min_value=10,
                max_value=300,
                value=self.refresh_interval,
                step=10
            )
        
        with col2:
            chart_theme = st.selectbox(
                "Chart Theme",
                ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"]
            )
        
        if st.button("💾 Save Dashboard Settings"):
            self.refresh_interval = refresh_interval
            st.success("✅ Dashboard settings updated!")
        
        # Data management
        st.subheader("🗃️ Data Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Export Performance Data"):
                # Export data functionality
                st.info("Export functionality would be implemented here")
        
        with col2:
            if st.button("🗑️ Clear Old Data"):
                # Clear old data functionality
                st.info("Data clearing functionality would be implemented here")
        
        with col3:
            if st.button("🔄 Reset All Settings"):
                # Reset settings functionality
                st.info("Settings reset functionality would be implemented here")
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to days"""
        if 'hour' in timeframe:
            hours = int(timeframe.split()[0])
            return hours / 24
        elif 'day' in timeframe:
            return int(timeframe.split()[0])
        else:
            return 7  # Default to 7 days


# Main application entry point
def main():
    """Main application entry point"""
    dashboard = AdvancedPIIDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()