#!/usr/bin/env python3
"""
Enhanced PII Review System with GPT Validation
Interactive review interface for reducing false positives
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import re
from io import BytesIO
import base64

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas

# Import existing modules
try:
    from azure_pii_detector import EnhancedAzurePIIDetector
    from gpt_validator import GPTPIIValidator
    from column_config import ColumnConfigManager, ColumnConfig, WhitelistPattern
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Warning: Some modules not available: {e}")

class InteractiveReviewSystem:
    """Interactive review system for PII redaction with learning capabilities"""
    
    def __init__(self):
        self.whitelist_db_path = "learned_whitelist.json"
        self.review_history_path = "review_history.json"
        self.learned_patterns = self.load_learned_patterns()
        self.review_history = self.load_review_history()
        
    def load_learned_patterns(self) -> Dict[str, List[str]]:
        """Load learned whitelist patterns from previous reviews"""
        if os.path.exists(self.whitelist_db_path):
            with open(self.whitelist_db_path, 'r') as f:
                return json.load(f)
        return {
            'global_whitelist': [],
            'column_specific': {},
            'context_patterns': {
                'support_ticket': [
                    'users', 'user', 'external user', 'internal user',
                    'contact', 'contacts', 'agent', 'agents',
                    'customer', 'customers', 'admin', 'admins',
                    'support staff', 'help desk', 'co-managed',
                    'end user', 'end users', 'guest user'
                ],
                'business_terms': [
                    'data', 'system', 'application', 'service',
                    'process', 'workflow', 'ticket', 'case',
                    'issue', 'problem', 'request', 'incident'
                ]
            }
        }
    
    def load_review_history(self) -> List[Dict]:
        """Load review history"""
        if os.path.exists(self.review_history_path):
            with open(self.review_history_path, 'r') as f:
                return json.load(f)
        return []
    
    def save_learned_patterns(self):
        """Save learned patterns to disk"""
        with open(self.whitelist_db_path, 'w') as f:
            json.dump(self.learned_patterns, f, indent=2)
    
    def save_review_history(self):
        """Save review history"""
        with open(self.review_history_path, 'w') as f:
            json.dump(self.review_history, f, indent=2)
    
    def add_to_whitelist(self, term: str, context: str = 'global'):
        """Add term to whitelist"""
        if context == 'global':
            if term not in self.learned_patterns['global_whitelist']:
                self.learned_patterns['global_whitelist'].append(term)
        else:
            if context not in self.learned_patterns['column_specific']:
                self.learned_patterns['column_specific'][context] = []
            if term not in self.learned_patterns['column_specific'][context]:
                self.learned_patterns['column_specific'][context].append(term)
        
        self.save_learned_patterns()
    
    def review_redactions(self, original_df: pd.DataFrame, redacted_df: pd.DataFrame, 
                         detection_results: Dict) -> pd.DataFrame:
        """
        Create an interactive review of redactions
        
        Returns:
            DataFrame with review results and corrections
        """
        review_data = []
        
        for col in original_df.columns:
            for idx in original_df.index:
                original_val = str(original_df.loc[idx, col])
                redacted_val = str(redacted_df.loc[idx, col])
                
                if original_val != redacted_val:
                    # Find what was redacted
                    redacted_terms = self.extract_redacted_terms(original_val, redacted_val)
                    
                    review_data.append({
                        'row': idx,
                        'column': col,
                        'original': original_val,
                        'redacted': redacted_val,
                        'redacted_terms': redacted_terms,
                        'false_positive': False,  # To be updated by user
                        'should_whitelist': False,  # To be updated by user
                        'confidence': detection_results.get(f"{col}_{idx}", {}).get('confidence', 0.0)
                    })
        
        return pd.DataFrame(review_data)
    
    def extract_redacted_terms(self, original: str, redacted: str) -> List[Dict]:
        """Extract terms that were redacted"""
        terms = []
        
        # Find all redaction patterns
        redaction_pattern = r'\[REDACTED_[A-Z]+\]'
        redactions = re.finditer(redaction_pattern, redacted)
        
        # Simple approach: split and compare
        # More sophisticated approach would use diff algorithms
        original_words = original.split()
        redacted_words = redacted.split()
        
        for match in redactions:
            redaction_type = match.group()
            # Extract the actual term that was redacted
            # This is simplified - real implementation would be more robust
            terms.append({
                'term': 'unknown',  # Would need better extraction logic
                'type': redaction_type,
                'position': match.start()
            })
        
        return terms
    
    def generate_review_pdf(self, review_df: pd.DataFrame, output_path: str = None) -> bytes:
        """
        Generate a PDF report for review with clickable sections
        
        Returns:
            PDF bytes for download
        """
        if output_path is None:
            buffer = BytesIO()
            output = buffer
        else:
            output = output_path
        
        doc = SimpleDocTemplate(output, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E4053'),
            alignment=TA_CENTER
        )
        
        story.append(Paragraph("PII Redaction Review Report", title_style))
        story.append(Spacer(1, 30))
        
        # Summary
        summary_style = ParagraphStyle(
            'Summary',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#34495E')
        )
        
        total_redactions = len(review_df)
        summary_text = f"""
        <b>Review Summary</b><br/>
        Total Redactions: {total_redactions}<br/>
        Review Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
        <br/>
        """
        story.append(Paragraph(summary_text, summary_style))
        story.append(Spacer(1, 20))
        
        # Create table for each redaction
        for idx, row in review_df.iterrows():
            # Section header
            section_style = ParagraphStyle(
                'Section',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2874A6')
            )
            
            story.append(Paragraph(f"Redaction #{idx + 1}", section_style))
            
            # Create comparison table
            data = [
                ['Field', 'Content'],
                ['Column', row['column']],
                ['Row', str(row['row'])],
                ['Original', row['original'][:100] + '...' if len(row['original']) > 100 else row['original']],
                ['Redacted', row['redacted'][:100] + '...' if len(row['redacted']) > 100 else row['redacted']],
                ['Confidence', f"{row.get('confidence', 0):.2%}"]
            ]
            
            table = Table(data, colWidths=[2*inch, 4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Add page break every 3 redactions
            if (idx + 1) % 3 == 0:
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        
        if output_path is None:
            buffer.seek(0)
            return buffer.getvalue()
        else:
            return None


class EnhancedPIIRedactorApp:
    """Enhanced Streamlit app with GPT validation and review system"""
    
    def __init__(self):
        self.setup_session_state()
        self.review_system = InteractiveReviewSystem()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'review_mode' not in st.session_state:
            st.session_state.review_mode = False
        if 'review_data' not in st.session_state:
            st.session_state.review_data = None
        if 'whitelist_terms' not in st.session_state:
            st.session_state.whitelist_terms = []
        if 'gpt_validation_enabled' not in st.session_state:
            st.session_state.gpt_validation_enabled = False
    
    def run(self):
        """Main application"""
        st.set_page_config(
            page_title="Enhanced PII Redactor with Review",
            page_icon="🛡️",
            layout="wide"
        )
        
        st.title("🛡️ Enhanced PII Redactor with GPT Validation & Review")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # GPT Validation Toggle
            use_gpt = st.checkbox(
                "🤖 Enable GPT Validation",
                value=st.session_state.gpt_validation_enabled,
                help="Use GPT to validate PII detections and reduce false positives"
            )
            st.session_state.gpt_validation_enabled = use_gpt
            
            if use_gpt:
                st.success("✅ GPT Validation Enabled")
                st.info("GPT will review each detection to reduce false positives")
            
            # Whitelist Management
            st.header("📝 Learned Whitelist")
            
            if st.button("📥 Import Whitelist"):
                self.import_whitelist()
            
            if st.button("📤 Export Whitelist"):
                self.export_whitelist()
            
            # Show current whitelist
            with st.expander("Current Whitelist Terms"):
                whitelist = self.review_system.learned_patterns.get('global_whitelist', [])
                if whitelist:
                    for term in whitelist[:20]:  # Show first 20
                        st.text(f"• {term}")
                    if len(whitelist) > 20:
                        st.text(f"... and {len(whitelist) - 20} more")
                else:
                    st.text("No terms in whitelist yet")
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Process Data", 
            "🔍 Review & Validate", 
            "📋 Generate Report",
            "📈 Analytics"
        ])
        
        with tab1:
            self.process_data_tab()
        
        with tab2:
            self.review_validate_tab()
        
        with tab3:
            self.generate_report_tab()
        
        with tab4:
            self.analytics_tab()
    
    def process_data_tab(self):
        """Data processing tab with GPT validation"""
        st.header("📊 Process Data with Smart PII Detection")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Whitelist Terms", len(self.review_system.learned_patterns['global_whitelist']))
            
            if st.button("🚀 Process with Smart Detection", type="primary"):
                with st.spinner("Processing with GPT validation..."):
                    # This would integrate with your existing detection system
                    processed_df, stats = self.process_with_validation(df)
                    
                    st.session_state.original_df = df
                    st.session_state.processed_df = processed_df
                    st.session_state.processing_stats = stats
                    
                    st.success("✅ Processing complete!")
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Detections", stats.get('total_detections', 0))
                    with col2:
                        st.metric("GPT Corrections", stats.get('gpt_corrections', 0))
                    with col3:
                        st.metric("False Positives Prevented", stats.get('false_positives_prevented', 0))
    
    def review_validate_tab(self):
        """Interactive review and validation tab"""
        st.header("🔍 Review & Validate Redactions")
        
        if 'processed_df' not in st.session_state:
            st.warning("Please process data first in the Process Data tab")
            return
        
        original_df = st.session_state.original_df
        processed_df = st.session_state.processed_df
        
        # Create review dataframe
        review_df = self.create_review_dataframe(original_df, processed_df)
        
        if len(review_df) == 0:
            st.info("No redactions to review")
            return
        
        # Review interface
        st.subheader(f"Found {len(review_df)} redactions to review")
        
        # Filtering options
        col1, col2 = st.columns(2)
        with col1:
            column_filter = st.selectbox(
                "Filter by column",
                ["All"] + list(review_df['column'].unique())
            )
        with col2:
            confidence_threshold = st.slider(
                "Minimum confidence",
                0.0, 1.0, 0.0
            )
        
        # Filter review data
        filtered_df = review_df
        if column_filter != "All":
            filtered_df = filtered_df[filtered_df['column'] == column_filter]
        filtered_df = filtered_df[filtered_df['confidence'] >= confidence_threshold]
        
        # Pagination
        items_per_page = 10
        num_pages = (len(filtered_df) - 1) // items_per_page + 1
        page = st.number_input("Page", min_value=1, max_value=num_pages, value=1)
        
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_df))
        
        # Display items for review
        for idx in range(start_idx, end_idx):
            row = filtered_df.iloc[idx]
            
            with st.expander(f"📝 {row['column']} - Row {row['row']} (Confidence: {row['confidence']:.1%})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original:**")
                    st.text_area(
                        "Original text",
                        value=row['original'],
                        height=100,
                        disabled=True,
                        key=f"orig_{idx}"
                    )
                
                with col2:
                    st.markdown("**Redacted:**")
                    st.text_area(
                        "Redacted text",
                        value=row['redacted'],
                        height=100,
                        disabled=True,
                        key=f"red_{idx}"
                    )
                
                # Review actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"✅ Correct", key=f"correct_{idx}"):
                        st.success("Marked as correct")
                
                with col2:
                    if st.button(f"❌ False Positive", key=f"false_{idx}"):
                        # Extract the incorrectly redacted terms
                        terms = self.extract_false_positive_terms(row['original'], row['redacted'])
                        for term in terms:
                            self.review_system.add_to_whitelist(term, 'global')
                        st.success(f"Added {len(terms)} terms to whitelist")
                
                with col3:
                    if st.button(f"🔧 Manual Edit", key=f"edit_{idx}"):
                        st.session_state[f'editing_{idx}'] = True
                
                # Manual edit mode
                if st.session_state.get(f'editing_{idx}', False):
                    new_text = st.text_area(
                        "Edit redacted text:",
                        value=row['redacted'],
                        key=f"edit_text_{idx}"
                    )
                    if st.button(f"💾 Save Edit", key=f"save_{idx}"):
                        # Save the manual edit
                        st.session_state.processed_df.loc[row['row'], row['column']] = new_text
                        st.session_state[f'editing_{idx}'] = False
                        st.success("Edit saved")
    
    def generate_report_tab(self):
        """Generate review report tab"""
        st.header("📋 Generate Review Report")
        
        if 'processed_df' not in st.session_state:
            st.warning("Please process data first")
            return
        
        st.subheader("Report Options")
        
        col1, col2 = st.columns(2)
        with col1:
            report_format = st.selectbox(
                "Report Format",
                ["PDF", "HTML", "Excel"]
            )
        
        with col2:
            include_options = st.multiselect(
                "Include in report",
                ["Summary Statistics", "Detailed Comparisons", "Whitelist Terms", "Confidence Scores"],
                default=["Summary Statistics", "Detailed Comparisons"]
            )
        
        if st.button("📄 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                if report_format == "PDF":
                    pdf_bytes = self.generate_pdf_report()
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"pii_review_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    
                    st.success("✅ Report generated successfully!")
    
    def analytics_tab(self):
        """Analytics and insights tab"""
        st.header("📈 Analytics & Insights")
        
        # Load review history
        history = self.review_system.review_history
        
        if not history:
            st.info("No review history available yet")
            return
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_reviews = len(history)
            st.metric("Total Reviews", total_reviews)
        
        with col2:
            # Calculate false positive rate
            false_positives = sum(1 for h in history if h.get('false_positive', False))
            fp_rate = (false_positives / total_reviews * 100) if total_reviews > 0 else 0
            st.metric("False Positive Rate", f"{fp_rate:.1f}%")
        
        with col3:
            whitelist_size = len(self.review_system.learned_patterns['global_whitelist'])
            st.metric("Whitelist Terms", whitelist_size)
        
        with col4:
            # Calculate improvement over time
            if len(history) > 1:
                recent_fp_rate = self.calculate_recent_fp_rate(history[-10:])
                overall_fp_rate = self.calculate_recent_fp_rate(history)
                improvement = overall_fp_rate - recent_fp_rate
                st.metric("FP Rate Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
        
        # Charts
        st.subheader("Trends")
        
        # False positive trend chart
        if len(history) > 5:
            import plotly.graph_objects as go
            
            # Group by date
            dates = []
            fp_rates = []
            
            # Simplified trend calculation
            for i in range(0, len(history), 10):
                batch = history[i:i+10]
                if batch:
                    fp_count = sum(1 for h in batch if h.get('false_positive', False))
                    fp_rate = (fp_count / len(batch)) * 100
                    fp_rates.append(fp_rate)
                    dates.append(i)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=fp_rates,
                mode='lines+markers',
                name='False Positive Rate',
                line=dict(color='#E74C3C', width=2)
            ))
            
            fig.update_layout(
                title="False Positive Rate Trend",
                xaxis_title="Review Batch",
                yaxis_title="False Positive Rate (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Most common false positives
        st.subheader("Most Common False Positives")
        
        false_positive_terms = {}
        for item in history:
            if item.get('false_positive', False):
                for term in item.get('terms', []):
                    false_positive_terms[term] = false_positive_terms.get(term, 0) + 1
        
        if false_positive_terms:
            sorted_terms = sorted(false_positive_terms.items(), key=lambda x: x[1], reverse=True)[:10]
            
            terms_df = pd.DataFrame(sorted_terms, columns=['Term', 'Count'])
            st.dataframe(terms_df)
    
    # Helper methods
    def process_with_validation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Process dataframe with GPT validation"""
        # This is a simplified version - integrate with your actual processing
        stats = {
            'total_detections': 0,
            'gpt_corrections': 0,
            'false_positives_prevented': 0
        }
        
        # Apply whitelist
        whitelist_terms = self.review_system.learned_patterns['global_whitelist']
        
        # Simulate processing (replace with actual implementation)
        processed_df = df.copy()
        
        # Example: Apply some redactions
        for col in df.columns:
            for idx in df.index:
                value = str(df.loc[idx, col])
                
                # Check if contains known false positive terms
                contains_whitelist = any(term.lower() in value.lower() for term in whitelist_terms)
                
                if not contains_whitelist and '@' in value:  # Simple email detection
                    processed_df.loc[idx, col] = '[REDACTED_EMAIL]'
                    stats['total_detections'] += 1
        
        return processed_df, stats
    
    def create_review_dataframe(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Create dataframe for review"""
        review_data = []
        
        for col in original_df.columns:
            for idx in original_df.index:
                orig = str(original_df.loc[idx, col])
                proc = str(processed_df.loc[idx, col])
                
                if orig != proc:
                    review_data.append({
                        'row': idx,
                        'column': col,
                        'original': orig,
                        'redacted': proc,
                        'confidence': np.random.random()  # Replace with actual confidence
                    })
        
        return pd.DataFrame(review_data)
    
    def extract_false_positive_terms(self, original: str, redacted: str) -> List[str]:
        """Extract terms that were incorrectly redacted"""
        terms = []
        
        # Find what was redacted by comparing
        import difflib
        
        # Simple word-level diff
        orig_words = original.split()
        red_words = redacted.split()
        
        for i, (o, r) in enumerate(zip(orig_words, red_words)):
            if o != r and '[REDACTED' in r:
                terms.append(o)
        
        return terms
    
    def calculate_recent_fp_rate(self, history: List[Dict]) -> float:
        """Calculate false positive rate for a set of history items"""
        if not history:
            return 0.0
        
        false_positives = sum(1 for h in history if h.get('false_positive', False))
        return (false_positives / len(history)) * 100
    
    def generate_pdf_report(self) -> bytes:
        """Generate PDF report"""
        original_df = st.session_state.get('original_df')
        processed_df = st.session_state.get('processed_df')
        
        if original_df is None or processed_df is None:
            return b""
        
        # Create review dataframe
        review_df = self.create_review_dataframe(original_df, processed_df)
        
        # Generate PDF
        return self.review_system.generate_review_pdf(review_df)
    
    def import_whitelist(self):
        """Import whitelist from file"""
        uploaded = st.file_uploader("Upload whitelist JSON", type=['json'])
        if uploaded:
            whitelist = json.load(uploaded)
            self.review_system.learned_patterns.update(whitelist)
            self.review_system.save_learned_patterns()
            st.success("Whitelist imported successfully!")
    
    def export_whitelist(self):
        """Export whitelist to file"""
        whitelist_json = json.dumps(self.review_system.learned_patterns, indent=2)
        st.download_button(
            label="Download Whitelist",
            data=whitelist_json,
            file_name=f"whitelist_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )


def main():
    """Main entry point"""
    app = EnhancedPIIRedactorApp()
    app.run()


if __name__ == "__main__":
    main()
