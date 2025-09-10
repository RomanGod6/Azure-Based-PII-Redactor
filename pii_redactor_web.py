#!/usr/bin/env python3
"""
PII Redactor Pro - Streamlit Web App
A modern web-based GUI for redacting PII from CSV files using Azure AI
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sqlite3
from pathlib import Path
import tempfile
import time
import json
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

# Azure AI imports
try:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Import enhanced PII detection and column configuration
try:
    from azure_pii_detector import EnhancedAzurePIIDetector
    from column_config import ColumnConfigManager
    COLUMN_CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Column configuration not available: {e}")
    COLUMN_CONFIG_AVAILABLE = False

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="PII Redactor Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PIIRedactorWebApp:
    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.azure_key = os.getenv("AZURE_KEY")
        self.text_analytics_client = None
        self.enhanced_detector = None
        self.column_config_manager = None
        self.setup_azure_client()
        self.setup_column_config()
        self.init_database()
    
    def setup_azure_client(self):
        """Setup Azure Text Analytics client"""
        if AZURE_AVAILABLE and self.azure_endpoint and self.azure_key:
            try:
                self.text_analytics_client = TextAnalyticsClient(
                    endpoint=self.azure_endpoint,
                    credential=AzureKeyCredential(self.azure_key)
                )
                return True
            except Exception as e:
                st.error(f"Azure connection failed: {str(e)}")
                return False
        return False
    
    def setup_column_config(self):
        """Setup column configuration manager"""
        if COLUMN_CONFIG_AVAILABLE:
            try:
                self.column_config_manager = ColumnConfigManager()
                if self.azure_endpoint and self.azure_key:
                    self.enhanced_detector = EnhancedAzurePIIDetector(
                        self.azure_endpoint, 
                        self.azure_key,
                        self.column_config_manager
                    )
                return True
            except Exception as e:
                st.error(f"Column configuration setup failed: {str(e)}")
                return False
        return False
    
    def init_database(self):
        """Initialize SQLite database for history"""
        try:
            conn = sqlite3.connect('pii_redactor_history.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS redaction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    filename TEXT,
                    rows_processed INTEGER,
                    columns_processed INTEGER,
                    cost REAL,
                    duration_seconds REAL,
                    status TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Database initialization failed: {str(e)}")
    
    def detect_and_redact_text(self, text):
        """Detect and redact PII in text using Azure AI or local fallback"""
        if not text or pd.isna(text):
            return str(text), 0.0
        
        text_str = str(text).strip()
        if not text_str:
            return text_str, 0.0
        
        cost = 0.0
        
        if self.text_analytics_client:
            try:
                # Use Azure AI
                documents = [text_str]
                response = self.text_analytics_client.recognize_pii_entities(documents)
                
                # Calculate cost (approximately $1 per 1000 text records)
                cost = len(text_str) * 0.001 / 1000
                
                redacted_text = text_str
                for doc in response:
                    if not doc.is_error:
                        # Sort entities by offset in reverse order to maintain positions
                        entities = sorted(doc.entities, key=lambda x: x.offset, reverse=True)
                        
                        for entity in entities:
                            start = entity.offset
                            end = start + entity.length
                            category = entity.category
                            
                            # Create redaction label
                            redaction_label = f"[REDACTED_{category.upper()}]"
                            
                            # Replace the entity text
                            redacted_text = redacted_text[:start] + redaction_label + redacted_text[end:]
                
                return redacted_text, cost
                
            except Exception as e:
                st.warning(f"Azure AI failed, using local detection: {str(e)}")
        
        # Fallback to local regex-based detection
        return self.local_pii_detection(text_str), 0.0
    
    def local_pii_detection(self, text):
        """Local regex-based PII detection as fallback"""
        import re
        
        # Common PII patterns
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            'SSN': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'CREDIT_CARD': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'IP_ADDRESS': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }
        
        redacted_text = text
        for category, pattern in patterns.items():
            redacted_text = re.sub(pattern, f'[REDACTED_{category}]', redacted_text)
        
        return redacted_text
    
    def process_dataframe(self, df, columns_to_process=None, column_configs=None, use_gpt_validation=False, azure_api_key=None):
        """Process entire dataframe with column-specific configurations"""
        if columns_to_process is None:
            columns_to_process = df.columns.tolist()
        
        processed_df = df.copy()
        total_cost = 0.0
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_cells = len(df) * len(columns_to_process)
        processed_cells = 0
        
        start_time = time.time()
        
        # Setup GPT validation if requested
        if use_gpt_validation and azure_api_key:
            try:
                from gpt_validator import GPTPIIValidator
                gpt_validator = GPTPIIValidator()  # Will auto-load from environment
                status_text.text("🤖 Azure GPT validation enabled - initializing...")
            except ImportError:
                st.warning("⚠️ GPT validation not available. Install requests package: pip install requests")
                gpt_validator = None
            except Exception as e:
                st.warning(f"⚠️ GPT validation setup failed: {str(e)}")
                gpt_validator = None
        else:
            gpt_validator = None
        
        # Use enhanced detector if available and configured
        if self.enhanced_detector and column_configs:
            # Setup enhanced detector with GPT validation
            if gpt_validator:
                self.enhanced_detector.setup_gpt_validation(gpt_validator, azure_api_key)
                status_text.text("🤖 Processing with Azure GPT validation and column-specific configurations...")
            else:
                status_text.text("Processing with column-specific configurations...")
                
            # Import ColumnConfig classes for proper object creation
            try:
                from column_config import ColumnConfig, WhitelistPattern, BlacklistPattern
                
                # Convert dictionary configs to ColumnConfig objects
                for col_name, config_dict in column_configs.items():
                    if col_name in columns_to_process:
                        # Create ColumnConfig object
                        col_config = ColumnConfig(column_name=col_name)
                        col_config.enabled = config_dict.get('enabled', True)
                        col_config.sensitivity_threshold = config_dict.get('sensitivity', 0.7)
                        col_config.custom_redaction_label = config_dict.get('custom_redaction_label', '')
                        
                        # Add whitelist patterns
                        whitelist_patterns = config_dict.get('whitelist_patterns', [])
                        for pattern in whitelist_patterns:
                            if pattern.strip():
                                col_config.add_whitelist_pattern(WhitelistPattern(pattern.strip()))
                        
                        # Add blacklist patterns
                        blacklist_patterns = config_dict.get('blacklist_patterns', [])
                        for pattern in blacklist_patterns:
                            if pattern.strip():
                                col_config.add_blacklist_pattern(BlacklistPattern(pattern.strip()))
                        
                        # Add excluded entity types
                        excluded_types = config_dict.get('excluded_entity_types', [])
                        for entity_type in excluded_types:
                            if entity_type.strip():
                                col_config.add_excluded_entity_type(entity_type.strip())
                        
                        # Set the configuration in the manager
                        self.column_config_manager.set_column_config(col_config)
                
                # Process using enhanced detector
                status_text.text("Processing with column-specific configurations...")
                processed_df, stats = self.enhanced_detector.detect_and_redact_dataframe(
                    df, columns_to_process
                )
                total_cost = stats.get('cost', 0.0)
                
                progress_bar.progress(1.0)
                duration = time.time() - start_time
                status_text.text("✅ Processing complete!")
                
                # Get GPT validation stats if available
                gpt_stats = {}
                if gpt_validator:
                    gpt_stats = {
                        'gpt_cost': gpt_validator.total_cost,
                        'gpt_enabled': True
                    }
                
                return processed_df, total_cost, duration, gpt_stats
                
            except ImportError as e:
                status_text.text(f"Column config import failed: {e}. Using standard processing...")
            except Exception as e:
                status_text.text(f"Enhanced processing failed: {e}. Using standard processing...")
        
        # Fallback to original processing
        for col in columns_to_process:
            if col in df.columns:
                status_text.text(f"Processing column: {col}")
                
                for idx in df.index:
                    original_value = df.loc[idx, col]
                    redacted_value, cost = self.detect_and_redact_text(original_value)
                    processed_df.loc[idx, col] = redacted_value
                    total_cost += cost
                    
                    processed_cells += 1
                    progress = processed_cells / total_cells
                    progress_bar.progress(progress)
                    
                    # Update ETA
                    if processed_cells > 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_cell = elapsed_time / processed_cells
                        remaining_cells = total_cells - processed_cells
                        eta_seconds = remaining_cells * avg_time_per_cell
                        eta_minutes = eta_seconds / 60
                        
                        if eta_minutes > 1:
                            status_text.text(f"Processing column: {col} - ETA: {eta_minutes:.1f} minutes")
                        else:
                            status_text.text(f"Processing column: {col} - ETA: {eta_seconds:.0f} seconds")
        
        duration = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        # Get GPT validation stats if available
        gpt_stats = {}
        if gpt_validator:
            gpt_stats = {
                'gpt_cost': gpt_validator.total_cost,
                'gpt_enabled': True
            }
        
        return processed_df, total_cost, duration, gpt_stats
    
    def save_to_history(self, filename, rows, columns, cost, duration, status):
        """Save processing details to history database"""
        try:
            conn = sqlite3.connect('pii_redactor_history.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO redaction_history 
                (timestamp, filename, rows_processed, columns_processed, cost, duration_seconds, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                filename,
                rows,
                columns,
                cost,
                duration,
                status
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Failed to save to history: {str(e)}")
    
    def get_history(self):
        """Get processing history from database"""
        try:
            conn = sqlite3.connect('pii_redactor_history.db')
            df = pd.read_sql_query("SELECT * FROM redaction_history ORDER BY timestamp DESC", conn)
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()
    
    def extract_redacted_terms(self, original: str, redacted: str) -> list:
        """Extract what was redacted by comparing original and redacted text"""
        import re
        terms = []
        
        # Find all redaction patterns in the redacted text
        pattern = r'\[REDACTED_([A-Z_]+)\]'
        redactions = re.finditer(pattern, redacted)
        
        # For each redaction, try to find what original term it replaced
        redacted_copy = redacted
        original_words = original.split()
        redacted_words = redacted.split()
        
        # Simple approach: find positions where words differ
        orig_idx = 0
        red_idx = 0
        
        while orig_idx < len(original_words) and red_idx < len(redacted_words):
            if '[REDACTED_' in redacted_words[red_idx]:
                # Found a redaction, try to map it back to original
                redaction_type = re.search(r'\[REDACTED_([A-Z_]+)\]', redacted_words[red_idx])
                if redaction_type:
                    entity_type = redaction_type.group(1)
                    # Assume the original term is at the same position
                    if orig_idx < len(original_words):
                        terms.append({
                            'original': original_words[orig_idx],
                            'type': entity_type,
                            'full_redaction': redacted_words[red_idx]
                        })
                orig_idx += 1
                red_idx += 1
            elif original_words[orig_idx] == redacted_words[red_idx]:
                orig_idx += 1
                red_idx += 1
            else:
                # Skip mismatched words
                orig_idx += 1
        
        return terms
    
    def apply_whitelist_to_text(self, original: str, redacted: str, whitelist: set) -> str:
        """Apply whitelist to restore false positives in a single text"""
        result = redacted
        
        # Get redacted terms
        terms = self.extract_redacted_terms(original, redacted)
        
        # Check each term against whitelist
        for term in terms:
            if term['original'].lower() in [w.lower() for w in whitelist]:
                # Replace the redaction with original term
                result = result.replace(term['full_redaction'], term['original'], 1)
        
        return result
    
    def apply_whitelist_to_dataframe(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Apply whitelist across entire dataframe"""
        corrected_df = processed_df.copy()
        whitelist = st.session_state.get('whitelist', set())
        
        if not whitelist:
            return corrected_df
        
        for col in processed_df.columns:
            for idx in processed_df.index:
                original_val = str(original_df.loc[idx, col])
                redacted_val = str(processed_df.loc[idx, col])
                
                if original_val != redacted_val and '[REDACTED_' in redacted_val:
                    # Apply whitelist
                    corrected_val = self.apply_whitelist_to_text(original_val, redacted_val, whitelist)
                    corrected_df.loc[idx, col] = corrected_val
        
        return corrected_df

def main():
    app = PIIRedactorWebApp()
    
    # Header
    st.title("🛡️ PII Redactor Pro")
    st.markdown("**Azure AI-Powered Data Protection for CSV Files**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Azure status
        if app.text_analytics_client:
            st.success("✅ Azure AI Connected")
        else:
            st.warning("⚠️ Azure AI Not Available")
            if not AZURE_AVAILABLE:
                st.error("Azure libraries not installed")
            elif not app.azure_endpoint or not app.azure_key:
                st.error("Azure credentials not found in .env")
        
        # Column configuration status
        if COLUMN_CONFIG_AVAILABLE and app.column_config_manager:
            st.success("✅ Column Configuration Available")
        else:
            st.warning("⚠️ Column Configuration Not Available")
        
        # File upload
        st.header("📁 Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your CSV file containing PII to be redacted"
        )
        
        if uploaded_file:
            # Load and preview data
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
                
                # Column selection
                st.header("🎯 Select Columns")
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect(
                    "Choose columns to process:",
                    all_columns,
                    default=all_columns,
                    help="Select which columns contain PII to redact"
                )
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                df = None
                selected_columns = []
        else:
            df = None
            selected_columns = []
                
        # Processing options
        st.header("🔧 Options")
        preview_only = st.checkbox("Preview mode (first 5 rows only)", value=False)
        local_only = st.checkbox("Use local detection only (no Azure)", value=False)
        
        # GPT Validation option
        use_gpt_validation = st.checkbox(
            "🤖 Enable GPT Validation (Reduces false positives)", 
            value=False,
            help="Uses Azure GPT to validate PII detections and reduce false positives. Uses your configured Azure API key."
        )
        
        # Load Azure API key from environment
        azure_api_key = os.getenv('AZURE_KEY') if use_gpt_validation else None
        
        if use_gpt_validation and not azure_api_key:
            st.error("❌ Azure API key not found in environment. Please check your .env file.")
        elif use_gpt_validation:
            st.success("✅ Azure GPT validation ready")
    
    # Main content area
    if df is not None:
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Data Preview", "⚙️ Column Config", "🚀 Process", "📈 Results", "🔍 Interactive Review", "📜 History"])
        
        with tab1:
            st.header("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Selected Columns", len(selected_columns))
        
        with tab2:
            st.header("⚙️ Column Configuration")
            
            if COLUMN_CONFIG_AVAILABLE and app.column_config_manager and selected_columns:
                # Initialize session state for column configs
                if 'column_configs' not in st.session_state:
                    st.session_state['column_configs'] = {}
                if 'applied_template' not in st.session_state:
                    st.session_state['applied_template'] = None
                
                st.markdown("### Configure PII detection for each column")
                st.markdown("*Customize how PII detection works for different types of data*")
                
                # Template Management Section - TOP PRIORITY
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown("#### 📋 **Template Management**")
                    template_type = st.selectbox(
                        "Choose a pre-configured template:",
                        ["None", "Zendesk Tickets", "Support Tickets", "Customer Data", "Employee Data", "Financial Records", "Medical Records"],
                        index=0,
                        key="template_select",
                        help="Templates provide optimized settings for common data types"
                    )
                
                with col2:
                    # Template status indicator
                    if st.session_state['applied_template']:
                        st.success(f"✅ **Template Applied:** {st.session_state['applied_template']}")
                    else:
                        st.info("ℹ️ **No template applied** - using custom settings")
                    
                    # Apply template button
                    if st.button("🚀 Apply Template", disabled=(template_type == "None"), type="primary"):
                        if template_type != "None":
                            st.session_state['applied_template'] = template_type
                            
                            # Apply template configurations immediately
                            if template_type == "Zendesk Tickets":
                                # Comprehensive Zendesk template - handles all common false positives
                                zendesk_configs = {
                                    # Subject lines - usually safe, minimal redaction
                                    'subject': {
                                        'enabled': True,
                                        'sensitivity': 0.9,
                                        'whitelist_patterns': [
                                            r'issue\s+with', r'problem\s+with', r'question\s+about',
                                            r'help\s+with', r'unable\s+to', r'error\s+in',
                                            r'request\s+for', r'access\s+to', r'permission'
                                        ],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': ['PersonType', 'Event', 'Product', 'Skill', 'Organization'],
                                        'custom_redaction_label': '[SUBJECT_INFO]',
                                        'column_type': 'text'
                                    },
                                    
                                    # Description/Body - main content with lots of false positives
                                    'description': {
                                        'enabled': True,
                                        'sensitivity': 0.85,
                                        'whitelist_patterns': [
                                            r'co-managed\s+users?', r'external\s+users?', r'internal\s+only',
                                            r'end\s+users?', r'admin\s+users?', r'guest\s+users?',
                                            r'system\s+users?', r'test\s+users?', r'demo\s+users?',
                                            r'contact\s+form', r'support\s+ticket', r'help\s+desk',
                                            r'user\s+interface', r'user\s+experience', r'user\s+guide',
                                            r'user\s+manual', r'user\s+account', r'user\s+profile',
                                            r'data\s+export', r'data\s+import', r'file\s+upload',
                                            r'system\s+error', r'server\s+error', r'application\s+error',
                                            r'browser\s+issue', r'login\s+problem', r'access\s+denied',
                                            r'permission\s+denied', r'feature\s+request', r'bug\s+report',
                                            r'ticket\s+#?\d+', r'case\s+#?\d+', r'ref\s+#?\d+'
                                        ],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': ['PersonType', 'Event', 'Product', 'Skill', 'DateTime'],
                                        'custom_redaction_label': '[TICKET_CONTENT]',
                                        'column_type': 'text'
                                    },
                                    
                                    # Comments/Notes - similar to description but may contain agent notes
                                    'body': {
                                        'enabled': True,
                                        'sensitivity': 0.8,
                                        'whitelist_patterns': [
                                            r'agent\s+notes?', r'internal\s+notes?', r'private\s+notes?',
                                            r'follow\s+up', r'escalated\s+to', r'assigned\s+to',
                                            r'status\s+changed', r'priority\s+set', r'resolved\s+by',
                                            r'customer\s+contacted', r'waiting\s+for', r'pending\s+review'
                                        ],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': ['PersonType', 'Event', 'Product', 'Skill'],
                                        'custom_redaction_label': '[COMMENT_CONTENT]',
                                        'column_type': 'text'
                                    },
                                    
                                    # Status fields - usually safe
                                    'status': {
                                        'enabled': False,  # Usually just "open", "closed", etc.
                                        'sensitivity': 0.7,
                                        'whitelist_patterns': [],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': [],
                                        'custom_redaction_label': '',
                                        'column_type': 'text'
                                    },
                                    
                                    # Priority fields - usually safe
                                    'priority': {
                                        'enabled': False,  # Usually just "high", "low", etc.
                                        'sensitivity': 0.7,
                                        'whitelist_patterns': [],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': [],
                                        'custom_redaction_label': '',
                                        'column_type': 'text'
                                    },
                                    
                                    # Tags - can contain sensitive info but lots of false positives
                                    'tags': {
                                        'enabled': True,
                                        'sensitivity': 0.9,
                                        'whitelist_patterns': [
                                            r'bug', r'feature', r'enhancement', r'urgent', r'high_priority',
                                            r'billing', r'technical', r'account', r'access', r'permissions',
                                            r'integration', r'api', r'mobile', r'web', r'desktop'
                                        ],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': ['PersonType', 'Organization', 'Product', 'Skill'],
                                        'custom_redaction_label': '[TAG]',
                                        'column_type': 'text'
                                    },
                                    
                                    # Custom fields - varies, moderate protection
                                    'custom_fields': {
                                        'enabled': True,
                                        'sensitivity': 0.75,
                                        'whitelist_patterns': [
                                            r'version\s+\d+', r'build\s+\d+', r'environment',
                                            r'production', r'staging', r'development', r'test'
                                        ],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': ['PersonType', 'Event', 'Product'],
                                        'custom_redaction_label': '[CUSTOM_FIELD]',
                                        'column_type': 'text'
                                    },
                                    
                                    # Author/User info - likely contains PII, but protect role-based terms
                                    'author_name': {
                                        'enabled': True,
                                        'sensitivity': 0.6,  # Lower to catch names but protect roles
                                        'whitelist_patterns': [
                                            r'support\s+agent', r'customer\s+service', r'tech\s+support',
                                            r'system\s+admin', r'help\s+desk', r'administrator',
                                            r'agent\s+\d+', r'user\s+\d+', r'customer\s+\d+'
                                        ],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': ['PersonType'],
                                        'custom_redaction_label': '[AUTHOR]',
                                        'column_type': 'name'
                                    },
                                    
                                    # Email fields - high PII but protect system emails
                                    'author_email': {
                                        'enabled': True,
                                        'sensitivity': 0.7,
                                        'whitelist_patterns': [
                                            r'noreply@', r'support@', r'help@', r'system@',
                                            r'admin@', r'notifications@', r'alerts@'
                                        ],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': [],
                                        'custom_redaction_label': '[EMAIL]',
                                        'column_type': 'email'
                                    },
                                    
                                    # Organization fields - protect common business terms
                                    'organization_id': {
                                        'enabled': True,
                                        'sensitivity': 0.8,
                                        'whitelist_patterns': [
                                            r'enterprise', r'business', r'professional', r'standard',
                                            r'trial', r'demo', r'test', r'sandbox'
                                        ],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': ['PersonType', 'Event'],
                                        'custom_redaction_label': '[ORG_ID]',
                                        'column_type': 'id'
                                    }
                                }
                                
                                # Apply configurations to matching columns
                                for col in selected_columns:
                                    col_key = f"config_{col}"
                                    col_lower = col.lower()
                                    
                                    # Smart column matching
                                    if any(keyword in col_lower for keyword in ['subject']):
                                        config = zendesk_configs['subject']
                                    elif any(keyword in col_lower for keyword in ['description', 'content', 'message']):
                                        config = zendesk_configs['description']
                                    elif any(keyword in col_lower for keyword in ['body', 'comment', 'note', 'text']):
                                        config = zendesk_configs['body']
                                    elif any(keyword in col_lower for keyword in ['status']):
                                        config = zendesk_configs['status']
                                    elif any(keyword in col_lower for keyword in ['priority']):
                                        config = zendesk_configs['priority']
                                    elif any(keyword in col_lower for keyword in ['tag']):
                                        config = zendesk_configs['tags']
                                    elif any(keyword in col_lower for keyword in ['custom', 'field']):
                                        config = zendesk_configs['custom_fields']
                                    elif any(keyword in col_lower for keyword in ['author', 'name']) and 'email' not in col_lower:
                                        config = zendesk_configs['author_name']
                                    elif any(keyword in col_lower for keyword in ['email', 'mail']):
                                        config = zendesk_configs['author_email']
                                    elif any(keyword in col_lower for keyword in ['org', 'company']):
                                        config = zendesk_configs['organization_id']
                                    else:
                                        # Default configuration for unknown columns
                                        config = {
                                            'enabled': True,
                                            'sensitivity': 0.75,
                                            'whitelist_patterns': [],
                                            'blacklist_patterns': [],
                                            'excluded_entity_types': ['PersonType', 'Event'],
                                            'custom_redaction_label': '[ZENDESK_DATA]',
                                            'column_type': 'auto'
                                        }
                                    
                                    st.session_state['column_configs'][col_key] = config.copy()
                            
                            elif template_type == "Support Tickets":
                                # Template for support ticket data - excludes PersonType which causes false positives
                                for col in selected_columns:
                                    col_key = f"config_{col}"
                                    if 'description' in col.lower() or 'comment' in col.lower() or 'note' in col.lower():
                                        st.session_state['column_configs'][col_key] = {
                                            'enabled': True,
                                            'sensitivity': 0.8,
                                            'whitelist_patterns': ['Co-managed', 'external user', 'internal only', 'contact'],
                                            'blacklist_patterns': [],
                                            'excluded_entity_types': ['PersonType', 'Event', 'Product'],
                                            'custom_redaction_label': '[REDACTED_INFO]',
                                            'column_type': 'text'
                                        }
                            elif template_type == "Customer Data":
                                # Standard customer data template
                                for col in selected_columns:
                                    col_key = f"config_{col}"
                                    st.session_state['column_configs'][col_key] = {
                                        'enabled': True,
                                        'sensitivity': 0.7,
                                        'whitelist_patterns': [],
                                        'blacklist_patterns': [],
                                        'excluded_entity_types': ['PersonType'],
                                        'custom_redaction_label': '',
                                        'column_type': 'auto'
                                    }
                            
                            st.success(f"✅ Applied '{template_type}' template to {len(selected_columns)} columns!")
                            st.rerun()
                
                with col3:
                    # Clear template button
                    if st.button("🗑️ Clear", help="Remove template and reset to defaults"):
                        st.session_state['applied_template'] = None
                        st.session_state['column_configs'] = {}
                        st.success("Template cleared!")
                        st.rerun()
                
                # Template details
                if st.session_state['applied_template']:
                    with st.expander(f"📋 Template Details: {st.session_state['applied_template']}", expanded=False):
                        if st.session_state['applied_template'] == "Zendesk Tickets":
                            st.markdown("""
                            **Zendesk Tickets Template** optimizes for support ticket data:
                            
                            🎯 **Smart Column Detection:**
                            - Subject lines: High sensitivity, protects common phrases
                            - Descriptions: Whitelists "co-managed users", "external user", etc.
                            - Comments: Protects agent terminology
                            - Status/Priority: Disabled (usually safe)
                            - Tags: High sensitivity, protects common tags
                            - Names: Lower sensitivity, protects role-based terms
                            - Emails: Protects system emails (noreply@, support@)
                            
                            🚫 **Excluded Entity Types:**
                            - PersonType (prevents "users", "contact" false positives)
                            - Event, Product, Skill (reduces business term flagging)
                            
                            ✅ **Perfect for:** Zendesk exports, customer support data
                            """)
                        elif st.session_state['applied_template'] == "Support Tickets":
                            st.markdown("""
                            **Support Tickets Template** for general support data:
                            - Moderate sensitivity settings
                            - Protects common support terminology
                            - Excludes PersonType, Event, Product entities
                            """)
                        elif st.session_state['applied_template'] == "Customer Data":
                            st.markdown("""
                            **Customer Data Template** for standard customer records:
                            - Balanced sensitivity settings
                            - Excludes PersonType to reduce false positives
                            - Good for CRM exports, customer databases
                            """)
                
                st.markdown("---")
                
                # Configuration for each selected column
                for col in selected_columns:
                    with st.expander(f"🔧 Configure Column: **{col}**", expanded=False):
                        col_key = f"config_{col}"
                        
                        # Initialize config if not exists
                        if col_key not in st.session_state['column_configs']:
                            st.session_state['column_configs'][col_key] = {
                                'enabled': True,
                                'sensitivity': 0.7,
                                'whitelist_patterns': [],
                                'blacklist_patterns': [],
                                'excluded_entity_types': [],
                                'custom_redaction_label': '',
                                'column_type': 'auto'
                            }
                        
                        config = st.session_state['column_configs'][col_key]
                        
                        # Show sample data
                        sample_values = df[col].dropna().head(3).tolist()
                        if sample_values:
                            st.markdown("**Sample data:**")
                            for i, val in enumerate(sample_values):
                                st.code(f"Row {i+1}: {str(val)[:100]}{'...' if len(str(val)) > 100 else ''}")
                        
                        # Configuration options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            config['enabled'] = st.checkbox(
                                "Enable PII Detection",
                                value=config['enabled'],
                                key=f"enabled_{col}"
                            )
                            
                            config['column_type'] = st.selectbox(
                                "Column Type",
                                ['auto', 'name', 'email', 'phone', 'address', 'id', 'date', 'text', 'numeric'],
                                index=['auto', 'name', 'email', 'phone', 'address', 'id', 'date', 'text', 'numeric'].index(config['column_type']),
                                key=f"type_{col}",
                                help="Hint for what type of data this column contains"
                            )
                        
                        with col2:
                            config['sensitivity'] = st.slider(
                                "Sensitivity Level",
                                min_value=0.1,
                                max_value=1.0,
                                value=config['sensitivity'],
                                step=0.1,
                                key=f"sensitivity_{col}",
                                help="Lower = more permissive, Higher = more strict"
                            )
                            
                            config['custom_redaction_label'] = st.text_input(
                                "Custom Redaction Label",
                                value=config['custom_redaction_label'],
                                key=f"label_{col}",
                                placeholder="e.g., [CUSTOMER_NAME]",
                                help="Leave empty to use default labels"
                            )
                        
                        # Whitelist patterns
                        st.markdown("**Whitelist Patterns** *(Never redact these patterns)*")
                        whitelist_input = st.text_area(
                            "Patterns to protect (one per line)",
                            value='\n'.join(config['whitelist_patterns']),
                            key=f"whitelist_{col}",
                            help="RegEx patterns or exact text that should never be redacted",
                            height=60
                        )
                        config['whitelist_patterns'] = [p.strip() for p in whitelist_input.split('\n') if p.strip()]
                        
                        # Blacklist patterns
                        st.markdown("**Blacklist Patterns** *(Always redact these patterns)*")
                        blacklist_input = st.text_area(
                            "Patterns to always redact (one per line)",
                            value='\n'.join(config['blacklist_patterns']),
                            key=f"blacklist_{col}",
                            help="RegEx patterns or exact text that should always be redacted",
                            height=60
                        )
                        config['blacklist_patterns'] = [p.strip() for p in blacklist_input.split('\n') if p.strip()]
                        
                        # Entity type exclusions
                        st.markdown("**Entity Type Exclusions** *(Skip these PII types entirely)*")
                        
                        # Common entity types that cause false positives
                        common_entity_types = [
                            'PersonType', 'Person', 'Organization', 'Location', 'Event', 
                            'Product', 'Skill', 'DateTime', 'Date', 'Time', 'Age',
                            'Quantity', 'Number', 'Ordinal', 'Percentage'
                        ]
                        
                        # Multi-select for common entity types
                        excluded_entities = st.multiselect(
                            "Select entity types to exclude:",
                            options=common_entity_types,
                            default=config.get('excluded_entity_types', []),
                            key=f"excluded_{col}",
                            help="These PII entity types will be completely ignored for this column"
                        )
                        config['excluded_entity_types'] = excluded_entities
                        
                        # Custom entity types input
                        custom_excluded = st.text_input(
                            "Additional entity types to exclude (comma-separated):",
                            value=','.join([e for e in config.get('excluded_entity_types', []) if e not in common_entity_types]),
                            key=f"custom_excluded_{col}",
                            help="Add other Azure entity types not in the list above"
                        )
                        
                        # Combine common and custom exclusions
                        if custom_excluded:
                            custom_list = [e.strip() for e in custom_excluded.split(',') if e.strip()]
                            config['excluded_entity_types'] = list(set(excluded_entities + custom_list))
                        else:
                            config['excluded_entity_types'] = excluded_entities
                        
                        # Show current exclusions
                        if config['excluded_entity_types']:
                            st.info(f"**Excluded entities:** {', '.join(config['excluded_entity_types'])}")
                        
                        # Preview button for this column
                        if st.button(f"🔍 Preview {col}", key=f"preview_{col}"):
                            if sample_values:
                                st.markdown("**Preview Results:**")
                                
                                # Show preview using simple detection for now
                                for i, val in enumerate(sample_values):
                                    # Use simple detection for preview
                                    redacted, _ = app.detect_and_redact_text(str(val))
                                    
                                    col_orig, col_red = st.columns(2)
                                    with col_orig:
                                        st.text_area(
                                            f"Original {i+1}:",
                                            value=str(val),
                                            height=60,
                                            disabled=True,
                                            key=f"orig_{col}_{i}"
                                        )
                                    with col_red:
                                        st.text_area(
                                            f"Redacted {i+1}:",
                                            value=redacted,
                                            height=60,
                                            disabled=True,
                                            key=f"red_{col}_{i}"
                                        )
                
                # Configuration management
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("💾 Save Configuration"):
                        # Save current configuration
                        config_data = {}
                        for col in selected_columns:
                            col_key = f"config_{col}"
                            if col_key in st.session_state['column_configs']:
                                config_data[col] = st.session_state['column_configs'][col_key]
                        
                        st.session_state['saved_column_configs'] = config_data
                        st.success("Configuration saved!")
                
                with col2:
                    if st.button("📥 Load Saved Config"):
                        if 'saved_column_configs' in st.session_state:
                            st.session_state['column_configs'] = {
                                f"config_{col}": config 
                                for col, config in st.session_state['saved_column_configs'].items()
                            }
                            st.session_state['applied_template'] = "Custom (Saved)"
                            st.success("Saved configuration loaded!")
                            st.rerun()
                        else:
                            st.warning("No saved configuration found!")
                
                with col3:
                    if st.button("🔄 Reset All"):
                        # Reset all configurations
                        st.session_state['column_configs'] = {}
                        st.session_state['applied_template'] = None
                        st.success("All configurations reset!")
                        st.rerun()
            
            elif not COLUMN_CONFIG_AVAILABLE:
                st.warning("⚠️ Column configuration is not available. Please ensure the column_config modules are properly installed.")
            elif not selected_columns:
                st.info("ℹ️ Please select columns in the sidebar to configure PII detection settings.")
            else:
                st.error("❌ Column configuration manager is not initialized.")

        with tab3:
            st.header("🚀 Start PII Redaction")
            
            if selected_columns:
                # Cost estimation
                if app.text_analytics_client and not local_only:
                    total_chars = sum(df[col].astype(str).str.len().sum() for col in selected_columns)
                    estimated_cost = total_chars * 0.001 / 1000
                    st.info(f"💰 Estimated cost: ${estimated_cost:.4f}")
                
                if st.button("🚀 Start Redaction", type="primary"):
                    # Show GPT validation status
                    if use_gpt_validation and azure_api_key:
                        st.info("🤖 GPT Validation: ENABLED - Will validate detections to reduce false positives")
                    else:
                        st.info("ℹ️ GPT Validation: DISABLED - Consider enabling to reduce false positives")
                        
                    with st.spinner("Processing PII redaction..."):
                        # Use local detection if requested
                        if local_only:
                            app.text_analytics_client = None
                        
                        # Get column configurations
                        column_configs = None
                        if 'saved_column_configs' in st.session_state:
                            column_configs = st.session_state['saved_column_configs']
                        elif 'column_configs' in st.session_state:
                            # Use current unsaved configs
                            column_configs = {}
                            for col in selected_columns:
                                col_key = f"config_{col}"
                                if col_key in st.session_state['column_configs']:
                                    column_configs[col] = st.session_state['column_configs'][col_key]
                        
                        # Process data
                        process_df = df.head(5) if preview_only else df
                        processed_df, total_cost, duration, gpt_stats = app.process_dataframe(
                            process_df, selected_columns, column_configs, use_gpt_validation, azure_api_key
                        )
                        
                        # Store results in session state
                        st.session_state['processed_df'] = processed_df
                        st.session_state['original_df'] = process_df
                        st.session_state['total_cost'] = total_cost
                        st.session_state['duration'] = duration
                        st.session_state['gpt_stats'] = gpt_stats
                        st.session_state['filename'] = uploaded_file.name
                        
                        # Save to history
                        app.save_to_history(
                            uploaded_file.name,
                            len(process_df),
                            len(selected_columns),
                            total_cost,
                            duration,
                            "Completed"
                        )
                        
                        st.success("✅ Processing complete!")
                        st.balloons()
            else:
                st.warning("Please select at least one column to process.")
        
        with tab4:
            st.header("📈 Redaction Results")
            
            if 'processed_df' in st.session_state:
                processed_df = st.session_state['processed_df']
                original_df = st.session_state['original_df']
                
                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Rows Processed", len(processed_df))
                with col2:
                    st.metric("Duration", f"{st.session_state['duration']:.1f}s")
                with col3:
                    st.metric("Azure Cost", f"${st.session_state['total_cost']:.4f}")
                with col4:
                    redaction_count = sum(
                        processed_df[col].astype(str).str.contains(r'\[REDACTED_', na=False).sum()
                        for col in selected_columns
                    )
                    st.metric("Items Redacted", redaction_count)
                with col5:
                    # GPT validation metrics
                    gpt_stats = st.session_state.get('gpt_stats', {})
                    if gpt_stats.get('gpt_enabled', False):
                        st.metric("🤖 GPT Cost", f"${gpt_stats.get('gpt_cost', 0):.6f}")
                    else:
                        st.metric("🤖 GPT", "Disabled")
                
                # GPT Validation Status
                gpt_stats = st.session_state.get('gpt_stats', {})
                if gpt_stats.get('gpt_enabled', False):
                    st.success(f"🤖 GPT Validation: Active - Helped reduce false positives (Cost: ${gpt_stats.get('gpt_cost', 0):.6f})")
                else:
                    st.info("🤖 GPT Validation: Not used - Enable in Options to reduce false positives")
                
                # View options
                view_mode = st.radio(
                    "View Mode:",
                    ["Redacted Data", "Original Data", "Side-by-Side Comparison"],
                    horizontal=True
                )
                
                if view_mode == "Redacted Data":
                    st.subheader("🛡️ Redacted Data")
                    st.dataframe(processed_df, use_container_width=True)
                elif view_mode == "Original Data":
                    st.subheader("📄 Original Data")
                    st.dataframe(original_df, use_container_width=True)
                else:  # Side-by-Side
                    st.subheader("🔀 Side-by-Side Comparison")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original**")
                        st.dataframe(original_df, use_container_width=True)
                    with col2:
                        st.write("**Redacted**")
                        st.dataframe(processed_df, use_container_width=True)
                
                # Download button
                csv = processed_df.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{st.session_state['filename'].replace('.csv', '')}_redacted_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download Redacted CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    type="primary"
                )
            else:
                st.info("No processed data yet. Go to the Process tab to redact PII.")
        
        with tab5:
            st.header("🔍 Interactive PII Review")
            
            # Initialize session state for interactive review
            if 'whitelist' not in st.session_state:
                st.session_state.whitelist = set()
            if 'false_positives' not in st.session_state:
                st.session_state.false_positives = []
            if 'auto_apply' not in st.session_state:
                st.session_state.auto_apply = True
            
            # Load whitelist from file if exists
            whitelist_file = 'pii_whitelist.json'
            if os.path.exists(whitelist_file) and not st.session_state.whitelist:
                try:
                    with open(whitelist_file, 'r') as f:
                        data = json.load(f)
                        st.session_state.whitelist = set(data.get('terms', []))
                        st.session_state.false_positives = data.get('false_positives', [])
                except Exception as e:
                    st.warning(f"Could not load whitelist: {e}")
            
            # Check if we have processed data to review
            if 'processed_df' in st.session_state and 'original_df' in st.session_state:
                original_df = st.session_state['original_df']
                processed_df = st.session_state['processed_df']
                
                # Sidebar controls for review
                with st.sidebar:
                    st.header("🔍 Review Controls")
                    
                    # Whitelist management
                    st.metric("Whitelist Terms", len(st.session_state.whitelist))
                    st.metric("False Positives Found", len(st.session_state.false_positives))
                    
                    # Auto-apply toggle
                    st.session_state.auto_apply = st.checkbox(
                        "Auto-apply whitelist globally",
                        value=st.session_state.auto_apply,
                        help="Automatically apply whitelist changes to all data"
                    )
                    
                    # Manual whitelist entry
                    st.subheader("Add to Whitelist")
                    new_term = st.text_input("Add term manually:")
                    if st.button("Add Term") and new_term:
                        st.session_state.whitelist.add(new_term.strip())
                        # Save whitelist
                        whitelist_data = {
                            'terms': list(st.session_state.whitelist),
                            'false_positives': st.session_state.false_positives,
                            'updated': datetime.now().isoformat()
                        }
                        with open(whitelist_file, 'w') as f:
                            json.dump(whitelist_data, f, indent=2)
                        st.success(f"Added '{new_term}' to whitelist")
                        st.rerun()
                    
                    # Export/Import whitelist
                    if st.button("💾 Save Whitelist"):
                        whitelist_data = {
                            'terms': list(st.session_state.whitelist),
                            'false_positives': st.session_state.false_positives,
                            'updated': datetime.now().isoformat()
                        }
                        with open(whitelist_file, 'w') as f:
                            json.dump(whitelist_data, f, indent=2)
                        st.success("Whitelist saved!")
                    
                    if st.button("📤 Export Whitelist"):
                        whitelist_json = json.dumps({
                            'terms': list(st.session_state.whitelist),
                            'false_positives': st.session_state.false_positives
                        }, indent=2)
                        
                        st.download_button(
                            "Download Whitelist JSON",
                            whitelist_json,
                            "pii_whitelist.json",
                            "application/json"
                        )
                    
                    # Show current whitelist
                    with st.expander("View/Edit Whitelist"):
                        if st.session_state.whitelist:
                            for i, term in enumerate(sorted(st.session_state.whitelist)):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.text(term)
                                with col2:
                                    if st.button("❌", key=f"remove_{i}"):
                                        st.session_state.whitelist.remove(term)
                                        # Save updated whitelist
                                        whitelist_data = {
                                            'terms': list(st.session_state.whitelist),
                                            'false_positives': st.session_state.false_positives,
                                            'updated': datetime.now().isoformat()
                                        }
                                        with open(whitelist_file, 'w') as f:
                                            json.dump(whitelist_data, f, indent=2)
                                        st.rerun()
                        else:
                            st.info("No terms in whitelist yet")
                
                # Apply whitelist if auto-apply is enabled
                if st.session_state.auto_apply and st.session_state.whitelist:
                    display_df = app.apply_whitelist_to_dataframe(original_df, processed_df)
                else:
                    display_df = processed_df
                
                # Main review interface
                st.subheader("Interactive Review Interface")
                
                # Navigation controls
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_redactions = 0
                    for col in processed_df.columns:
                        total_redactions += processed_df[col].astype(str).str.contains(r'\[REDACTED_', na=False).sum()
                    st.metric("Total Redactions", total_redactions)
                
                with col2:
                    if st.session_state.whitelist:
                        whitelist_applied = 0
                        for col in processed_df.columns:
                            for idx in processed_df.index:
                                if display_df.loc[idx, col] != processed_df.loc[idx, col]:
                                    whitelist_applied += 1
                        st.metric("Whitelist Applied", whitelist_applied)
                    else:
                        st.metric("Whitelist Applied", 0)
                
                with col3:
                    if st.button("🔄 Refresh All", help="Reapply whitelist to all data"):
                        st.rerun()
                
                with col4:
                    if st.button("🧹 Clear Whitelist", help="Remove all whitelist terms"):
                        if st.session_state.whitelist:
                            st.session_state.whitelist.clear()
                            st.session_state.false_positives.clear()
                            # Save empty whitelist
                            with open(whitelist_file, 'w') as f:
                                json.dump({'terms': [], 'false_positives': []}, f)
                            st.success("Whitelist cleared!")
                            st.rerun()
                
                # Interactive data display
                st.subheader("Review Redacted Data")
                st.info("💡 Click on cells below to reveal original text and mark false positives")
                
                # Show data in interactive format
                for row_idx in range(min(20, len(processed_df))):  # Show first 20 rows
                    with st.expander(f"Row {row_idx + 1}", expanded=False):
                        for col in processed_df.columns:
                            original_val = str(original_df.iloc[row_idx][col])
                            redacted_val = str(processed_df.iloc[row_idx][col])
                            current_val = str(display_df.iloc[row_idx][col])
                            
                            if original_val != redacted_val:  # This cell has redactions
                                st.markdown(f"**{col}:**")
                                
                                col1, col2, col3 = st.columns([2, 2, 1])
                                
                                with col1:
                                    st.markdown("*Current:*")
                                    st.code(current_val)
                                
                                with col2:
                                    if st.button(f"👁️ Reveal Original", key=f"reveal_{row_idx}_{col}"):
                                        st.markdown("*Original:*")
                                        st.success(original_val)
                                        
                                        # Find redacted terms and offer actions
                                        redacted_terms = app.extract_redacted_terms(original_val, redacted_val)
                                        for term_data in redacted_terms:
                                            term = term_data['original']
                                            if st.button(f"❌ Mark '{term}' as False Positive", key=f"fp_{row_idx}_{col}_{term}"):
                                                st.session_state.whitelist.add(term)
                                                st.session_state.false_positives.append({
                                                    'term': term,
                                                    'type': term_data.get('type', 'unknown'),
                                                    'context': original_val[:100]
                                                })
                                                # Save whitelist
                                                whitelist_data = {
                                                    'terms': list(st.session_state.whitelist),
                                                    'false_positives': st.session_state.false_positives,
                                                    'updated': datetime.now().isoformat()
                                                }
                                                with open(whitelist_file, 'w') as f:
                                                    json.dump(whitelist_data, f, indent=2)
                                                st.success(f"Added '{term}' to whitelist!")
                                                if st.session_state.auto_apply:
                                                    st.rerun()
                                
                                with col3:
                                    if current_val != redacted_val:
                                        st.success("✅ Restored")
                                    else:
                                        st.warning("🔒 Redacted")
                            else:
                                st.markdown(f"**{col}:** {original_val}")
                
                # Download corrected data
                st.subheader("Export Corrected Data")
                corrected_csv = display_df.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                corrected_filename = f"{st.session_state['filename'].replace('.csv', '')}_corrected_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download Corrected CSV",
                    data=corrected_csv,
                    file_name=corrected_filename,
                    mime="text/csv",
                    type="primary"
                )
                
                # Show summary of changes
                changes_made = sum(
                    1 for col in processed_df.columns 
                    for idx in processed_df.index 
                    if str(display_df.loc[idx, col]) != str(processed_df.loc[idx, col])
                )
                
                if changes_made > 0:
                    st.success(f"✅ Applied whitelist to {changes_made} cells")
                else:
                    st.info("ℹ️ No whitelist changes applied yet")
                    
            else:
                st.info("📄 No processed data available for review. Please go to the Process tab first to redact PII from your data.")
                
                # Option to upload already processed files
                st.subheader("Or Upload Previously Processed Files")
                
                col1, col2 = st.columns(2)
                with col1:
                    uploaded_original = st.file_uploader(
                        "Upload Original CSV",
                        type=['csv'],
                        key='review_original_upload'
                    )
                
                with col2:
                    uploaded_processed = st.file_uploader(
                        "Upload Processed/Redacted CSV", 
                        type=['csv'],
                        key='review_processed_upload'
                    )
                
                if uploaded_original and uploaded_processed:
                    if st.button("📥 Load Files for Review", type="primary"):
                        try:
                            st.session_state['original_df'] = pd.read_csv(uploaded_original)
                            st.session_state['processed_df'] = pd.read_csv(uploaded_processed)
                            st.session_state['filename'] = uploaded_original.name
                            st.success("Files loaded successfully! You can now review the redactions.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading files: {e}")
        
        with tab6:
            st.header("📜 Processing History")
            
            history_df = app.get_history()
            if not history_df.empty:
                # Summary charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Cost over time
                    fig_cost = px.line(
                        history_df, 
                        x='timestamp', 
                        y='cost',
                        title="Cost Over Time",
                        labels={'cost': 'Cost ($)', 'timestamp': 'Date'}
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
                
                with col2:
                    # Rows processed
                    fig_rows = px.bar(
                        history_df.head(10), 
                        x='filename', 
                        y='rows_processed',
                        title="Rows Processed by File",
                        labels={'rows_processed': 'Rows', 'filename': 'File'}
                    )
                    fig_rows.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_rows, use_container_width=True)
                
                # History table
                st.subheader("Recent Processing History")
                history_display = history_df.copy()
                history_display['timestamp'] = pd.to_datetime(history_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                history_display['cost'] = history_display['cost'].apply(lambda x: f"${x:.4f}")
                history_display['duration_seconds'] = history_display['duration_seconds'].apply(lambda x: f"{x:.1f}s")
                
                st.dataframe(
                    history_display[['timestamp', 'filename', 'rows_processed', 'columns_processed', 'cost', 'duration_seconds', 'status']],
                    use_container_width=True
                )
                
                # Total metrics
                st.subheader("📊 Total Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files Processed", len(history_df))
                with col2:
                    st.metric("Total Rows Processed", history_df['rows_processed'].sum())
                with col3:
                    st.metric("Total Cost", f"${history_df['cost'].sum():.4f}")
            else:
                st.info("No processing history yet.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to PII Redactor Pro! 🛡️
        
        **Protect sensitive data with AI-powered PII detection and redaction.**
        
        ### 🚀 Features:
        - **Azure AI Integration**: Industry-leading PII detection with 100+ entity types
        - **Interactive Review**: Click-to-reveal redacted text, mark false positives, batch whitelist actions
        - **Smart Whitelist**: Persistent whitelist that applies corrections across all tickets instantly
        - **Local Fallback**: Works offline with regex-based detection  
        - **Real-time Processing**: Live progress tracking with ETA
        - **Cost Monitoring**: Track Azure AI usage costs
        - **Export Ready**: Download redacted and corrected CSV files instantly
        - **Processing History**: Track all your redaction activities
        
        ### 📋 Getting Started:
        1. **Upload a CSV file** using the sidebar
        2. **Select columns** containing PII data
        3. **Choose your options** (preview mode, local detection)
        4. **Start redaction** and monitor progress
        5. **Use Interactive Review** to fix false positives with click-to-reveal
        6. **Download corrected results** when complete
        
        ### 💡 Tips:
        - Use **preview mode** to test redaction on first 5 rows
        - Enable **local detection** if Azure is unavailable
        - Check the **Interactive Review tab** to fix false positives - click any "[REDACTED_TYPE]" to reveal and whitelist
        - Whitelist terms apply globally across ALL data automatically
        - Check the **History tab** to monitor costs and usage
        
        **Ready to protect your data? Upload a CSV file to begin!** ⬆️
        """)

if __name__ == "__main__":
    main()
