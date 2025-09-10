#!/usr/bin/env python3
"""
Interactive PII Review Web Application
Click on redacted text to reveal original and mark false positives
Automatically applies learnings across all tickets
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Set
import re
import hashlib

# Page config
st.set_page_config(
    page_title="Interactive PII Reviewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InteractivePIIReviewer:
    """Interactive web interface for reviewing PII redactions"""
    
    def __init__(self):
        self.init_session_state()
        self.load_whitelist()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'original_data' not in st.session_state:
            st.session_state.original_data = None
        if 'current_row' not in st.session_state:
            st.session_state.current_row = 0
        if 'current_col' not in st.session_state:
            st.session_state.current_col = 0
        if 'whitelist' not in st.session_state:
            st.session_state.whitelist = set()
        if 'reviewed_items' not in st.session_state:
            st.session_state.reviewed_items = {}
        if 'false_positives' not in st.session_state:
            st.session_state.false_positives = []
        if 'review_mode' not in st.session_state:
            st.session_state.review_mode = 'cell'  # 'cell' or 'all'
        if 'auto_apply' not in st.session_state:
            st.session_state.auto_apply = True
            
    def load_whitelist(self):
        """Load saved whitelist from file"""
        whitelist_file = 'pii_whitelist.json'
        if os.path.exists(whitelist_file):
            with open(whitelist_file, 'r') as f:
                data = json.load(f)
                st.session_state.whitelist = set(data.get('terms', []))
                st.session_state.false_positives = data.get('false_positives', [])
    
    def save_whitelist(self):
        """Save whitelist to file"""
        whitelist_file = 'pii_whitelist.json'
        data = {
            'terms': list(st.session_state.whitelist),
            'false_positives': st.session_state.false_positives,
            'updated': datetime.now().isoformat()
        }
        with open(whitelist_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def extract_redacted_terms(self, original: str, redacted: str) -> List[Dict]:
        """Extract what was redacted by comparing original and redacted text"""
        terms = []
        
        # Find all redaction patterns
        pattern = r'\[REDACTED_([A-Z_]+)\]'
        
        # Split both strings into words for comparison
        orig_words = original.split()
        red_words = redacted.split()
        
        # Track redacted terms
        i, j = 0, 0
        while i < len(orig_words) and j < len(red_words):
            if '[REDACTED' in red_words[j]:
                # Found a redaction
                redaction_type = re.search(pattern, red_words[j])
                if redaction_type:
                    terms.append({
                        'original': orig_words[i],
                        'type': redaction_type.group(1),
                        'position': i,
                        'full_redaction': red_words[j]
                    })
                i += 1
                j += 1
            elif orig_words[i] == red_words[j]:
                # Words match, move both
                i += 1
                j += 1
            else:
                # Mismatch, try to sync
                i += 1
        
        return terms
    
    def apply_whitelist_to_text(self, original: str, redacted: str) -> str:
        """Apply whitelist to restore false positives"""
        result = redacted
        
        # Get redacted terms
        terms = self.extract_redacted_terms(original, redacted)
        
        # Check each term against whitelist
        for term in terms:
            if term['original'].lower() in [w.lower() for w in st.session_state.whitelist]:
                # Replace the redaction with original term
                result = result.replace(term['full_redaction'], term['original'], 1)
        
        return result
    
    def apply_whitelist_to_dataframe(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Apply whitelist across entire dataframe"""
        corrected_df = processed_df.copy()
        
        for col in processed_df.columns:
            for idx in processed_df.index:
                original_val = str(original_df.loc[idx, col])
                redacted_val = str(processed_df.loc[idx, col])
                
                if original_val != redacted_val:
                    # Apply whitelist
                    corrected_val = self.apply_whitelist_to_text(original_val, redacted_val)
                    corrected_df.loc[idx, col] = corrected_val
        
        return corrected_df
    
    def render_clickable_text(self, original: str, redacted: str, cell_id: str):
        """Render text with clickable redacted sections"""
        if original == redacted:
            # No redactions
            st.text(original)
            return
        
        # Extract redacted terms
        terms = self.extract_redacted_terms(original, redacted)
        
        if not terms:
            st.text(redacted)
            return
        
        # Create clickable interface
        st.markdown("**Click on redacted terms to review:**")
        
        # Display the text with buttons for redacted parts
        cols = st.columns(len(terms) + 1) if terms else [st.container()]
        
        # Build the display text
        display_parts = []
        words = redacted.split()
        
        col_idx = 0
        for i, word in enumerate(words):
            if '[REDACTED' in word:
                # This is a redacted term
                for term in terms:
                    if term['full_redaction'] == word:
                        with cols[col_idx % len(cols)]:
                            if st.button(
                                word,
                                key=f"reveal_{cell_id}_{i}",
                                help=f"Click to reveal: {term['original']}"
                            ):
                                st.success(f"Original: **{term['original']}**")
                                
                                # Options for this term
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if st.button("✅ Correct Redaction", key=f"correct_{cell_id}_{i}"):
                                        st.info("Marked as correctly redacted")
                                
                                with col2:
                                    if st.button("❌ False Positive", key=f"fp_{cell_id}_{i}"):
                                        # Add to whitelist
                                        st.session_state.whitelist.add(term['original'])
                                        st.session_state.false_positives.append({
                                            'term': term['original'],
                                            'type': term['type'],
                                            'context': original[:50]
                                        })
                                        self.save_whitelist()
                                        st.success(f"Added '{term['original']}' to whitelist")
                                        
                                        if st.session_state.auto_apply:
                                            st.rerun()
                                
                                with col3:
                                    if st.button("🔍 Show Context", key=f"context_{cell_id}_{i}"):
                                        st.info(f"Full text: {original}")
                        
                        col_idx += 1
                        break
        
        # Show the full text below
        st.markdown("**Current text:**")
        st.code(redacted)
        
        # If whitelist items would change this text, show preview
        corrected = self.apply_whitelist_to_text(original, redacted)
        if corrected != redacted:
            st.markdown("**After applying whitelist:**")
            st.success(corrected)
    
    def render_main_interface(self):
        """Render the main review interface"""
        st.title("🔍 Interactive PII Review System")
        
        # Sidebar controls
        with st.sidebar:
            st.header("📊 Data Management")
            
            # File upload
            uploaded_original = st.file_uploader(
                "Upload Original CSV",
                type=['csv'],
                key='original_upload'
            )
            
            uploaded_processed = st.file_uploader(
                "Upload Processed/Redacted CSV",
                type=['csv'],
                key='processed_upload'
            )
            
            if uploaded_original and uploaded_processed:
                if st.button("📥 Load Files", type="primary"):
                    st.session_state.original_data = pd.read_csv(uploaded_original)
                    st.session_state.processed_data = pd.read_csv(uploaded_processed)
                    st.success("Files loaded successfully!")
            
            st.divider()
            
            # Whitelist management
            st.header("📝 Whitelist Management")
            
            st.metric("Whitelist Terms", len(st.session_state.whitelist))
            st.metric("False Positives Found", len(st.session_state.false_positives))
            
            # Auto-apply toggle
            st.session_state.auto_apply = st.checkbox(
                "Auto-apply whitelist",
                value=st.session_state.auto_apply,
                help="Automatically apply whitelist changes to all data"
            )
            
            # Export/Import whitelist
            if st.button("💾 Save Whitelist"):
                self.save_whitelist()
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
            with st.expander("View Whitelist"):
                if st.session_state.whitelist:
                    for term in sorted(st.session_state.whitelist):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(term)
                        with col2:
                            if st.button("❌", key=f"remove_{term}"):
                                st.session_state.whitelist.remove(term)
                                st.rerun()
        
        # Main content area
        if st.session_state.original_data is not None and st.session_state.processed_data is not None:
            
            # Apply whitelist if auto-apply is on
            if st.session_state.auto_apply:
                display_df = self.apply_whitelist_to_dataframe(
                    st.session_state.original_data,
                    st.session_state.processed_data
                )
            else:
                display_df = st.session_state.processed_data
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Cell-by-Cell Review",
                "📊 Full Comparison",
                "📈 Statistics",
                "✅ Final Results"
            ])
            
            with tab1:
                self.render_cell_review(display_df)
            
            with tab2:
                self.render_full_comparison(display_df)
            
            with tab3:
                self.render_statistics()
            
            with tab4:
                self.render_final_results(display_df)
        
        else:
            # Welcome screen
            st.markdown("""
            ## Welcome to the Interactive PII Reviewer! 
            
            ### How it works:
            
            1. **Upload your files** - Original and processed/redacted CSVs
            2. **Click on redacted text** - See what was redacted
            3. **Mark false positives** - Build your whitelist
            4. **Auto-apply across all data** - Your choices apply everywhere
            5. **Export clean data** - Download corrected CSV
            
            ### Features:
            
            ✅ **Click to reveal** - Click any [REDACTED] text to see original
            ✅ **One-click whitelist** - Mark false positives instantly  
            ✅ **Auto-apply** - Changes apply across ALL tickets immediately
            ✅ **Smart learning** - System remembers your choices
            ✅ **Export ready** - Download corrected data anytime
            
            **Get started by uploading your files in the sidebar! →**
            """)
    
    def render_cell_review(self, display_df: pd.DataFrame):
        """Render cell-by-cell review interface"""
        st.header("🎯 Cell-by-Cell Review")
        
        original_df = st.session_state.original_data
        
        # Navigation controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Previous"):
                if st.session_state.current_col > 0:
                    st.session_state.current_col -= 1
                elif st.session_state.current_row > 0:
                    st.session_state.current_row -= 1
                    st.session_state.current_col = len(display_df.columns) - 1
                st.rerun()
        
        with col2:
            # Jump to specific cell
            row_col1, row_col2 = st.columns(2)
            with row_col1:
                st.session_state.current_row = st.number_input(
                    "Row",
                    min_value=0,
                    max_value=len(display_df) - 1,
                    value=st.session_state.current_row
                )
            with row_col2:
                col_names = display_df.columns.tolist()
                selected_col = st.selectbox(
                    "Column",
                    col_names,
                    index=st.session_state.current_col
                )
                st.session_state.current_col = col_names.index(selected_col)
        
        with col3:
            if st.button("Next ➡️"):
                if st.session_state.current_col < len(display_df.columns) - 1:
                    st.session_state.current_col += 1
                elif st.session_state.current_row < len(display_df) - 1:
                    st.session_state.current_row += 1
                    st.session_state.current_col = 0
                st.rerun()
        
        # Progress bar
        total_cells = len(display_df) * len(display_df.columns)
        current_cell = st.session_state.current_row * len(display_df.columns) + st.session_state.current_col + 1
        st.progress(current_cell / total_cells)
        st.caption(f"Cell {current_cell} of {total_cells}")
        
        # Display current cell
        st.divider()
        
        row = st.session_state.current_row
        col = display_df.columns[st.session_state.current_col]
        
        original_val = str(original_df.loc[row, col])
        redacted_val = str(display_df.loc[row, col])
        
        # Cell info
        st.markdown(f"### 📍 Row {row}, Column: {col}")
        
        # Show if there are redactions
        if original_val != redacted_val:
            st.warning("⚠️ This cell contains redactions")
            
            # Clickable interface
            cell_id = f"{row}_{col}"
            self.render_clickable_text(original_val, redacted_val, cell_id)
            
            # Quick actions
            st.divider()
            st.markdown("### Quick Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ All Correct", key=f"all_correct_{cell_id}"):
                    st.success("Marked all redactions as correct")
            
            with col2:
                if st.button("❌ All False Positives", key=f"all_fp_{cell_id}"):
                    # Add all terms to whitelist
                    terms = self.extract_redacted_terms(original_val, redacted_val)
                    for term in terms:
                        st.session_state.whitelist.add(term['original'])
                    self.save_whitelist()
                    st.success(f"Added {len(terms)} terms to whitelist")
                    if st.session_state.auto_apply:
                        st.rerun()
            
            with col3:
                if st.button("⏭️ Skip", key=f"skip_{cell_id}"):
                    # Move to next cell with redactions
                    self.move_to_next_redaction()
                    st.rerun()
        
        else:
            st.success("✅ No redactions in this cell")
            st.text(original_val)
            
            if st.button("⏭️ Next Redaction"):
                self.move_to_next_redaction()
                st.rerun()
    
    def move_to_next_redaction(self):
        """Move to the next cell containing redactions"""
        original_df = st.session_state.original_data
        processed_df = st.session_state.processed_data
        
        # Start from current position
        start_row = st.session_state.current_row
        start_col = st.session_state.current_col
        
        # Search for next redaction
        for row in range(start_row, len(processed_df)):
            for col_idx in range(len(processed_df.columns)):
                # Skip cells before current position in first row
                if row == start_row and col_idx <= start_col:
                    continue
                
                col = processed_df.columns[col_idx]
                if str(original_df.loc[row, col]) != str(processed_df.loc[row, col]):
                    st.session_state.current_row = row
                    st.session_state.current_col = col_idx
                    return
        
        # No more redactions found
        st.info("No more redactions found!")
    
    def render_full_comparison(self, display_df: pd.DataFrame):
        """Render full dataset comparison"""
        st.header("📊 Full Dataset Comparison")
        
        original_df = st.session_state.original_data
        
        # Filter to show only cells with changes
        show_only_changes = st.checkbox("Show only cells with redactions", value=True)
        
        if show_only_changes:
            # Find cells with changes
            changes = []
            for col in display_df.columns:
                for idx in display_df.index:
                    if str(original_df.loc[idx, col]) != str(display_df.loc[idx, col]):
                        changes.append({
                            'Row': idx,
                            'Column': col,
                            'Original': str(original_df.loc[idx, col])[:100],
                            'Redacted': str(display_df.loc[idx, col])[:100],
                            'Action': ''
                        })
            
            if changes:
                changes_df = pd.DataFrame(changes)
                
                # Editable dataframe for batch actions
                st.markdown("### Cells with Redactions")
                st.dataframe(changes_df, use_container_width=True)
                
                # Batch actions
                st.markdown("### Batch Actions")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Add all originals to whitelist"):
                        for change in changes:
                            terms = self.extract_redacted_terms(
                                change['Original'],
                                change['Redacted']
                            )
                            for term in terms:
                                st.session_state.whitelist.add(term['original'])
                        self.save_whitelist()
                        st.success(f"Added terms from {len(changes)} cells to whitelist")
                        if st.session_state.auto_apply:
                            st.rerun()
                
                with col2:
                    if st.button("Export changes report"):
                        csv = changes_df.to_csv(index=False)
                        st.download_button(
                            "Download Changes CSV",
                            csv,
                            "redaction_changes.csv",
                            "text/csv"
                        )
            else:
                st.success("No changes found between original and processed data!")
        
        else:
            # Show full comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Data")
                st.dataframe(original_df, use_container_width=True)
            
            with col2:
                st.markdown("### Processed Data (with whitelist applied)")
                st.dataframe(display_df, use_container_width=True)
    
    def render_statistics(self):
        """Render statistics about redactions"""
        st.header("📈 Redaction Statistics")
        
        original_df = st.session_state.original_data
        processed_df = st.session_state.processed_data
        
        # Calculate statistics
        total_cells = len(original_df) * len(original_df.columns)
        cells_with_redactions = 0
        redaction_types = {}
        
        for col in processed_df.columns:
            for idx in processed_df.index:
                original_val = str(original_df.loc[idx, col])
                redacted_val = str(processed_df.loc[idx, col])
                
                if original_val != redacted_val:
                    cells_with_redactions += 1
                    
                    # Count redaction types
                    pattern = r'\[REDACTED_([A-Z_]+)\]'
                    matches = re.findall(pattern, redacted_val)
                    for match in matches:
                        redaction_types[match] = redaction_types.get(match, 0) + 1
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cells", total_cells)
        
        with col2:
            st.metric("Cells with Redactions", cells_with_redactions)
        
        with col3:
            redaction_rate = (cells_with_redactions / total_cells * 100) if total_cells > 0 else 0
            st.metric("Redaction Rate", f"{redaction_rate:.1f}%")
        
        with col4:
            st.metric("Whitelist Terms", len(st.session_state.whitelist))
        
        # Redaction types breakdown
        if redaction_types:
            st.markdown("### Redaction Types")
            
            types_df = pd.DataFrame(
                [(k, v) for k, v in redaction_types.items()],
                columns=['Type', 'Count']
            ).sort_values('Count', ascending=False)
            
            st.bar_chart(types_df.set_index('Type'))
            
            # Table view
            st.dataframe(types_df, use_container_width=True)
        
        # False positives analysis
        if st.session_state.false_positives:
            st.markdown("### False Positives Found")
            
            fp_df = pd.DataFrame(st.session_state.false_positives)
            st.dataframe(fp_df, use_container_width=True)
            
            # Most common false positive types
            if 'type' in fp_df.columns:
                fp_types = fp_df['type'].value_counts()
                st.markdown("### False Positives by Type")
                st.bar_chart(fp_types)
    
    def render_final_results(self, display_df: pd.DataFrame):
        """Render final results and export options"""
        st.header("✅ Final Results")
        
        st.markdown("""
        ### Your Corrected Data
        
        This is your data with all whitelist terms applied. 
        False positives have been restored to their original values.
        """)
        
        # Show the corrected dataframe
        st.dataframe(display_df, use_container_width=True)
        
        # Export options
        st.markdown("### Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export corrected CSV
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Corrected CSV",
                csv,
                f"corrected_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                type="primary"
            )
        
        with col2:
            # Export whitelist
            whitelist_json = json.dumps({
                'terms': list(st.session_state.whitelist),
                'false_positives': st.session_state.false_positives,
                'created': datetime.now().isoformat()
            }, indent=2)
            
            st.download_button(
                "📋 Download Whitelist",
                whitelist_json,
                f"whitelist_{datetime.now().strftime('%Y%m%d')}.json",
                "application/json"
            )
        
        with col3:
            # Export review report
            report = self.generate_review_report(display_df)
            st.download_button(
                "📊 Download Report",
                report,
                f"review_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )
        
        # Summary
        st.markdown("### Summary")
        
        original_df = st.session_state.original_data
        
        # Count corrections
        corrections_made = 0
        for col in display_df.columns:
            for idx in display_df.index:
                proc_val = str(st.session_state.processed_data.loc[idx, col])
                corrected_val = str(display_df.loc[idx, col])
                if proc_val != corrected_val:
                    corrections_made += 1
        
        st.success(f"""
        ✅ **Review Complete!**
        
        - **Whitelist terms:** {len(st.session_state.whitelist)}
        - **Corrections made:** {corrections_made}
        - **False positives identified:** {len(st.session_state.false_positives)}
        
        Your whitelist has been saved and will be automatically applied to future data processing.
        """)
    
    def generate_review_report(self, display_df: pd.DataFrame) -> str:
        """Generate a text report of the review"""
        report = []
        report.append("PII REDACTION REVIEW REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("WHITELIST TERMS:")
        report.append("-" * 30)
        for term in sorted(st.session_state.whitelist):
            report.append(f"  - {term}")
        report.append("")
        
        report.append("FALSE POSITIVES IDENTIFIED:")
        report.append("-" * 30)
        for fp in st.session_state.false_positives[:20]:  # First 20
            report.append(f"  Term: {fp.get('term', 'N/A')}")
            report.append(f"  Type: {fp.get('type', 'N/A')}")
            report.append(f"  Context: {fp.get('context', 'N/A')}")
            report.append("")
        
        if len(st.session_state.false_positives) > 20:
            report.append(f"  ... and {len(st.session_state.false_positives) - 20} more")
        
        return "\n".join(report)
    
    def run(self):
        """Run the application"""
        self.render_main_interface()


def main():
    """Main entry point"""
    app = InteractivePIIReviewer()
    app.run()


if __name__ == "__main__":
    main()
