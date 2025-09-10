"""
Column Configuration GUI
Advanced interface for customizing PII detection rules per column
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from typing import Dict, List, Optional, Any
import re

# Import our configuration classes
try:
    from column_config import (
        ColumnConfigManager, ColumnConfig, DetectionMode, DataType,
        WhitelistPattern, BlacklistPattern, EntityRule
    )
except ImportError:
    print("Warning: column_config module not found. Some features may be limited.")


class ColumnConfigGUI:
    """Advanced GUI for column-level PII configuration"""
    
    def __init__(self, parent_window, config_manager: ColumnConfigManager = None):
        self.parent = parent_window
        self.config_manager = config_manager or ColumnConfigManager()
        self.current_column = None
        self.current_df = None
        
        # Create the main configuration window
        self.window = ctk.CTkToplevel(parent_window)
        self.window.title("Column PII Configuration - Advanced Settings")
        self.window.geometry("1200x800")
        self.window.transient(parent_window)
        self.window.grab_set()
        
        # Configure grid weights
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        
        self.create_gui()
        
    def create_gui(self):
        """Create the main GUI layout"""
        # Main container with padding
        main_frame = ctk.CTkFrame(self.window)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🛠️ Column PII Configuration",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=15)
        
        # Left sidebar - Column selection and templates
        sidebar = ctk.CTkFrame(main_frame, width=300)
        sidebar.grid(row=1, column=0, sticky="ns", padx=(0, 10))
        sidebar.grid_propagate(False)
        
        self.create_sidebar(sidebar)
        
        # Right side - Configuration details
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.grid(row=1, column=1, sticky="nsew")
        config_frame.grid_columnconfigure(0, weight=1)
        config_frame.grid_rowconfigure(0, weight=1)
        
        self.create_config_area(config_frame)
        
        # Bottom buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        self.create_bottom_buttons(button_frame)
        
    def create_sidebar(self, parent):
        """Create the left sidebar with column selection"""
        # Column selection section
        col_section = ctk.CTkFrame(parent)
        col_section.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            col_section,
            text="📊 Select Column",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        # Column dropdown
        self.column_var = ctk.StringVar()
        self.column_dropdown = ctk.CTkComboBox(
            col_section,
            variable=self.column_var,
            values=["No columns loaded"],
            command=self.on_column_selected,
            width=250
        )
        self.column_dropdown.pack(padx=10, pady=5)
        
        # Load CSV button
        load_btn = ctk.CTkButton(
            col_section,
            text="📁 Load CSV for Analysis",
            command=self.load_csv_for_analysis,
            height=35
        )
        load_btn.pack(padx=10, pady=5, fill="x")
        
        # Quick actions
        actions_frame = ctk.CTkFrame(col_section)
        actions_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            actions_frame,
            text="Quick Actions",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=(5, 2))
        
        quick_buttons = [
            ("🔍 Analyze Column", self.analyze_current_column),
            ("📋 Copy Config", self.copy_config),
            ("📌 Paste Config", self.paste_config),
            ("🔄 Reset to Default", self.reset_to_default)
        ]
        
        for text, command in quick_buttons:
            btn = ctk.CTkButton(
                actions_frame,
                text=text,
                command=command,
                height=30,
                width=200
            )
            btn.pack(padx=5, pady=2, fill="x")
        
        # Templates section
        template_section = ctk.CTkFrame(parent)
        template_section.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            template_section,
            text="📋 Apply Template",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        # Template buttons
        templates = [
            ("📧 Email Column", "email"),
            ("📞 Phone Column", "phone"),
            ("🆔 ID/Reference", "id_reference"),
            ("👤 Name Column", "name"),
            ("💰 Financial Data", "financial"),
            ("🏷️ Product/Category", "product_category"),
            ("💬 Comments/Text", "comments")
        ]
        
        for text, template_name in templates:
            btn = ctk.CTkButton(
                template_section,
                text=text,
                command=lambda t=template_name: self.apply_template(t),
                height=30,
                width=250
            )
            btn.pack(padx=10, pady=2, fill="x")
        
        # Configuration management
        config_mgmt_section = ctk.CTkFrame(parent)
        config_mgmt_section.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            config_mgmt_section,
            text="⚙️ Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        mgmt_buttons = [
            ("💾 Save Config", self.save_configuration),
            ("📂 Load Config", self.load_configuration),
            ("📤 Export Config", self.export_configuration),
            ("🗑️ Clear All", self.clear_all_configs)
        ]
        
        for text, command in mgmt_buttons:
            btn = ctk.CTkButton(
                config_mgmt_section,
                text=text,
                command=command,
                height=30,
                width=250
            )
            btn.pack(padx=10, pady=2, fill="x")
    
    def create_config_area(self, parent):
        """Create the main configuration area"""
        # Create tabview for different configuration sections
        self.config_tabview = ctk.CTkTabview(parent)
        self.config_tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Basic settings tab
        self.basic_tab = self.config_tabview.add("🎯 Basic Settings")
        self.create_basic_settings_tab()
        
        # Entity rules tab
        self.entity_tab = self.config_tabview.add("🏷️ Entity Rules")
        self.create_entity_rules_tab()
        
        # Patterns tab
        self.patterns_tab = self.config_tabview.add("🔍 Custom Patterns")
        self.create_patterns_tab()
        
        # Advanced tab
        self.advanced_tab = self.config_tabview.add("⚡ Advanced")
        self.create_advanced_tab()
        
        # Preview tab
        self.preview_tab = self.config_tabview.add("👁️ Preview")
        self.create_preview_tab()
        
    def create_basic_settings_tab(self):
        """Create the basic settings configuration"""
        # Scrollable frame
        scrollable = ctk.CTkScrollableFrame(self.basic_tab)
        scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Column info section
        info_frame = ctk.CTkFrame(scrollable)
        info_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            info_frame,
            text="Column Information",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Column name (read-only)
        name_frame = ctk.CTkFrame(info_frame)
        name_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(name_frame, text="Column Name:", width=120).pack(side="left", padx=5)
        self.column_name_label = ctk.CTkLabel(
            name_frame,
            text="No column selected",
            font=ctk.CTkFont(weight="bold")
        )
        self.column_name_label.pack(side="left", padx=5)
        
        # Detection mode
        mode_frame = ctk.CTkFrame(info_frame)
        mode_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(mode_frame, text="Detection Mode:", width=120).pack(side="left", padx=5)
        self.detection_mode_var = ctk.StringVar(value="balanced")
        self.detection_mode_menu = ctk.CTkOptionMenu(
            mode_frame,
            variable=self.detection_mode_var,
            values=["disabled", "conservative", "balanced", "aggressive", "custom"],
            command=self.on_detection_mode_changed,
            width=200
        )
        self.detection_mode_menu.pack(side="left", padx=5)
        
        # Help text for detection modes
        mode_help = ctk.CTkTextbox(info_frame, height=80, width=400)
        mode_help.pack(padx=15, pady=5)
        mode_help.insert("0.0", 
            "🔴 Disabled: Skip PII detection entirely\n"
            "🟡 Conservative: Only high-confidence PII (90%+)\n"
            "🟢 Balanced: Standard detection (70%+)\n"
            "🟠 Aggressive: Detect everything possible (50%+)\n"
            "⚙️ Custom: Use only your custom rules"
        )
        mode_help.configure(state="disabled")
        
        # Expected data type
        type_frame = ctk.CTkFrame(info_frame)
        type_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(type_frame, text="Expected Data Type:", width=120).pack(side="left", padx=5)
        self.data_type_var = ctk.StringVar(value="text")
        self.data_type_menu = ctk.CTkOptionMenu(
            type_frame,
            variable=self.data_type_var,
            values=[
                "text", "email", "phone", "name", "address", "id_number",
                "financial", "date", "numeric", "categorical", "product_code",
                "reference", "description", "comments", "url", "medical", "legal"
            ],
            width=200
        )
        self.data_type_menu.pack(side="left", padx=5)
        
        # Confidence threshold
        conf_frame = ctk.CTkFrame(info_frame)
        conf_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(conf_frame, text="Min Confidence:", width=120).pack(side="left", padx=5)
        self.confidence_var = ctk.DoubleVar(value=0.7)
        self.confidence_slider = ctk.CTkSlider(
            conf_frame,
            variable=self.confidence_var,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            width=200
        )
        self.confidence_slider.pack(side="left", padx=5)
        
        self.confidence_label = ctk.CTkLabel(conf_frame, text="0.70")
        self.confidence_label.pack(side="left", padx=5)
        
        # Update label when slider changes
        self.confidence_slider.configure(command=self.update_confidence_label)
        
        # Business context
        context_frame = ctk.CTkFrame(scrollable)
        context_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            context_frame,
            text="Business Context",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        ctk.CTkLabel(context_frame, text="Describe what this column contains:").pack(anchor="w", padx=15)
        self.business_context_text = ctk.CTkTextbox(context_frame, height=60)
        self.business_context_text.pack(fill="x", padx=15, pady=5)
        
        # Domain keywords
        keywords_frame = ctk.CTkFrame(context_frame)
        keywords_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(keywords_frame, text="Domain Keywords (comma-separated):").pack(anchor="w", padx=5)
        self.domain_keywords_entry = ctk.CTkEntry(keywords_frame, placeholder_text="e.g., customer, support, ticket")
        self.domain_keywords_entry.pack(fill="x", padx=5, pady=5)
        
        # Additional options
        options_frame = ctk.CTkFrame(scrollable)
        options_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            options_frame,
            text="Additional Options",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Checkboxes for various options
        self.require_review_var = ctk.BooleanVar()
        self.preserve_formatting_var = ctk.BooleanVar()
        self.partial_redaction_var = ctk.BooleanVar()
        
        ctk.CTkCheckBox(
            options_frame,
            text="Require human review for this column",
            variable=self.require_review_var
        ).pack(anchor="w", padx=15, pady=2)
        
        ctk.CTkCheckBox(
            options_frame,
            text="Preserve original formatting",
            variable=self.preserve_formatting_var
        ).pack(anchor="w", padx=15, pady=2)
        
        ctk.CTkCheckBox(
            options_frame,
            text="Allow partial redaction (e.g., J*** Smith)",
            variable=self.partial_redaction_var
        ).pack(anchor="w", padx=15, pady=2)
        
    def create_entity_rules_tab(self):
        """Create the entity-specific rules configuration"""
        # Scrollable frame
        scrollable = ctk.CTkScrollableFrame(self.entity_tab)
        scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ctk.CTkFrame(scrollable)
        header_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            header_frame,
            text="Entity-Specific Detection Rules",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=15)
        
        help_text = ctk.CTkTextbox(header_frame, height=60)
        help_text.pack(fill="x", padx=15, pady=5)
        help_text.insert("0.0", 
            "Configure how each type of PII entity should be handled. You can enable/disable detection, "
            "set confidence thresholds, and customize replacement text for each entity type."
        )
        help_text.configure(state="disabled")
        
        # Entity rules container
        self.entity_rules_frame = ctk.CTkFrame(scrollable)
        self.entity_rules_frame.pack(fill="x", pady=10)
        
        # Will be populated with entity rules
        self.entity_rule_widgets = {}
        self.create_entity_rule_widgets()
        
    def create_entity_rule_widgets(self):
        """Create widgets for common entity types"""
        common_entities = [
            ("Person", "Names of individuals"),
            ("Email", "Email addresses"),
            ("PhoneNumber", "Phone numbers"),
            ("Address", "Physical addresses"),
            ("SSN", "Social Security Numbers"),
            ("CreditCardNumber", "Credit card numbers"),
            ("Organization", "Company/organization names"),
            ("Location", "Geographic locations"),
            ("Date", "Dates and timestamps"),
            ("USBankAccountNumber", "US bank account numbers"),
            ("URL", "Web URLs"),
            ("IPAddress", "IP addresses")
        ]
        
        for entity_type, description in common_entities:
            self.create_entity_rule_widget(entity_type, description)
    
    def create_entity_rule_widget(self, entity_type: str, description: str):
        """Create a widget for configuring a specific entity type"""
        entity_frame = ctk.CTkFrame(self.entity_rules_frame)
        entity_frame.pack(fill="x", padx=10, pady=5)
        
        # Header with entity name and description
        header_frame = ctk.CTkFrame(entity_frame)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        # Enable/disable checkbox
        enabled_var = ctk.BooleanVar(value=True)
        enabled_check = ctk.CTkCheckBox(
            header_frame,
            text=f"{entity_type}",
            variable=enabled_var,
            font=ctk.CTkFont(weight="bold")
        )
        enabled_check.pack(side="left", padx=5)
        
        # Description
        desc_label = ctk.CTkLabel(
            header_frame,
            text=f"- {description}",
            text_color="gray"
        )
        desc_label.pack(side="left", padx=5)
        
        # Configuration row
        config_frame = ctk.CTkFrame(entity_frame)
        config_frame.pack(fill="x", padx=10, pady=5)
        
        # Confidence threshold
        ctk.CTkLabel(config_frame, text="Confidence:", width=80).pack(side="left", padx=5)
        confidence_var = ctk.DoubleVar(value=0.8)
        confidence_slider = ctk.CTkSlider(
            config_frame,
            variable=confidence_var,
            from_=0.0,
            to=1.0,
            width=120
        )
        confidence_slider.pack(side="left", padx=5)
        
        confidence_label = ctk.CTkLabel(config_frame, text="0.80", width=40)
        confidence_label.pack(side="left", padx=2)
        
        # Custom replacement
        ctk.CTkLabel(config_frame, text="Replace with:", width=80).pack(side="left", padx=5)
        replacement_var = ctk.StringVar(value=f"[{entity_type.upper()}]")
        replacement_entry = ctk.CTkEntry(
            config_frame,
            textvariable=replacement_var,
            width=120
        )
        replacement_entry.pack(side="left", padx=5)
        
        # Store widget references
        self.entity_rule_widgets[entity_type] = {
            'enabled_var': enabled_var,
            'confidence_var': confidence_var,
            'confidence_label': confidence_label,
            'replacement_var': replacement_var,
            'confidence_slider': confidence_slider
        }
        
        # Update label when slider changes
        def update_conf_label(value):
            confidence_label.configure(text=f"{value:.2f}")
        
        confidence_slider.configure(command=update_conf_label)
    
    def create_patterns_tab(self):
        """Create the custom patterns configuration"""
        # Main container with two sections
        main_container = ctk.CTkFrame(self.patterns_tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)
        
        # Whitelist patterns (left side)
        whitelist_frame = ctk.CTkFrame(main_container)
        whitelist_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        ctk.CTkLabel(
            whitelist_frame,
            text="✅ Whitelist Patterns",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))
        
        ctk.CTkLabel(
            whitelist_frame,
            text="Patterns that should NEVER be redacted",
            text_color="gray"
        ).pack(pady=(0, 10))
        
        # Whitelist controls
        wl_controls = ctk.CTkFrame(whitelist_frame)
        wl_controls.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            wl_controls,
            text="➕ Add Whitelist",
            command=self.add_whitelist_pattern,
            width=120,
            height=30
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            wl_controls,
            text="🗑️ Remove",
            command=self.remove_whitelist_pattern,
            width=100,
            height=30
        ).pack(side="left", padx=5)
        
        # Whitelist list
        self.whitelist_frame = ctk.CTkScrollableFrame(whitelist_frame)
        self.whitelist_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Blacklist patterns (right side)
        blacklist_frame = ctk.CTkFrame(main_container)
        blacklist_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        ctk.CTkLabel(
            blacklist_frame,
            text="❌ Blacklist Patterns",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))
        
        ctk.CTkLabel(
            blacklist_frame,
            text="Patterns that should ALWAYS be redacted",
            text_color="gray"
        ).pack(pady=(0, 10))
        
        # Blacklist controls
        bl_controls = ctk.CTkFrame(blacklist_frame)
        bl_controls.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            bl_controls,
            text="➕ Add Blacklist",
            command=self.add_blacklist_pattern,
            width=120,
            height=30
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            bl_controls,
            text="🗑️ Remove",
            command=self.remove_blacklist_pattern,
            width=100,
            height=30
        ).pack(side="left", padx=5)
        
        # Blacklist list
        self.blacklist_frame = ctk.CTkScrollableFrame(blacklist_frame)
        self.blacklist_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Pattern lists
        self.whitelist_patterns = []
        self.blacklist_patterns = []
    
    def create_advanced_tab(self):
        """Create the advanced configuration options"""
        scrollable = ctk.CTkScrollableFrame(self.advanced_tab)
        scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Advanced detection settings
        detection_frame = ctk.CTkFrame(scrollable)
        detection_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            detection_frame,
            text="Advanced Detection Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Context-aware detection
        context_frame = ctk.CTkFrame(detection_frame)
        context_frame.pack(fill="x", padx=15, pady=5)
        
        self.context_aware_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            context_frame,
            text="Enable context-aware detection",
            variable=self.context_aware_var
        ).pack(anchor="w", padx=5, pady=5)
        
        help_text = ctk.CTkLabel(
            context_frame,
            text="Uses surrounding text and column context to improve accuracy",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        help_text.pack(anchor="w", padx=20)
        
        # Data validation
        validation_frame = ctk.CTkFrame(scrollable)
        validation_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            validation_frame,
            text="Data Validation Rules",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Custom validator
        validator_frame = ctk.CTkFrame(validation_frame)
        validator_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(validator_frame, text="Custom Validation Function:").pack(anchor="w", padx=5)
        self.custom_validator_text = ctk.CTkTextbox(validator_frame, height=100)
        self.custom_validator_text.pack(fill="x", padx=5, pady=5)
        self.custom_validator_text.insert("0.0", 
            "# Python function that returns True if value should be redacted\n"
            "# Example:\n"
            "# def custom_validator(value, column_name, row_data):\n"
            "#     return len(value) > 10 and '@' in value\n"
        )
        
        # Performance settings
        perf_frame = ctk.CTkFrame(scrollable)
        perf_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            perf_frame,
            text="Performance & Cost Control",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Batch size
        batch_frame = ctk.CTkFrame(perf_frame)
        batch_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(batch_frame, text="Azure API Batch Size:", width=150).pack(side="left", padx=5)
        self.batch_size_var = ctk.IntVar(value=25)
        batch_spinner = ctk.CTkEntry(batch_frame, textvariable=self.batch_size_var, width=80)
        batch_spinner.pack(side="left", padx=5)
        
        # Cost limit
        cost_frame = ctk.CTkFrame(perf_frame)
        cost_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(cost_frame, text="Max Cost per Run ($):", width=150).pack(side="left", padx=5)
        self.max_cost_var = ctk.DoubleVar(value=10.0)
        cost_entry = ctk.CTkEntry(cost_frame, textvariable=self.max_cost_var, width=80)
        cost_entry.pack(side="left", padx=5)
        
    def create_preview_tab(self):
        """Create the preview tab for testing configurations"""
        # Main container
        container = ctk.CTkFrame(self.preview_tab)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)
        
        # Controls
        controls_frame = ctk.CTkFrame(container)
        controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ctk.CTkLabel(
            controls_frame,
            text="🔬 Test Your Configuration",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=15)
        
        # Input area
        input_frame = ctk.CTkFrame(controls_frame)
        input_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(input_frame, text="Test Data (one value per line):").pack(anchor="w", padx=5)
        self.test_input = ctk.CTkTextbox(input_frame, height=100)
        self.test_input.pack(fill="x", padx=5, pady=5)
        
        # Test button
        test_btn = ctk.CTkButton(
            controls_frame,
            text="🧪 Test Configuration",
            command=self.test_configuration,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        test_btn.pack(pady=10)
        
        # Results area
        results_frame = ctk.CTkFrame(container)
        results_frame.grid(row=1, column=0, sticky="nsew")
        
        ctk.CTkLabel(
            results_frame,
            text="Test Results",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 5))
        
        # Results tree
        self.results_tree = ttk.Treeview(
            results_frame,
            columns=("original", "redacted", "entities", "action"),
            show="headings",
            height=15
        )
        
        # Configure columns
        self.results_tree.heading("original", text="Original")
        self.results_tree.heading("redacted", text="Redacted")
        self.results_tree.heading("entities", text="Entities Found")
        self.results_tree.heading("action", text="Action Taken")
        
        self.results_tree.column("original", width=200)
        self.results_tree.column("redacted", width=200)
        self.results_tree.column("entities", width=150)
        self.results_tree.column("action", width=100)
        
        # Scrollbars for results
        results_scroll_y = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        results_scroll_x = ttk.Scrollbar(results_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=results_scroll_y.set, xscrollcommand=results_scroll_x.set)
        
        # Pack results
        self.results_tree.pack(side="left", fill="both", expand=True, padx=(15, 0), pady=15)
        results_scroll_y.pack(side="right", fill="y", pady=15)
        results_scroll_x.pack(side="bottom", fill="x", padx=15)
    
    def create_bottom_buttons(self, parent):
        """Create bottom action buttons"""
        # Left side - status
        self.status_label = ctk.CTkLabel(
            parent,
            text="Ready to configure columns",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=20, pady=15)
        
        # Right side - action buttons
        button_container = ctk.CTkFrame(parent)
        button_container.pack(side="right", padx=20, pady=15)
        
        buttons = [
            ("❌ Cancel", self.cancel_config, "red"),
            ("💾 Save & Apply", self.save_and_apply, "green"),
            ("✅ Apply Configuration", self.apply_configuration, "blue")
        ]
        
        for text, command, color in buttons:
            btn = ctk.CTkButton(
                button_container,
                text=text,
                command=command,
                fg_color=color,
                hover_color=f"dark{color}",
                width=150,
                height=40
            )
            btn.pack(side="left", padx=5)
    
    # Event handlers
    def on_column_selected(self, column_name: str):
        """Handle column selection"""
        self.current_column = column_name
        self.load_column_config()
        self.update_status(f"Configuring column: {column_name}")
    
    def on_detection_mode_changed(self, mode: str):
        """Handle detection mode change"""
        self.update_entity_rules_state()
    
    def update_confidence_label(self, value):
        """Update confidence threshold label"""
        self.confidence_label.configure(text=f"{value:.2f}")
    
    def load_column_config(self):
        """Load configuration for the selected column"""
        if not self.current_column:
            return
        
        config = self.config_manager.get_column_config(self.current_column)
        
        # Update basic settings
        self.column_name_label.configure(text=self.current_column)
        self.detection_mode_var.set(config.detection_mode.value)
        self.data_type_var.set(config.expected_data_type.value)
        self.confidence_var.set(config.min_confidence)
        self.update_confidence_label(config.min_confidence)
        
        # Update checkboxes
        self.require_review_var.set(config.require_human_review)
        self.preserve_formatting_var.set(config.preserve_formatting)
        self.partial_redaction_var.set(config.partial_redaction)
        
        # Update text fields
        self.business_context_text.delete("0.0", "end")
        if config.business_context:
            self.business_context_text.insert("0.0", config.business_context)
        
        self.domain_keywords_entry.delete(0, "end")
        if config.domain_keywords:
            self.domain_keywords_entry.insert(0, ", ".join(config.domain_keywords))
        
        # Update entity rules
        for entity_type, widgets in self.entity_rule_widgets.items():
            if entity_type in config.entity_rules:
                rule = config.entity_rules[entity_type]
                widgets['enabled_var'].set(rule.enabled)
                widgets['confidence_var'].set(rule.confidence_threshold)
                widgets['confidence_label'].configure(text=f"{rule.confidence_threshold:.2f}")
                if rule.custom_replacement:
                    widgets['replacement_var'].set(rule.custom_replacement)
            else:
                widgets['enabled_var'].set(True)
                widgets['confidence_var'].set(0.8)
                widgets['confidence_label'].configure(text="0.80")
                widgets['replacement_var'].set(f"[{entity_type.upper()}]")
        
        # Update patterns
        self.load_patterns(config)
    
    def load_patterns(self, config: ColumnConfig):
        """Load whitelist and blacklist patterns"""
        # Clear existing patterns
        for widget in self.whitelist_frame.winfo_children():
            widget.destroy()
        for widget in self.blacklist_frame.winfo_children():
            widget.destroy()
        
        self.whitelist_patterns.clear()
        self.blacklist_patterns.clear()
        
        # Load whitelist patterns
        for pattern in config.whitelist_patterns:
            self.add_whitelist_pattern_widget(pattern)
        
        # Load blacklist patterns
        for pattern in config.blacklist_patterns:
            self.add_blacklist_pattern_widget(pattern)
    
    def add_whitelist_pattern(self):
        """Add a new whitelist pattern"""
        pattern = WhitelistPattern(
            pattern="",
            description="New whitelist pattern"
        )
        self.add_whitelist_pattern_widget(pattern)
    
    def add_whitelist_pattern_widget(self, pattern: WhitelistPattern):
        """Add whitelist pattern widget"""
        frame = ctk.CTkFrame(self.whitelist_frame)
        frame.pack(fill="x", pady=2)
        
        # Pattern input
        pattern_entry = ctk.CTkEntry(frame, placeholder_text="Pattern", width=150)
        pattern_entry.pack(side="left", padx=2)
        pattern_entry.insert(0, pattern.pattern)
        
        # Description input
        desc_entry = ctk.CTkEntry(frame, placeholder_text="Description", width=120)
        desc_entry.pack(side="left", padx=2)
        desc_entry.insert(0, pattern.description)
        
        # Regex checkbox
        regex_var = ctk.BooleanVar(value=pattern.regex)
        regex_check = ctk.CTkCheckBox(frame, text="Regex", variable=regex_var, width=50)
        regex_check.pack(side="left", padx=2)
        
        # Case sensitive checkbox
        case_var = ctk.BooleanVar(value=pattern.case_sensitive)
        case_check = ctk.CTkCheckBox(frame, text="Case", variable=case_var, width=50)
        case_check.pack(side="left", padx=2)
        
        # Store pattern data
        pattern_data = {
            'frame': frame,
            'pattern_entry': pattern_entry,
            'desc_entry': desc_entry,
            'regex_var': regex_var,
            'case_var': case_var
        }
        self.whitelist_patterns.append(pattern_data)
    
    def add_blacklist_pattern(self):
        """Add a new blacklist pattern"""
        pattern = BlacklistPattern(
            pattern="",
            description="New blacklist pattern",
            replacement="[REDACTED]"
        )
        self.add_blacklist_pattern_widget(pattern)
    
    def add_blacklist_pattern_widget(self, pattern: BlacklistPattern):
        """Add blacklist pattern widget"""
        frame = ctk.CTkFrame(self.blacklist_frame)
        frame.pack(fill="x", pady=2)
        
        # Pattern input
        pattern_entry = ctk.CTkEntry(frame, placeholder_text="Pattern", width=120)
        pattern_entry.pack(side="left", padx=2)
        pattern_entry.insert(0, pattern.pattern)
        
        # Replacement input
        replacement_entry = ctk.CTkEntry(frame, placeholder_text="Replace with", width=100)
        replacement_entry.pack(side="left", padx=2)
        replacement_entry.insert(0, pattern.replacement)
        
        # Description input
        desc_entry = ctk.CTkEntry(frame, placeholder_text="Description", width=100)
        desc_entry.pack(side="left", padx=2)
        desc_entry.insert(0, pattern.description)
        
        # Regex checkbox
        regex_var = ctk.BooleanVar(value=pattern.regex)
        regex_check = ctk.CTkCheckBox(frame, text="Regex", variable=regex_var, width=50)
        regex_check.pack(side="left", padx=2)
        
        # Store pattern data
        pattern_data = {
            'frame': frame,
            'pattern_entry': pattern_entry,
            'replacement_entry': replacement_entry,
            'desc_entry': desc_entry,
            'regex_var': regex_var
        }
        self.blacklist_patterns.append(pattern_data)
    
    def remove_whitelist_pattern(self):
        """Remove selected whitelist pattern"""
        if self.whitelist_patterns:
            pattern_data = self.whitelist_patterns.pop()
            pattern_data['frame'].destroy()
    
    def remove_blacklist_pattern(self):
        """Remove selected blacklist pattern"""
        if self.blacklist_patterns:
            pattern_data = self.blacklist_patterns.pop()
            pattern_data['frame'].destroy()
    
    def update_entity_rules_state(self):
        """Update entity rules based on detection mode"""
        mode = self.detection_mode_var.get()
        disabled = (mode == "disabled")
        
        for widgets in self.entity_rule_widgets.values():
            state = "disabled" if disabled else "normal"
            widgets['confidence_slider'].configure(state=state)
    
    def apply_template(self, template_name: str):
        """Apply a predefined template"""
        if not self.current_column:
            messagebox.showwarning("Warning", "Please select a column first")
            return
        
        self.config_manager.apply_template(self.current_column, template_name)
        self.load_column_config()
        self.update_status(f"Applied {template_name} template to {self.current_column}")
    
    def analyze_current_column(self):
        """Analyze the current column data"""
        if not self.current_column or self.current_df is None:
            messagebox.showwarning("Warning", "Please select a column and load CSV data first")
            return
        
        analysis = self.config_manager.analyze_column_data(self.current_df, self.current_column)
        
        # Show analysis results
        analysis_window = ctk.CTkToplevel(self.window)
        analysis_window.title(f"Analysis: {self.current_column}")
        analysis_window.geometry("600x500")
        
        # Analysis content
        content = ctk.CTkTextbox(analysis_window)
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        content.insert("0.0", f"Column Analysis: {self.current_column}\n")
        content.insert("end", "="*50 + "\n\n")
        
        content.insert("end", f"Unique values: {analysis.get('unique_values', 'N/A')}\n")
        content.insert("end", f"Null percentage: {analysis.get('null_percentage', 'N/A'):.1f}%\n\n")
        
        content.insert("end", "Sample values:\n")
        for value in analysis.get('sample_values', []):
            content.insert("end", f"  • {value}\n")
        
        content.insert("end", "\nData type suggestions:\n")
        for data_type, confidence in analysis.get('data_type_suggestions', []):
            content.insert("end", f"  • {data_type}: {confidence:.1%} confidence\n")
        
        content.configure(state="disabled")
    
    def copy_config(self):
        """Copy current column configuration"""
        if not self.current_column:
            messagebox.showwarning("Warning", "Please select a column first")
            return
        
        # Store config in clipboard-like storage
        self._copied_config = self.get_current_config()
        self.update_status(f"Copied configuration from {self.current_column}")
    
    def paste_config(self):
        """Paste configuration to current column"""
        if not self.current_column:
            messagebox.showwarning("Warning", "Please select a column first")
            return
        
        if not hasattr(self, '_copied_config') or not self._copied_config:
            messagebox.showwarning("Warning", "No configuration copied")
            return
        
        # Apply copied config
        self.config_manager.set_column_config(self._copied_config)
        self._copied_config.column_name = self.current_column
        self.load_column_config()
        self.update_status(f"Pasted configuration to {self.current_column}")
    
    def reset_to_default(self):
        """Reset current column to default configuration"""
        if not self.current_column:
            messagebox.showwarning("Warning", "Please select a column first")
            return
        
        # Remove from config manager to get default
        if self.current_column in self.config_manager.configs:
            del self.config_manager.configs[self.current_column]
        
        self.load_column_config()
        self.update_status(f"Reset {self.current_column} to default configuration")
    
    def test_configuration(self):
        """Test the current configuration with sample data"""
        test_data = self.test_input.get("0.0", "end").strip().split("\n")
        test_data = [line.strip() for line in test_data if line.strip()]
        
        if not test_data:
            messagebox.showwarning("Warning", "Please enter test data")
            return
        
        # Clear results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Test each value
        for value in test_data:
            # Simulate PII detection (you would integrate with actual Azure detection here)
            redacted, entities, action = self.simulate_pii_detection(value)
            
            self.results_tree.insert("", "end", values=(
                value[:50] + "..." if len(value) > 50 else value,
                redacted[:50] + "..." if len(redacted) > 50 else redacted,
                ", ".join(entities) if entities else "None",
                action
            ))
    
    def simulate_pii_detection(self, value: str) -> tuple[str, list, str]:
        """Simulate PII detection for testing"""
        # This is a simplified simulation - integrate with actual Azure detection
        entities = []
        action = "No change"
        redacted = value
        
        # Simple pattern matching for demo
        if "@" in value and "." in value:
            entities.append("Email")
            redacted = "[EMAIL]"
            action = "Redacted"
        elif re.match(r'\d{3}-\d{3}-\d{4}', value):
            entities.append("Phone")
            redacted = "[PHONE]"
            action = "Redacted"
        
        return redacted, entities, action
    
    def get_current_config(self) -> ColumnConfig:
        """Get current configuration from UI"""
        if not self.current_column:
            return None
        
        # Create config from current UI state
        config = ColumnConfig(column_name=self.current_column)
        
        # Basic settings
        config.detection_mode = DetectionMode(self.detection_mode_var.get())
        config.expected_data_type = DataType(self.data_type_var.get())
        config.min_confidence = self.confidence_var.get()
        config.require_human_review = self.require_review_var.get()
        config.preserve_formatting = self.preserve_formatting_var.get()
        config.partial_redaction = self.partial_redaction_var.get()
        
        # Text fields
        config.business_context = self.business_context_text.get("0.0", "end").strip()
        keywords_text = self.domain_keywords_entry.get().strip()
        config.domain_keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
        
        # Entity rules
        config.entity_rules = {}
        for entity_type, widgets in self.entity_rule_widgets.items():
            rule = EntityRule(
                entity_type=entity_type,
                enabled=widgets['enabled_var'].get(),
                confidence_threshold=widgets['confidence_var'].get(),
                custom_replacement=widgets['replacement_var'].get()
            )
            config.entity_rules[entity_type] = rule
        
        # Patterns
        config.whitelist_patterns = []
        for pattern_data in self.whitelist_patterns:
            if pattern_data['pattern_entry'].get().strip():
                pattern = WhitelistPattern(
                    pattern=pattern_data['pattern_entry'].get(),
                    description=pattern_data['desc_entry'].get(),
                    regex=pattern_data['regex_var'].get(),
                    case_sensitive=pattern_data['case_var'].get()
                )
                config.whitelist_patterns.append(pattern)
        
        config.blacklist_patterns = []
        for pattern_data in self.blacklist_patterns:
            if pattern_data['pattern_entry'].get().strip():
                pattern = BlacklistPattern(
                    pattern=pattern_data['pattern_entry'].get(),
                    description=pattern_data['desc_entry'].get(),
                    replacement=pattern_data['replacement_entry'].get(),
                    regex=pattern_data['regex_var'].get()
                )
                config.blacklist_patterns.append(pattern)
        
        return config
    
    def load_csv_for_analysis(self):
        """Load CSV file for column analysis"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File for Analysis",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                import pandas as pd
                self.current_df = pd.read_csv(file_path)
                
                # Update column dropdown
                columns = list(self.current_df.columns)
                self.column_dropdown.configure(values=columns)
                
                if columns:
                    self.column_dropdown.set(columns[0])
                    self.on_column_selected(columns[0])
                
                self.update_status(f"Loaded CSV with {len(columns)} columns")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def save_configuration(self):
        """Save current configuration"""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                self.config_manager.save_config(file_path)
                messagebox.showinfo("Success", "Configuration saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def load_configuration(self):
        """Load configuration from file"""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                self.config_manager.load_config(file_path)
                self.load_column_config()
                messagebox.showinfo("Success", "Configuration loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
    
    def export_configuration(self):
        """Export configuration as JSON"""
        config = self.get_current_config()
        if not config:
            messagebox.showwarning("Warning", "No configuration to export")
            return
        
        # Convert to JSON
        config_dict = {
            'column_name': config.column_name,
            'detection_mode': config.detection_mode.value,
            'expected_data_type': config.expected_data_type.value,
            'min_confidence': config.min_confidence,
            'require_human_review': config.require_human_review,
            'preserve_formatting': config.preserve_formatting,
            'partial_redaction': config.partial_redaction,
            'business_context': config.business_context,
            'domain_keywords': config.domain_keywords,
            'entity_rules': {k: {
                'enabled': v.enabled,
                'confidence_threshold': v.confidence_threshold,
                'custom_replacement': v.custom_replacement
            } for k, v in config.entity_rules.items()},
            'whitelist_patterns': [{
                'pattern': p.pattern,
                'description': p.description,
                'regex': p.regex,
                'case_sensitive': p.case_sensitive
            } for p in config.whitelist_patterns],
            'blacklist_patterns': [{
                'pattern': p.pattern,
                'description': p.description,
                'replacement': p.replacement,
                'regex': p.regex
            } for p in config.blacklist_patterns]
        }
        
        export_window = ctk.CTkToplevel(self.window)
        export_window.title("Export Configuration")
        export_window.geometry("600x500")
        
        text_area = ctk.CTkTextbox(export_window)
        text_area.pack(fill="both", expand=True, padx=20, pady=20)
        text_area.insert("0.0", json.dumps(config_dict, indent=2))
    
    def clear_all_configs(self):
        """Clear all configurations"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all configurations?"):
            self.config_manager.configs.clear()
            self.load_column_config()
            self.update_status("All configurations cleared")
    
    def cancel_config(self):
        """Cancel configuration"""
        self.window.destroy()
    
    def save_and_apply(self):
        """Save current configuration and apply"""
        if not self.current_column:
            messagebox.showwarning("Warning", "Please select a column first")
            return
        
        config = self.get_current_config()
        self.config_manager.set_column_config(config)
        self.update_status(f"Configuration saved for {self.current_column}")
        messagebox.showinfo("Success", "Configuration saved successfully!")
    
    def apply_configuration(self):
        """Apply configuration and close"""
        self.save_and_apply()
        self.window.destroy()
    
    def update_status(self, message: str):
        """Update status message"""
        self.status_label.configure(text=message)


def open_column_config_gui(parent_window, config_manager=None):
    """Open the column configuration GUI"""
    return ColumnConfigGUI(parent_window, config_manager)
