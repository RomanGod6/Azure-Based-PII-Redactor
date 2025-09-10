"""
Column-Level PII Configuration System
Advanced customization for PII detection on a per-column basis
"""

import json
import re
from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd


class DetectionMode(Enum):
    """Detection modes for columns"""
    AGGRESSIVE = "aggressive"      # Detect everything possible
    BALANCED = "balanced"         # Standard detection with some filtering
    CONSERVATIVE = "conservative" # Only detect high-confidence PII
    CUSTOM = "custom"            # Use custom rules only
    DISABLED = "disabled"        # Skip PII detection entirely


class DataType(Enum):
    """Expected data types for columns"""
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    NAME = "name"
    ADDRESS = "address"
    ID_NUMBER = "id_number"
    FINANCIAL = "financial"
    DATE = "date"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    PRODUCT_CODE = "product_code"
    REFERENCE = "reference"
    DESCRIPTION = "description"
    COMMENTS = "comments"
    URL = "url"
    MEDICAL = "medical"
    LEGAL = "legal"


@dataclass
class WhitelistPattern:
    """Pattern that should NOT be redacted"""
    pattern: str
    description: str
    regex: bool = False
    case_sensitive: bool = False


@dataclass
class BlacklistPattern:
    """Pattern that should ALWAYS be redacted"""
    pattern: str
    description: str
    replacement: str
    regex: bool = False
    case_sensitive: bool = False


@dataclass
class EntityRule:
    """Rules for specific PII entity types"""
    entity_type: str
    enabled: bool = True
    confidence_threshold: float = 0.8
    custom_replacement: Optional[str] = None
    context_aware: bool = True


@dataclass
class ColumnConfig:
    """Configuration for a single column"""
    column_name: str
    detection_mode: DetectionMode = DetectionMode.BALANCED
    expected_data_type: DataType = DataType.TEXT
    
    # Entity-specific rules
    entity_rules: Dict[str, EntityRule] = None
    
    # Custom patterns
    whitelist_patterns: List[WhitelistPattern] = None
    blacklist_patterns: List[BlacklistPattern] = None
    
    # Entity type exclusions
    excluded_entity_types: List[str] = None
    
    # Validation rules
    min_confidence: float = 0.7
    require_human_review: bool = False
    
    # Context awareness
    business_context: Optional[str] = None
    domain_keywords: List[str] = None
    
    # Advanced settings
    preserve_formatting: bool = False
    partial_redaction: bool = False
    custom_validator: Optional[str] = None
    
    def __post_init__(self):
        if self.entity_rules is None:
            self.entity_rules = {}
        if self.whitelist_patterns is None:
            self.whitelist_patterns = []
        if self.blacklist_patterns is None:
            self.blacklist_patterns = []
        if self.excluded_entity_types is None:
            self.excluded_entity_types = []
        if self.domain_keywords is None:
            self.domain_keywords = []
    
    def add_excluded_entity_type(self, entity_type: str):
        """Add an entity type to exclude from detection"""
        if entity_type not in self.excluded_entity_types:
            self.excluded_entity_types.append(entity_type)
    
    def remove_excluded_entity_type(self, entity_type: str):
        """Remove an entity type from exclusion list"""
        if entity_type in self.excluded_entity_types:
            self.excluded_entity_types.remove(entity_type)
    
    def is_entity_type_excluded(self, entity_type: str) -> bool:
        """Check if an entity type is excluded"""
        return entity_type in self.excluded_entity_types


class ColumnConfigManager:
    """Manages column configurations and applies custom rules"""
    
    def __init__(self):
        self.configs: Dict[str, ColumnConfig] = {}
        self.global_settings = {
            'default_mode': DetectionMode.BALANCED,
            'auto_detect_data_types': True,
            'smart_suggestions': True,
            'preserve_data_relationships': True
        }
        
        # Predefined templates for common column types
        self.templates = self._create_templates()
        
    def _create_templates(self) -> Dict[str, ColumnConfig]:
        """Create predefined templates for common column types"""
        templates = {}
        
        # Email template
        templates['email'] = ColumnConfig(
            column_name="email_template",
            detection_mode=DetectionMode.CONSERVATIVE,
            expected_data_type=DataType.EMAIL,
            entity_rules={
                'Email': EntityRule('Email', True, 0.9, '[EMAIL]'),
                'Person': EntityRule('Person', False),  # Don't redact names in email context
            },
            whitelist_patterns=[
                WhitelistPattern(
                    pattern=r"noreply@|support@|admin@|info@|no-reply@",
                    description="System/service emails",
                    regex=True,
                    case_sensitive=False
                )
            ]
        )
        
        # Phone template
        templates['phone'] = ColumnConfig(
            column_name="phone_template",
            detection_mode=DetectionMode.BALANCED,
            expected_data_type=DataType.PHONE,
            entity_rules={
                'PhoneNumber': EntityRule('PhoneNumber', True, 0.8, '[PHONE]'),
                'Number': EntityRule('Number', False),  # Don't redact other numbers
            },
            whitelist_patterns=[
                WhitelistPattern(
                    pattern=r"1-800-|1-888-|1-877-|1-866-",
                    description="Toll-free numbers",
                    regex=True
                )
            ]
        )
        
        # ID/Reference template
        templates['id_reference'] = ColumnConfig(
            column_name="id_template",
            detection_mode=DetectionMode.CONSERVATIVE,
            expected_data_type=DataType.ID_NUMBER,
            entity_rules={
                'SSN': EntityRule('SSN', True, 0.95, '[SSN]'),
                'Number': EntityRule('Number', False),  # Don't redact reference numbers
            },
            whitelist_patterns=[
                WhitelistPattern(
                    pattern=r"^(REF|ID|TKT|ORD)-?\d+$",
                    description="Reference/Order/Ticket IDs",
                    regex=True,
                    case_sensitive=False
                )
            ]
        )
        
        # Name template
        templates['name'] = ColumnConfig(
            column_name="name_template",
            detection_mode=DetectionMode.BALANCED,
            expected_data_type=DataType.NAME,
            entity_rules={
                'Person': EntityRule('Person', True, 0.8, '[NAME]'),
                'Organization': EntityRule('Organization', False),  # Handle separately
            },
            blacklist_patterns=[
                BlacklistPattern(
                    pattern=r"\b(mr|mrs|ms|dr|prof)\s+\w+",
                    description="Titles with names",
                    replacement="[TITLE_NAME]",
                    regex=True,
                    case_sensitive=False
                )
            ]
        )
        
        # Financial template
        templates['financial'] = ColumnConfig(
            column_name="financial_template",
            detection_mode=DetectionMode.AGGRESSIVE,
            expected_data_type=DataType.FINANCIAL,
            entity_rules={
                'CreditCardNumber': EntityRule('CreditCardNumber', True, 0.9, '[CREDIT_CARD]'),
                'USBankAccountNumber': EntityRule('USBankAccountNumber', True, 0.9, '[BANK_ACCOUNT]'),
                'InternationalBankingAccountNumber': EntityRule('InternationalBankingAccountNumber', True, 0.9, '[IBAN]'),
            }
        )
        
        # Product/Category template
        templates['product_category'] = ColumnConfig(
            column_name="product_template",
            detection_mode=DetectionMode.DISABLED,
            expected_data_type=DataType.CATEGORICAL,
            domain_keywords=['product', 'category', 'type', 'model', 'sku']
        )
        
        # Comments/Description template
        templates['comments'] = ColumnConfig(
            column_name="comments_template",
            detection_mode=DetectionMode.BALANCED,
            expected_data_type=DataType.COMMENTS,
            entity_rules={
                'Person': EntityRule('Person', True, 0.85, '[NAME]'),
                'Email': EntityRule('Email', True, 0.9, '[EMAIL]'),
                'PhoneNumber': EntityRule('PhoneNumber', True, 0.85, '[PHONE]'),
            },
            require_human_review=True,
            business_context="Customer feedback and support comments"
        )
        
        return templates
    
    def get_column_config(self, column_name: str) -> ColumnConfig:
        """Get configuration for a column"""
        return self.configs.get(column_name, self._create_default_config(column_name))
    
    def set_column_config(self, config: ColumnConfig):
        """Set configuration for a column"""
        self.configs[config.column_name] = config
    
    def _create_default_config(self, column_name: str) -> ColumnConfig:
        """Create default configuration based on column name analysis"""
        config = ColumnConfig(column_name=column_name)
        
        # Auto-detect data type based on column name
        name_lower = column_name.lower()
        
        if any(word in name_lower for word in ['email', 'e-mail', 'mail']):
            config.expected_data_type = DataType.EMAIL
            config.detection_mode = DetectionMode.CONSERVATIVE
        elif any(word in name_lower for word in ['phone', 'tel', 'mobile', 'cell']):
            config.expected_data_type = DataType.PHONE
            config.detection_mode = DetectionMode.BALANCED
        elif any(word in name_lower for word in ['name', 'author', 'user', 'customer']):
            config.expected_data_type = DataType.NAME
            config.detection_mode = DetectionMode.BALANCED
        elif any(word in name_lower for word in ['address', 'street', 'city', 'zip', 'postal']):
            config.expected_data_type = DataType.ADDRESS
            config.detection_mode = DetectionMode.BALANCED
        elif any(word in name_lower for word in ['id', 'ref', 'ticket', 'order', 'number']):
            config.expected_data_type = DataType.ID_NUMBER
            config.detection_mode = DetectionMode.CONSERVATIVE
        elif any(word in name_lower for word in ['comment', 'description', 'note', 'body', 'content']):
            config.expected_data_type = DataType.COMMENTS
            config.detection_mode = DetectionMode.BALANCED
        elif any(word in name_lower for word in ['product', 'category', 'type', 'status', 'priority']):
            config.expected_data_type = DataType.CATEGORICAL
            config.detection_mode = DetectionMode.DISABLED
        elif any(word in name_lower for word in ['date', 'time', 'created', 'updated']):
            config.expected_data_type = DataType.DATE
            config.detection_mode = DetectionMode.CONSERVATIVE
        
        return config
    
    def apply_template(self, column_name: str, template_name: str):
        """Apply a predefined template to a column"""
        if template_name in self.templates:
            template = self.templates[template_name]
            config = ColumnConfig(
                column_name=column_name,
                detection_mode=template.detection_mode,
                expected_data_type=template.expected_data_type,
                entity_rules=template.entity_rules.copy(),
                whitelist_patterns=template.whitelist_patterns.copy(),
                blacklist_patterns=template.blacklist_patterns.copy(),
                min_confidence=template.min_confidence,
                require_human_review=template.require_human_review,
                business_context=template.business_context,
                domain_keywords=template.domain_keywords.copy()
            )
            self.set_column_config(config)
    
    def analyze_column_data(self, df: pd.DataFrame, column_name: str, sample_size: int = 100) -> Dict[str, Any]:
        """Analyze column data to suggest configuration"""
        if column_name not in df.columns:
            return {}
        
        col_data = df[column_name].dropna().head(sample_size)
        
        analysis = {
            'data_type_suggestions': [],
            'pattern_suggestions': [],
            'confidence_score': 0.0,
            'sample_values': col_data.head(5).tolist(),
            'unique_values': col_data.nunique(),
            'null_percentage': (df[column_name].isnull().sum() / len(df)) * 100
        }
        
        # Analyze patterns in the data
        text_data = col_data.astype(str)
        
        # Email detection
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = text_data.str.contains(email_pattern, regex=True, na=False).sum()
        if email_matches > 0:
            analysis['data_type_suggestions'].append(('EMAIL', email_matches / len(text_data)))
        
        # Phone detection
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phone_matches = text_data.str.contains(phone_pattern, regex=True, na=False).sum()
        if phone_matches > 0:
            analysis['data_type_suggestions'].append(('PHONE', phone_matches / len(text_data)))
        
        # ID patterns
        id_pattern = r'^[A-Z]{2,4}-?\d{4,}$'
        id_matches = text_data.str.contains(id_pattern, regex=True, na=False).sum()
        if id_matches > 0:
            analysis['data_type_suggestions'].append(('ID_NUMBER', id_matches / len(text_data)))
        
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b'
        ]
        date_matches = 0
        for pattern in date_patterns:
            date_matches += text_data.str.contains(pattern, regex=True, na=False).sum()
        if date_matches > 0:
            analysis['data_type_suggestions'].append(('DATE', date_matches / len(text_data)))
        
        # Sort suggestions by confidence
        analysis['data_type_suggestions'].sort(key=lambda x: x[1], reverse=True)
        
        return analysis
    
    def should_redact_entity(self, column_name: str, entity_type: str, confidence: float, text: str) -> tuple[bool, str]:
        """Determine if an entity should be redacted based on column configuration"""
        config = self.get_column_config(column_name)
        
        # Check if detection is disabled
        if config.detection_mode == DetectionMode.DISABLED:
            return False, text
        
        # Check whitelist patterns first
        for pattern in config.whitelist_patterns:
            if self._matches_pattern(text, pattern):
                return False, text
        
        # Check blacklist patterns
        for pattern in config.blacklist_patterns:
            if self._matches_pattern(text, pattern):
                return True, pattern.replacement
        
        # Check entity-specific rules
        if entity_type in config.entity_rules:
            rule = config.entity_rules[entity_type]
            if not rule.enabled:
                return False, text
            if confidence < rule.confidence_threshold:
                return False, text
            return True, rule.custom_replacement or f'[{entity_type.upper()}]'
        
        # Apply detection mode logic
        if config.detection_mode == DetectionMode.CONSERVATIVE:
            return confidence >= 0.9, f'[{entity_type.upper()}]'
        elif config.detection_mode == DetectionMode.BALANCED:
            return confidence >= config.min_confidence, f'[{entity_type.upper()}]'
        elif config.detection_mode == DetectionMode.AGGRESSIVE:
            return confidence >= 0.5, f'[{entity_type.upper()}]'
        
        return False, text
    
    def _matches_pattern(self, text: str, pattern: WhitelistPattern) -> bool:
        """Check if text matches a pattern"""
        pattern_str = pattern.pattern
        flags = 0 if pattern.case_sensitive else re.IGNORECASE
        
        try:
            if pattern.regex:
                return bool(re.search(pattern_str, text, flags))
            else:
                if pattern.case_sensitive:
                    return pattern_str in text
                else:
                    return pattern_str.lower() in text.lower()
        except re.error:
            return False
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file"""
        config_data = {
            'global_settings': self.global_settings,
            'column_configs': {}
        }
        
        for col_name, config in self.configs.items():
            config_data['column_configs'][col_name] = asdict(config)
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def load_config(self, filepath: str):
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            self.global_settings = config_data.get('global_settings', {})
            
            self.configs = {}
            for col_name, config_dict in config_data.get('column_configs', {}).items():
                # Convert dict back to ColumnConfig
                config = ColumnConfig(column_name=col_name)
                config.__dict__.update(config_dict)
                
                # Convert enum strings back to enums
                if isinstance(config.detection_mode, str):
                    config.detection_mode = DetectionMode(config.detection_mode)
                if isinstance(config.expected_data_type, str):
                    config.expected_data_type = DataType(config.expected_data_type)
                
                # Convert entity rules
                if config.entity_rules:
                    converted_rules = {}
                    for entity_type, rule_dict in config.entity_rules.items():
                        if isinstance(rule_dict, dict):
                            rule = EntityRule(entity_type)
                            rule.__dict__.update(rule_dict)
                            converted_rules[entity_type] = rule
                        else:
                            converted_rules[entity_type] = rule_dict
                    config.entity_rules = converted_rules
                
                self.configs[col_name] = config
                
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def get_template_names(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())
    
    def duplicate_config(self, source_column: str, target_column: str):
        """Duplicate configuration from one column to another"""
        if source_column in self.configs:
            source_config = self.configs[source_column]
            new_config = ColumnConfig(
                column_name=target_column,
                detection_mode=source_config.detection_mode,
                expected_data_type=source_config.expected_data_type,
                entity_rules=source_config.entity_rules.copy(),
                whitelist_patterns=source_config.whitelist_patterns.copy(),
                blacklist_patterns=source_config.blacklist_patterns.copy(),
                min_confidence=source_config.min_confidence,
                require_human_review=source_config.require_human_review,
                business_context=source_config.business_context,
                domain_keywords=source_config.domain_keywords.copy()
            )
            self.set_column_config(new_config)
