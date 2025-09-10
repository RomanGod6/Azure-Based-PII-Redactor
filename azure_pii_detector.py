"""
Enhanced PII Detection Module with Azure AI Integration
This module provides the actual Azure AI implementation for PII detection
"""

import re
from typing import Dict, List, Tuple, Optional
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import pandas as pd

# Import column configuration system
try:
    from column_config import ColumnConfigManager, ColumnConfig, DetectionMode
except ImportError:
    print("Warning: Column configuration system not available")


class AzurePIIDetector:
    """Advanced PII detector using Azure AI services"""
    
    # PII entity types mapping to redaction labels
    PII_ENTITY_MAP = {
        'Person': '[NAME]',
        'PersonType': '[PERSON_TYPE]',
        'Location': '[LOCATION]',
        'Organization': '[ORGANIZATION]',
        'Event': '[EVENT]',
        'Product': '[PRODUCT]',
        'Skill': '[SKILL]',
        'Email': '[EMAIL]',
        'URL': '[URL]',
        'IPAddress': '[IP_ADDRESS]',
        'PhoneNumber': '[PHONE]',
        'Address': '[ADDRESS]',
        'InternationalBankingAccountNumber': '[IBAN]',
        'CreditCardNumber': '[CREDIT_CARD]',
        'SSN': '[SSN]',
        'DateTime': '[DATE_TIME]',
        'Date': '[DATE]',
        'Time': '[TIME]',
        'Duration': '[DURATION]',
        'Age': '[AGE]',
        'Quantity': '[QUANTITY]',
        'Percentage': '[PERCENTAGE]',
        'Number': '[NUMBER]',
        'Ordinal': '[ORDINAL]',
        'MonetaryAmount': '[AMOUNT]',
        'Temperature': '[TEMPERATURE]',
        # Healthcare specific
        'HealthcareEntity': '[HEALTH_INFO]',
        'MedicalCode': '[MEDICAL_CODE]',
        'Gene': '[GENE]',
        'DrugName': '[DRUG]',
        'BodyStructure': '[BODY_PART]',
        'Diagnosis': '[DIAGNOSIS]',
        'SymptomOrSign': '[SYMPTOM]',
        'MedicationClass': '[MED_CLASS]',
        'TreatmentName': '[TREATMENT]',
        # Financial specific
        'USBankAccountNumber': '[US_BANK_ACCOUNT]',
        'UKNationalInsuranceNumber': '[UK_NI_NUMBER]',
        'CanadaSocialInsuranceNumber': '[CA_SIN]',
        'AustralianBusinessNumber': '[AU_ABN]',
        'AustralianCompanyNumber': '[AU_ACN]',
        'AustralianTaxFileNumber': '[AU_TFN]',
        'AustralianMedicalAccountNumber': '[AU_MEDICAL]',
        'BrazilCPFNumber': '[BR_CPF]',
        'BrazilLegalEntityNumber': '[BR_CNPJ]',
        'BrazilNationalIDRG': '[BR_RG]',
        'IndiaUniqueIdentificationNumber': '[IN_AADHAAR]',
        'IndiaPermanentAccountNumber': '[IN_PAN]',
        'JapanResidentRegistrationNumber': '[JP_RESIDENT]',
        'JapanSocialInsuranceNumber': '[JP_SIN]',
        'JapanPassportNumber': '[JP_PASSPORT]',
        'JapanDriverLicenseNumber': '[JP_DRIVER_LICENSE]',
        'JapanBankAccountNumber': '[JP_BANK_ACCOUNT]',
        'JapanMyNumberCorporate': '[JP_CORP_NUMBER]',
        'JapanMyNumberPersonal': '[JP_MY_NUMBER]',
        # IDs and Documents
        'USDriverLicenseNumber': '[US_DRIVER_LICENSE]',
        'USPassportNumber': '[US_PASSPORT]',
        'UKDriverLicenseNumber': '[UK_DRIVER_LICENSE]',
        'UKElectoralRollNumber': '[UK_ELECTORAL]',
        'UKNationalHealthNumber': '[UK_NHS]',
        'UKPassportNumber': '[UK_PASSPORT]',
        'UKUniqueTaxpayerNumber': '[UK_UTR]',
        'CanadaDriverLicenseNumber': '[CA_DRIVER_LICENSE]',
        'CanadaHealthNumber': '[CA_HEALTH]',
        'CanadaPassportNumber': '[CA_PASSPORT]',
        'CanadaBankAccountNumber': '[CA_BANK_ACCOUNT]',
        'AustralianDriverLicenseNumber': '[AU_DRIVER_LICENSE]',
        'AustralianPassportNumber': '[AU_PASSPORT]',
        'EUPassportNumber': '[EU_PASSPORT]',
        'EUDriverLicenseNumber': '[EU_DRIVER_LICENSE]',
        'EUSocialSecurityNumber': '[EU_SSN]',
        'EUTaxIdentificationNumber': '[EU_TIN]',
        'EUNationalIdentificationNumber': '[EU_NID]',
        'EUDebitCardNumber': '[EU_DEBIT_CARD]',
        'GermanyDriverLicenseNumber': '[DE_DRIVER_LICENSE]',
        'GermanyPassportNumber': '[DE_PASSPORT]',
        'GermanyIdentityCardNumber': '[DE_ID_CARD]',
        'GermanyTaxIdentificationNumber': '[DE_TAX_ID]',
        'FranceDriverLicenseNumber': '[FR_DRIVER_LICENSE]',
        'FranceHealthInsuranceNumber': '[FR_HEALTH]',
        'FranceNationalID': '[FR_NID]',
        'FrancePassportNumber': '[FR_PASSPORT]',
        'FranceSocialSecurityNumber': '[FR_SSN]',
        'FranceTaxIdentificationNumber': '[FR_TAX_ID]',
        'ItalyDriverLicenseNumber': '[IT_DRIVER_LICENSE]',
        'ItalyFiscalCode': '[IT_FISCAL_CODE]',
        'ItalyPassportNumber': '[IT_PASSPORT]',
        'ItalyValueAddedTaxNumber': '[IT_VAT]',
        'SpainSocialSecurityNumber': '[ES_SSN]',
        'SpainTaxIdentificationNumber': '[ES_TAX_ID]',
        'SpainDriverLicenseNumber': '[ES_DRIVER_LICENSE]',
        'SpainPassportNumber': '[ES_PASSPORT]',
    }
    
    def __init__(self, endpoint: str, key: str, config_manager: ColumnConfigManager = None):
        """
        Initialize the Azure PII Detector
        
        Args:
            endpoint: Azure Cognitive Services endpoint
            key: Azure Cognitive Services API key
            config_manager: Column configuration manager for custom rules
        """
        self.client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        self.cost_per_1000_chars = 0.001  # $1 per 1M characters
        self.config_manager = config_manager or ColumnConfigManager()
        
    def detect_pii_batch(self, texts: List[str], language: str = "en") -> List[Dict]:
        """
        Detect PII in a batch of texts
        
        Args:
            texts: List of text strings to analyze
            language: Language code (default: "en")
            
        Returns:
            List of dictionaries containing PII detection results
        """
        try:
            # Filter out empty texts
            valid_texts = [(i, text) for i, text in enumerate(texts) if text and not pd.isna(text)]
            
            if not valid_texts:
                return [{'text': text, 'redacted': text, 'entities': []} for text in texts]
            
            # Extract just the text values for API call
            text_values = [text for _, text in valid_texts]
            
            # Call Azure API
            response = self.client.recognize_pii_entities(
                documents=text_values,
                language=language,
                categories_filter=None,  # Detect all PII categories
                string_index_type="UnicodeCodePoint"
            )
            
            # Process results
            results = [{'text': text, 'redacted': text, 'entities': []} for text in texts]
            
            for idx, result in enumerate(response):
                original_idx = valid_texts[idx][0]
                original_text = valid_texts[idx][1]
                
                if result.is_error:
                    results[original_idx] = {
                        'text': original_text,
                        'redacted': original_text,
                        'entities': [],
                        'error': str(result.error)
                    }
                else:
                    redacted_text, entities = self._process_pii_result(original_text, result)
                    results[original_idx] = {
                        'text': original_text,
                        'redacted': redacted_text,
                        'entities': entities
                    }
            
            return results
            
        except Exception as e:
            print(f"Error in batch PII detection: {str(e)}")
            # Return original texts if error occurs
            return [{'text': text, 'redacted': text, 'entities': [], 'error': str(e)} for text in texts]
    
    def _process_pii_result(self, text: str, result) -> Tuple[str, List[Dict]]:
        """
        Process PII detection result and redact text
        
        Args:
            text: Original text
            result: Azure PII detection result
            
        Returns:
            Tuple of (redacted_text, entities_list)
        """
        entities = []
        redacted = text
        
        # Sort entities by position (reverse order to maintain positions)
        sorted_entities = sorted(result.entities, key=lambda e: e.offset, reverse=True)
        
        for entity in sorted_entities:
            # Get redaction label
            label = self.PII_ENTITY_MAP.get(entity.category, f'[{entity.category.upper()}]')
            
            # Store entity info
            entities.append({
                'text': entity.text,
                'category': entity.category,
                'subcategory': getattr(entity, 'subcategory', None),
                'confidence': entity.confidence_score,
                'offset': entity.offset,
                'length': entity.length,
                'redaction': label
            })
            
            # Redact in text (working backwards to maintain positions)
            redacted = redacted[:entity.offset] + label + redacted[entity.offset + entity.length:]
        
        return redacted, entities
    
    def detect_and_redact_dataframe(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None,
                                   batch_size: int = 25) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and redact PII in a pandas DataFrame with column-specific configuration
        
        Args:
            df: Input DataFrame
            columns: Specific columns to process (None = all columns)
            batch_size: Number of texts to process in each API call
            
        Returns:
            Tuple of (redacted_df, statistics_dict)
        """
        redacted_df = df.copy()
        columns_to_process = columns or df.columns.tolist()
        
        stats = {
            'total_cells': 0,
            'cells_with_pii': 0,
            'entities_found': {},
            'cost': 0.0,
            'columns_processed': {},
            'skipped_columns': [],
            'custom_rules_applied': 0
        }
        
        for col in columns_to_process:
            if col not in df.columns:
                continue
            
            # Get column configuration
            config = self.config_manager.get_column_config(col)
            
            # Check if detection is disabled for this column
            if config.detection_mode == DetectionMode.DISABLED:
                stats['skipped_columns'].append(col)
                continue
            
            col_data = df[col].astype(str).tolist()
            column_stats = {
                'total_cells': len(col_data),
                'redacted_cells': 0,
                'entities': {},
                'custom_patterns_matched': 0
            }
            
            # Process in batches
            all_results = []
            for i in range(0, len(col_data), batch_size):
                batch = col_data[i:i + batch_size]
                batch_results = self.detect_pii_batch_with_config(batch, col, config)
                all_results.extend(batch_results)
                
                # Calculate cost
                batch_chars = sum(len(str(text)) for text in batch if text and not pd.isna(text))
                stats['cost'] += (batch_chars / 1000) * self.cost_per_1000_chars
            
            # Apply redactions with column-specific logic
            for idx, result in enumerate(all_results):
                original_value = col_data[idx]
                
                # Apply custom patterns first (whitelist/blacklist)
                processed_value = self.apply_custom_patterns(original_value, config)
                
                # If value was changed by custom patterns, use that
                if processed_value != original_value:
                    redacted_df.at[idx, col] = processed_value
                    column_stats['custom_patterns_matched'] += 1
                    column_stats['redacted_cells'] += 1
                    stats['custom_rules_applied'] += 1
                else:
                    # Apply Azure detection with column configuration
                    final_value = self.apply_azure_detection_with_config(
                        original_value, result, config
                    )
                    redacted_df.at[idx, col] = final_value
                    
                    if final_value != original_value:
                        column_stats['redacted_cells'] += 1
                
                # Update statistics
                stats['total_cells'] += 1
                if result['entities']:
                    stats['cells_with_pii'] += 1
                    for entity in result['entities']:
                        category = entity['category']
                        stats['entities_found'][category] = stats['entities_found'].get(category, 0) + 1
                        column_stats['entities'][category] = column_stats['entities'].get(category, 0) + 1
            
            stats['columns_processed'][col] = column_stats
        
        return redacted_df, stats
    
    def get_supported_pii_categories(self) -> List[str]:
        """Get list of all supported PII categories"""
        return list(self.PII_ENTITY_MAP.keys())
    
    def estimate_cost(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> float:
        """
        Estimate the cost of processing a DataFrame
        
        Args:
            df: Input DataFrame
            columns: Specific columns to process (None = all columns)
            
        Returns:
            Estimated cost in USD
        """
        columns_to_process = columns or df.columns.tolist()
        total_chars = 0
        
        for col in columns_to_process:
            if col in df.columns:
                col_chars = df[col].astype(str).str.len().sum()
                total_chars += col_chars
        
        return (total_chars / 1000) * self.cost_per_1000_chars
    
    def detect_pii_batch_with_config(self, texts: List[str], column_name: str, 
                                   config: ColumnConfig, language: str = "en") -> List[Dict]:
        """
        Detect PII in a batch of texts with column-specific configuration
        
        Args:
            texts: List of text strings to analyze
            column_name: Name of the column being processed
            config: Column configuration
            language: Language code (default: "en")
            
        Returns:
            List of dictionaries containing PII detection results
        """
        # If detection is disabled, return original texts
        if config.detection_mode == DetectionMode.DISABLED:
            return [{'text': text, 'redacted': text, 'entities': []} for text in texts]
        
        # Use standard Azure detection
        return self.detect_pii_batch(texts, language)
    
    def apply_custom_patterns(self, text: str, config: ColumnConfig) -> str:
        """
        Apply custom whitelist and blacklist patterns
        
        Args:
            text: Input text
            config: Column configuration
            
        Returns:
            Text after applying custom patterns
        """
        if not text or pd.isna(text):
            return text
        
        text_str = str(text)
        
        # Check whitelist patterns first (these should NOT be redacted)
        for pattern in config.whitelist_patterns:
            if self._matches_pattern_obj(text_str, pattern):
                return text_str  # Return original text unchanged
        
        # Check blacklist patterns (these should ALWAYS be redacted)
        for pattern in config.blacklist_patterns:
            if self._matches_pattern_obj(text_str, pattern):
                return pattern.replacement
        
        return text_str  # No custom patterns matched
    
    def apply_azure_detection_with_config(self, original_text: str, azure_result: Dict, 
                                        config: ColumnConfig) -> str:
        """
        Apply Azure detection results with column configuration
        
        Args:
            original_text: Original text
            azure_result: Result from Azure PII detection
            config: Column configuration
            
        Returns:
            Final redacted text
        """
        if not azure_result.get('entities'):
            return original_text
        
        redacted_text = original_text
        
        # Sort entities by position (reverse order to maintain positions)
        sorted_entities = sorted(azure_result['entities'], key=lambda e: e['offset'], reverse=True)
        
        for entity in sorted_entities:
            entity_type = entity['category']
            confidence = entity['confidence']
            entity_text = entity['text']
            
            # Check if this entity should be redacted based on column config
            should_redact, replacement = self.config_manager.should_redact_entity(
                config.column_name, entity_type, confidence, entity_text
            )
            
            if should_redact:
                # Apply redaction (working backwards to maintain positions)
                start = entity['offset']
                end = start + entity['length']
                redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
        
        return redacted_text
    
    def _matches_pattern_obj(self, text: str, pattern) -> bool:
        """
        Check if text matches a pattern object (WhitelistPattern or BlacklistPattern)
        
        Args:
            text: Text to check
            pattern: Pattern object
            
        Returns:
            True if text matches pattern
        """
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
    
    def get_column_config_manager(self) -> ColumnConfigManager:
        """Get the column configuration manager"""
        return self.config_manager
    
    def set_column_config_manager(self, config_manager: ColumnConfigManager):
        """Set the column configuration manager"""
        self.config_manager = config_manager


class LocalPIIDetector:
    """
    Fallback local PII detector using regex patterns
    Use this when Azure AI is not available
    """
    
    PATTERNS = {
        '[EMAIL]': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '[PHONE]': r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        '[SSN]': r'\b\d{3}-\d{2}-\d{4}\b',
        '[CREDIT_CARD]': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        '[DATE]': [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        ],
        '[IP_ADDRESS]': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        '[URL]': r'https?://[^\s]+',
        '[ZIP_CODE]': r'\b\d{5}(?:-\d{4})?\b',
    }
    
    def redact_text(self, text: str) -> str:
        """
        Redact PII from text using regex patterns
        
        Args:
            text: Input text
            
        Returns:
            Redacted text
        """
        if not text or pd.isna(text):
            return text
        
        redacted = str(text)
        
        for label, patterns in self.PATTERNS.items():
            if isinstance(patterns, list):
                for pattern in patterns:
                    redacted = re.sub(pattern, label, redacted, flags=re.IGNORECASE)
            else:
                redacted = re.sub(patterns, label, redacted, flags=re.IGNORECASE)
        
        return redacted
    
    def redact_dataframe(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Redact PII from DataFrame columns
        
        Args:
            df: Input DataFrame
            columns: Specific columns to process (None = all columns)
            
        Returns:
            Redacted DataFrame
        """
        redacted_df = df.copy()
        columns_to_process = columns or df.columns.tolist()
        
        for col in columns_to_process:
            if col in df.columns:
                redacted_df[col] = df[col].apply(self.redact_text)
        
        return redacted_df


class EnhancedAzurePIIDetector(AzurePIIDetector):
    """
    Enhanced Azure PII Detector with column-specific configuration support
    and optional GPT-based validation
    """
    
    def __init__(self, endpoint: str, key: str, column_config_manager=None, use_gpt_validation: bool = False, openai_api_key: str = None):
        """
        Initialize the Enhanced Azure PII Detector
        
        Args:
            endpoint: Azure Cognitive Services endpoint
            key: Azure Cognitive Services API key
            column_config_manager: ColumnConfigManager instance for column-specific rules
            use_gpt_validation: Whether to use GPT for PII validation
            openai_api_key: OpenAI API key for GPT validation
        """
        super().__init__(endpoint, key)
        self.column_config_manager = column_config_manager
        self.use_gpt_validation = use_gpt_validation
        self.gpt_validator = None
        
        if use_gpt_validation:
            try:
                from gpt_validator import GPTPIIValidator
                self.gpt_validator = GPTPIIValidator(openai_api_key)
                print("✅ GPT validation enabled")
            except ImportError:
                print("⚠️ GPT validation not available - missing dependencies")
                self.use_gpt_validation = False
            except Exception as e:
                print(f"⚠️ GPT validation setup failed: {e}")
                self.use_gpt_validation = False
    
    def setup_gpt_validation(self, gpt_validator, api_key: str = None):
        """
        Setup or update GPT validation for the detector
        
        Args:
            gpt_validator: GPTPIIValidator instance
            api_key: API key (not used with Azure GPT validator)
        """
        self.gpt_validator = gpt_validator
        self.use_gpt_validation = gpt_validator is not None
        if self.use_gpt_validation:
            print("✅ GPT validation setup complete")
        else:
            print("⚠️ GPT validation disabled")
    
    def detect_and_redact_dataframe(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None,
                                   batch_size: int = 25) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and redact PII in a pandas DataFrame with column-specific configurations
        
        Args:
            df: Input DataFrame
            columns: Specific columns to process (None = all columns)
            batch_size: Number of texts to process in each API call
            
        Returns:
            Tuple of (redacted_df, statistics_dict)
        """
        redacted_df = df.copy()
        columns_to_process = columns or df.columns.tolist()
        
        stats = {
            'total_cells': 0,
            'cells_with_pii': 0,
            'entities_found': {},
            'cost': 0.0,
            'column_stats': {}
        }
        
        for col in columns_to_process:
            if col not in df.columns:
                continue
            
            # Get column-specific configuration
            col_config = None
            if self.column_config_manager:
                try:
                    col_config = self.column_config_manager.get_column_config(col)
                except:
                    col_config = None
            
            # Skip if column is disabled
            if col_config and not col_config.enabled:
                continue
            
            col_data = df[col].astype(str).tolist()
            
            # Process in batches with column-specific rules
            all_results = []
            for i in range(0, len(col_data), batch_size):
                batch = col_data[i:i + batch_size]
                
                if col_config:
                    # Apply column-specific processing
                    batch_results = self._process_batch_with_config(batch, col, col_config)
                else:
                    # Use standard processing
                    batch_results = self.detect_pii_batch(batch)
                
                all_results.extend(batch_results)
                
                # Calculate cost
                batch_chars = sum(len(str(text)) for text in batch if text and not pd.isna(text))
                stats['cost'] += (batch_chars / 1000) * self.cost_per_1000_chars
            
            # Apply redactions
            column_entities = 0
            for idx, result in enumerate(all_results):
                redacted_df.at[idx, col] = result['redacted']
                
                # Update statistics
                stats['total_cells'] += 1
                if result['entities']:
                    stats['cells_with_pii'] += 1
                    column_entities += len(result['entities'])
                    for entity in result['entities']:
                        category = entity['category']
                        stats['entities_found'][category] = stats['entities_found'].get(category, 0) + 1
            
            # Store column-specific stats
            stats['column_stats'][col] = {
                'entities_found': column_entities,
                'cells_processed': len(all_results)
            }
        
        return redacted_df, stats
    
    def _process_batch_with_config(self, texts: List[str], column_name: str, config) -> List[Dict]:
        """
        Process a batch of texts with column-specific configuration and optional GPT validation
        
        Args:
            texts: List of text strings to analyze
            column_name: Name of the column being processed
            config: ColumnConfig object with column-specific configuration
            
        Returns:
            List of dictionaries containing PII detection results
        """
        # Get standard Azure results first
        results = self.detect_pii_batch(texts)
        
        # Apply column-specific modifications and GPT validation
        for i, result in enumerate(results):
            if 'entities' not in result or not result['entities']:
                continue
            
            original_text = result['text']
            print(f"🔍 Processing text: '{original_text[:100]}...' with {len(result['entities'])} entities")
            
            # Debug: Show detected entities
            for entity in result['entities']:
                print(f"  📍 Entity: '{entity['text']}' (Type: {entity['category']}, Confidence: {entity.get('confidence_score', 'N/A')})")
            
            # Step 1: Apply GPT validation if enabled
            if self.use_gpt_validation and self.gpt_validator:
                try:
                    print(f"🤖 GPT validation: Processing {len(result['entities'])} entities for column '{column_name}'")
                    
                    # Determine context based on column name
                    context = self._get_validation_context(column_name)
                    
                    # Validate with GPT
                    validation_results = self.gpt_validator.validate_pii_detection(
                        original_text, result['entities'], context
                    )
                    
                    print(f"🤖 GPT validation: Received {len(validation_results)} validation results")
                    
                    # Apply GPT validation results
                    corrected_text, validated_entities = self.gpt_validator.apply_validation_results(
                        original_text, result['entities'], validation_results
                    )
                    
                    # Update entities list with GPT-validated entities
                    result['entities'] = validated_entities
                    print(f"🤖 GPT validation: Kept {len(validated_entities)} entities after validation")
                    
                except Exception as e:
                    print(f"⚠️ GPT validation error: {e}")
                    # Continue with original entities on error
                    
                    # Add validation info to result
                    result['gpt_validated'] = True
                    result['validation_results'] = validation_results
                    
                except Exception as e:
                    print(f"GPT validation failed for text: {str(e)}")
                    # Continue with original entities if GPT fails
                    result['gpt_validated'] = False
            else:
                result['gpt_validated'] = False
            
            # Step 2: Apply column-specific rules to remaining entities
            modified_entities = []
            
            # Apply whitelist patterns (remove entities that match whitelist)
            whitelist_patterns = [p.pattern for p in config.whitelist_patterns] if config.whitelist_patterns else []
            for entity in result['entities']:
                entity_text = entity['text']
                entity_category = entity['category']
                should_whitelist = False
                
                # Skip if entity type is excluded
                if config.is_entity_type_excluded(entity_category):
                    continue
                
                # Check whitelist patterns
                for pattern in whitelist_patterns:
                    try:
                        if re.search(pattern, entity_text, re.IGNORECASE):
                            should_whitelist = True
                            break
                    except re.error:
                        # If regex is invalid, try exact match
                        if pattern.lower() in entity_text.lower():
                            should_whitelist = True
                            break
                
                if not should_whitelist:
                    modified_entities.append(entity)
            
            # Apply blacklist patterns (add forced redactions)
            blacklist_patterns = [p.pattern for p in config.blacklist_patterns] if config.blacklist_patterns else []
            for pattern in blacklist_patterns:
                try:
                    matches = list(re.finditer(pattern, original_text, re.IGNORECASE))
                    for match in matches:
                        # Add as a forced entity
                        modified_entities.append({
                            'text': match.group(),
                            'category': 'FORCED_REDACTION',
                            'subcategory': None,
                            'confidence': 1.0,
                            'offset': match.start(),
                            'length': match.end() - match.start(),
                            'redaction': config.custom_redaction_label or '[REDACTED]'
                        })
                except re.error:
                    # If regex is invalid, try exact string match
                    if pattern.lower() in original_text.lower():
                        start_idx = original_text.lower().find(pattern.lower())
                        modified_entities.append({
                            'text': pattern,
                            'category': 'FORCED_REDACTION',
                            'subcategory': None,
                            'confidence': 1.0,
                            'offset': start_idx,
                            'length': len(pattern),
                            'redaction': config.custom_redaction_label or '[REDACTED]'
                        })
            
            # Apply sensitivity threshold
            sensitivity_threshold = config.sensitivity_threshold
            filtered_entities = [
                entity for entity in modified_entities 
                if entity['confidence'] >= sensitivity_threshold or entity['category'] == 'FORCED_REDACTION'
            ]
            
            # Rebuild redacted text with filtered entities
            redacted_text = original_text
            sorted_entities = sorted(filtered_entities, key=lambda e: e['offset'], reverse=True)
            
            for entity in sorted_entities:
                label = entity.get('redaction') or self.PII_ENTITY_MAP.get(entity['category'], f'[{entity["category"].upper()}]')
                if config.custom_redaction_label:
                    label = config.custom_redaction_label
                
                start = entity['offset']
                end = start + entity['length']
                redacted_text = redacted_text[:start] + label + redacted_text[end:]
            
            # Update result
            results[i]['entities'] = filtered_entities
            results[i]['redacted'] = redacted_text
        
        return results
    
    def _get_validation_context(self, column_name: str) -> str:
        """Determine validation context based on column name"""
        col_lower = column_name.lower()
        
        if any(keyword in col_lower for keyword in ['zendesk', 'ticket', 'support']):
            return "zendesk"
        elif any(keyword in col_lower for keyword in ['description', 'comment', 'note', 'body']):
            return "support_ticket"  
        elif any(keyword in col_lower for keyword in ['customer', 'client']):
            return "customer_data"
        else:
            return "general"
