"""
GPT-based PII Validation System
Uses Azure OpenAI GPT to validate whether detected PII is actually sensitive
"""

import requests
import os
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

@dataclass
class ValidationResult:
    """Result of GPT validation"""
    is_real_pii: bool
    confidence: float
    explanation: str
    should_redact: bool
    suggested_action: str

class GPTPIIValidator:
    """
    Uses GPT-4o-mini to validate PII detection results
    Acts as a smart filter to reduce false positives
    """
    
    def __init__(self, azure_api_key: Optional[str] = None, azure_endpoint: Optional[str] = None, 
                 deployment_name: Optional[str] = None, api_version: Optional[str] = None):
        """
        Initialize GPT PII Validator with Azure OpenAI
        
        Args:
            azure_api_key: Azure API key (if None, loads from AZURE_KEY env var)
            azure_endpoint: Azure OpenAI endpoint (if None, loads from AZURE_GPT_ENDPOINT env var)
            deployment_name: Deployment name (if None, loads from AZURE_GPT_DEPLOYMENT env var)
            api_version: API version (if None, loads from AZURE_GPT_API_VERSION env var)
        """
        self.azure_api_key = azure_api_key or os.getenv('AZURE_KEY')
        self.azure_endpoint = azure_endpoint or os.getenv('AZURE_GPT_ENDPOINT', 'https://dgrif-malg2o2i-eastus2.cognitiveservices.azure.com')
        self.deployment_name = deployment_name or os.getenv('AZURE_GPT_DEPLOYMENT', 'gpt-5-chat')
        self.api_version = api_version or os.getenv('AZURE_GPT_API_VERSION', '2025-01-01-preview')
        
        if not self.azure_api_key:
            raise ValueError("Azure API key not found. Set AZURE_KEY environment variable or pass azure_api_key parameter.")
        
        self.base_url = f"{self.azure_endpoint}/openai/deployments/{self.deployment_name}/chat/completions"
        self.model = self.deployment_name
        self.cost_per_1k_tokens = 0.00015  # Approximate cost
        self.total_cost = 0.0
        
        # Validation prompts for different contexts
        self.base_prompt = """
You are a PII validation expert. Your job is to determine if text flagged as PII is actually sensitive personal information or just business terminology.

Context: This text was flagged by Azure AI as containing PII, but it might be a false positive.

Guidelines:
- REAL PII: Actual names, emails, phone numbers, addresses, SSNs, etc. of real people
- NOT PII: Job titles, user roles, business terms, product names, generic references
- BUSINESS TERMS: "users", "customers", "contacts", "agents", "admins" are usually NOT PII
- CONTEXT MATTERS: "external users" in support context is NOT PII

Examples:
- "John Smith called" → REAL PII (actual person name)
- "external users can't access" → NOT PII (user role/type)
- "contact form submission" → NOT PII (business process)
- "support agent assigned" → NOT PII (job role)
- "customer service team" → NOT PII (department)

Respond with JSON only:
{
    "is_real_pii": boolean,
    "confidence": 0.0-1.0,
    "explanation": "brief reason",
    "should_redact": boolean,
    "suggested_action": "keep_original|redact|partial_redact"
}
"""
    
    def validate_pii_detection(self, 
                              original_text: str, 
                              detected_entities: List[Dict],
                              context: str = "general") -> Dict[str, ValidationResult]:
        """
        Validate multiple PII detections in text using GPT
        
        Args:
            original_text: Original text before redaction
            detected_entities: List of detected PII entities from Azure
            context: Context hint (e.g., "support_ticket", "customer_data")
            
        Returns:
            Dictionary mapping entity text to ValidationResult
        """
        if not detected_entities:
            return {}
        
        results = {}
        
        # Group entities by type for batch processing
        entity_groups = {}
        for entity in detected_entities:
            entity_type = entity.get('category', 'Unknown')
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity)
        
        # Process each entity type
        for entity_type, entities in entity_groups.items():
            try:
                batch_results = self._validate_entity_batch(
                    original_text, entities, entity_type, context
                )
                results.update(batch_results)
            except Exception as e:
                print(f"Error validating {entity_type} entities: {e}")
                # Default to keeping redaction on error
                for entity in entities:
                    results[entity['text']] = ValidationResult(
                        is_real_pii=True,
                        confidence=0.5,
                        explanation=f"Validation failed: {str(e)}",
                        should_redact=True,
                        suggested_action="redact"
                    )
        
        return results
    
    def _validate_entity_batch(self, 
                              original_text: str,
                              entities: List[Dict],
                              entity_type: str,
                              context: str) -> Dict[str, ValidationResult]:
        """Validate a batch of entities of the same type"""
        
        # Create context-aware prompt
        context_prompt = self._get_context_prompt(context, entity_type)
        
        # Prepare entities for validation
        entity_texts = [entity['text'] for entity in entities]
        
        validation_prompt = f"""
{self.base_prompt}

{context_prompt}

Original text: "{original_text}"

Entity type flagged: {entity_type}
Entities to validate: {entity_texts}

For each entity, determine if it's real PII or a false positive.
"""
        
        try:
            print(f"🤖 Making Azure GPT API call to {self.base_url}")
            print(f"🤖 Validating {len(entity_texts)} entities of type {entity_type}")
            
            # Prepare Azure OpenAI request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.azure_api_key}"
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a PII validation expert."},
                    {"role": "user", "content": validation_prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.1  # Low temperature for consistent results
            }
            
            # Call Azure GPT
            response = requests.post(
                f"{self.base_url}?api-version={self.api_version}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"🤖 Azure GPT response status: {response.status_code}")
            
            if response.status_code != 200:
                raise Exception(f"Azure OpenAI request failed: {response.status_code} - {response.text}")
            
            response_data = response.json()
            
            # Calculate cost (approximate)
            tokens_used = response_data.get('usage', {}).get('total_tokens', 100)  # Fallback estimate
            cost = (tokens_used / 1000) * self.cost_per_1k_tokens
            self.total_cost += cost
            
            print(f"🤖 GPT validation cost: ${cost:.6f} (Total: ${self.total_cost:.6f})")
            
            # Parse response
            response_text = response_data['choices'][0]['message']['content']
            return self._parse_validation_response(response_text, entities)
            
        except Exception as e:
            print(f"GPT validation error: {e}")
            # Default to keeping redaction
            return {
                entity['text']: ValidationResult(
                    is_real_pii=True,
                    confidence=0.5,
                    explanation=f"GPT validation failed: {str(e)}",
                    should_redact=True,
                    suggested_action="redact"
                ) for entity in entities
            }
    
    def _get_context_prompt(self, context: str, entity_type: str) -> str:
        """Get context-specific validation prompt"""
        
        context_prompts = {
            "support_ticket": f"""
CONTEXT: Support ticket/customer service data
- "users", "customers", "contacts" are usually user TYPES, not individual people
- "external users", "internal users", "admin users" are ROLES, not PII
- "agent", "support staff", "customer service" are JOB TITLES, not PII
- Only actual individual names like "John Smith" should be considered PII
- System/role references should NOT be redacted

Entity type: {entity_type}
Special attention: PersonType entities are often false positives in support context.
""",
            
            "zendesk": f"""
CONTEXT: Zendesk support ticket data
- Very high likelihood of false positives for PersonType
- "Co-managed users", "external users", "end users" are USER CATEGORIES, not PII
- "contact", "customer", "agent" in business context are ROLES, not individual people
- "user interface", "user experience" are PRODUCT TERMS, not PII
- Only redact if it's clearly an individual person's name

Entity type: {entity_type}
""",
            
            "customer_data": f"""
CONTEXT: Customer database/CRM data
- Higher likelihood that Person entities are real individual names
- But still check for generic terms like "customer", "client", "user"
- Email domains and system emails may not be PII

Entity type: {entity_type}
""",
            
            "general": f"""
CONTEXT: General business data
- Consider business vs personal context
- Job titles, roles, and categories are typically not PII

Entity type: {entity_type}
"""
        }
        
        return context_prompts.get(context, context_prompts["general"])
    
    def _parse_validation_response(self, response_text: str, entities: List[Dict]) -> Dict[str, ValidationResult]:
        """Parse GPT response and create ValidationResult objects"""
        
        results = {}
        
        try:
            # Try to parse as JSON
            if response_text.strip().startswith('{'):
                # Single entity response
                parsed = json.loads(response_text)
                if entities:
                    entity_text = entities[0]['text']
                    results[entity_text] = ValidationResult(
                        is_real_pii=parsed.get('is_real_pii', True),
                        confidence=parsed.get('confidence', 0.5),
                        explanation=parsed.get('explanation', 'No explanation'),
                        should_redact=parsed.get('should_redact', True),
                        suggested_action=parsed.get('suggested_action', 'redact')
                    )
            
            elif response_text.strip().startswith('['):
                # Multiple entities response
                parsed_list = json.loads(response_text)
                for i, parsed in enumerate(parsed_list):
                    if i < len(entities):
                        entity_text = entities[i]['text']
                        results[entity_text] = ValidationResult(
                            is_real_pii=parsed.get('is_real_pii', True),
                            confidence=parsed.get('confidence', 0.5),
                            explanation=parsed.get('explanation', 'No explanation'),
                            should_redact=parsed.get('should_redact', True),
                            suggested_action=parsed.get('suggested_action', 'redact')
                        )
            
            else:
                # Try to extract JSON from text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    if entities:
                        entity_text = entities[0]['text']
                        results[entity_text] = ValidationResult(
                            is_real_pii=parsed.get('is_real_pii', True),
                            confidence=parsed.get('confidence', 0.5),
                            explanation=parsed.get('explanation', 'No explanation'),
                            should_redact=parsed.get('should_redact', True),
                            suggested_action=parsed.get('suggested_action', 'redact')
                        )
        
        except json.JSONDecodeError:
            # Fallback: parse as natural language
            for entity in entities:
                entity_text = entity['text']
                
                # Simple keyword analysis of GPT response
                response_lower = response_text.lower()
                entity_lower = entity_text.lower()
                
                # Look for clear indicators
                is_not_pii = any(phrase in response_lower for phrase in [
                    f"{entity_lower} is not pii",
                    f"{entity_lower} is not personal",
                    "false positive",
                    "not sensitive",
                    "business term",
                    "job title",
                    "role",
                    "category"
                ])
                
                is_pii = any(phrase in response_lower for phrase in [
                    f"{entity_lower} is pii",
                    f"{entity_lower} is personal",
                    "should be redacted",
                    "sensitive",
                    "individual name"
                ])
                
                if is_not_pii:
                    should_redact = False
                    confidence = 0.8
                elif is_pii:
                    should_redact = True
                    confidence = 0.8
                else:
                    should_redact = True  # Default to safe
                    confidence = 0.3
                
                results[entity_text] = ValidationResult(
                    is_real_pii=should_redact,
                    confidence=confidence,
                    explanation="Parsed from natural language response",
                    should_redact=should_redact,
                    suggested_action="redact" if should_redact else "keep_original"
                )
        
        # Ensure all entities have results
        for entity in entities:
            entity_text = entity['text']
            if entity_text not in results:
                results[entity_text] = ValidationResult(
                    is_real_pii=True,
                    confidence=0.5,
                    explanation="Could not parse GPT response",
                    should_redact=True,
                    suggested_action="redact"
                )
        
        return results
    
    def apply_validation_results(self, 
                                original_text: str,
                                detected_entities: List[Dict],
                                validation_results: Dict[str, ValidationResult]) -> Tuple[str, List[Dict]]:
        """
        Apply validation results to create a corrected redaction
        
        Args:
            original_text: Original text
            detected_entities: Original detected entities
            validation_results: GPT validation results
            
        Returns:
            Tuple of (corrected_text, filtered_entities)
        """
        
        # Filter entities based on validation
        valid_entities = []
        for entity in detected_entities:
            entity_text = entity['text']
            validation = validation_results.get(entity_text)
            
            if validation and validation.should_redact:
                valid_entities.append(entity)
            # Entities that shouldn't be redacted are filtered out
        
        # Rebuild redacted text with only validated entities
        corrected_text = original_text
        
        # Sort entities by position (reverse order to maintain positions)
        sorted_entities = sorted(valid_entities, key=lambda e: e['offset'], reverse=True)
        
        for entity in sorted_entities:
            start = entity['offset']
            end = start + entity['length']
            
            # Use original redaction label
            redaction_label = entity.get('redaction', f"[REDACTED_{entity['category'].upper()}]")
            
            corrected_text = corrected_text[:start] + redaction_label + corrected_text[end:]
        
        return corrected_text, valid_entities
    
    def get_cost_summary(self) -> Dict[str, float]:
        """Get cost summary for GPT validation"""
        return {
            'total_cost': self.total_cost,
            'cost_per_validation': self.cost_per_1k_tokens,
            'model': self.model
        }

# Quick test function
def test_validator():
    """Test the GPT validator with sample data"""
    
    # Sample problematic text
    test_text = "Co-managed users can not see note that comes in internal only if the external user is not a contact"
    
    # Sample detected entities (what Azure would return)
    test_entities = [
        {
            'text': 'users',
            'category': 'PersonType',
            'confidence': 0.8,
            'offset': 11,
            'length': 5,
            'redaction': '[REDACTED_PERSONTYPE]'
        },
        {
            'text': 'user',
            'category': 'PersonType', 
            'confidence': 0.7,
            'offset': 81,
            'length': 4,
            'redaction': '[REDACTED_PERSONTYPE]'
        },
        {
            'text': 'contact',
            'category': 'PersonType',
            'confidence': 0.6,
            'offset': 98,
            'length': 7,
            'redaction': '[REDACTED_PERSONTYPE]'
        }
    ]
    
    try:
        validator = GPTPIIValidator()
        
        print("Testing GPT PII Validator...")
        print(f"Original: {test_text}")
        print(f"Detected entities: {[e['text'] for e in test_entities]}")
        
        # Validate
        validation_results = validator.validate_pii_detection(
            test_text, test_entities, context="support_ticket"
        )
        
        print("\nValidation Results:")
        for entity_text, result in validation_results.items():
            print(f"  '{entity_text}': {'❌ NOT PII' if not result.should_redact else '✅ IS PII'} "
                  f"(confidence: {result.confidence:.2f}) - {result.explanation}")
        
        # Apply corrections
        corrected_text, filtered_entities = validator.apply_validation_results(
            test_text, test_entities, validation_results
        )
        
        print(f"\nCorrected: {corrected_text}")
        print(f"Cost: ${validator.get_cost_summary()['total_cost']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_validator()
