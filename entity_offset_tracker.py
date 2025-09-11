#!/usr/bin/env python3
"""
Entity Offset Tracking System
Prevents entity position corruption through multiple text transformations
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class TrackedEntity:
    """Entity with position tracking through transformations"""
    text: str
    category: str
    original_offset: int
    original_length: int
    current_offset: int
    current_length: int
    confidence: float
    transformation_history: List[str]
    is_valid: bool = True
    replacement_text: Optional[str] = None


class EntityOffsetTracker:
    """
    Tracks entity positions through multiple text transformations
    Ensures accurate redaction even after text modifications
    """
    
    def __init__(self, original_text: str, original_entities: List[Dict]):
        """Initialize with original text and entities"""
        self.original_text = original_text
        self.current_text = original_text
        self.entities: List[TrackedEntity] = []
        
        # Convert original entities to tracked entities
        for entity in original_entities:
            tracked = TrackedEntity(
                text=entity['text'],
                category=entity['category'],
                original_offset=entity['offset'],
                original_length=entity['length'],
                current_offset=entity['offset'],
                current_length=entity['length'],
                confidence=entity.get('confidence', 0.0),
                transformation_history=[]
            )
            self.entities.append(tracked)
        
        # Sort entities by position for processing
        self.entities.sort(key=lambda e: e.current_offset)
    
    def remove_entities(self, entities_to_remove: List[str], reason: str) -> int:
        """
        Remove entities by text content and track the reason
        
        Args:
            entities_to_remove: List of entity texts to remove
            reason: Reason for removal (e.g., "GPT_validation", "ML_filter")
            
        Returns:
            Number of entities removed
        """
        removed_count = 0
        
        for entity in self.entities:
            if entity.text in entities_to_remove and entity.is_valid:
                entity.is_valid = False
                entity.transformation_history.append(f"REMOVED_{reason}")
                removed_count += 1
        
        return removed_count
    
    def add_forced_entities(self, new_entities: List[Dict], reason: str) -> int:
        """
        Add new entities (e.g., from blacklist patterns) and update positions
        
        Args:
            new_entities: List of entity dicts to add
            reason: Reason for addition (e.g., "blacklist_pattern")
            
        Returns:
            Number of entities added
        """
        added_count = 0
        
        for entity_dict in new_entities:
            # Verify entity position is still valid
            start = entity_dict['offset']
            end = start + entity_dict['length']
            
            if 0 <= start < len(self.current_text) and end <= len(self.current_text):
                actual_text = self.current_text[start:end]
                
                # Only add if text matches
                if actual_text.lower() == entity_dict['text'].lower():
                    tracked = TrackedEntity(
                        text=entity_dict['text'],
                        category=entity_dict['category'],
                        original_offset=start,
                        original_length=entity_dict['length'],
                        current_offset=start,
                        current_length=entity_dict['length'],
                        confidence=entity_dict.get('confidence', 1.0),
                        transformation_history=[f"ADDED_{reason}"],
                        replacement_text=entity_dict.get('redaction', f"[{entity_dict['category'].upper()}]")
                    )
                    self.entities.append(tracked)
                    added_count += 1
        
        # Re-sort entities by position
        self.entities.sort(key=lambda e: e.current_offset)
        return added_count
    
    def update_confidence_thresholds(self, threshold: float, reason: str) -> int:
        """
        Remove entities below confidence threshold
        
        Args:
            threshold: Minimum confidence to keep entity
            reason: Reason for filtering
            
        Returns:
            Number of entities filtered out
        """
        filtered_count = 0
        
        for entity in self.entities:
            if entity.is_valid and entity.confidence < threshold:
                entity.is_valid = False
                entity.transformation_history.append(f"LOW_CONFIDENCE_{reason}")
                filtered_count += 1
        
        return filtered_count
    
    def validate_entity_positions(self, reason: str = "position_validation") -> int:
        """
        Validate that all entities still point to correct text positions
        
        Args:
            reason: Reason for validation check
            
        Returns:
            Number of entities invalidated due to position errors
        """
        invalidated_count = 0
        
        for entity in self.entities:
            if not entity.is_valid:
                continue
            
            start = entity.current_offset
            end = start + entity.current_length
            
            # Check bounds
            if start < 0 or end > len(self.current_text):
                entity.is_valid = False
                entity.transformation_history.append(f"INVALID_BOUNDS_{reason}")
                invalidated_count += 1
                continue
            
            # Check if text still matches
            actual_text = self.current_text[start:end]
            if actual_text.lower() != entity.text.lower():
                # Try to find the text nearby (within 10 characters)
                found_position = self._find_text_near_position(entity.text, start, 10)
                
                if found_position is not None:
                    # Update position
                    entity.current_offset = found_position
                    entity.transformation_history.append(f"REPOSITIONED_{reason}")
                else:
                    # Cannot find the text - invalidate entity
                    entity.is_valid = False
                    entity.transformation_history.append(f"TEXT_NOT_FOUND_{reason}")
                    invalidated_count += 1
        
        return invalidated_count
    
    def _find_text_near_position(self, text: str, original_position: int, search_radius: int) -> Optional[int]:
        """Find text within search radius of original position"""
        text_lower = text.lower()
        current_lower = self.current_text.lower()
        
        # Search in expanding radius
        for radius in range(1, search_radius + 1):
            # Search backwards
            start_pos = max(0, original_position - radius)
            if start_pos + len(text) <= len(current_lower):
                if current_lower[start_pos:start_pos + len(text)] == text_lower:
                    return start_pos
            
            # Search forwards
            start_pos = original_position + radius
            if start_pos + len(text) <= len(current_lower):
                if current_lower[start_pos:start_pos + len(text)] == text_lower:
                    return start_pos
        
        return None
    
    def get_valid_entities(self) -> List[Dict]:
        """Get all valid entities as dictionary list"""
        valid_entities = []
        
        for entity in self.entities:
            if entity.is_valid:
                entity_dict = {
                    'text': entity.text,
                    'category': entity.category,
                    'offset': entity.current_offset,
                    'length': entity.current_length,
                    'confidence': entity.confidence,
                    'redaction': entity.replacement_text or f"[{entity.category.upper()}]"
                }
                valid_entities.append(entity_dict)
        
        return valid_entities
    
    def generate_redacted_text(self, custom_replacements: Dict[str, str] = None) -> str:
        """
        Generate final redacted text with all valid entities
        
        Args:
            custom_replacements: Custom replacement text for specific categories
            
        Returns:
            Redacted text string
        """
        # Get valid entities sorted by position (reverse order for processing)
        valid_entities = [e for e in self.entities if e.is_valid]
        valid_entities.sort(key=lambda e: e.current_offset, reverse=True)
        
        redacted_text = self.current_text
        
        for entity in valid_entities:
            start = entity.current_offset
            end = start + entity.current_length
            
            # Determine replacement text
            if custom_replacements and entity.category in custom_replacements:
                replacement = custom_replacements[entity.category]
            elif entity.replacement_text:
                replacement = entity.replacement_text
            else:
                replacement = f"[{entity.category.upper()}]"
            
            # Apply replacement
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
        
        return redacted_text
    
    def get_processing_summary(self) -> Dict:
        """Get summary of all entity processing"""
        total_entities = len(self.entities)
        valid_entities = len([e for e in self.entities if e.is_valid])
        invalid_entities = total_entities - valid_entities
        
        # Count transformations
        transformation_counts = {}
        for entity in self.entities:
            for transformation in entity.transformation_history:
                transformation_counts[transformation] = transformation_counts.get(transformation, 0) + 1
        
        return {
            'total_entities_processed': total_entities,
            'valid_entities_remaining': valid_entities,
            'entities_filtered_out': invalid_entities,
            'transformation_history': transformation_counts,
            'accuracy_preservation_rate': valid_entities / max(total_entities, 1)
        }
    
    def get_debug_info(self) -> Dict:
        """Get detailed debug information"""
        debug_info = {
            'original_text_length': len(self.original_text),
            'current_text_length': len(self.current_text),
            'entities': []
        }
        
        for i, entity in enumerate(self.entities):
            entity_info = {
                'index': i,
                'text': entity.text,
                'category': entity.category,
                'original_position': f"{entity.original_offset}-{entity.original_offset + entity.original_length}",
                'current_position': f"{entity.current_offset}-{entity.current_offset + entity.current_length}",
                'confidence': entity.confidence,
                'is_valid': entity.is_valid,
                'transformation_history': entity.transformation_history
            }
            
            # Add current text at position if valid
            if entity.is_valid and entity.current_offset < len(self.current_text):
                start = entity.current_offset
                end = min(start + entity.current_length, len(self.current_text))
                entity_info['current_text_at_position'] = self.current_text[start:end]
            
            debug_info['entities'].append(entity_info)
        
        return debug_info


# Integration helper functions
def create_tracker_from_azure_results(text: str, azure_result: Dict) -> EntityOffsetTracker:
    """Create tracker from Azure detection results"""
    entities = azure_result.get('entities', [])
    return EntityOffsetTracker(text, entities)


def apply_tracking_to_processing_pipeline(original_text: str, entities: List[Dict], 
                                        processing_steps: List[Tuple[str, callable]]) -> Tuple[str, List[Dict], Dict]:
    """
    Apply entity tracking through a processing pipeline
    
    Args:
        original_text: Original text
        entities: Original entities
        processing_steps: List of (step_name, processing_function) tuples
        
    Returns:
        Tuple of (final_text, final_entities, processing_summary)
    """
    tracker = EntityOffsetTracker(original_text, entities)
    
    for step_name, process_func in processing_steps:
        try:
            # Apply processing step
            process_func(tracker)
            
            # Validate positions after each step
            tracker.validate_entity_positions(f"after_{step_name}")
            
        except Exception as e:
            print(f"⚠️ Error in processing step '{step_name}': {e}")
            # Continue with remaining steps
    
    final_text = tracker.generate_redacted_text()
    final_entities = tracker.get_valid_entities()
    summary = tracker.get_processing_summary()
    
    return final_text, final_entities, summary


# Example usage
if __name__ == "__main__":
    # Test with sample data
    test_text = "Co-managed users can not see note that comes in internal only if the external user is not a contact"
    test_entities = [
        {'text': 'users', 'category': 'PersonType', 'offset': 11, 'length': 5, 'confidence': 0.8},
        {'text': 'user', 'category': 'PersonType', 'offset': 81, 'length': 4, 'confidence': 0.7},
        {'text': 'contact', 'category': 'PersonType', 'offset': 98, 'length': 7, 'confidence': 0.6}
    ]
    
    print("🧪 Testing Entity Offset Tracker")
    print(f"Original text: {test_text}")
    print(f"Original entities: {[e['text'] for e in test_entities]}")
    
    # Create tracker
    tracker = EntityOffsetTracker(test_text, test_entities)
    
    # Simulate processing steps
    tracker.remove_entities(['users', 'user'], 'GPT_validation')  # Remove false positives
    tracker.update_confidence_thresholds(0.65, 'confidence_filter')  # Remove low confidence
    
    # Validate positions
    invalidated = tracker.validate_entity_positions('final_check')
    
    # Generate results
    final_text = tracker.generate_redacted_text()
    final_entities = tracker.get_valid_entities()
    summary = tracker.get_processing_summary()
    
    print(f"\n📊 Processing Results:")
    print(f"Final text: {final_text}")
    print(f"Final entities: {[e['text'] for e in final_entities]}")
    print(f"Processing summary: {summary}")
    
    # Debug info
    debug = tracker.get_debug_info()
    print(f"\n🔍 Debug info: {debug}")
    
    print("✅ Test completed")