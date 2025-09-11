#!/usr/bin/env python3
"""
Advanced Confidence Scoring and Feedback Loop System
Dynamic confidence adjustment based on user feedback and performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle
import hashlib
from collections import defaultdict, deque
import statistics

# ML imports for confidence modeling
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@dataclass
class ConfidenceSignal:
    """Individual signal that affects confidence scoring"""
    signal_name: str
    value: float
    weight: float
    source: str  # 'azure', 'gpt', 'ml', 'pattern', 'feedback'
    timestamp: datetime
    context: Dict


@dataclass
class ConfidenceScore:
    """Comprehensive confidence score for a detection"""
    entity_text: str
    entity_type: str
    base_confidence: float
    adjusted_confidence: float
    signals: List[ConfidenceSignal]
    factors: Dict[str, float]
    reliability_score: float
    recommendation: str  # 'redact', 'keep', 'review'


@dataclass
class FeedbackPattern:
    """Pattern learned from user feedback"""
    pattern_type: str  # 'text_pattern', 'context_pattern', 'entity_pattern'
    pattern_value: str
    confidence_adjustment: float
    feedback_count: int
    accuracy_improvement: float
    last_updated: datetime


class AdvancedConfidenceScorer:
    """
    Advanced confidence scoring system with dynamic adjustment based on feedback
    Uses multiple signals and machine learning to optimize detection confidence
    """
    
    def __init__(self, db_path: str = "confidence_scoring.db"):
        """Initialize confidence scoring system"""
        self.db_path = db_path
        self.ml_available = ML_AVAILABLE
        
        # Confidence models
        self.confidence_predictor = None
        self.reliability_predictor = None
        self.feature_scaler = None
        
        # Learned patterns from feedback
        self.feedback_patterns: Dict[str, FeedbackPattern] = {}
        self.entity_confidence_history: Dict[str, List[float]] = defaultdict(list)
        self.context_confidence_adjustments: Dict[str, float] = {}
        
        # Signal weights (dynamic, updated based on performance)
        self.signal_weights = {
            'azure_confidence': 0.4,
            'gpt_validation': 0.25,
            'pattern_match': 0.15,
            'context_relevance': 0.1,
            'historical_accuracy': 0.1
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high_confidence': 0.95,
            'medium_confidence': 0.80,
            'low_confidence': 0.60,
            'review_threshold': 0.70
        }
        
        # Performance tracking
        self.scoring_performance = {
            'total_scores_generated': 0,
            'accuracy_improvements': [],
            'false_positive_reductions': [],
            'user_agreement_rate': 0.0
        }
        
        # Initialize system
        self._initialize_database()
        if self.ml_available:
            self._initialize_ml_models()
        
        print("✅ Advanced Confidence Scorer initialized")
        print(f"   🧠 ML Models: {'✅ Available' if self.ml_available else '❌ Unavailable'}")
        print(f"   📊 Confidence thresholds: {self.confidence_thresholds}")
    
    def _initialize_database(self):
        """Initialize database for confidence scoring data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Confidence scores table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS confidence_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    entity_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    base_confidence REAL NOT NULL,
                    adjusted_confidence REAL NOT NULL,
                    reliability_score REAL NOT NULL,
                    signals TEXT NOT NULL,
                    factors TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    session_id TEXT
                )
            """)
            
            # Feedback patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_value TEXT NOT NULL,
                    confidence_adjustment REAL NOT NULL,
                    feedback_count INTEGER NOT NULL,
                    accuracy_improvement REAL NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)
            
            # Confidence signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS confidence_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    signal_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    weight REAL NOT NULL,
                    source TEXT NOT NULL,
                    context TEXT NOT NULL,
                    entity_text TEXT NOT NULL
                )
            """)
            
            # Performance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scoring_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    context TEXT
                )
            """)
            
            conn.commit()
    
    def _initialize_ml_models(self):
        """Initialize ML models for confidence prediction"""
        if not self.ml_available:
            return
        
        try:
            # Confidence predictor - predicts adjusted confidence
            self.confidence_predictor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Reliability predictor - predicts how reliable a confidence score is
            self.reliability_predictor = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            # Feature scaler
            self.feature_scaler = StandardScaler()
            
            print("✅ ML models for confidence scoring initialized")
            
        except Exception as e:
            print(f"⚠️ ML model initialization failed: {e}")
            self.ml_available = False
    
    def calculate_confidence_score(self, 
                                 entity_text: str,
                                 entity_type: str,
                                 base_confidence: float,
                                 context: Dict,
                                 session_id: str = None) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score for an entity detection
        
        Args:
            entity_text: The detected entity text
            entity_type: The type of entity (Person, Email, etc.)
            base_confidence: Base confidence from Azure AI
            context: Context information (column name, full text, etc.)
            session_id: Processing session ID
            
        Returns:
            ConfidenceScore object with detailed scoring information
        """
        
        signals = []
        
        # Signal 1: Azure base confidence
        azure_signal = ConfidenceSignal(
            signal_name='azure_confidence',
            value=base_confidence,
            weight=self.signal_weights['azure_confidence'],
            source='azure',
            timestamp=datetime.now(),
            context={'entity_type': entity_type}
        )
        signals.append(azure_signal)
        
        # Signal 2: GPT validation (if available)
        gpt_signal = self._calculate_gpt_signal(entity_text, entity_type, context)
        if gpt_signal:
            signals.append(gpt_signal)
        
        # Signal 3: Pattern matching confidence
        pattern_signal = self._calculate_pattern_signal(entity_text, entity_type, context)
        signals.append(pattern_signal)
        
        # Signal 4: Context relevance
        context_signal = self._calculate_context_signal(entity_text, entity_type, context)
        signals.append(context_signal)
        
        # Signal 5: Historical accuracy
        historical_signal = self._calculate_historical_signal(entity_text, entity_type)
        signals.append(historical_signal)
        
        # Signal 6: Learned feedback patterns
        feedback_signal = self._calculate_feedback_signal(entity_text, entity_type, context)
        if feedback_signal:
            signals.append(feedback_signal)
        
        # Calculate adjusted confidence
        adjusted_confidence = self._calculate_adjusted_confidence(signals, base_confidence)
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(signals, context)
        
        # Determine recommendation
        recommendation = self._determine_recommendation(adjusted_confidence, reliability_score)
        
        # Create confidence score object
        confidence_score = ConfidenceScore(
            entity_text=entity_text,
            entity_type=entity_type,
            base_confidence=base_confidence,
            adjusted_confidence=adjusted_confidence,
            signals=signals,
            factors=self._extract_factors(signals),
            reliability_score=reliability_score,
            recommendation=recommendation
        )
        
        # Save to database
        self._save_confidence_score(confidence_score, session_id)
        
        # Update performance tracking
        self.scoring_performance['total_scores_generated'] += 1
        
        return confidence_score
    
    def _calculate_gpt_signal(self, entity_text: str, entity_type: str, context: Dict) -> Optional[ConfidenceSignal]:
        """Calculate GPT validation signal"""
        # This would integrate with GPT validation results
        # For now, return a placeholder
        
        # Check if this entity type commonly has false positives
        false_positive_prone_types = ['PersonType', 'Organization', 'Event']
        
        if entity_type in false_positive_prone_types:
            # Lower confidence for commonly false positive types
            gpt_confidence = 0.6
        else:
            # Higher confidence for clearly PII types
            gpt_confidence = 0.9
        
        return ConfidenceSignal(
            signal_name='gpt_validation',
            value=gpt_confidence,
            weight=self.signal_weights['gpt_validation'],
            source='gpt',
            timestamp=datetime.now(),
            context={'validation_type': 'heuristic'}
        )
    
    def _calculate_pattern_signal(self, entity_text: str, entity_type: str, context: Dict) -> ConfidenceSignal:
        """Calculate pattern matching confidence signal"""
        text_lower = entity_text.lower()
        
        # Business terms that are likely false positives
        business_terms = {
            'user', 'users', 'customer', 'customers', 'client', 'clients',
            'contact', 'contacts', 'agent', 'agents', 'admin', 'admins',
            'support', 'service', 'team', 'staff', 'system', 'automated'
        }
        
        # Check for business term patterns
        if entity_type in ['PersonType', 'Person'] and text_lower in business_terms:
            pattern_confidence = 0.2  # Low confidence for business terms
        elif entity_type == 'Email' and any(term in text_lower for term in ['noreply', 'support', 'admin']):
            pattern_confidence = 0.3  # Low confidence for system emails
        elif entity_type in ['Person'] and len(entity_text.split()) == 1 and text_lower in business_terms:
            pattern_confidence = 0.1  # Very low confidence for single business words
        else:
            pattern_confidence = 0.8  # Default higher confidence
        
        return ConfidenceSignal(
            signal_name='pattern_match',
            value=pattern_confidence,
            weight=self.signal_weights['pattern_match'],
            source='pattern',
            timestamp=datetime.now(),
            context={'pattern_type': 'business_term_check'}
        )
    
    def _calculate_context_signal(self, entity_text: str, entity_type: str, context: Dict) -> ConfidenceSignal:
        """Calculate context relevance signal"""
        column_name = context.get('column_name', '').lower()
        full_text = context.get('full_text', '').lower()
        
        # Context-based confidence adjustments
        context_confidence = 0.7  # Default
        
        # Support/ticket context adjustments
        if any(term in column_name for term in ['subject', 'description', 'comment']):
            if entity_type == 'PersonType':
                context_confidence = 0.3  # Low confidence for PersonType in support context
            elif entity_type == 'Person' and entity_text.lower() in ['user', 'customer', 'agent']:
                context_confidence = 0.2  # Very low for role-like terms
        
        # Email column context
        elif 'email' in column_name:
            if entity_type == 'Email':
                context_confidence = 0.9  # High confidence for emails in email columns
            elif entity_type == 'Person':
                context_confidence = 0.4  # Lower confidence for names in email columns
        
        # Tags/category context
        elif any(term in column_name for term in ['tag', 'category', 'type', 'status']):
            context_confidence = 0.2  # Low confidence for most entities in categorical columns
        
        return ConfidenceSignal(
            signal_name='context_relevance',
            value=context_confidence,
            weight=self.signal_weights['context_relevance'],
            source='context',
            timestamp=datetime.now(),
            context={'column_name': column_name}
        )
    
    def _calculate_historical_signal(self, entity_text: str, entity_type: str) -> ConfidenceSignal:
        """Calculate historical accuracy signal"""
        # Get historical confidence for this entity
        entity_key = f"{entity_type}:{entity_text.lower()}"
        historical_confidences = self.entity_confidence_history.get(entity_key, [])
        
        if historical_confidences:
            # Calculate average historical confidence
            avg_historical = statistics.mean(historical_confidences[-10:])  # Last 10 occurrences
            historical_confidence = avg_historical
        else:
            # No history, use entity type average
            historical_confidence = self._get_entity_type_average_confidence(entity_type)
        
        return ConfidenceSignal(
            signal_name='historical_accuracy',
            value=historical_confidence,
            weight=self.signal_weights['historical_accuracy'],
            source='historical',
            timestamp=datetime.now(),
            context={'history_count': len(historical_confidences)}
        )
    
    def _calculate_feedback_signal(self, entity_text: str, entity_type: str, context: Dict) -> Optional[ConfidenceSignal]:
        """Calculate feedback-based signal"""
        # Check learned patterns
        text_lower = entity_text.lower()
        
        for pattern_key, pattern in self.feedback_patterns.items():
            if pattern.pattern_type == 'text_pattern' and pattern.pattern_value == text_lower:
                return ConfidenceSignal(
                    signal_name='feedback_pattern',
                    value=max(0.0, min(1.0, 0.5 + pattern.confidence_adjustment)),
                    weight=0.15,  # Additional weight for feedback
                    source='feedback',
                    timestamp=datetime.now(),
                    context={'pattern_key': pattern_key, 'feedback_count': pattern.feedback_count}
                )
        
        return None
    
    def _calculate_adjusted_confidence(self, signals: List[ConfidenceSignal], base_confidence: float) -> float:
        """Calculate final adjusted confidence using weighted signals with validation"""
        
        # Validate inputs
        if not signals or base_confidence < 0 or base_confidence > 1:
            return max(0.0, min(1.0, base_confidence))
        
        # Use ML model if available and properly trained
        if (self.ml_available and self.confidence_predictor and 
            hasattr(self.confidence_predictor, 'n_features_in_')):
            try:
                features = self._extract_features_for_ml(signals, base_confidence)
                if len(features) == self.confidence_predictor.n_features_in_:
                    scaled_features = self.feature_scaler.transform([features])
                    predicted_confidence = self.confidence_predictor.predict(scaled_features)[0]
                    # Ensure ML prediction is reasonable (within 0.3 of base confidence)
                    if abs(predicted_confidence - base_confidence) <= 0.3:
                        return max(0.0, min(1.0, predicted_confidence))
            except Exception as e:
                # ML prediction failed, fall back to weighted approach
                pass
        
        # Enhanced weighted average approach with signal validation
        total_weight = 0
        weighted_sum = 0
        valid_signals = 0
        
        for signal in signals:
            # Validate signal values
            if 0.0 <= signal.value <= 1.0 and signal.weight > 0:
                weighted_sum += signal.value * signal.weight
                total_weight += signal.weight
                valid_signals += 1
        
        if total_weight > 0 and valid_signals > 0:
            raw_confidence = weighted_sum / total_weight
            
            # Apply conservative adjustment - prevent over-inflation
            # If signals disagree significantly, reduce confidence
            signal_values = [s.value for s in signals if 0.0 <= s.value <= 1.0]
            if len(signal_values) > 1:
                signal_variance = np.var(signal_values)
                # High variance means signals disagree - reduce confidence
                variance_penalty = min(0.2, signal_variance * 0.5)
                raw_confidence = max(base_confidence * 0.8, raw_confidence - variance_penalty)
            
            # Ensure adjusted confidence doesn't exceed base confidence by too much
            max_boost = 0.15  # Maximum 15% boost over base confidence
            adjusted_confidence = min(raw_confidence, base_confidence + max_boost)
            
            # Final bounds checking
            return max(0.0, min(1.0, adjusted_confidence))
        else:
            # No valid signals, return base confidence
            return max(0.0, min(1.0, base_confidence))
    
    def _calculate_reliability_score(self, signals: List[ConfidenceSignal], context: Dict) -> float:
        """Calculate reliability score for the confidence assessment"""
        
        # Factors that affect reliability
        signal_count = len(signals)
        signal_variance = np.var([s.value for s in signals]) if signals else 0
        
        # More signals generally mean higher reliability
        signal_reliability = min(1.0, signal_count / 6.0)  # 6 is our target signal count
        
        # Lower variance means signals agree more
        agreement_reliability = 1.0 - min(1.0, signal_variance * 2)
        
        # Context completeness
        context_completeness = len(context) / 5.0  # Assuming 5 key context fields
        context_reliability = min(1.0, context_completeness)
        
        # Average reliability
        overall_reliability = (signal_reliability + agreement_reliability + context_reliability) / 3.0
        
        return max(0.0, min(1.0, overall_reliability))
    
    def _determine_recommendation(self, confidence: float, reliability: float) -> str:
        """Determine recommendation based on confidence and reliability"""
        
        # High confidence and high reliability
        if confidence >= self.confidence_thresholds['high_confidence'] and reliability >= 0.8:
            return 'redact'
        
        # Low confidence or low reliability
        elif confidence < self.confidence_thresholds['low_confidence'] or reliability < 0.5:
            return 'keep'
        
        # Medium confidence or uncertain cases
        elif confidence >= self.confidence_thresholds['review_threshold']:
            return 'review'
        
        else:
            return 'keep'
    
    def add_user_feedback(self, entity_text: str, entity_type: str, 
                         user_decision: str, confidence_score: ConfidenceScore,
                         context: Dict = None):
        """Add user feedback to improve confidence scoring"""
        
        # Update entity confidence history
        entity_key = f"{entity_type}:{entity_text.lower()}"
        
        # Determine actual confidence based on user decision
        if user_decision == 'correct_redaction':
            actual_confidence = 1.0
        elif user_decision == 'false_positive':
            actual_confidence = 0.0
        elif user_decision == 'uncertain':
            actual_confidence = 0.5
        else:
            actual_confidence = confidence_score.adjusted_confidence
        
        # Add to history
        self.entity_confidence_history[entity_key].append(actual_confidence)
        
        # Keep only last 50 entries per entity
        if len(self.entity_confidence_history[entity_key]) > 50:
            self.entity_confidence_history[entity_key] = self.entity_confidence_history[entity_key][-50:]
        
        # Learn patterns from feedback
        self._learn_feedback_patterns(entity_text, entity_type, user_decision, confidence_score, context)
        
        # Update signal weights based on performance
        self._update_signal_weights(confidence_score, user_decision)
        
        # Retrain ML models if enough feedback
        if self.scoring_performance['total_scores_generated'] % 20 == 0:
            self._retrain_confidence_models()
        
        print(f"📚 Feedback added for '{entity_text}' ({entity_type}): {user_decision}")
    
    def _learn_feedback_patterns(self, entity_text: str, entity_type: str, 
                                user_decision: str, confidence_score: ConfidenceScore,
                                context: Dict = None):
        """Learn patterns from user feedback"""
        
        # Text pattern learning
        text_pattern_key = f"text:{entity_text.lower()}"
        
        if text_pattern_key not in self.feedback_patterns:
            self.feedback_patterns[text_pattern_key] = FeedbackPattern(
                pattern_type='text_pattern',
                pattern_value=entity_text.lower(),
                confidence_adjustment=0.0,
                feedback_count=0,
                accuracy_improvement=0.0,
                last_updated=datetime.now()
            )
        
        pattern = self.feedback_patterns[text_pattern_key]
        pattern.feedback_count += 1
        pattern.last_updated = datetime.now()
        
        # Adjust confidence based on feedback
        if user_decision == 'false_positive':
            pattern.confidence_adjustment -= 0.1
        elif user_decision == 'correct_redaction':
            pattern.confidence_adjustment += 0.05
        
        # Bounds checking
        pattern.confidence_adjustment = max(-0.5, min(0.5, pattern.confidence_adjustment))
        
        # Context pattern learning
        if context and 'column_name' in context:
            context_pattern_key = f"context:{entity_type}:{context['column_name']}"
            
            if context_pattern_key not in self.feedback_patterns:
                self.feedback_patterns[context_pattern_key] = FeedbackPattern(
                    pattern_type='context_pattern',
                    pattern_value=f"{entity_type}:{context['column_name']}",
                    confidence_adjustment=0.0,
                    feedback_count=0,
                    accuracy_improvement=0.0,
                    last_updated=datetime.now()
                )
            
            context_pattern = self.feedback_patterns[context_pattern_key]
            context_pattern.feedback_count += 1
            context_pattern.last_updated = datetime.now()
            
            if user_decision == 'false_positive':
                context_pattern.confidence_adjustment -= 0.05
            elif user_decision == 'correct_redaction':
                context_pattern.confidence_adjustment += 0.02
            
            context_pattern.confidence_adjustment = max(-0.3, min(0.3, context_pattern.confidence_adjustment))
        
        # Save patterns to database
        self._save_feedback_patterns()
    
    def _update_signal_weights(self, confidence_score: ConfidenceScore, user_decision: str):
        """Update signal weights based on performance with validation and limits"""
        
        # Validate inputs
        if not confidence_score or not confidence_score.signals or not user_decision:
            return
        
        # Map user decision to actual confidence with more nuanced scoring
        decision_mapping = {
            'correct_redaction': 1.0,
            'false_positive': 0.0,
            'false_negative': 1.0,  # Should have been redacted
            'uncertain': 0.5,
            'partial_correct': 0.7
        }
        
        actual_confidence = decision_mapping.get(user_decision, 0.5)
        
        # Track signal performance over time
        if not hasattr(self, 'signal_performance_history'):
            self.signal_performance_history = defaultdict(list)
        
        # Calculate signal accuracy and update history
        signal_accuracies = {}
        for signal in confidence_score.signals:
            if signal.signal_name in self.signal_weights:
                # Calculate how close signal was to actual outcome
                error = abs(signal.value - actual_confidence)
                accuracy = 1.0 - error  # Higher is better
                signal_accuracies[signal.signal_name] = accuracy
                
                # Add to performance history (keep last 50 measurements)
                self.signal_performance_history[signal.signal_name].append(accuracy)
                if len(self.signal_performance_history[signal.signal_name]) > 50:
                    self.signal_performance_history[signal.signal_name].pop(0)
        
        # Update weights based on recent performance (only if we have enough data)
        min_samples = 5
        for signal_name in self.signal_weights:
            if len(self.signal_performance_history[signal_name]) >= min_samples:
                # Calculate recent average performance
                recent_performance = np.mean(self.signal_performance_history[signal_name][-10:])
                overall_performance = np.mean(self.signal_performance_history[signal_name])
                
                # Very conservative weight adjustment (max 1% change)
                if recent_performance > 0.7:  # Good performance
                    adjustment = min(0.01, (recent_performance - 0.7) * 0.05)
                elif recent_performance < 0.4:  # Poor performance
                    adjustment = max(-0.01, (recent_performance - 0.4) * 0.05)
                else:
                    adjustment = 0
                
                # Apply adjustment with bounds checking
                new_weight = self.signal_weights[signal_name] + adjustment
                self.signal_weights[signal_name] = max(0.05, min(0.6, new_weight))  # Keep weights reasonable
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.signal_weights.values())
        if total_weight > 0:
            for key in self.signal_weights:
                self.signal_weights[key] /= total_weight
        
        # Log significant weight changes
        if hasattr(self, '_last_weights'):
            for signal_name, new_weight in self.signal_weights.items():
                old_weight = self._last_weights.get(signal_name, new_weight)
                if abs(new_weight - old_weight) > 0.05:
                    print(f"📊 Signal weight updated: {signal_name} {old_weight:.3f} → {new_weight:.3f}")
        
        self._last_weights = self.signal_weights.copy()
    
    def _retrain_confidence_models(self):
        """Retrain ML models with accumulated feedback"""
        if not self.ml_available:
            return
        
        try:
            # Get training data from database
            training_data = self._prepare_ml_training_data()
            
            if len(training_data) < 10:  # Need minimum samples
                return
            
            X, y = zip(*training_data)
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.feature_scaler.fit(X_train)
            X_train_scaled = self.feature_scaler.transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train confidence predictor
            self.confidence_predictor.fit(X_train_scaled, y_train)
            
            # Evaluate performance
            y_pred = self.confidence_predictor.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"🧠 ML models retrained - R²: {r2:.3f}, MSE: {mse:.3f}")
            
        except Exception as e:
            print(f"⚠️ ML model retraining failed: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Get confidence scoring performance metrics"""
        
        # Calculate recent performance
        recent_feedback_count = sum(len(history) for history in self.entity_confidence_history.values())
        
        # Calculate pattern learning effectiveness
        pattern_count = len(self.feedback_patterns)
        active_patterns = len([p for p in self.feedback_patterns.values() if p.feedback_count >= 3])
        
        return {
            'total_scores_generated': self.scoring_performance['total_scores_generated'],
            'entities_with_feedback': len(self.entity_confidence_history),
            'total_feedback_count': recent_feedback_count,
            'learned_patterns': pattern_count,
            'active_patterns': active_patterns,
            'ml_model_trained': self.ml_available and hasattr(self.confidence_predictor, 'predict'),
            'signal_weights': self.signal_weights.copy(),
            'confidence_thresholds': self.confidence_thresholds.copy()
        }
    
    def export_learned_patterns(self, filepath: str):
        """Export learned patterns for backup or sharing"""
        export_data = {
            'feedback_patterns': {k: asdict(v) for k, v in self.feedback_patterns.items()},
            'entity_confidence_history': {k: v for k, v in self.entity_confidence_history.items()},
            'signal_weights': self.signal_weights,
            'confidence_thresholds': self.confidence_thresholds,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📤 Learned patterns exported to {filepath}")
    
    def import_learned_patterns(self, filepath: str):
        """Import learned patterns from backup"""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            # Import feedback patterns
            for key, pattern_dict in import_data.get('feedback_patterns', {}).items():
                self.feedback_patterns[key] = FeedbackPattern(**pattern_dict)
            
            # Import confidence history
            self.entity_confidence_history.update(import_data.get('entity_confidence_history', {}))
            
            # Import weights and thresholds
            self.signal_weights.update(import_data.get('signal_weights', {}))
            self.confidence_thresholds.update(import_data.get('confidence_thresholds', {}))
            
            print(f"📥 Learned patterns imported from {filepath}")
            
        except Exception as e:
            print(f"⚠️ Failed to import patterns: {e}")
    
    # Helper methods
    def _extract_factors(self, signals: List[ConfidenceSignal]) -> Dict[str, float]:
        """Extract factor analysis from signals"""
        factors = {}
        for signal in signals:
            factors[signal.signal_name] = signal.value * signal.weight
        return factors
    
    def _extract_features_for_ml(self, signals: List[ConfidenceSignal], base_confidence: float) -> List[float]:
        """Extract features for ML model"""
        features = [base_confidence]
        
        # Add signal values
        signal_dict = {s.signal_name: s.value for s in signals}
        expected_signals = ['azure_confidence', 'gpt_validation', 'pattern_match', 'context_relevance', 'historical_accuracy']
        
        for signal_name in expected_signals:
            features.append(signal_dict.get(signal_name, 0.5))  # Default to 0.5 if missing
        
        # Add meta features
        features.extend([
            len(signals),  # Number of signals
            np.var([s.value for s in signals]) if signals else 0,  # Signal variance
            np.mean([s.weight for s in signals]) if signals else 0  # Average weight
        ])
        
        return features
    
    def _get_entity_type_average_confidence(self, entity_type: str) -> float:
        """Get average confidence for an entity type"""
        type_confidences = []
        
        for key, confidences in self.entity_confidence_history.items():
            if key.startswith(f"{entity_type}:"):
                type_confidences.extend(confidences)
        
        if type_confidences:
            return statistics.mean(type_confidences)
        else:
            # Default confidences by entity type
            defaults = {
                'Email': 0.9,
                'PhoneNumber': 0.85,
                'Person': 0.8,
                'SSN': 0.95,
                'CreditCardNumber': 0.95,
                'PersonType': 0.3,  # Often false positive
                'Organization': 0.7,
                'Location': 0.75
            }
            return defaults.get(entity_type, 0.7)
    
    def _prepare_ml_training_data(self) -> List[Tuple[List[float], float]]:
        """Prepare training data for ML models"""
        training_data = []
        
        # This would extract training data from stored confidence scores and feedback
        # For now, return empty list
        return training_data
    
    def _save_confidence_score(self, score: ConfidenceScore, session_id: str = None):
        """Save confidence score to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO confidence_scores
                (timestamp, entity_text, entity_type, base_confidence, adjusted_confidence,
                 reliability_score, signals, factors, recommendation, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                score.entity_text,
                score.entity_type,
                score.base_confidence,
                score.adjusted_confidence,
                score.reliability_score,
                json.dumps([asdict(s) for s in score.signals], default=str),
                json.dumps(score.factors),
                score.recommendation,
                session_id
            ))
            conn.commit()
    
    def _save_feedback_patterns(self):
        """Save feedback patterns to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for key, pattern in self.feedback_patterns.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO feedback_patterns
                    (pattern_type, pattern_value, confidence_adjustment, feedback_count,
                     accuracy_improvement, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_type,
                    pattern.pattern_value,
                    pattern.confidence_adjustment,
                    pattern.feedback_count,
                    pattern.accuracy_improvement,
                    pattern.last_updated.isoformat()
                ))
            
            conn.commit()


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Advanced Confidence Scorer - Example Usage")
    
    # Initialize scorer
    scorer = AdvancedConfidenceScorer("test_confidence.db")
    
    # Test confidence scoring
    test_cases = [
        {
            'entity_text': 'user',
            'entity_type': 'PersonType',
            'base_confidence': 0.85,
            'context': {
                'column_name': 'subject',
                'full_text': 'External user cannot access dashboard'
            }
        },
        {
            'entity_text': 'john.smith@company.com',
            'entity_type': 'Email',
            'base_confidence': 0.95,
            'context': {
                'column_name': 'customer_email',
                'full_text': 'Customer john.smith@company.com reported an issue'
            }
        },
        {
            'entity_text': 'support',
            'entity_type': 'PersonType',
            'base_confidence': 0.75,
            'context': {
                'column_name': 'description',
                'full_text': 'Please contact support for assistance'
            }
        }
    ]
    
    print("\n📊 Testing confidence scoring:")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        
        confidence_score = scorer.calculate_confidence_score(
            entity_text=case['entity_text'],
            entity_type=case['entity_type'],
            base_confidence=case['base_confidence'],
            context=case['context'],
            session_id='test_session'
        )
        
        print(f"Entity: '{case['entity_text']}' ({case['entity_type']})")
        print(f"Base confidence: {case['base_confidence']:.3f}")
        print(f"Adjusted confidence: {confidence_score.adjusted_confidence:.3f}")
        print(f"Reliability: {confidence_score.reliability_score:.3f}")
        print(f"Recommendation: {confidence_score.recommendation}")
        print(f"Signals: {len(confidence_score.signals)}")
        
        # Simulate user feedback
        if i == 1:  # First case - simulate false positive feedback
            scorer.add_user_feedback(
                case['entity_text'],
                case['entity_type'],
                'false_positive',
                confidence_score,
                case['context']
            )
        elif i == 2:  # Second case - simulate correct detection
            scorer.add_user_feedback(
                case['entity_text'],
                case['entity_type'],
                'correct_redaction',
                confidence_score,
                case['context']
            )
    
    # Show performance metrics
    print("\n📈 Performance metrics:")
    metrics = scorer.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Confidence scoring example completed")