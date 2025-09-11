#!/usr/bin/env python3
"""
Test actual PII detection performance to debug 0.0% metrics
"""

import pandas as pd
import os
from dotenv import load_dotenv
from azure_pii_detector import EnhancedAzurePIIDetector
from gpt_validator import GPTPIIValidator
from performance_monitor import PerformanceMonitor

# Load environment
load_dotenv()

def create_test_data():
    """Create test data with known PII"""
    test_data = {
        'customer_name': [
            'John Smith',
            'Jane Doe',
            'Michael Johnson',
            'Sarah Wilson',
            'David Brown'
        ],
        'email': [
            'john.smith@example.com',
            'jane.doe@company.org',
            'mjohnson@business.net',
            'sarah.w@service.com',
            'david.brown@enterprise.co'
        ],
        'phone': [
            '(555) 123-4567',
            '555-987-6543',
            '(555) 555-0123',
            '555.456.7890',
            '5551234567'
        ],
        'description': [
            'Customer John Smith called about his account',
            'Jane needs help with billing for jane.doe@company.org',
            'Technical support ticket for phone number 555-123-4567',
            'Password reset request from Sarah Wilson',
            'Account verification for SSN 123-45-6789'
        ]
    }
    return pd.DataFrame(test_data)

def test_azure_detection():
    """Test Azure AI detection"""
    print("🔍 Testing Azure AI Detection...")
    
    # Get credentials
    endpoint = os.getenv('AZURE_ENDPOINT')
    key = os.getenv('AZURE_KEY')
    
    if not endpoint or not key:
        print("❌ Azure credentials not found")
        return False
    
    try:
        # Create detector
        detector = EnhancedAzurePIIDetector(endpoint, key)
        
        # Create test data
        df = create_test_data()
        print(f"   📋 Test data: {len(df)} rows, {len(df.columns)} columns")
        
        # Process data
        redacted_df, stats = detector.detect_and_redact_dataframe(df)
        
        print(f"   📊 Processing results:")
        print(f"      Total cells processed: {stats['total_cells']}")
        print(f"      Cells with PII: {stats['cells_with_pii']}")
        print(f"      Entities found: {sum(stats['entities_found'].values())}")
        print(f"      Cost: ${stats['cost']:.4f}")
        
        if stats['cells_with_pii'] > 0:
            print("✅ Azure detection working correctly")
            
            # Show some results
            print("\n   🔍 Detection samples:")
            for col in df.columns:
                if col in stats.get('column_stats', {}):
                    col_stats = stats['column_stats'][col]
                    if col_stats['entities_found'] > 0:
                        print(f"      {col}: {col_stats['entities_found']} entities")
                        original = df[col].iloc[0]
                        redacted = redacted_df[col].iloc[0]
                        if original != redacted:
                            print(f"         '{original}' → '{redacted}'")
            
            return True
        else:
            print("⚠️ No PII detected in test data")
            return False
            
    except Exception as e:
        print(f"❌ Azure detection error: {e}")
        return False

def test_gpt_validation():
    """Test GPT validation"""
    print("\n🔍 Testing GPT Validation...")
    
    try:
        # Get credentials
        endpoint = os.getenv('AZURE_GPT_ENDPOINT')
        key = os.getenv('AZURE_KEY')
        deployment = os.getenv('AZURE_GPT_DEPLOYMENT')
        
        if not all([endpoint, key, deployment]):
            print("❌ GPT credentials incomplete")
            return False
        
        # Create GPT validator
        validator = GPTPIIValidator(
            azure_api_key=key,
            azure_endpoint=endpoint,
            deployment_name=deployment,
            api_version=os.getenv('AZURE_GPT_API_VERSION', '2025-01-01-preview')
        )
        
        # Test validation with sample entities
        test_text = "Customer John Smith called about his account"
        test_entities = [
            {
                'text': 'John Smith',
                'category': 'Person',
                'confidence_score': 1.0,
                'offset': 9,
                'length': 10
            }
        ]
        
        validation_results = validator.validate_pii_detection(
            test_text, test_entities, "customer_support"
        )
        
        print(f"   📊 Validation results: {len(validation_results)} items")
        
        if validation_results:
            for result in validation_results:
                print(f"      Entity: '{result['entity_text']}' → Should redact: {result['should_redact']}")
                if result.get('explanation'):
                    print(f"         Reason: {result['explanation']}")
            
            print("✅ GPT validation working correctly")
            return True
        else:
            print("⚠️ No validation results")
            return False
            
    except Exception as e:
        print(f"❌ GPT validation error: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring system"""
    print("\n🔍 Testing Performance Monitoring...")
    
    try:
        # Create performance monitor
        monitor = PerformanceMonitor("test_performance.db")
        
        # Start a test session
        session_id = monitor.start_session("test_session", 100)
        
        # Simulate processing
        monitor.update_session_progress(session_id, 50, detections=10)
        monitor.update_session_progress(session_id, 100, detections=20)
        
        # Add some feedback
        monitor.add_accuracy_feedback(
            session_id,
            "Test customer John Smith",
            [{"text": "John Smith", "category": "Person", "confidence": 1.0}],
            {
                "marked_correct": [True],
                "feedback_type": "correct",
                "confidence_before": 1.0,
                "confidence_after": 1.0
            }
        )
        
        # End session
        monitor.end_session(session_id, {
            'accuracy_estimate': 0.95,
            'cost_breakdown': {'azure': 0.001},
            'confidence_distribution': {'high': 90, 'medium': 10}
        })
        
        # Get metrics
        metrics = monitor.get_real_time_metrics()
        
        print(f"   📊 Real-time metrics:")
        print(f"      Current accuracy: {metrics['current_accuracy']:.1%}")
        print(f"      Processing speed: {metrics['processing_speed']:.1f} records/sec")
        print(f"      False positive rate: {metrics['false_positive_rate']:.1%}")
        
        print("✅ Performance monitoring working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring error: {e}")
        return False

def test_enhanced_pipeline():
    """Test the full enhanced detection pipeline"""
    print("\n🔍 Testing Enhanced Detection Pipeline...")
    
    try:
        from enhanced_ml_detector import create_enhanced_detector
        
        # Get credentials
        endpoint = os.getenv('AZURE_ENDPOINT')
        key = os.getenv('AZURE_KEY')
        
        # Create enhanced detector
        detector = create_enhanced_detector(
            azure_endpoint=endpoint,
            azure_key=key,
            enable_gpt=True,
            openai_key=key
        )
        
        # Create test data
        df = create_test_data()
        
        # Process with enhanced pipeline
        result = detector.detect_and_validate_comprehensive(
            df, 
            columns=['customer_name', 'email', 'description']
        )
        
        print(f"   📊 Enhanced pipeline results:")
        print(f"      Accuracy: {result['accuracy']:.1%}")
        print(f"      Confidence: {result['confidence']:.1%}")
        print(f"      Entities detected: {result['total_entities']}")
        print(f"      Processing layers used: {', '.join(result['processing_layers'])}")
        
        if result['accuracy'] > 0:
            print("✅ Enhanced pipeline working correctly")
            return True
        else:
            print("⚠️ Enhanced pipeline showing 0% accuracy")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced pipeline error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 PII Detection Performance Test\n")
    
    azure_success = test_azure_detection()
    gpt_success = test_gpt_validation()
    monitor_success = test_performance_monitoring()
    enhanced_success = test_enhanced_pipeline()
    
    print(f"\n📊 Test Results:")
    print(f"   Azure Detection: {'✅' if azure_success else '❌'}")
    print(f"   GPT Validation: {'✅' if gpt_success else '❌'}")
    print(f"   Performance Monitor: {'✅' if monitor_success else '❌'}")
    print(f"   Enhanced Pipeline: {'✅' if enhanced_success else '❌'}")
    
    total_success = sum([azure_success, gpt_success, monitor_success, enhanced_success])
    
    if total_success == 4:
        print(f"\n🎉 All systems working! Performance should now show >0%")
    else:
        print(f"\n⚠️ {4-total_success} systems need attention")
    
    print("\n💡 Next steps:")
    print("   1. Run your main application")
    print("   2. Process some data")
    print("   3. Check if performance metrics now show >0%")