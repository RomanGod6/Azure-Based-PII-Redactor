#!/usr/bin/env python3
"""
Detailed debugging of 3-row sample to find exact failure point
"""

import pandas as pd
import os
import json
from dotenv import load_dotenv
from enhanced_ml_detector import create_enhanced_detector

# Load environment
load_dotenv()

def test_3_rows_detailed():
    """Test the exact same pipeline used by the GUI on 3 rows"""
    print("🔍 Testing Enhanced Detection Pipeline on 3 Rows")
    print("=" * 60)
    
    try:
        # Load the clean test sample
        df = pd.read_csv('clean_test_sample.csv')
        print(f"📋 Loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"📝 Columns: {list(df.columns)}")
        
        # Select PII columns like the GUI would
        pii_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['name', 'email', 'subject', 'description', 'body', 'author']):
                pii_columns.append(col)
        
        print(f"🎯 Selected PII columns: {pii_columns}")
        
        # Show sample data
        print(f"\n📋 Sample Data Preview:")
        for col in pii_columns[:3]:
            if col in df.columns and not df[col].isna().all():
                sample_val = str(df[col].iloc[0])[:100]
                print(f"   {col}: '{sample_val}{'...' if len(str(df[col].iloc[0])) > 100 else ''}'")
        
        # Get credentials
        endpoint = os.getenv('AZURE_ENDPOINT')
        key = os.getenv('AZURE_KEY')
        
        if not endpoint or not key:
            print("❌ Azure credentials not found")
            return False
        
        print(f"\n🔧 Creating Enhanced Detector...")
        
        # Create enhanced detector (same as GUI)
        detector = create_enhanced_detector(
            azure_endpoint=endpoint,
            azure_key=key,
            enable_gpt=True,
            openai_key=key
        )
        
        print(f"✅ Enhanced detector created successfully")
        
        # Process with same settings as GUI
        print(f"\n🚀 Starting Enhanced Detection Pipeline...")
        print(f"   📊 Confidence threshold: 0.95")
        print(f"   🎯 Selected columns: {len(pii_columns)}")
        print(f"   🔄 Enable learning: True")
        
        # Call the exact same method as GUI
        result = detector.detect_and_validate_comprehensive(
            df,
            columns=pii_columns,
            enable_learning=True,
            confidence_threshold=0.95  # Same as GUI default
        )
        
        print(f"\n📊 DETAILED RESULTS:")
        print(f"=" * 40)
        
        # Extract the tuple results
        processed_df, stats = result
        
        print(f"🔢 Basic Stats:")
        print(f"   Total cells processed: {stats.get('total_cells', 'N/A')}")
        print(f"   Total detections: {stats.get('total_detections', 'N/A')}")
        print(f"   Processing time: {stats.get('total_processing_time', 'N/A'):.2f}s")
        
        print(f"\n🔍 Detection Layers:")
        layers = stats.get('detection_layers', {})
        for layer_name, layer_data in layers.items():
            detections = layer_data.get('detections', 0)
            corrections = layer_data.get('corrections', 0)
            print(f"   {layer_name}: {detections} detections, {corrections} corrections")
        
        print(f"\n⚖️ Performance Metrics:")
        perf = stats.get('performance_metrics', {})
        print(f"   Estimated accuracy: {perf.get('estimated_accuracy', 'N/A'):.1%}")
        print(f"   Detection rate: {perf.get('detection_rate', 'N/A'):.1%}")
        
        print(f"\n📋 Column-by-Column Analysis:")
        for col in pii_columns:
            if col in df.columns:
                original_sample = str(df[col].iloc[0])[:50] + "..." if len(str(df[col].iloc[0])) > 50 else str(df[col].iloc[0])
                processed_sample = str(processed_df[col].iloc[0])[:50] + "..." if len(str(processed_df[col].iloc[0])) > 50 else str(processed_df[col].iloc[0])
                
                changed = original_sample != processed_sample
                print(f"   📝 {col}:")
                print(f"      Original: '{original_sample}'")
                print(f"      Processed: '{processed_sample}'")
                print(f"      Changed: {'✅ Yes' if changed else '❌ No'}")
        
        # Count actual changes
        total_changes = 0
        for col in pii_columns:
            if col in df.columns:
                for idx in range(len(df)):
                    if str(df[col].iloc[idx]) != str(processed_df[col].iloc[idx]):
                        total_changes += 1
        
        print(f"\n🎯 FINAL ANALYSIS:")
        print(f"   Layers found detections: {'✅ Yes' if any(layer.get('detections', 0) > 0 for layer in layers.values()) else '❌ No'}")
        print(f"   Final data changed: {'✅ Yes' if total_changes > 0 else '❌ No'}")
        print(f"   Total changes made: {total_changes}")
        print(f"   Expected vs Actual: Expected {stats.get('total_detections', 0)}, Got {total_changes}")
        
        if stats.get('total_detections', 0) > 0 and total_changes == 0:
            print(f"\n🚨 ISSUE IDENTIFIED:")
            print(f"   ❌ Detections were found but not applied to final data!")
            print(f"   🔍 Possible causes:")
            print(f"      - Confidence filtering too strict")
            print(f"      - Error in applying redactions")
            print(f"      - Column mapping issue")
            
            # Debug confidence filtering
            print(f"\n🔬 Confidence Analysis:")
            if 'column_stats' in stats:
                for col_name, col_stat in stats['column_stats'].items():
                    entities = col_stat.get('entities_detected', [])
                    if entities:
                        print(f"   📊 {col_name}: {len(entities)} entities")
                        for i, entity in enumerate(entities[:3]):  # Show first 3
                            conf = entity.get('confidence', 'N/A')
                            text = entity.get('text', 'N/A')
                            print(f"      {i+1}. '{text}' (confidence: {conf})")
        
        return total_changes > 0
        
    except Exception as e:
        print(f"❌ Detailed test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Detailed 3-Row Debug Test\n")
    
    success = test_3_rows_detailed()
    
    print(f"\n" + "=" * 60)
    print(f"🏁 Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if not success:
        print(f"\n💡 Next steps:")
        print(f"   1. Check confidence threshold settings")
        print(f"   2. Verify column selection logic")
        print(f"   3. Debug redaction application process")