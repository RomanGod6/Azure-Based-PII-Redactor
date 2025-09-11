#!/usr/bin/env python3
"""
Debug the exact GUI flow to find where detections are lost
"""

import pandas as pd
import os
from dotenv import load_dotenv
from enhanced_ml_detector import create_enhanced_detector

# Load environment
load_dotenv()

def simulate_gui_flow():
    """Simulate the exact GUI processing flow"""
    print("🔍 Simulating GUI Processing Flow")
    print("=" * 60)
    
    try:
        # Load clean test sample (same as GUI would)
        df = pd.read_csv('clean_test_sample.csv')
        print(f"📋 Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Simulate GUI column selection - ALL COLUMNS like user selected
        selected_columns = ['subject', 'description', 'form_name', 'author_id', 'author_name', 'author_type', 'author_email', 'body', 'html_body']
        print(f"🎯 Selected columns (as user did): {selected_columns}")
        
        # Simulate GUI settings
        show_confidence = True
        enable_learning = True
        confidence_threshold = 0.95  # High threshold like GUI
        
        print(f"⚙️ Settings:")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Show confidence: {show_confidence}")
        print(f"   Enable learning: {enable_learning}")
        
        # Get credentials
        endpoint = os.getenv('AZURE_ENDPOINT')
        key = os.getenv('AZURE_KEY')
        
        # Create detector (same as GUI)
        print(f"\n🔧 Creating detector...")
        detector = create_enhanced_detector(
            azure_endpoint=endpoint,
            azure_key=key,
            enable_gpt=True,
            openai_key=key
        )
        
        # Step 1: Call enhanced detection (same as GUI line 592-597)
        print(f"\n🚀 Step 1: Enhanced Detection...")
        processed_df, stats = detector.detect_and_validate_comprehensive(
            df,
            columns=selected_columns,
            enable_learning=enable_learning,
            confidence_threshold=confidence_threshold
        )
        
        print(f"✅ Enhanced detection complete")
        print(f"   📊 Stats total_detections: {stats.get('total_detections', 'MISSING')}")
        print(f"   📊 Stats type: {type(stats)}")
        print(f"   📊 Stats keys: {list(stats.keys())}")
        
        # Step 2: Apply confidence scoring (same as GUI line 602-604)
        if show_confidence:
            print(f"\n⚖️ Step 2: Applying confidence scoring...")
            # This is where the GUI might be losing detections!
            
            # Simulate confidence scoring (simplified)
            try:
                print(f"   📋 Before confidence scoring:")
                print(f"      Total detections: {stats.get('total_detections', 0)}")
                
                # Count actual changes in dataframe
                actual_changes = 0
                for col in selected_columns:
                    if col in df.columns:
                        for idx in range(len(df)):
                            original = str(df[col].iloc[idx])
                            processed = str(processed_df[col].iloc[idx])
                            if original != processed:
                                actual_changes += 1
                
                print(f"      Actual changes in dataframe: {actual_changes}")
                
                # The GUI might be calling apply_confidence_scoring which could modify stats
                # For now, let's skip this step to see if it's the problem
                print(f"   ⚠️ Skipping confidence scoring to test...")
                
            except Exception as e:
                print(f"   ❌ Confidence scoring error: {e}")
        
        # Step 3: Count final results (same as GUI display logic)
        print(f"\n📊 Step 3: Final Result Analysis...")
        
        # This is what GUI displays
        gui_total_detections = stats.get('total_detections', 0)
        print(f"   GUI would show: {gui_total_detections} total detections")
        
        # Count actual changes
        actual_changes = 0
        changes_by_column = {}
        
        for col in selected_columns:
            if col in df.columns:
                col_changes = 0
                for idx in range(len(df)):
                    original = str(df[col].iloc[idx])
                    processed = str(processed_df[col].iloc[idx])
                    if original != processed:
                        actual_changes += 1
                        col_changes += 1
                        print(f"      Change in {col}[{idx}]: '{original}' → '{processed}'")
                
                changes_by_column[col] = col_changes
        
        print(f"\n🎯 COMPARISON:")
        print(f"   Stats says: {gui_total_detections} detections")
        print(f"   Actually changed: {actual_changes} cells")
        print(f"   Changes by column: {changes_by_column}")
        
        # If there's a mismatch, that's our problem
        if gui_total_detections != actual_changes:
            print(f"\n🚨 MISMATCH DETECTED!")
            print(f"   The stats object is not accurately reflecting actual changes")
            print(f"   This explains why GUI shows 0 but backend works")
            
            # Check if it's a stats calculation issue
            print(f"\n🔬 Deep dive into stats:")
            if 'detection_layers' in stats:
                layers = stats['detection_layers']
                for layer_name, layer_data in layers.items():
                    print(f"      {layer_name}: {layer_data}")
        
        return actual_changes > 0
        
    except Exception as e:
        print(f"❌ GUI flow simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 GUI Flow Debug Test\n")
    
    success = simulate_gui_flow()
    
    print(f"\n" + "=" * 60)
    print(f"🏁 Result: {'✅ WORKING' if success else '❌ BROKEN'}")