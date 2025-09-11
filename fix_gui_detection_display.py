#!/usr/bin/env python3
"""
Quick fix for GUI detection display issue
"""

import pandas as pd

def count_actual_detections(original_df, processed_df, selected_columns):
    """Count actual detections by comparing dataframes"""
    actual_detections = 0
    
    for col in selected_columns:
        if col in original_df.columns and col in processed_df.columns:
            for idx in range(len(original_df)):
                original_val = str(original_df[col].iloc[idx])
                processed_val = str(processed_df[col].iloc[idx])
                if original_val != processed_val:
                    actual_detections += 1
    
    return actual_detections

def test_detection_counting():
    """Test the detection counting logic"""
    print("🔍 Testing Detection Counting Logic")
    
    # Load test data
    df = pd.read_csv('clean_test_sample.csv')
    
    # Create a mock "processed" dataframe with some redactions
    processed_df = df.copy()
    processed_df.loc[0, 'author_name'] = '[NAME]'  # Dwayne Baker -> [NAME]
    processed_df.loc[1, 'author_name'] = '[NAME]'  # Sarah Wilson -> [NAME]
    processed_df.loc[0, 'body'] = processed_df.loc[0, 'body'].replace('Phillip', '[NAME]').replace('Dwayne', '[NAME]')
    
    selected_columns = ['subject', 'description', 'author_name', 'author_email', 'body', 'html_body']
    
    # Count detections
    detections = count_actual_detections(df, processed_df, selected_columns)
    
    print(f"📊 Original data:")
    print(f"   author_name[0]: '{df.loc[0, 'author_name']}'")
    print(f"   author_name[1]: '{df.loc[1, 'author_name']}'")
    print(f"   body[0]: '{df.loc[0, 'body'][:50]}...'")
    
    print(f"\n📊 Processed data:")
    print(f"   author_name[0]: '{processed_df.loc[0, 'author_name']}'")
    print(f"   author_name[1]: '{processed_df.loc[1, 'author_name']}'")
    print(f"   body[0]: '{processed_df.loc[0, 'body'][:50]}...'")
    
    print(f"\n🎯 Detected changes: {detections}")
    
    if detections > 0:
        print("✅ Detection counting works correctly")
        return True
    else:
        print("❌ Detection counting failed")
        return False

if __name__ == "__main__":
    test_detection_counting()