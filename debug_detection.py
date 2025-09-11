#!/usr/bin/env python3
"""
Debug PII detection on ZDTicketsAutotask.csv
"""

import pandas as pd
import os
from dotenv import load_dotenv
from azure_pii_detector import EnhancedAzurePIIDetector

# Load environment
load_dotenv()

def test_small_sample():
    """Test detection on a small sample"""
    print("🔍 Testing PII detection on small sample...")
    
    # Create test data from what we saw in the CSV
    test_data = {
        'subject': ['Co-managed users can not see note that comes in internal only if the external user is not a contact'],
        'description': ['Autotask ticket - Type: incident'],
        'author_name': ['Dwayne'],
        'body': ['Hello Phillip, Thank you for bringing this issue to our attention. My name is Dwayne, and I was tasked with reviewing this request for you.']
    }
    
    df = pd.DataFrame(test_data)
    print(f"   📋 Test data: {len(df)} rows, {len(df.columns)} columns")
    
    # Get credentials
    endpoint = os.getenv('AZURE_ENDPOINT')
    key = os.getenv('AZURE_KEY')
    
    if not endpoint or not key:
        print("❌ Azure credentials not found")
        return False
    
    try:
        # Create detector
        detector = EnhancedAzurePIIDetector(endpoint, key)
        
        # Process data
        redacted_df, stats = detector.detect_and_redact_dataframe(df)
        
        print(f"   📊 Detection results:")
        print(f"      Total cells processed: {stats['total_cells']}")
        print(f"      Cells with PII: {stats['cells_with_pii']}")
        print(f"      Entities found: {sum(stats['entities_found'].values())}")
        
        # Show detailed results
        for col in df.columns:
            print(f"\n   🔍 Column '{col}':")
            original = df[col].iloc[0]
            redacted = redacted_df[col].iloc[0]
            print(f"      Original: '{original}'")
            print(f"      Redacted: '{redacted}'")
            if original != redacted:
                print(f"      ✅ PII detected and redacted")
            else:
                print(f"      ❌ No PII detected")
        
        if stats['cells_with_pii'] > 0:
            print(f"\n✅ PII detection working - found {sum(stats['entities_found'].values())} entities")
            return True
        else:
            print(f"\n❌ No PII detected in test data")
            return False
            
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return False

def test_csv_sample():
    """Test detection on actual CSV sample"""
    print("\n🔍 Testing PII detection on CSV sample...")
    
    try:
        # Read first 5 rows of the CSV
        df = pd.read_csv('ZDTicketsAutotask.csv', nrows=5)
        print(f"   📋 CSV sample: {len(df)} rows, {len(df.columns)} columns")
        
        # Show column names
        print(f"   📝 Columns: {list(df.columns)}")
        
        # Focus on likely PII columns
        pii_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['name', 'email', 'subject', 'description', 'body', 'author']):
                pii_columns.append(col)
        
        print(f"   🎯 PII columns selected: {pii_columns}")
        
        if not pii_columns:
            print("   ⚠️ No obvious PII columns found")
            return False
        
        # Get credentials
        endpoint = os.getenv('AZURE_ENDPOINT')
        key = os.getenv('AZURE_KEY')
        
        # Create detector
        detector = EnhancedAzurePIIDetector(endpoint, key)
        
        # Process only PII columns
        sample_df = df[pii_columns]
        redacted_df, stats = detector.detect_and_redact_dataframe(sample_df)
        
        print(f"   📊 Detection results:")
        print(f"      Total cells processed: {stats['total_cells']}")
        print(f"      Cells with PII: {stats['cells_with_pii']}")
        print(f"      Entities found: {sum(stats['entities_found'].values())}")
        
        # Show sample results
        for col in pii_columns[:3]:  # Show first 3 columns
            if col in df.columns:
                print(f"\n   🔍 Column '{col}' (first row):")
                original = str(df[col].iloc[0])[:100] + "..." if len(str(df[col].iloc[0])) > 100 else str(df[col].iloc[0])
                redacted = str(redacted_df[col].iloc[0])[:100] + "..." if len(str(redacted_df[col].iloc[0])) > 100 else str(redacted_df[col].iloc[0])
                print(f"      Original: '{original}'")
                print(f"      Redacted: '{redacted}'")
        
        return stats['cells_with_pii'] > 0
        
    except Exception as e:
        print(f"❌ CSV test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Debug PII Detection\n")
    
    test1_success = test_small_sample()
    test2_success = test_csv_sample()
    
    print(f"\n📊 Debug Results:")
    print(f"   Small sample test: {'✅' if test1_success else '❌'}")
    print(f"   CSV sample test: {'✅' if test2_success else '❌'}")
    
    if test1_success and not test2_success:
        print(f"\n💡 Issue likely with CSV parsing or column selection")
    elif not test1_success:
        print(f"\n💡 Issue with PII detection configuration")
    else:
        print(f"\n🎉 PII detection working correctly")