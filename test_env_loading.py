#!/usr/bin/env python3
"""
Test environment loading in application files
"""

import sys
import os

def test_enhanced_app_env():
    """Test if enhanced app loads environment properly"""
    print("🔍 Testing enhanced app environment loading...")
    
    try:
        # Import the enhanced app (this will trigger load_dotenv)
        from enhanced_pii_redactor_app import EnhancedPIIRedactorApp
        
        # Check if environment variables are loaded
        endpoint = os.getenv('AZURE_ENDPOINT')
        key = os.getenv('AZURE_KEY')
        
        print(f"   Endpoint: {endpoint}")
        print(f"   Key: {'SET' if key else 'NOT SET'}")
        
        if endpoint and key:
            print("✅ Enhanced app loads environment correctly")
            return True
        else:
            print("❌ Enhanced app failed to load environment")
            return False
            
    except Exception as e:
        print(f"❌ Error testing enhanced app: {e}")
        return False

def test_web_app_env():
    """Test if web app loads environment properly"""
    print("\n🔍 Testing web app environment loading...")
    
    try:
        # Import the web app (this will trigger load_dotenv)
        import pii_redactor_web
        
        # Check if environment variables are loaded
        endpoint = os.getenv('AZURE_ENDPOINT')
        key = os.getenv('AZURE_KEY')
        
        print(f"   Endpoint: {endpoint}")
        print(f"   Key: {'SET' if key else 'NOT SET'}")
        
        if endpoint and key:
            print("✅ Web app loads environment correctly")
            return True
        else:
            print("❌ Web app failed to load environment")
            return False
            
    except Exception as e:
        print(f"❌ Error testing web app: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Environment Loading Test\n")
    
    # Clear environment first to simulate fresh start
    os.environ.pop('AZURE_ENDPOINT', None)
    os.environ.pop('AZURE_KEY', None)
    
    enhanced_success = test_enhanced_app_env()
    web_success = test_web_app_env()
    
    print(f"\n📊 Test Results:")
    print(f"   Enhanced App: {'✅' if enhanced_success else '❌'}")
    print(f"   Web App: {'✅' if web_success else '❌'}")
    
    if enhanced_success and web_success:
        print("\n🎉 All applications load environment correctly!")
    else:
        print("\n⚠️ Some applications need fixing")