#!/usr/bin/env python3
"""
Test Azure AI connectivity and environment loading
"""

import os
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

def test_azure_connection():
    """Test Azure AI connectivity"""
    print("🔍 Testing Azure AI Connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    endpoint = os.getenv('AZURE_ENDPOINT')
    key = os.getenv('AZURE_KEY')
    
    print(f"Endpoint: {endpoint}")
    print(f"Key: {'SET' if key else 'NOT SET'}")
    
    if not endpoint or not key:
        print("❌ Azure credentials not found in environment")
        return False
    
    try:
        # Create client
        client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        
        # Test with simple text
        test_text = ["John Smith lives in Seattle and works at Microsoft."]
        
        print("🔍 Testing PII detection...")
        response = client.recognize_pii_entities(documents=test_text, language="en")
        
        for result in response:
            if result.is_error:
                print(f"❌ Azure AI Error: {result.error}")
                return False
            else:
                print(f"✅ Azure AI Connection Successful!")
                print(f"   Detected {len(result.entities)} PII entities:")
                for entity in result.entities:
                    print(f"   - {entity.text} ({entity.category}) - Confidence: {entity.confidence_score:.2f}")
                return True
                
    except Exception as e:
        print(f"❌ Connection Failed: {str(e)}")
        return False

def test_gpt_connection():
    """Test Azure GPT connectivity"""
    print("\n🔍 Testing Azure GPT Connection...")
    
    # Load environment variables
    load_dotenv()
    
    gpt_endpoint = os.getenv('AZURE_GPT_ENDPOINT')
    key = os.getenv('AZURE_KEY')
    deployment = os.getenv('AZURE_GPT_DEPLOYMENT')
    api_version = os.getenv('AZURE_GPT_API_VERSION')
    
    print(f"GPT Endpoint: {gpt_endpoint}")
    print(f"Deployment: {deployment}")
    print(f"API Version: {api_version}")
    print(f"Key: {'SET' if key else 'NOT SET'}")
    
    if not all([gpt_endpoint, key, deployment]):
        print("❌ Azure GPT credentials incomplete")
        return False
    
    try:
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=key,
            api_version=api_version,
            azure_endpoint=gpt_endpoint
        )
        
        # Test simple completion
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Say 'Hello, Azure GPT is working!'"}],
            max_tokens=50
        )
        
        print("✅ Azure GPT Connection Successful!")
        print(f"   Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ GPT Connection Failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Azure Connection Test\n")
    
    azure_success = test_azure_connection()
    gpt_success = test_gpt_connection()
    
    print(f"\n📊 Test Results:")
    print(f"   Azure AI: {'✅' if azure_success else '❌'}")
    print(f"   Azure GPT: {'✅' if gpt_success else '❌'}")
    
    if not azure_success:
        print("\n💡 Troubleshooting Steps:")
        print("   1. Check your .env file has correct AZURE_ENDPOINT and AZURE_KEY")
        print("   2. Verify your Azure AI service is running")
        print("   3. Check your Azure portal for correct endpoint and key")
    
    if not gpt_success:
        print("\n💡 GPT Troubleshooting Steps:")
        print("   1. Check AZURE_GPT_ENDPOINT and AZURE_GPT_DEPLOYMENT")
        print("   2. Verify your GPT deployment name is correct")
        print("   3. Check if GPT service is available in your region")