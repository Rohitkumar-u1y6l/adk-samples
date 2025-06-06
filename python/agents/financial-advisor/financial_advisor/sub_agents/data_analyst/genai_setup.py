import os
import json
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

def setup_genai():
    """Set up Google Generative AI with either API key or service account"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Try API key first
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        try:
            genai.configure(api_key=api_key)
            print("✓ Successfully configured GenAI with API key")
            return True
        except Exception as e:
            print(f"⚠️ Warning: Failed to configure with API key: {str(e)}")
    
    # Try service account credentials
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path:
        try:
            if not Path(creds_path).exists():
                print(f"❌ Error: Service account JSON file not found at {creds_path}")
                return False
                
            # Verify JSON is valid
            with open(creds_path) as f:
                json.load(f)
                
            # The environment variable is enough, no need to explicitly configure
            print("✓ Successfully configured GenAI with service account")
            return True
            
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid service account JSON file")
            return False
        except Exception as e:
            print(f"❌ Error configuring with service account: {str(e)}")
            return False
    
    print("❌ Error: No authentication method found")
    print("\nPlease set either:")
    print("1. GOOGLE_API_KEY environment variable, or")
    print("2. GOOGLE_APPLICATION_CREDENTIALS pointing to service account JSON file")
    print("\nYou can set these in the .env file or using:")
    print("export GOOGLE_API_KEY='your-api-key'")
    print("export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account.json'")
    return False
