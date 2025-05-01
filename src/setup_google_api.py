#!/usr/bin/env python3
"""
Setup script for Google API credentials needed for the web search tool.

This script helps users create or update a .env file with the necessary 
Google API key and Custom Search Engine ID for web search functionality.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def main():
    # First, check if .env exists and load it
    env_path = Path('.env')
    env_exists = env_path.exists()
    
    if env_exists:
        print("Found existing .env file. Will update with Google API credentials.")
        load_dotenv()
    else:
        print("No .env file found. Will create a new one.")
    
    # Check for existing OPENAI_API_KEY
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    # Get Google API credentials
    google_api_key = input("\nEnter your Google API Key: ").strip()
    if not google_api_key:
        print("Error: Google API Key is required for web search functionality.")
        sys.exit(1)
    
    google_cx = input("Enter your Google Custom Search Engine ID: ").strip()
    if not google_cx:
        print("Error: Google Custom Search Engine ID is required for web search functionality.")
        sys.exit(1)
    
    # Write to .env file
    with open('.env', 'w') as f:
        # Preserve OpenAI key if it exists
        if openai_key:
            f.write(f"OPENAI_API_KEY={openai_key}\n")
        else:
            openai_key = input("\nEnter your OpenAI API Key (required for the forecasting system): ").strip()
            if openai_key:
                f.write(f"OPENAI_API_KEY={openai_key}\n")
            else:
                print("Warning: No OpenAI API Key provided. You'll need to set it before using the forecasting system.")
        
        # Add Google API credentials
        f.write(f"GOOGLE_API_KEY={google_api_key}\n")
        f.write(f"GOOGLE_CX={google_cx}\n")
    
    print("\nCredentials saved to .env file.")
    print("\nTo use the web search functionality, you need:")
    print("1. A Google API Key with Custom Search API enabled")
    print("2. A Custom Search Engine ID configured for searching the web")
    print("\nFor instructions on setting these up, visit:")
    print("https://developers.google.com/custom-search/v1/introduction")
    print("https://programmablesearchengine.google.com/")

if __name__ == "__main__":
    main() 