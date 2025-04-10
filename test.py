#!/usr/bin/env python3
import subprocess
import json
import time
import sys

# API endpoint
API_URL = "http://localhost:8000/embed"

def run_curl_command(text):
    """Run curl command to test the embedding API"""
    curl_command = [
        "curl",
        "-s",  # silent mode
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({"text": text}),
        API_URL
    ]
    
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error making API request: {e}")
        print(f"Error output: {e.stderr}")
        return None

def test_embedding_api():
    """Test the embedding API with various inputs"""
    print("Testing embedding API...")
    
    # Test cases
    test_cases = [
        "This is a simple test sentence.",
        "Another example with different words and context.",
        "Testing with a longer paragraph. This text should be embedded into a vector of floating point numbers. The model should capture the semantic meaning of this text."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {text[:50]}..." if len(text) > 50 else f"\nTest case {i}: {text}")
        
        # Make API request
        response_text = run_curl_command(text)
        if not response_text:
            print("No response received")
            continue
        
        try:
            response = json.loads(response_text)
            
            # Check if embedding exists
            if "embedding" in response:
                embedding = response["embedding"]
                print(f"✓ Received embedding with {len(embedding)} dimensions")
                
                # Print first few values of the embedding
                preview = [round(val, 4) for val in embedding[:5]]
                print(f"First 5 values: {preview}...")
            else:
                print("✗ No embedding found in response")
                print(f"Response: {response}")
        
        except json.JSONDecodeError:
            print("✗ Failed to parse JSON response")
            print(f"Raw response: {response_text}")

if __name__ == "__main__":
    # Check if server is running
    print("Checking if embedding server is running...")
    try:
        # Simple HEAD request to check if server is up
        subprocess.run(["curl", "-s", "-I", "http://localhost:8000"], 
                       check=True, capture_output=True)
        print("Server appears to be running.")
    except subprocess.CalledProcessError:
        print("Warning: Server might not be running. Start the server with 'python main.py'")
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            sys.exit(1)
    
    # Run tests
    test_embedding_api() 