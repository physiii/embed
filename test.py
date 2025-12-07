#!/usr/bin/env python3
import subprocess
import json
import time
import sys
import os

# API endpoints (override with EMBED_API_BASE, e.g., http://localhost:8001)
API_BASE = os.environ.get("EMBED_API_BASE", "http://localhost:8000").rstrip("/")
API_URL = f"{API_BASE}/embed"
API_URL_BATCH = f"{API_BASE}/embed_batch"

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

def run_curl_command_batch(texts):
    """Run curl command to test the batch embedding API"""
    curl_command = [
        "curl",
        "-s",  # silent mode
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({"texts": texts}),
        API_URL_BATCH,
    ]

    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error making batch API request: {e}")
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

def test_batch_embedding_api():
    """Test the batch embedding API for correct shape and alignment"""
    print("\nTesting batch embedding API...")

    test_batches = [
        [
            "Batch test sentence one.",
            "Batch test sentence two, slightly longer to vary length.",
            "Short",
        ],
        [
            "Another batch to ensure multiple calls work.",
            "Checking behavior with a different second sentence.",
        ],
    ]

    for i, batch in enumerate(test_batches, 1):
        print(f"\nBatch {i}: {len(batch)} texts")
        response_text = run_curl_command_batch(batch)
        if not response_text:
            print("No response received")
            continue

        try:
            response = json.loads(response_text)
            if "embeddings" not in response:
                print("✗ No embeddings found in response")
                print(f"Response: {response}")
                continue

            embeddings = response["embeddings"]
            if len(embeddings) != len(batch):
                print(f"✗ Mismatched embedding count: expected {len(batch)}, got {len(embeddings)}")
                continue

            dims = len(embeddings[0]) if embeddings else 0
            print(f"✓ Received {len(embeddings)} embeddings with dimension {dims}")
            preview = [round(val, 4) for val in embeddings[0][:5]] if embeddings else []
            print(f"First embedding first 5 values: {preview}...")
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
    test_batch_embedding_api()