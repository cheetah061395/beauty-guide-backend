#!/usr/bin/env python3
"""
Test script for Perfect Corp AI Makeup VTO integration
"""
import requests
import json
import base64

def test_perfect_corp_integration():
    # Test photo path - convert screenshot to test data
    test_photo_path = "/Users/shirleyjiang/Desktop/Screenshot 2026-01-08 at 3.18.50 PM.png"
    
    # Read and encode the test photo
    with open(test_photo_path, 'rb') as f:
        image_data = f.read()
    
    # Sample makeup analysis data (simulating AI recommendations)
    sample_analysis = {
        "generalFeedback": "Your current look has a beautiful natural base. To enhance your features, we can add some warmth with a subtle lip color and soft blush to bring out your natural glow.",
        "specificSuggestions": [
            "Add a soft pink lip color to enhance your natural lip tone",
            "Apply a gentle peachy blush to add warmth to your cheeks", 
            "Consider a light eyeshadow to define your eyes while keeping the look natural"
        ],
        "productRecommendations": [
            "Pink lipstick for a fresh, natural look",
            "Peach blush for warmth",
            "Natural eyeshadow palette"
        ],
        "techniques": [
            "Apply lipstick with light pressure for a natural finish",
            "Blend blush upward toward temples",
            "Use neutral tones for everyday wear"
        ],
        "followUpQuestion": "Would you like tips on achieving this natural enhanced look?"
    }
    
    # Prepare the multipart form data
    files = {
        'file': ('test_photo.png', image_data, 'image/png'),
        'analysis': (None, json.dumps(sample_analysis), 'application/json')
    }
    
    # Make the API call to our backend
    print("Testing Perfect Corp AI Makeup VTO integration...")
    print(f"Image size: {len(image_data)} bytes")
    print("Analysis summary:", sample_analysis['generalFeedback'][:100] + "...")
    
    try:
        response = requests.post(
            'http://localhost:8000/generate-after-photo',
            files=files,
            timeout=120  # 2 minutes timeout for API processing
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print(f"Generated image URL: {result.get('generated_image_url', 'None')}")
            print(f"Processing time: {result.get('processing_time', 'Unknown')}s")
            print(f"Method used: {result.get('method', 'Unknown')}")
            
            if result.get('success'):
                print("✅ Perfect Corp AI Makeup VTO integration working!")
            else:
                print("❌ Error occurred:")
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        else:
            print("❌ HTTP Error:")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - API may be taking too long to process")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_perfect_corp_integration()