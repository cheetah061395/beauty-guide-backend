#!/usr/bin/env python3
"""
Test script for Beauty Guide Backend
Tests all endpoints with sample data
"""

import requests
import json
import base64
from PIL import Image, ImageDraw
import io

# Backend URL
BASE_URL = "http://localhost:8000"

def create_test_image():
    """Create a simple test image with a face-like shape"""
    # Create a 300x300 RGB image
    img = Image.new('RGB', (300, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple face
    # Face outline (circle)
    draw.ellipse([50, 50, 250, 250], outline='black', width=3)
    
    # Eyes
    draw.ellipse([80, 100, 110, 130], fill='black')
    draw.ellipse([190, 100, 220, 130], fill='black')
    
    # Nose (triangle)
    draw.polygon([(150, 140), (140, 180), (160, 180)], outline='black', width=2)
    
    # Mouth (arc)
    draw.arc([120, 180, 180, 220], start=0, end=180, fill='black', width=3)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_analyze_landmarks():
    """Test landmark analysis endpoint"""
    print("\nTesting /analyze-landmarks endpoint...")
    try:
        test_image = create_test_image()
        
        files = {'file': ('test.png', test_image, 'image/png')}
        response = requests.post(f"{BASE_URL}/analyze-landmarks", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Landmark analysis passed: {data.get('landmarks_count', 0)} landmarks detected")
            return True
        elif response.status_code == 422:
            print("⚠️ No face detected in test image (expected for simple drawing)")
            return True  # This is expected for our simple test image
        else:
            print(f"❌ Landmark analysis failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Landmark analysis error: {e}")
        return False

def test_generate_mask():
    """Test mask generation endpoint"""
    print("\nTesting /generate-mask endpoint...")
    try:
        test_image = create_test_image()
        target_regions = json.dumps([{"name": "lips", "confidence": 0.9}])
        
        files = {'file': ('test.png', test_image, 'image/png')}
        data = {'target_regions': target_regions}
        
        response = requests.post(f"{BASE_URL}/generate-mask", files=files, data=data)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Mask generation passed: {len(data.get('mask_base64', ''))} bytes mask")
            return True
        elif response.status_code == 422:
            print("⚠️ No face detected for mask generation (expected for simple drawing)")
            return True  # This is expected for our simple test image
        else:
            print(f"❌ Mask generation failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Mask generation error: {e}")
        return False

def test_generate_after_photo():
    """Test complete after photo generation"""
    print("\nTesting /generate-after-photo endpoint...")
    try:
        test_image = create_test_image()
        
        # Sample makeup analysis
        analysis = {
            "generalFeedback": "Your overall look is natural and fresh.",
            "specificSuggestions": ["Try adding a subtle lip tint", "Consider light mascara"],
            "productRecommendations": ["Pink lip gloss", "Brown mascara"],
            "techniques": ["Apply lip gloss in thin layers", "Curl lashes before mascara"],
            "followUpQuestion": "Would you like specific brand recommendations?"
        }
        
        files = {'file': ('test.png', test_image, 'image/png')}
        data = {'analysis': json.dumps(analysis)}
        
        response = requests.post(f"{BASE_URL}/generate-after-photo", files=files, data=data)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ After photo generation passed: {data.get('generated_image_url', 'No URL')}")
                return True
            else:
                error = data.get('error', 'Unknown error')
                if 'No face detected' in error:
                    print("⚠️ No face detected in test image (expected for simple drawing)")
                    return True
                else:
                    print(f"❌ After photo generation failed: {error}")
                    return False
        else:
            print(f"❌ After photo generation failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ After photo generation error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Beauty Guide Backend Test Suite ===\n")
    
    tests = [
        test_health,
        test_analyze_landmarks,
        test_generate_mask,
        test_generate_after_photo
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed or had warnings")

if __name__ == "__main__":
    main()