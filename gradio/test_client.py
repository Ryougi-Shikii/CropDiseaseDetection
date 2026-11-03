"""
Simple Test Client for Crop Disease ViT Inference Server
Run this script to verify your Hugging Face Space endpoint or local test server.
"""

import sys
import os
import requests

# Set your target URL here:
# For local testing:
# TARGET_URL = "http://localhost:7860"
# For Hugging Face Spaces:
TARGET_URL = "https://<YOUR-HF-USERNAME>-<YOUR-SPACE-NAME>.hf.space"

def test_health():
    print(f"Testing health endpoint on {TARGET_URL}/api/health ...")
    try:
        res = requests.get(f"{TARGET_URL}/api/health", timeout=10)
        print("Status code:", res.status_code)
        print("Response:", res.json())
    except Exception as e:
        print("Health check failed:", e)

def test_prediction(image_path: str):
    print(f"\nTesting leaf diagnosis on {image_path} ...")
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    url = f"{TARGET_URL}/api/predict"
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            res = requests.post(url, files=files, timeout=30)
            
        print("Status code:", res.status_code)
        data = res.json()
        print(f"Predicted Crop: {data.get('crop')}")
        print(f"Condition:      {data.get('condition')}")
        print(f"Confidence:     {data.get('confidence')}%")
        print(f"Severity:       {data.get('severity')}")
        print(f"Latency:        {data.get('telemetry', {}).get('latency_ms')} ms")
        print("\nOrganic Treatments:")
        for t in data.get('organic_treatment', []):
            print(f" - {t}")
    except Exception as e:
        print("Prediction request failed:", e)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        TARGET_URL = sys.argv[1].rstrip('/')
    
    test_health()
    
    # Try sample image
    sample_img = os.path.join(os.path.dirname(__file__), "examples", "Tomato_LeafMold.JPG")
    if os.path.exists(sample_img):
        test_prediction(sample_img)
