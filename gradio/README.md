---
title: Crop Disease Detection ViT
emoji: 🌾
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🌾 Vision Transformer Crop Disease Detection & Fast Inference API

High-performance AI crop disease detection powered by Vision Transformer (`google/vit-base-patch16-224`) fine-tuned across 38 crop disease classes with **98.97% validation accuracy**.

This Space provides both:
1. **Interactive Gradio Web Interface** for instant visual leaf diagnostics.
2. **Permanent High-Speed REST API** (`/api/predict`) designed to power external web frontends (such as [anni88git/CropDiseaseDetection](https://github.com/anni88git/CropDiseaseDetection)), mobile apps, and IoT field devices.

---

## 🚀 Why Use This Hugging Face Space Over Render?

| Feature | Render Free Tier | Hugging Face Space Docker |
| :--- | :--- | :--- |
| **RAM** | 512 MB (Often crashes with ViT OOM) | **16 GB RAM** (Plenty of headroom) |
| **CPU** | 0.1 CPU shared (Severe throttling) | **2 dedicated vCPUs** |
| **Inference Latency** | 10 – 30+ seconds (or timeout) | **~150 – 250 ms** (Lightning fast) |
| **Cold Starts** | 60+ seconds spin-up delay | Kept warm or fast wake |
| **Permanent Link** | Yes (but slow/sleeping) | **Permanent `https://<user>-<space>.hf.space`** |
| **CORS** | Requires manual proxy | **Built-in `Access-Control-Allow-Origin: *`** |

---

## 🌐 Permanent Link Format

Your permanent URL is derived directly from your Hugging Face username and Space name:

```
https://<YOUR_HF_USERNAME>-<YOUR_SPACE_NAME>.hf.space
```

*(Note: Replace spaces and underscores with hyphens in lowercase. For example, user `Tamaki` and space `crop-disease-detection` yields `https://tamaki-crop-disease-detection.hf.space`)*

---

## 🔌 API Endpoints Reference

### 1. Leaf Prediction (`POST /api/predict`)
Accepts multipart form-data with the image file. Compatible directly with `anni88git` frontend.

- **URL:** `https://<YOUR-SPACE>.hf.space/api/predict`
- **Method:** `POST`
- **Body:** `multipart/form-data` with field `file`

#### Sample Response:
```json
{
  "success": true,
  "predicted_class": "Tomato___Early_blight",
  "crop": "Tomato",
  "condition": "Early Blight",
  "is_healthy": false,
  "confidence": 99.12,
  "confidence_level": "High",
  "severity": "Moderate to High",
  "pathogen": "Alternaria solani (Fungus)",
  "description": "Fungal pathogen causing dark concentric rings (target-like spots)...",
  "symptoms": ["Concentric brown rings on lower leaves", "Yellowing halo around spots"],
  "organic_treatment": ["Apply copper-based fungicides", "Prune infected bottom leaves"],
  "chemical_treatment": ["Chlorothalonil or Mancozeb sprays"],
  "prevention": ["Avoid overhead irrigation", "Mulch base of plants"],
  "top5": [
    {
      "class_name": "Tomato___Early_blight",
      "crop": "Tomato",
      "condition": "Early Blight",
      "confidence": 99.12,
      "is_healthy": false
    }
  ],
  "telemetry": {
    "latency_ms": 142.5,
    "input_resolution": "384x384",
    "model_architecture": "Vision Transformer (ViT-Base-Patch16-224 · 98.97% Accuracy)",
    "classes_evaluated": 38
  }
}
```

---

### 2. Base64 Prediction (`POST /api/predict/json`)
- **URL:** `https://<YOUR-SPACE>.hf.space/api/predict/json`
- **Method:** `POST`
- **Body:** `{"image_base64": "data:image/jpeg;base64,..."}`

---

### 3. List All 38 Classes (`GET /api/classes`)
- **URL:** `https://<YOUR-SPACE>.hf.space/api/classes`
- **Method:** `GET`
- Returns full agronomic directory with symptoms, treatments, and pathogens for all 38 classes.

---

### 4. Health Check (`GET /api/health` and `/ping`)
- **URL:** `https://<YOUR-SPACE>.hf.space/api/health`
- Returns model loaded status and service health.

---

## 🛠️ Connecting External Projects (anni88git Frontend)

To make `anni88git/CropDiseaseDetection` (or your Render frontend) call this Hugging Face Space:

In `static/index.html`:
```javascript
// Replace this line:
// const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:8080' : '';

// With your permanent Hugging Face Space URL:
const API_BASE = 'https://<YOUR-HF-USERNAME>-<YOUR-SPACE-NAME>.hf.space';
```

Now, all leaf analysis requests will bypass Render's slow CPU and compute directly on Hugging Face in ~150ms!
