import os
import io
import time
import json
import base64
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any

from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr

from disease_info import get_disease_info, DISEASE_DB

# ==============================================================================
# Model & Pipeline Configuration
# ==============================================================================
HF_MODEL_ID = "RuougiShiki/crop-disease-vit"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check local model paths (supporting both standalone space and repo subdirectory)
CANDIDATE_MODEL_DIRS = [
    os.path.join(BASE_DIR, 'best_vit_plant_model'),
    os.path.join(BASE_DIR, '..', 'best_vit_plant_model')
]

LOCAL_MODEL_DIR = None
for candidate in CANDIDATE_MODEL_DIRS:
    if os.path.exists(os.path.join(candidate, 'model.safetensors')):
        LOCAL_MODEL_DIR = candidate
        break

# Fallback classes definition
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Set CPU thread optimization
torch.set_num_threads(min(4, os.cpu_count() or 4))

print("Loading Vision Transformer pipeline...")
t0 = time.time()
model_source = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR else HF_MODEL_ID
print(f"Loading weights from: {model_source}")

try:
    processor = AutoImageProcessor.from_pretrained(model_source)
    model = AutoModelForImageClassification.from_pretrained(model_source)
    model.eval()

    # Load classes from model config if present
    if hasattr(model.config, 'id2label') and model.config.id2label:
        CLASS_NAMES = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    # Warm-up pass to eliminate cold start latency
    with torch.inference_mode():
        dummy_input = torch.zeros((1, 3, 224, 224))
        _ = model(dummy_input)

    print(f"Vision Transformer loaded & warmed up in {time.time() - t0:.2f}s")
except Exception as e:
    print(f"Error loading model from {model_source}: {e}")
    processor = None
    model = None

# ==============================================================================
# Core Inference Function
# ==============================================================================
def run_model_inference(image: Image.Image) -> Dict[str, Any]:
    """Execute ViT inference with fast preprocessing and comprehensive agronomic metadata."""
    if model is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Vision Transformer is not ready or model weights failed to load."
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Fast downscale for high-megapixel camera uploads to save CPU time
    if image.width > 384 or image.height > 384:
        image.thumbnail((384, 384), Image.Resampling.BILINEAR)

    t_start = time.time()
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    top5_probs, top5_indices = torch.topk(probabilities, 5)
    latency_ms = round((time.time() - t_start) * 1000, 1)

    best_idx = int(top5_indices[0].item())
    best_class = model.config.id2label[best_idx] if hasattr(model.config, 'id2label') and best_idx in model.config.id2label else CLASS_NAMES[best_idx]
    confidence = float(top5_probs[0].item() * 100.0)

    top5 = []
    for prob, idx in zip(top5_probs, top5_indices):
        c_idx = int(idx.item())
        cls_name = model.config.id2label[c_idx] if hasattr(model.config, 'id2label') and c_idx in model.config.id2label else CLASS_NAMES[c_idx]
        cls_info = get_disease_info(cls_name)
        top5.append({
            "class_name": cls_name,
            "crop": cls_info['crop'],
            "condition": cls_info['condition'],
            "confidence": round(float(prob.item() * 100.0), 2),
            "is_healthy": cls_info['is_healthy']
        })

    disease_data = get_disease_info(best_class)
    is_unknown = confidence < 60.0

    return {
        "success": True,
        "predicted_class": best_class,
        "crop": disease_data['crop'],
        "condition": disease_data['condition'],
        "is_healthy": disease_data['is_healthy'],
        "confidence": round(confidence, 2),
        "confidence_level": "High" if confidence >= 85 else ("Moderate" if confidence >= 65 else "Low"),
        "is_unknown": is_unknown,
        "severity": disease_data['severity'],
        "pathogen": disease_data['pathogen'],
        "description": disease_data['description'],
        "symptoms": disease_data['symptoms'],
        "organic_treatment": disease_data['organic_treatment'],
        "chemical_treatment": disease_data['chemical_treatment'],
        "prevention": disease_data['prevention'],
        "top5": top5,
        "telemetry": {
            "latency_ms": latency_ms,
            "input_resolution": f"{image.width}x{image.height}",
            "model_architecture": "Vision Transformer (ViT-Base-Patch16-224 · 98.97% Accuracy)",
            "classes_evaluated": len(CLASS_NAMES)
        }
    }


# ==============================================================================
# Gradio UI Inference Wrapper
# ==============================================================================
def gradio_predict(image):
    """Processes leaf image from Gradio UI and outputs structured markdown, confidences, and treatments."""
    if image is None:
        return (
            "⚠️ Please upload or capture a crop leaf image to start the diagnosis.",
            {},
            "No active diagnosis.",
            "No active diagnosis.",
            "No active diagnosis.",
            {}
        )

    res = run_model_inference(image)
    crop = res["crop"]
    condition = res["condition"]
    conf = res["confidence"]
    is_healthy = res["is_healthy"]
    is_unknown = res["is_unknown"]
    severity = res["severity"]
    pathogen = res["pathogen"]

    if is_unknown:
        status_md = f"""
### ⚠️ Low Diagnostic Confidence ({conf:.1f}%)
The Vision Transformer detected **{crop} - {condition}** with only **{conf:.1f}%** confidence.
This image may not be a supported agricultural crop leaf, or the lighting/angle is obscured.
"""
    elif is_healthy:
        status_md = f"""
### ✅ Healthy Leaf Verified
**Crop:** {crop}  
**Diagnostic Status:** Healthy (No pathological lesions detected)  
**Confidence:** `{conf:.2f}%` ({res['confidence_level']} Confidence)  
**Inference Latency:** `{res['telemetry']['latency_ms']} ms`
"""
    else:
        status_md = f"""
### 🚨 Disease Detected: {condition}
**Crop:** {crop}  
**Pathogen:** {pathogen}  
**Severity Level:** `{severity}`  
**Diagnostic Confidence:** `{conf:.2f}%` ({res['confidence_level']} Confidence)  
**Inference Latency:** `{res['telemetry']['latency_ms']} ms`
"""

    # Top-5 dict for gr.Label
    label_dict = {
        f"{item['crop']} — {item['condition']}": item["confidence"] / 100.0
        for item in res["top5"]
    }

    # Clinical Symptoms Markdown
    symptoms_list = "\n".join([f"- {s}" for s in res.get("symptoms", [])])
    symptoms_md = f"""
**Clinical Overview:**  
{res['description']}

**Symptoms:**  
{symptoms_list}
"""

    # Treatments Markdown
    organic_list = "\n".join([f"- {o}" for o in res.get("organic_treatment", [])])
    chemical_list = "\n".join([f"- {c}" for c in res.get("chemical_treatment", [])])
    treatments_md = f"""
#### 🌿 Certified Organic Remedies
{organic_list if organic_list else "- None required (Plant is healthy)."}

#### 🧪 Chemical Treatment & Dosages
{chemical_list if chemical_list else "- No chemical application needed."}
"""

    # Prevention Markdown
    prevention_list = "\n".join([f"- {p}" for p in res.get("prevention", [])])
    prevention_md = f"""
#### 🛡️ Preventative Field Measures
{prevention_list if prevention_list else "- Maintain standard crop scouting and irrigation hygiene."}
"""

    return (
        status_md,
        label_dict,
        symptoms_md,
        treatments_md,
        prevention_md,
        res
    )


# ==============================================================================
# FastAPI REST Endpoints (Compatible with anni88git Frontend)
# ==============================================================================
fastapi_app = FastAPI(
    title="Crop Disease ViT Inference Server",
    description="High-performance Vision Transformer API for crop disease detection",
    version="2.0.0"
)

# Enable CORS for external frontends (e.g. Render, Vercel, Netlify, localhost)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Base64PredictRequest(BaseModel):
    image_base64: Optional[str] = None
    sample_name: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    user_name: Optional[str] = "Farmer"
    user_purpose: Optional[str] = "Detection"
    context: Optional[str] = None
    diagnosis: Optional[dict] = None

@fastapi_app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    """Primary REST endpoint called by frontend via FormData."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        return run_model_inference(image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

@fastapi_app.post("/api/predict/json")
async def api_predict_json(req: Base64PredictRequest):
    """JSON inference endpoint accepting base64 encoded images."""
    try:
        if req.sample_name:
            examples_dir = os.path.join(BASE_DIR, 'examples')
            sample_path = os.path.join(examples_dir, os.path.basename(req.sample_name))
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=404, detail="Sample image not found.")
            image = Image.open(sample_path)
            return run_model_inference(image)
        elif req.image_base64:
            b64_str = req.image_base64
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            data = base64.b64decode(b64_str)
            image = Image.open(io.BytesIO(data))
            return run_model_inference(image)
        else:
            raise HTTPException(status_code=400, detail="image_base64 or sample_name is required.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@fastapi_app.get("/api/classes")
async def api_classes():
    """Return all 38 supported crop disease classes with agronomic knowledge."""
    classes_list = []
    for cls in CLASS_NAMES:
        info = get_disease_info(cls)
        classes_list.append({
            "id": cls,
            "crop": info['crop'],
            "condition": info['condition'],
            "is_healthy": info['is_healthy'],
            "severity": info.get('severity', 'Standard'),
            "pathogen": info.get('pathogen', 'N/A'),
            "description": info.get('description', ''),
            "symptoms": info.get('symptoms', []),
            "organic_treatment": info.get('organic_treatment', []),
            "chemical_treatment": info.get('chemical_treatment', []),
            "prevention": info.get('prevention', [])
        })
    return {"classes": classes_list, "total": len(classes_list)}

@fastapi_app.get("/api/health")
async def api_health():
    """Service health and telemetry status."""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "model_architecture": "Vision Transformer (ViT-Base-Patch16-224)",
        "model_accuracy": "98.97%",
        "classes_count": len(CLASS_NAMES),
        "engine": "Hugging Face Space Docker (Gradio + FastAPI)"
    }

@fastapi_app.get("/ping")
async def ping():
    return {"pong": True}

@fastapi_app.post("/api/chat")
async def chat_with_bot(req: ChatRequest):
    """Agronomic AI chat endpoint with Groq LLM integration or smart agricultural fallback."""
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        system_prompt = (
            f"You are an expert agronomist assisting {req.user_name or 'Farmer'}. "
            "Provide field-actionable, scientific advice. Structure answers clearly: "
            "1) Diagnosis cause, 2) Organic & chemical dosages, 3) Prevention. Keep under 180 words."
        )
        if req.diagnosis:
            system_prompt += f"\nACTIVE LEAF DIAGNOSIS: {req.diagnosis.get('crop')} - {req.diagnosis.get('condition')} ({req.diagnosis.get('confidence')}%)"
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message}
            ],
            "temperature": 0.2,
            "max_tokens": 400
        }
        try:
            url_req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
            )
            with urllib.request.urlopen(url_req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {"success": True, "reply": data["choices"][0]["message"]["content"].strip()}
        except Exception as e:
            pass

    # Agronomic fallback if GROQ_API_KEY is not configured
    if req.diagnosis:
        d = req.diagnosis
        return {
            "success": True,
            "reply": f"For {d.get('crop')} diagnosed with {d.get('condition')} (Severity: {d.get('severity', 'Moderate')}):\n\n"
                     f"• Recommended Action: Apply certified fungicide/treatment: {', '.join(d.get('organic_treatment', ['Prune affected foliage']))}.\n"
                     f"• Preventative Measure: {', '.join(d.get('prevention', ['Ensure proper canopy aeration and sanitize tools']))}."
        }
    return {
        "success": True,
        "reply": "Please upload a crop leaf image first to diagnose potential pathologies and receive customized treatment guidelines."
    }


# ==============================================================================
# Gradio Blocks Web Interface
# ==============================================================================
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.header-box {
    text-align: center;
    padding: 24px;
    background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
    border-radius: 14px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(6, 78, 59, 0.15);
}
.header-box h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    color: #f0fdf4;
}
.header-box p {
    font-size: 1.05rem;
    margin-top: 8px;
    color: #a7f3d0;
}
.badge-row {
    margin-top: 12px;
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255, 255, 255, 0.15);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #ecfdf5;
}
"""

with gr.Blocks(title="AI Crop Disease Detector & Fast API", css=custom_css, theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="green")) as demo:
    gr.HTML("""
        <div class="header-box">
            <h1>🌾 Vision Transformer Crop Disease Diagnostic Engine</h1>
            <p>High-accuracy leaf pathology detection powered by ViT-Base (98.97% accuracy on 38 disease classes)</p>
            <div class="badge-row">
                <span class="badge">🚀 Hugging Face Space Docker</span>
                <span class="badge">⚡ Sub-second ViT Inference</span>
                <span class="badge">🌐 Permanent REST API for External Apps</span>
                <span class="badge">🌿 38 Plant Classes</span>
            </div>
        </div>
    """)

    with gr.Row():
        # Left Column: Input and Controls
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Crop Leaf Image",
                sources=["upload", "clipboard"]
            )
            with gr.Row():
                submit_btn = gr.Button("🔍 Diagnose Leaf", variant="primary", scale=2)
                clear_btn = gr.Button("🧹 Clear", scale=1)

            # Example images if available
            examples_dir = os.path.join(BASE_DIR, "examples")
            if os.path.exists(examples_dir):
                example_files = [
                    os.path.join(examples_dir, f)
                    for f in sorted(os.listdir(examples_dir))
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ][:6]
                if example_files:
                    gr.Examples(
                        examples=example_files,
                        inputs=image_input,
                        label="Clickable Sample Test Images"
                    )

        # Right Column: Diagnostic Results
        with gr.Column(scale=1):
            diagnostic_banner = gr.Markdown("### 🌿 Diagnostic Ready\nUpload a leaf photograph on the left to begin.")
            label_output = gr.Label(num_top_classes=5, label="Top 5 Predicted Probabilities")

            with gr.Tabs():
                with gr.Tab("🩺 Clinical Symptoms & Pathogen"):
                    symptoms_display = gr.Markdown("Symptoms and clinical details will appear here after diagnosis.")
                
                with gr.Tab("🌿 Treatments & Remedies"):
                    treatments_display = gr.Markdown("Certified organic and chemical recommendations will appear here.")

                with gr.Tab("🛡️ Field Prevention"):
                    prevention_display = gr.Markdown("Prevention best practices will appear here.")

                with gr.Tab("⚡ Raw JSON Response"):
                    json_display = gr.JSON(label="Structured Response (as returned to external APIs)")

    # Permanent API Documentation Section
    with gr.Accordion("🌐 Permanent API Link & Integration Guide (Connect anni88git or External Web Apps)", open=False):
        gr.Markdown("""
### Permanent API Access Instructions
When hosted on Hugging Face Spaces, this app exposes a **permanent link** accessible globally with **CORS enabled**.

#### 1. Direct REST API Endpoint:
```
POST https://<YOUR-HF-USERNAME>-<YOUR-SPACE-NAME>.hf.space/api/predict
Content-Type: multipart/form-data
Body: file: <Image File>
```

#### 2. JavaScript / Fetch Integration (for anni88git `index.html`):
In `index.html`, set:
```javascript
const API_BASE = 'https://<YOUR-HF-USERNAME>-<YOUR-SPACE-NAME>.hf.space';

// Inference call:
const form = new FormData();
form.append('file', imageFile);

const response = await fetch(API_BASE + '/api/predict', {
  method: 'POST',
  body: form
});
const result = await response.json();
console.log(result.predicted_class, result.confidence, result.organic_treatment);
```

#### 3. Python `requests` Client:
```python
import requests

url = "https://<YOUR-HF-USERNAME>-<YOUR-SPACE-NAME>.hf.space/api/predict"
with open("leaf.jpg", "rb") as f:
    res = requests.post(url, files={"file": f})
print(res.json())
```

#### 4. Official Gradio Client:
```python
from gradio_client import Client, handle_file

client = Client("https://<YOUR-HF-USERNAME>-<YOUR-SPACE-NAME>.hf.space")
result = client.predict(
    image=handle_file("leaf.jpg"),
    api_name="/predict"
)
```
""")

    # Event Handlers
    submit_btn.click(
        fn=gradio_predict,
        inputs=[image_input],
        outputs=[diagnostic_banner, label_output, symptoms_display, treatments_display, prevention_display, json_display],
        api_name="predict"
    )

    def clear_all():
        return None, "### 🌿 Diagnostic Ready\nUpload a leaf photograph on the left to begin.", {}, "", "", "", {}

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, diagnostic_banner, label_output, symptoms_display, treatments_display, prevention_display, json_display]
    )

# Mount Gradio UI onto FastAPI root so both the Web UI and REST API run simultaneously on port 7860
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Gradio + FastAPI service on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
