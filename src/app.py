import os
import io
import webbrowser
from PIL import Image
from ultralytics import YOLO
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

# 1. โหลดโมเดล YOLO (best.pt)
current_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(current_project_dir, "runs", "classify", "runs_classify", "custom_classifier_exp", "weights", "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")

model = YOLO(model_path)

# 2. UI หน้าเว็บ (HTML + CSS + JavaScript)
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐱🐶 Cat vs Dog Classifier</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
        .card { background-color: #1e293b; border-radius: 16px; padding: 32px; width: 100%; max-width: 460px; border: 1px solid #334155; text-align: center; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); }
        h1 { font-size: 24px; margin-bottom: 8px; color: #38bdf8; }
        p.subtitle { font-size: 14px; color: #94a3b8; margin-bottom: 24px; }
        .upload-area { border: 2px dashed #475569; border-radius: 12px; padding: 24px; cursor: pointer; background-color: #0f172a; margin-bottom: 20px; transition: 0.3s; }
        .upload-area:hover { border-color: #38bdf8; }
        .preview-img { max-width: 100%; max-height: 220px; border-radius: 8px; display: none; margin: 0 auto 12px auto; object-fit: cover; }
        input[type="file"] { display: none; }
        .btn { background-color: #0284c7; color: white; border: none; padding: 12px; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; width: 100%; transition: 0.2s; }
        .btn:hover { background-color: #0369a1; }
        .btn:disabled { background-color: #475569; cursor: not-allowed; }
        .result-container { margin-top: 24px; display: none; text-align: left; }
        .bar-group { margin-bottom: 12px; }
        .bar-label { display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 4px; }
        .progress-bg { background-color: #334155; height: 10px; border-radius: 5px; overflow: hidden; }
        .progress-fill { height: 100%; background-color: #38bdf8; width: 0%; transition: width 0.4s ease-out; }
        .status-text { margin-top: 12px; font-size: 14px; color: #cbd5e1; font-weight: 500; }
    </style>
</head>
<body>

    <div class="card">
        <h1>🐱🐶 Cat vs Dog Classifier</h1>
        <p class="subtitle">อัปโหลดภาพเพื่อจำแนกสุนัขหรือแมว</p>

        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <img id="preview" class="preview-img" alt="Preview">
            <span id="uploadText">📁 คลิกเพื่อเลือกรูปภาพ</span>
            <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])">
        </div>

        <button id="submitBtn" class="btn" onclick="predictImage()" disabled>🔍 วิเคราะห์รูปภาพ</button>
        <p id="status" class="status-text"></p>

        <div id="results" class="result-container">
            <div class="bar-group">
                <div class="bar-label">
                    <span>🐱 Cats</span>
                    <span id="catPercent">0%</span>
                </div>
                <div class="progress-bg">
                    <div id="catBar" class="progress-fill"></div>
                </div>
            </div>

            <div class="bar-group">
                <div class="bar-label">
                    <span>🐶 Dogs</span>
                    <span id="dogPercent">0%</span>
                </div>
                <div class="progress-bg">
                    <div id="dogBar" class="progress-fill"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        function handleFile(file) {
            if (!file) return;
            selectedFile = file;

            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.getElementById('preview');
                img.src = e.target.result;
                img.style.display = 'block';
                document.getElementById('uploadText').style.display = 'none';
            };
            reader.readAsDataURL(file);

            document.getElementById('submitBtn').disabled = false;
            document.getElementById('results').style.display = 'none';
            document.getElementById('status').innerText = '';
        }

        async function predictImage() {
            if (!selectedFile) return;

            const btn = document.getElementById('submitBtn');
            const status = document.getElementById('status');

            btn.disabled = true;
            status.innerText = '⏳ กำลังวิเคราะห์รูปภาพ...';

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('API Error');

                const data = await response.json();
                const catConf = (data.probabilities.cats * 100).toFixed(2);
                const dogConf = (data.probabilities.dogs * 100).toFixed(2);

                document.getElementById('catPercent').innerText = `${catConf}%`;
                document.getElementById('catBar').style.width = `${catConf}%`;

                document.getElementById('dogPercent').innerText = `${dogConf}%`;
                document.getElementById('dogBar').style.width = `${dogConf}%`;

                document.getElementById('results').style.display = 'block';
                status.innerText = `🏆 ผลลัพธ์: ${data.top1.toUpperCase()} (${(data.confidence * 100).toFixed(2)}%)`;

            } catch (err) {
                status.innerText = '❌ เกิดข้อผิดพลาดในการประมวลผล';
                console.error(err);
            } finally {
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)
    result = results[0]

    probs = {}
    for idx, conf in enumerate(result.probs.data.tolist()):
        cname = result.names[idx]
        probs[cname] = float(conf)

    top1_name = result.names[result.probs.top1]
    top1_conf = float(result.probs.top1conf.item())

    return JSONResponse({
        "top1": top1_name,
        "confidence": top1_conf,
        "probabilities": probs
    })

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)