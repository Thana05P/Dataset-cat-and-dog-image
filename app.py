import os
import gradio as gr
from PIL import Image
from ultralytics import YOLO

# 1. โหลดโมเดล best.pt
model_path = os.path.join("runs", "classify", "runs_classify", "custom_classifier_exp", "weights", "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")

model = YOLO(model_path)

# 2. ฟังก์ชันสำหรับ Predict
def predict_image(img):
    if img is None:
        return None
    
    results = model(img)
    result = results[0]
    
    confidences = {}
    for idx, conf in enumerate(result.probs.data.tolist()):
        class_name = result.names[idx]
        confidences[class_name] = float(conf)
        
    return confidences

# CSS สำหรับซ่อนปุ่มขยายจอ (Fullscreen/Maximize button)
custom_css = """
button[aria-label="fullscreen"], 
button[title="Full screen"],
.fullscreen-button {
    display: none !important;
}
"""

# 3. สร้าง Web Interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(
        type="pil", 
        sources=["upload"],                # เปิดเฉพาะปุ่มอัปโหลด
        label="🖼️ อัปโหลดรูปภาพสุนัขหรือแมว"
    ),
    outputs=gr.Label(num_top_classes=2, label="📊 ผลการจำแนกประเภท (Probability)"),
    title="🐱🐶 Cat vs Dog Classification Web App",
    description="อัปโหลดรูปภาพเพื่อทดสอบการจำแนกสุนัขหรือแมวด้วยโมเดล YOLO11 Custom Classifier",
    theme="soft",
    css=custom_css,                        # ซ่อนปุ่ม Fullscreen ด้วย CSS
    flagging_mode="never"                  # ปิดปุ่ม Flag ด้านล่าง
)

# 4. เริ่มรันเว็บเซิร์ฟเวอร์
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)