"""
สคริปต์ทำนายผล Image Classification พร้อมแสดงภาพและค่าเปอร์เซ็นต์ความมั่นใจ
"""

import sys
import io
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import numpy as np
from ultralytics import YOLO

def classify_and_visualize(image_path=None, model_path="yolo11n-cls.pt"):
    print("=" * 65)
    print(" 🔮 RUNNING IMAGE CLASSIFICATION INFERENCE")
    print("=" * 65)

    # 1. โหลดโมเดล
    print(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)

    # 2. จัดเตรียมรูปภาพนำเข้า (หากไม่ได้ระบุ จะสร้างรูปจำลองขึ้นมาทดสอบ)
    if image_path is None or not cv2.haveImageReader(image_path):
        print("🖼️ Generating synthetic image for demonstration...")
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(image, (150, 150), 80, (0, 200, 255), -1)
        cv2.putText(image, "TEST", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    else:
        image = cv2.imread(image_path)

    # 3. รันการทำนายผล (Inference)
    results = model.predict(source=image, verbose=False)
    result = results[0]

    # 4. ดึง Top-1 และ Top-5 Predictions
    top1_idx = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    top1_name = result.names[top1_idx]

    top5_indices = result.probs.top5
    top5_confs = result.probs.top5conf.tolist()

    print(f"\n🏆 Top-1 Prediction: '{top1_name}' ({top1_conf*100:.2f}%)")
    print("\n📊 Top-5 Probabilities Ranking:")
    for rank, (idx, conf) in enumerate(zip(top5_indices, top5_confs), 1):
        cname = result.names[idx]
        print(f"   {rank}. {cname:20s} : {conf*100:6.2f}%")

    # 5. วาด Overlay ข้อความลงบนภาพ
    annotated_img = image.copy()
    label_text = f"Top-1: {top1_name} ({top1_conf*100:.1f}%)"
    cv2.rectangle(annotated_img, (10, 10), (280, 50), (0, 0, 0), -1)
    cv2.putText(annotated_img, label_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # บันทึกภาพผลลัพธ์
    output_path = "classification_result.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"\n💾 Annotated result saved to: '{output_path}'")

if __name__ == '__main__':
    classify_and_visualize()
