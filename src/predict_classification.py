"""
สคริปต์ทำนายผล Image Classification แสดงผลจัดอันดับ Probabilities Ranking
"""

import sys
import io
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import glob
import cv2
from ultralytics import YOLO

def classify_and_visualize(image_path=None, model_path=None):
    print("=" * 65)
    print(" 🔮 RUNNING IMAGE CLASSIFICATION INFERENCE")
    print("=" * 65)

    current_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # 1. โหลดโมเดลที่เทรนเสร็จแล้ว (best.pt)
    if model_path is None:
        model_path = os.path.join(current_project_dir, "runs", "classify", "runs_classify", "custom_classifier_exp", "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
        return

    print(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)

    # 2. เลือกลำดับภาพทดสอบจาก Dataset/test
    if image_path is None or not os.path.exists(image_path):
        test_images = glob.glob(os.path.join(current_project_dir, "Dataset", "test", "*", "*.*"))
        if test_images:
            image_path = test_images[0]
            print(f"📂 Auto-selected test image: {image_path}")
        else:
            print("❌ ไม่พบรูปภาพในโฟลเดอร์ Dataset/test/")
            return
    else:
        print(f"📂 Loading image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ ไม่สามารถเปิดไฟล์รูปภาพ: {image_path}")
        return

    # 3. รันการทำนายผล
    results = model.predict(source=image, verbose=False)
    result = results[0]

    # 4. ดึง Top-1 และจัดอันดับ Ranking
    top1_idx = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    top1_name = result.names[top1_idx]

    # จัดอันดับคลาสทั้งหมดจากความมั่นใจมากไปน้อย
    sorted_indices = result.probs.top5 if hasattr(result.probs, 'top5') else [top1_idx]

    print(f"\n🏆 Prediction: '{top1_name}' ({top1_conf * 100:.2f}%)")
    print(f"\n📊 Probabilities Ranking:")
    for rank, idx in enumerate(sorted_indices, 1):
        cname = result.names[idx]
        conf = float(result.probs.data[idx])
        print(f"   {rank}. {cname:20s} : {conf * 100:6.2f}%")

    # 5. วาด Overlay ข้อความลงบนภาพ (ขยาย/ย่อตามขนาดภาพจริงอัตโนมัติ)
    annotated_img = image.copy()
    label_text = f"Top-1: {top1_name} ({top1_conf * 100:.1f}%)"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    
    # คำนวณขนาดความกว้าง-สูงของข้อความจริง
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    # วาดกล่องสีดำรองพื้นพอดีตัวอักษร
    margin = 8
    cv2.rectangle(
        annotated_img, 
        (10, 10), 
        (10 + text_w + margin * 2, 10 + text_h + margin * 2), 
        (0, 0, 0), 
        -1
    )
    
    # วาดข้อความสีเขียว
    cv2.putText(
        annotated_img, 
        label_text, 
        (10 + margin, 10 + text_h + margin), 
        font, 
        font_scale, 
        (0, 255, 0), 
        thickness
    )
    
    # 6. บันทึกภาพลง reports/figures/
    output_dir = os.path.join(current_project_dir, "reports", "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "classification_result.jpg")
    cv2.imwrite(output_path, annotated_img)
    print(f"\n💾 Annotated result saved to: '{output_path}'")

if __name__ == '__main__':
    classify_and_visualize()