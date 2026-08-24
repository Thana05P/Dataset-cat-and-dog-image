"""
สคริปต์ฝึกสอนโมเดล Image Classification ด้วย Ultralytics YOLO
เหมาะสำหรับงาน: จำแนกขยะ, แยกโรคพืช, คัดเกรดผลไม้, ตรวจตำหนิชิ้นส่วน
"""

import sys
import io
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
from ultralytics import YOLO

def train_custom_classifier():
    print("=" * 65)
    print(" 🚀 STARTING YOLO CLASSIFICATION TRAINING")
    print("=" * 65)

    # 1. เลือกรุ่น Pretrained Model (เช่น yolo11n-cls.pt หรือ yolov8n-cls.pt)
    model_name = "yolo11n-cls.pt"
    print(f"📦 Loading base model: {model_name}")
    model = YOLO(model_name)

    # 2. กำหนด Path ชุดข้อมูล (Dataset Path)
    # หมายเหตุ: โฟลเดอร์ต้องมีโครงสร้าง train/ และ val/
    dataset_path = "Dataset"  # สามารถเปลี่ยนเป็น path โฟลเดอร์ dataset ของนักศึกษาได้
    
    print(f"📂 Dataset Target: {dataset_path}")
    print("⚙️ Training Hyperparameters:")
    print("   - Image Size (imgsz) : 224")
    print("   - Epochs             : 5 (สำหรับ Demo)")
    print("   - Batch Size         : 16")
    print("   - Device             : cpu / 0 (auto)")

    # 3. สั่งฝึกสอนโมเดล (Train Model)
    results = model.train(
        data=dataset_path,
        epochs=150,
        imgsz=224,
        batch=16,
        project="runs_classify",
        name="custom_classifier_exp",
        exist_ok=True,
        verbose=True
    )

    print("\n" + "=" * 65)
    print(" ✅ TRAINING FINISHED SUCCESSFULLY!")
    print(f" 💾 Model Weights Saved to: runs_classify/custom_classifier_exp/weights/best.pt")
    print("=" * 65)

if __name__ == '__main__':
    train_custom_classifier()
