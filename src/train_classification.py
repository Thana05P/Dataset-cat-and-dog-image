"""
สคริปต์ฝึกสอนโมเดล Image Classification ด้วย Ultralytics YOLO
จำแนกภาพสุนัขและแมว (Cat vs Dog Classification)
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

    # 1. หาตำแหน่ง Root Directory ของโปรเจกต์ (ถอยออกจาก src/ มา 1 ระดับ)
    current_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # 2. เลือกรุ่น Pretrained Model
    model_name = "yolo11n-cls.pt"
    print(f"📦 Loading base model: {model_name}")
    model = YOLO(model_name)

    # 3. กำหนด Path ชุดข้อมูล และโฟลเดอร์บันทึกผลลัพธ์
    dataset_path = os.path.join(current_project_dir, "Dataset")
    runs_dir = os.path.join(current_project_dir, "runs", "classify", "runs_classify")
    
    print(f"📂 Dataset Target: {dataset_path}")
    print("⚙️ Training Hyperparameters:")
    print("   - Image Size (imgsz) : 224")
    print("   - Epochs             : 150")
    print("   - Batch Size         : 16")
    print("   - Device             : cpu / 0 (auto)")

    # 4. สั่งฝึกสอนโมเดล (Train Model)
    results = model.train(
        data=dataset_path,
        epochs=150,
        imgsz=224,
        batch=16,
        project=runs_dir,
        name="custom_classifier_exp",
        exist_ok=True,
        verbose=True
    )

    save_weight_path = os.path.join(runs_dir, "custom_classifier_exp", "weights", "best.pt")
    print("\n" + "=" * 65)
    print(" ✅ TRAINING FINISHED SUCCESSFULLY!")
    print(f" 💾 Model Weights Saved to: {save_weight_path}")
    print("=" * 65)

if __name__ == '__main__':
    train_custom_classifier()