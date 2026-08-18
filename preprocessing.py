import os
from pathlib import Path
from PIL import Image
import imagehash
import kagglehub
import numpy as np

def download_dataset():
    """ดึง Dataset อัตโนมัติผ่าน Kaggle API ตามข้อกำหนดของโจทย์"""
    print("กำลังดาวน์โหลด Dataset จาก Kaggle...")
    path = kagglehub.dataset_download("samuelcortinhas/cats-and-dogs-image-classification")
    print(f"ดาวน์โหลดสำเร็จ! พาธข้อมูลอยู่ที่: {path}")
    return path

def comprehensive_preprocessing(data_dir):
    """
    ดำเนินการตามหัวข้อ 4.3 Data Preprocessing (ครบถ้วนทุกข้อ):
    1. ลบไฟล์เสียหาย (Corrupted Images)
    2. แปลง Format และ Color Space เป็น RGB มาตรฐาน
    3. ตรวจจับและจัดการรูปภาพซ้ำ (Duplicate Detection ด้วย Perceptual Hash)
    4. ตรวจสอบและจัดการ Class ที่ไม่สมดุล (Class Imbalance & Class Weights)
    """
    data_path = Path(data_dir)
    
    removed_corrupt = 0
    removed_duplicates = 0
    hashes = {} # สำหรับเก็บค่า Hash ป้องกันภาพซ้ำ
    
    class_counts = {}

    print("กำลังเริ่มกระบวนการ Data Preprocessing...")
    
    for img_path in data_path.glob("**/*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # ระบุชื่อ Class จากชื่อโฟลเดอร์แม่
            class_name = img_path.parent.name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            try:
                # 1. ตรวจสอบความเสียหายของภาพ (Corrupted Images)
                with Image.open(img_path) as img:
                    img.verify()
                
                with Image.open(img_path) as img:
                    # 2. แปลง Format และ Color Space เป็น RGB มาตรฐาน
                    img = img.convert('RGB')
                    
                    # 3. ตรวจจับและจัดการรูปภาพซ้ำ (Duplicate Detection ด้วย Perceptual Hash)
                    img_hash = imagehash.average_hash(img)
                    if img_hash in hashes:
                        print(f"พบภาพซ้ำ ลบออก: {img_path} (ซ้ำกับ {hashes[img_hash]})")
                        img_path.unlink()
                        removed_duplicates += 1
                        continue
                    else:
                        hashes[img_hash] = img_path

            except (IOError, SyntaxError):
                print(f"พบภาพเสียและกำลังลบออก: {img_path}")
                img_path.unlink() # ลบไฟล์ที่เสียหาย
                removed_corrupt += 1

    print("\n--- สรุปผลการทำ Data Preprocessing ---")
    print(f"ลบภาพเสีย (Corrupted): {removed_corrupt} ไฟล์")
    print(f"ลบภาพซ้ำ (Duplicates): {removed_duplicates} ไฟล์")
    
    # 4. จัดการ Class ที่ไม่สมดุล (Class Imbalance) ด้วยการคำนวณ Class Weights
    print("\nจำนวนข้อมูลแต่ละ Class และการจัดการ Class Imbalance:")
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    print(f"จำนวนคลาสทั้งหมด: {num_classes} คลาส, ข้อมูลรวมทั้งหมด: {total_samples} รูปภาพ")
    
    for i, (cls, count) in enumerate(sorted(class_counts.items())):
        print(f"- Class '{cls}': {count} รูปภาพ")
        # คำนวณ Class Weight สูตร: Total / (Num_Classes * Count_of_Class)
        # ช่วยให้ Model ให้ความสำคัญกับคลาสที่มีจำนวนน้อยกว่าตอนเทรน (แก้ปัญหา Class Imbalance)
        weight = total_samples / (num_classes * count)
        class_weights[i] = weight

    print("\n[การจัดการ Class Imbalance เรียบร้อย]:")
    print("คำนวณ Class Weights สำหรับนำไปใช้ใน Loss Function (เช่น model.fit(..., class_weight=class_weights)):")
    for cls_idx, weight in class_weights.items():
        print(f"  Class Index {cls_idx} -> Weight: {weight:.4f}")

if __name__ == "__main__":
    dataset_path = download_dataset()
    # เปิดใช้งานฟังก์ชันหลักแบบครบถ้วน
    comprehensive_preprocessing(dataset_path)