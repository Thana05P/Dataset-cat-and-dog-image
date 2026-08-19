import os
from pathlib import Path
from PIL import Image
import imagehash
import kagglehub

def download_dataset():
    """ดึง Dataset อัตโนมัติผ่าน Kaggle API ตามข้อกำหนดของโจทย์"""
    print("กำลังดาวน์โหลด Dataset จาก Kaggle...")
    path = kagglehub.dataset_download("samuelcortinhas/cats-and-dogs-image-classification")
    print(f"ดาวน์โหลดสำเร็จ! พาธข้อมูลอยู่ที่: {path}")
    return path

def comprehensive_preprocessing(data_dir):
    """
    ดำเนินการตามหัวข้อ 4.3 Data Preprocessing:
    1. ลบไฟล์เสียหาย (Corrupted Images)
    2. แปลง Format และ Color Space เป็น RGB มาตรฐาน
    3. ตรวจจับและจัดการรูปภาพซ้ำ (Duplicate Detection ด้วย Perceptual Hash)
    4. ตรวจสอบความสมดุลของข้อมูลระหว่าง Class (Class Imbalance)
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
                # 1. ตรวจสอบความเสียหายของภาพ
                with Image.open(img_path) as img:
                    img.verify()
                
                with Image.open(img_path) as img:
                    # 4. แปลง Color Space เป็น RGB เสมอ
                    img = img.convert('RGB')
                    
                    # 2. ตรวจจับรูปภาพซ้ำ (Duplicate Detection ด้วย Perceptual Hash)
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
                img_path.unlink() # 1. ลบไฟล์ที่เสียหาย
                removed_corrupt += 1

    print("\n--- สรุปผลการทำ Data Preprocessing ---")
    print(f"ลบภาพเสีย (Corrupted): {removed_corrupt} ไฟล์")
    print(f"ลบภาพซ้ำ (Duplicates): {removed_duplicates} ไฟล์")
    
    # 3. รายงานสถิติ Class Imbalance เพื่อวางแผนจัดการต่อ (เช่น Oversampling/Undersampling)
    print("\nจำนวนข้อมูลแต่ละ Class (Class Imbalance Check):")
    for cls, count in class_counts.items():
        print(f"- Class '{cls}': {count} รูปภาพ")

if __name__ == "__main__":
    dataset_path = download_dataset()
    # เปิดใช้งานฟังก์ชันหลัก
    comprehensive_preprocessing(dataset_path)