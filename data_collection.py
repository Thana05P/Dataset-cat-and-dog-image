import os
import shutil
from pathlib import Path
import kagglehub

def get_dataset_path():
    """
    ดาวน์โหลด Dataset จาก Kaggle และคัดลอกไฟล์มาไว้ในโฟลเดอร์โปรเจกต์
    พร้อมนับจำนวนไฟล์ทั้งหมดที่ดึงมาได้
    """
    print("[Data Collection] กำลังดาวน์โหลด Dataset จาก Kaggle...")
    
    # ดาวน์โหลดผ่าน kagglehub
    download_path = kagglehub.dataset_download("samuelcortinhas/cats-and-dogs-image-classification")
    
    current_dir = Path(os.path.abspath(__file__)).parent
    local_dataset_dir = current_dir / 'Dataset-cat-and-dog-image'
    
    # คัดลอกไฟล์ลงโฟลเดอร์โปรเจกต์
    if not local_dataset_dir.exists():
        print(f"📂 กำลังคัดลอกไฟล์มาไว้ที่: {local_dataset_dir}")
        shutil.copytree(download_path, local_dataset_dir)
        print("✅ คัดลอกโฟลเดอร์ Dataset สำเร็จ!")
    else:
        print("📂 พบโฟลเดอร์ Dataset ในโปรเจกต์อยู่แล้ว")

    # --- ส่วนที่เพิ่มมา: นับจำนวนไฟล์รูปภาพ ---
    total_images = 0
    class_counts = {}
    
    # วิ่งหาไฟล์นามสกุลรูปภาพ
    for img_path in local_dataset_dir.rglob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            total_images += 1
            class_name = img_path.parent.name # ชื่อโฟลเดอร์ย่อย (เช่น cats, dogs)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
    print("\n" + "="*40)
    print("--- สรุปจำนวนรูปภาพที่ดึงมา ---")
    for cls, count in class_counts.items():
        print(f"คลาส '{cls}': {count} รูป")
    print(f"รวมทั้งหมด: {total_images} รูป")
    print("="*40)
    # --------------------------------------

    return str(local_dataset_dir)

if __name__ == "__main__":
    get_dataset_path()