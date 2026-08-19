import os
import shutil
from pathlib import Path
from PIL import Image, ImageFile
import imagehash

# อนุญาตให้โหลดภาพที่ไม่สมบูรณ์เพื่อป้องกัน Error
ImageFile.LOAD_TRUNCATED_IMAGES = True

def comprehensive_preprocessing():
    """
    1. ดึงข้อมูลจากโฟลเดอร์ 'Dataset-cat-and-dog-image' ในปัจจุบัน
    2. ทำความสะอาด (ลบภาพเสีย, ภาพซ้ำ, แปลง RGB)
    3. บันทึกผลลัพธ์ที่ Clean แล้วลงในโฟลเดอร์ 'data/processed/' ในโฟลเดอร์ปัจจุบัน
    """
    current_project_dir = Path(os.path.abspath(__file__)).parent
    data_path = current_project_dir / "Dataset-cat-and-dog-image"
    output_dir = current_project_dir / "data" / "processed"
    
    if not data_path.exists():
        print(f"❌ ไม่พบโฟลเดอร์ Dataset ที่พาธ: {data_path}")
        print("กรุณานำโฟลเดอร์ 'Dataset-cat-and-dog-image' มาวางไว้ในโฟลเดอร์เดียวกับสคริปต์นี้ก่อน")
        return

    # สร้างโฟลเดอร์ปลายทางสำหรับเก็บข้อมูลที่ clean แล้ว
    output_dir.mkdir(parents=True, exist_ok=True)

    removed_corrupt = 0
    removed_duplicates = 0
    hashes = {} 
    saved_counts = {}

    print(f"[*] เริ่มกระบวนการ Data Preprocessing และย้ายข้อมูลมาที่โฟลเดอร์: {output_dir}\n")
    
    for img_path in data_path.glob("**/*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # ข้ามโฟลเดอร์หลัก
            if img_path.parent.name in ["Dataset-cat-and-dog-image"]:
                continue
                
            class_name = img_path.parent.name
            class_out_dir = output_dir / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 1. ตรวจสอบความเสียหายของภาพ
                with Image.open(img_path) as img:
                    img.verify()
                
                with Image.open(img_path) as img:
                    # 2. แปลง Color Space เป็น RGB เสมอ
                    img = img.convert('RGB')
                    
                    # 3. ตรวจจับรูปภาพซ้ำ (Duplicate Detection ด้วย Perceptual Hash)
                    img_hash = imagehash.average_hash(img)
                    if img_hash in hashes:
                        print(f"พบภาพซ้ำ ข้ามการบันทึก: {img_path.name}")
                        removed_duplicates += 1
                        continue
                    else:
                        hashes[img_hash] = img_path
                        
                        # 4. บันทึกภาพที่ clean แล้วลงในโฟลเดอร์ปัจจุบัน (data/processed/)
                        out_file = class_out_dir / img_path.name
                        img.save(out_file)
                        
                        saved_counts[class_name] = saved_counts.get(class_name, 0) + 1

            except (IOError, SyntaxError):
                print(f"พบภาพเสีย ข้ามการบันทึก: {img_path.name}")
                removed_corrupt += 1

    print("\n" + "="*45)
    print("--- สรุปผลการทำ Data Preprocessing & Moving ---")
    print(f"ภาพเสีย (Corrupted) ที่คัดทิ้ง: {removed_corrupt} ไฟล์")
    print(f"ภาพซ้ำ (Duplicates) ที่คัดทิ้ง: {removed_duplicates} ไฟล์")
    print("-" * 45)
    print("จำนวนไฟล์ที่ Clean และบันทึกมาไว้ในโฟลเดอร์ปัจจุบันสำเร็จ:")
    for cls, count in saved_counts.items():
        print(f"  - Class '{cls}': {count} รูปภาพ")
    print(f"  - บันทึกไว้ที่: {output_dir}")
    print("="*45)

if __name__ == "__main__":
    comprehensive_preprocessing()