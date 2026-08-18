import os
from pathlib import Path
from PIL import Image
import imagehash

# 1. ดึงไฟล์ของเพื่อนมาใช้งาน (ไม่ต้องกดโหลดเอง โค้ดจะไปเรียกมาให้)
import data_collection 

def comprehensive_preprocessing(data_dir):
    data_path = Path(data_dir)
    
    current_project_dir = Path(os.path.abspath(__file__)).parent
    output_dir = current_project_dir / 'data' / 'processed'
    
    removed_corrupt = 0
    removed_duplicates = 0
    hashes = {}          
    class_counts = {}    

    print(f"\n[Preprocessing] กำลังเริ่มกระบวนการ Clean รูปภาพ...\nไฟล์ที่สำเร็จจะถูกบันทึกลง: {output_dir}\n")
    
    for img_path in data_path.glob("**/*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            class_name = img_path.parent.name
            class_out_dir = output_dir / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img_hash = imagehash.average_hash(img)
                    
                    if img_hash in hashes:
                        removed_duplicates += 1
                        continue
                    else:
                        hashes[img_hash] = img_path.name
                        out_file = class_out_dir / img_path.name
                        img.save(out_file)
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

            except (IOError, SyntaxError):
                removed_corrupt += 1

    print("\n" + "="*40)
    print("--- สรุปผลการทำ Data Preprocessing ---")
    print(f"ภาพเสีย (Corrupted) ที่คัดทิ้ง: {removed_corrupt} ไฟล์")
    print(f"ภาพซ้ำ (Duplicates) ที่คัดทิ้ง: {removed_duplicates} ไฟล์")
    print("="*40)
    
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    class_weights = {}
    
    print("\n[คำนวณ Class Weights สำหรับแก้ปัญหา Class Imbalance]:")
    for i, (cls, count) in enumerate(sorted(class_counts.items())):
        weight = total_samples / (num_classes * count) if count > 0 else 0
        class_weights[i] = weight
        print(f"- Class '{cls}': {count} รูปภาพ -> Weight: {weight:.4f}")

    print("\n✅ เสร็จสิ้น! รูปภาพที่พร้อมใช้งานถูกบันทึกไว้ที่โฟลเดอร์ data/processed")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # ไฮไลต์อยู่ตรงนี้ครับ!
    # โค้ดจะไปสั่งรันฟังก์ชันโหลดข้อมูลจากไฟล์ data_collection ของเพื่อนอัตโนมัติ
    # ---------------------------------------------------------
    dataset_path = data_collection.get_dataset_path()
    
    # เมื่อโหลดเสร็จ ก็เอาข้อมูลที่ได้มาทำ Preprocessing ต่อเลย
    comprehensive_preprocessing(dataset_path)