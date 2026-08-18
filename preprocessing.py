import os
from pathlib import Path
from PIL import Image
import imagehash
import kagglehub
import numpy as np

def download_dataset():
    """
    ฟังก์ชันสำหรับดึง Dataset อัตโนมัติผ่าน Kaggle API
    ตอบโจทย์ข้อ 4.1: โหลดข้อมูลอัตโนมัติเมื่อรันโค้ด โดยไม่แนบไฟล์รูปใน Github
    """
    print("กำลังดาวน์โหลด Dataset จาก Kaggle...")
    # kagglehub จะจัดการโหลดและเก็บไว้ใน Cache ของเครื่องให้อัตโนมัติ
    path = kagglehub.dataset_download("samuelcortinhas/cats-and-dogs-image-classification")
    print(f"ดาวน์โหลดสำเร็จ! พาธข้อมูลดิบอยู่ที่: {path}")
    return path

def comprehensive_preprocessing(data_dir):
    """
    ฟังก์ชันสำหรับทำ Data Cleaning (ข้อ 4.3) และ Image Processing (ข้อ 4.4)
    กระบวนการ: จะอ่านไฟล์ต้นฉบับ -> คัดกรอง/ปรับแต่ง -> เซฟไฟล์ที่สมบูรณ์ลงโฟลเดอร์โปรเจกต์
    """
    data_path = Path(data_dir)
    
    # 1. สร้างโฟลเดอร์ปลายทาง 'data/processed' ให้อยู่ในโฟลเดอร์โปรเจกต์ปัจจุบัน
    # เพื่อให้เพื่อนคนอื่น (เช่น คนทำ data_split) สามารถอ้างอิง path ไปใช้งานต่อได้ง่ายๆ
    current_project_dir = Path(os.path.abspath(__file__)).parent
    output_dir = current_project_dir / 'data' / 'processed'
    
    # ตัวแปรสำหรับเก็บสถิติการลบข้อมูล
    removed_corrupt = 0
    removed_duplicates = 0
    hashes = {}          # Dictionary สำหรับเก็บค่า Hash ของรูปภาพ (ใช้เช็ครูปซ้ำ)
    class_counts = {}    # Dictionary สำหรับนับจำนวนรูปภาพที่รอดชีวิตในแต่ละคลาส

    print(f"กำลังเริ่มกระบวนการ Data Preprocessing...\nไฟล์ที่สำเร็จจะถูกบันทึกลง: {output_dir}\n")
    
    # ใช้ .glob("**/*.*") เพื่อกวาดหาไฟล์ทุกประเภทในทุกโฟลเดอร์ย่อย (รวมถึงโฟลเดอร์ train และ test เดิม)
    for img_path in data_path.glob("**/*.*"):
        
        # กรองเอาเฉพาะไฟล์ที่มีนามสกุลเป็นรูปภาพ
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            
            # ดึงชื่อคลาส (cats หรือ dogs) มาจากชื่อโฟลเดอร์ที่รูปนั้นอยู่
            class_name = img_path.parent.name
            
            # สร้างโฟลเดอร์ปลายทางรอไว้เลย เช่น data/processed/cats/
            class_out_dir = output_dir / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                # ---------------------------------------------------------
                # ขั้นตอนที่ 1: ตรวจสอบไฟล์เสีย (Corrupted Images Detection)
                # ---------------------------------------------------------
                # เปิดไฟล์เพื่อเช็คโครงสร้างเบื้องต้น ถ้าไฟล์พังมันจะโยน Error (เข้า except) ทันที
                with Image.open(img_path) as img:
                    img.verify()
                
                # ---------------------------------------------------------
                # ขั้นตอนที่ 2: Image Processing (Format & Resize)
                # ---------------------------------------------------------
                with Image.open(img_path) as img:
                    # แปลง Color Space ทุกรูปให้เป็น RGB ป้องกันปัญหารูปขาวดำ (Grayscale) หลุดรอดมา
                    img = img.convert('RGB')
                    
                    # (โจทย์ข้อ 4.4) Resize รูปภาพให้เป็นขนาด 224x224
                    # เหตุผล: เป็นขนาดมาตรฐานที่โมเดล CNN ชื่อดัง (เช่น ResNet, VGG) ใช้รับข้อมูล
                    img = img.resize((224, 224))
                    
                    # ---------------------------------------------------------
                    # ขั้นตอนที่ 3: ตรวจจับรูปซ้ำ (Duplicate Detection)
                    # ---------------------------------------------------------
                    # ใช้ Perceptual Hash (average_hash) เพื่อแปลงภาพเป็นรหัส
                    # ถ้ารหัสซ้ำกัน แสดงว่าเป็นภาพเดียวกัน (แม้ขนาดไฟล์เดิมจะไม่เท่ากันก็ตาม)
                    img_hash = imagehash.average_hash(img)
                    
                    if img_hash in hashes:
                        # ถ้าเจอว่ารหัสนี้เคยมีแล้ว = รูปซ้ำ -> ข้ามการบันทึก (ทิ้งไป)
                        print(f"[-] พบภาพซ้ำ ข้ามการบันทึก: {img_path.name}")
                        removed_duplicates += 1
                        continue
                    else:
                        # ถ้ารหัสใหม่ = ไม่ซ้ำ -> จดจำรหัสไว้ แล้วทำการบันทึก
                        hashes[img_hash] = img_path.name
                        
                        # ---------------------------------------------------------
                        # ขั้นตอนที่ 4: บันทึกภาพที่ Clean แล้ว
                        # ---------------------------------------------------------
                        out_file = class_out_dir / img_path.name
                        img.save(out_file)
                        
                        # นับจำนวนว่าคลาสนี้มีรูปที่ใช้งานได้เพิ่มขึ้น 1 รูป
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

            except (IOError, SyntaxError):
                # ถ้าไฟล์เปิดไม่ได้ (ไฟล์พัง/ไฟล์ดัมมี่) โค้ดจะตกมาที่นี่และข้ามการบันทึกไป
                print(f"[x] พบภาพเสีย ข้ามการบันทึก: {img_path.name}")
                removed_corrupt += 1

    # ---------------------------------------------------------
    # สรุปผลการทำงาน (Report)
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("--- สรุปผลการทำ Data Preprocessing ---")
    print(f"ภาพเสีย (Corrupted) ที่คัดทิ้ง: {removed_corrupt} ไฟล์")
    print(f"ภาพซ้ำ (Duplicates) ที่คัดทิ้ง: {removed_duplicates} ไฟล์")
    print("="*40)
    
    # ---------------------------------------------------------
    # ขั้นตอนที่ 5: จัดการ Class Imbalance (ข้อ 4.3)
    # ---------------------------------------------------------
    print("\nจำนวนข้อมูลที่ใช้งานได้ (ใน data/processed) และการคำนวณ Class Weights:")
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    class_weights = {}
    
    print(f"จำนวนคลาสทั้งหมด: {num_classes} คลาส, ข้อมูลรวมทั้งหมด: {total_samples} รูปภาพ")
    
    # ลูปเพื่อคำนวณน้ำหนัก (Weight) ให้แต่ละคลาส
    # คลาสไหนมีรูปน้อย จะได้ค่า Weight เยอะ (เพื่อให้โมเดลให้ความสำคัญตอนเทรนมากขึ้น)
    for i, (cls, count) in enumerate(sorted(class_counts.items())):
        print(f"- Class '{cls}': {count} รูปภาพ")
        weight = total_samples / (num_classes * count) if count > 0 else 0
        class_weights[i] = weight

    print("\n[คำนวณ Class Weights สำหรับแก้ปัญหา Class Imbalance เรียบร้อย]:")
    for cls_idx, weight in class_weights.items():
        print(f"  Class Index {cls_idx} -> Weight: {weight:.4f}")
    
    print("\n✅ เสร็จสิ้น! รูปภาพที่พร้อมใช้งานถูกบันทึกไว้ที่โฟลเดอร์ data/processed")
    print("คุณสามารถรันไฟล์ data_split.py เป็นลำดับถัดไปได้เลย")

if __name__ == "__main__":
    # เริ่มต้นการทำงาน 
    # 1. โหลดข้อมูลดิบ
    dataset_path = download_dataset()
    
    # 2. นำข้อมูลดิบมาทำ Preprocessing
    comprehensive_preprocessing(dataset_path)