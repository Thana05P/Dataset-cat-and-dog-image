import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance
import imagehash
import numpy as np
import matplotlib.pyplot as plt

# ตั้งค่า Path ปัจจุบันของโปรเจกต์
current_project_dir = Path(os.path.abspath(__file__)).parent

def save_step_by_step_plot(original, resized, enhanced, augmented, save_path):
    """
    ฟังก์ชันแสดงภาพ Before/After ของทุกขั้นตอน เพื่อยืนยันผลลัพธ์
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original)
    axes[0].set_title("1. Original\n(ภาพต้นฉบับ)")
    axes[0].axis('off')
    
    axes[1].imshow(resized)
    axes[1].set_title("2. Resized (224x224, LANCZOS)\n(ปรับขนาดไม่ให้ภาพแตก)")
    axes[1].axis('off')
    
    axes[2].imshow(enhanced)
    axes[2].set_title("3. Sharpened\n(เพิ่มความคมชัด)")
    axes[2].axis('off')
    
    axes[3].imshow(augmented)
    axes[3].set_title("4. Augmented\n(Flip/Brightness)")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def comprehensive_preprocessing():
    data_path = current_project_dir / 'Dataset-cat-and-dog-image'
    
    if not data_path.exists():
        print(f"❌ ไม่พบโฟลเดอร์ {data_path.name} ในเครื่อง กรุณาตรวจสอบตำแหน่งโฟลเดอร์")
        sys.exit(1)

    output_dir = current_project_dir / 'data' / 'processed'
    report_dir = current_project_dir / 'reports' / 'figures'
    
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    initial_counts = {}
    cleaned_counts = {}
    removed_corrupt = 0
    removed_duplicates = 0
    hashes = {}          
    sample_saved = {'cats': False, 'dogs': False}

    print(f"[Preprocessing] กำลังสำรวจและนับจำนวนรูปภาพจากโฟลเดอร์ {data_path.name}...\n")
    
    # 1. นับจำนวนไฟล์เริ่มต้นแยกตามคลาส
    for img_path in data_path.glob("**/*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            c_name = img_path.parent.name
            initial_counts[c_name] = initial_counts.get(c_name, 0) + 1

    print("--- จำนวนไฟล์เริ่มต้น (Raw Dataset) ---")
    for c_name, count in initial_counts.items():
        print(f"- คลาส '{c_name}': {count} ไฟล์")
    print("-" * 40)

    print(f"[Preprocessing] เริ่มกระบวนการ Image Processing, Cleaning และลบไฟล์ขยะ...\n")
    
    for img_path in data_path.glob("**/*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            class_name = img_path.parent.name
            class_out_dir = output_dir / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                # ตรวจสอบความสมบูรณ์ของไฟล์ภาพ
                with Image.open(img_path) as img:
                    img.verify()
                
                with Image.open(img_path) as img:
                    original_img = img.copy() 
                    img = img.convert('RGB')
                    
                    # 1. Resize ด้วย LANCZOS เพื่อรักษาความคมชัด ขอบไม่แตก
                    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # 2. เพิ่มความคมชัด (Sharpen) แทนการใช้ฟิลเตอร์เบลอ
                    enhancer_sharp = ImageEnhance.Sharpness(img_resized)
                    img_enhanced = enhancer_sharp.enhance(1.5) 
                    
                    # 3. Data Augmentation (Flip & Brightness)
                    img_augmented = img_enhanced.copy()
                    if np.random.rand() > 0.5:
                        img_augmented = img_augmented.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    enhancer_bright = ImageEnhance.Brightness(img_augmented)
                    img_augmented = enhancer_bright.enhance(np.random.uniform(0.8, 1.2)) 
                    
                    # 4. ตรวจสอบภาพซ้ำ (Duplicate Detection)
                    img_hash = imagehash.average_hash(img_augmented)
                    if img_hash in hashes:
                        removed_duplicates += 1
                        # หากเจอภาพซ้ำ สามารถลบทิ้งจากต้นทางได้เลย (หรือข้ามการเซฟ)
                        try:
                            os.remove(img_path)
                        except OSError:
                            pass
                        continue
                    else:
                        hashes[img_hash] = img_path.name
                        
                        # บันทึกภาพตัวอย่าง Before/After 4 ขั้นตอน
                        if class_name in sample_saved and not sample_saved[class_name]:
                            plot_path = report_dir / f"step_by_step_{class_name}.png"
                            save_step_by_step_plot(original_img, img_resized, img_enhanced, img_augmented, plot_path)
                            print(f"📸 บันทึกภาพขั้นตอนประมวลผลของ {class_name} ไว้ที่: {plot_path}")
                            sample_saved[class_name] = True
                        
                        # บันทึกไฟล์ที่ผ่านการ Clean ลงโฟลเดอร์ processed
                        out_file = class_out_dir / img_path.name
                        img_augmented.save(out_file)
                        
                        cleaned_counts[class_name] = cleaned_counts.get(class_name, 0) + 1

            except (IOError, SyntaxError):
                # หากเจอไฟล์เสีย (Corrupted) สั่งลบทิ้งจากเครื่องทันที
                removed_corrupt += 1
                try:
                    os.remove(img_path)
                except OSError:
                    pass

    # แสดงรายงานสรุปผล
    print("\n" + "="*45)
    print("--- สรุปผลการทำ Data Preprocessing & Cleaning ---")
    print(f"ภาพเสีย (Corrupted) ที่ตรวจพบและลบทิ้ง: {removed_corrupt} ไฟล์")
    print(f"ภาพซ้ำ (Duplicates) ที่ตรวจพบและคัดทิ้ง: {removed_duplicates} ไฟล์")
    print("-" * 45)
    print("จำนวนไฟล์ที่เหลือหลังผ่านการจัดการ (Cleaned & Processed):")
    for c_name, count in cleaned_counts.items():
        print(f"  - คลาส '{c_name}': {count} ไฟล์")
    print(f"  - รวมทั้งหมดที่พร้อมใช้งาน: {sum(cleaned_counts.values())} ไฟล์")
    print("="*45)
    print(f"\n✅ เสร็จสิ้น! ไฟล์ถูกคลีนและจัดเก็บเรียบร้อย")


if __name__ == "__main__":
    comprehensive_preprocessing()