import os
import sys
import urllib.request
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import imagehash
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# ตั้งค่า Path ปัจจุบันของไฟล์ที่รันอยู่ (เพื่อให้ทำงานในโฟลเดอร์นี้ทันที)
# =====================================================================
current_project_dir = Path(os.path.abspath(__file__)).parent

# =====================================================================
# ระบบดาวน์โหลดไฟล์ data_collection.py มาไว้ในโฟลเดอร์ปัจจุบันอัตโนมัติ
# =====================================================================
module_name = "data_collection.py"
module_path = current_project_dir / module_name

if not module_path.exists():
    print(f"🔍 ไม่พบไฟล์ {module_name} กำลังดาวน์โหลดจาก GitHub...")
    github_raw_url = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/Dataset-cat-and-dog-image/feature/data-collection/data_collection.py"
    try:
        urllib.request.urlretrieve(github_raw_url, module_path)
        print(f"✅ ดาวน์โหลด {module_name} สำเร็จ!\n")
    except Exception as e:
        print(f"❌ ดาวน์โหลดล้มเหลว: {e}")
        sys.exit(1)

import data_collection 
# =====================================================================


def save_before_after_plot(original_img, processed_img, save_path):
    """ฟังก์ชันสร้างภาพ Before/After สำหรับนำเสนอ"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Before (Original)")
    axes[0].axis('off')
    
    axes[1].imshow(processed_img)
    axes[1].set_title("After (Processed + Augmented)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def comprehensive_preprocessing(data_dir):
    data_path = Path(data_dir)
    
    # กำหนดให้โฟลเดอร์จัดเก็บข้อมูลและรายงาน สร้างอยู่ภายใต้ Path ปัจจุบันนี้โดยตรง
    output_dir = current_project_dir / 'data' / 'processed'
    report_dir = current_project_dir / 'reports' / 'figures'
    
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    removed_corrupt = 0
    removed_duplicates = 0
    hashes = {}          
    class_counts = {}    
    sample_saved = {'cats': False, 'dogs': False}

    print(f"[Preprocessing] กำลังเริ่ม Clean และ Process รูปภาพในพื้นที่ทำงานปัจจุบัน...\n")
    
    for img_path in data_path.glob("**/*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            class_name = img_path.parent.name
            class_out_dir = output_dir / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                with Image.open(img_path) as img:
                    original_img = img.copy() 
                    
                    img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img = img.filter(ImageFilter.MedianFilter(size=3))
                    
                    if np.random.rand() > 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(np.random.uniform(0.8, 1.2)) 
                    
                    img_array_normalized = np.array(img) / 255.0
                    
                    img_hash = imagehash.average_hash(img)
                    if img_hash in hashes:
                        removed_duplicates += 1
                        continue
                    else:
                        hashes[img_hash] = img_path.name
                        
                        if class_name in sample_saved and not sample_saved[class_name]:
                            plot_path = report_dir / f"before_after_{class_name}.png"
                            save_before_after_plot(original_img, img, plot_path)
                            print(f"📸 สร้างรูป Before/After ของ {class_name} ไว้ที่: {plot_path}")
                            sample_saved[class_name] = True
                        
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
    print(f"\n✅ เสร็จสิ้น! โฟลเดอร์งานถูกสร้างและจัดการเรียบร้อยใน: {current_project_dir}")


if __name__ == "__main__":
    # ดึง Path ข้อมูลจาก data_collection.py มาประมวลผลต่อ
    dataset_path = data_collection.get_dataset_path()
    comprehensive_preprocessing(dataset_path)