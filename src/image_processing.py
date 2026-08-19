import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import kagglehub

# --- ตั้งค่าฟอนต์ภาษาไทยสำหรับ Matplotlib เพื่อป้องกันตัวอักษรเพี้ยน (ภาษาเอเลี่ยน) ---
plt.rcParams['font.family'] = 'Tahoma' 
plt.rcParams['axes.unicode_minus'] = False # ป้องกันเครื่องหมายลบเพี้ยน

def get_sample_image():
    """ดึงตัวอย่างภาพจาก Dataset เพื่อมาทดสอบกระบวนการ Image Processing"""
    print("กำลังดาวน์โหลดและค้นหาภาพตัวอย่างจาก Dataset...")
    path = kagglehub.dataset_download("samuelcortinhas/cats-and-dogs-image-classification")
    image_paths = list(Path(path).glob("**/*.jpg")) + list(Path(path).glob("**/*.png"))
    if image_paths:
        return image_paths[0]
    return None

def show_before_after_pipeline(img_path):
    """
    แสดงภาพ Before/After ของทุกขั้นตอนใน 4.4 Image Processing เพื่อยืนยันผลลัพธ์
    """
    # 1. ภาพต้นฉบับ (Before)
    original_img = Image.open(img_path).convert('RGB')
    
    # 2. Resize: ปรับขนาดมาตรฐานพร้อมเหตุผล
    TARGET_SIZE = (150, 150)
    resized_img = original_img.resize(TARGET_SIZE)
    
    # 3. Noise Reduction / Denoising: กรองสัญญาณรบกวนเบื้องต้น
    denoised_img = original_img.filter(ImageFilter.SMOOTH)
    
    # 4. Data Augmentation - Flip: พลิกภาพแนวนอน
    flipped_img = original_img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 5. Data Augmentation - Brightness: ปรับเพิ่มความสว่าง
    enhancer = ImageEnhance.Brightness(original_img)
    bright_img = enhancer.enhance(1.3) # เพิ่มความสว่าง 30%

    # --- จัดเตรียมแสดงผลภาพ Before/After เปรียบเทียบในหน้าต่างเดียว ---
    plt.figure(figsize=(16, 9))
    
    # ภาพที่ 1: Original
    plt.subplot(2, 3, 1)
    plt.imshow(original_img)
    plt.title("1. Original Image (Before)\n[ข้อมูลดิบก่อนประมวลผล]")
    plt.axis('off')
    
    # ภาพที่ 2: Resize
    plt.subplot(2, 3, 2)
    plt.imshow(resized_img)
    plt.title(f"2. Resize ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})\n[เหตุผล: ควบคุมขนาด Input ให้เท่ากันเพื่อเข้า Model]")
    plt.axis('off')
    
    # ภาพที่ 3: Noise Reduction
    plt.subplot(2, 3, 3)
    plt.imshow(denoised_img)
    plt.title("3. Denoising / Noise Reduction\n[เหตุผล: ลดจุดรบกวนและเม็ดพิกเซลเพี้ยน]")
    plt.axis('off')
    
    # ภาพที่ 4: Augmentation (Flip)
    plt.subplot(2, 3, 4)
    plt.imshow(flipped_img)
    plt.title("4. Augmentation: Horizontal Flip\n[เหตุผล: เพิ่มความหลากหลาย ป้องกัน Overfitting]")
    plt.axis('off')
    
    # ภาพที่ 5: Augmentation (Brightness)
    plt.subplot(2, 3, 5)
    plt.imshow(bright_img)
    plt.title("5. Augmentation: Brightness Adjust\n[เหตุผล: จำลองสภาพแสงที่แตกต่างในโลกจริง]")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sample_path = get_sample_image()
    if sample_path:
        print(f"กำลังประมวลผลภาพตัวอย่างเพื่อแสดงผล Before/After จาก: {sample_path}")
        show_before_after_pipeline(sample_path)
    else:
        print("ไม่พบไฟล์รูปภาพใน Dataset")