import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import kagglehub

# ============================================================
# ตั้งค่าฟอนต์ภาษาไทยสำหรับ Matplotlib
# ============================================================
plt.rcParams["font.family"] = "Tahoma"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# ตั้งค่าขนาดภาพมาตรฐาน
# ============================================================
TARGET_SIZE = (150, 150)

# ============================================================
# 1. ดาวน์โหลด Dataset และค้นหาภาพตัวอย่าง
# ============================================================
def get_sample_image():
    """
    ดาวน์โหลด Dataset Cats and Dogs จาก Kaggle
    และค้นหาภาพตัวอย่างสำหรับทดสอบ Image Processing
    """

    print("=" * 60)
    print("กำลังดาวน์โหลดและค้นหาภาพตัวอย่างจาก Dataset...")
    print("=" * 60)

    try:
        path = kagglehub.dataset_download(
            "samuelcortinhas/cats-and-dogs-image-classification"
        )

        print(f"Dataset อยู่ที่: {path}")

        # ค้นหาไฟล์ภาพ JPG และ PNG
        image_paths = (
            list(Path(path).glob("**/*.jpg"))
            + list(Path(path).glob("**/*.jpeg"))
            + list(Path(path).glob("**/*.png"))
        )

        if image_paths:
            print(f"พบภาพทั้งหมด: {len(image_paths)} ภาพ")
            print(f"เลือกภาพตัวอย่าง: {image_paths[0]}")
            return image_paths[0]

        print("ไม่พบไฟล์รูปภาพใน Dataset")
        return None

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการดาวน์โหลด Dataset: {e}")
        return None

# ============================================================
# 2. Image Processing Pipeline
# ============================================================
def process_image(img_path):
    """
    ประมวลผลภาพตามขั้นตอนของข้อ 4.4 Image Processing
    ขั้นตอน:
    1. Original Image
    2. Resize
    3. Denoising
    4. Normalization
    5. Horizontal Flip
    6. Brightness Adjustment
    """

    # --------------------------------------------------------
    # 1. Original Image
    # --------------------------------------------------------
    original_img = Image.open(img_path).convert("RGB")

    # --------------------------------------------------------
    # 2. Resize
    # --------------------------------------------------------
    resized_img = original_img.resize(
        TARGET_SIZE,
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # 3. Noise Reduction / Denoising
    # --------------------------------------------------------
    # ใช้ Median Filter เพื่อลด Noise และรักษาขอบของวัตถุ
    denoised_img = resized_img.filter(ImageFilter.MedianFilter(size=3))

    # --------------------------------------------------------
    # 4. Normalization
    # --------------------------------------------------------
    img_array = np.array(denoised_img, dtype=np.float32)

    # แปลงค่าพิกเซลจาก 0-255 เป็น 0-1
    normalized_array = img_array / 255.0

    # --------------------------------------------------------
    # 5. Data Augmentation - Horizontal Flip
    # --------------------------------------------------------
    flipped_img = denoised_img.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    # --------------------------------------------------------
    # 6. Data Augmentation - Brightness
    # --------------------------------------------------------
    enhancer = ImageEnhance.Brightness(denoised_img)

    # เพิ่มความสว่าง 30%
    bright_img = enhancer.enhance(1.3)

    return {
        "original": original_img,
        "resized": resized_img,
        "denoised": denoised_img,
        "normalized": normalized_array,
        "flipped": flipped_img,
        "bright": bright_img,
        "original_array": np.array(original_img),
        "resized_array": np.array(resized_img),
        "denoised_array": img_array,
    }

# ============================================================
# 3. แสดงข้อมูลของภาพ
# ============================================================
def print_image_information(images):
    """
    แสดงข้อมูลขนาดภาพและค่าพิกเซล
    """

    original_array = images["original_array"]
    resized_array = images["resized_array"]
    denoised_array = images["denoised_array"]
    normalized_array = images["normalized"]

    print("\n")
    print("=" * 60)
    print("IMAGE INFORMATION")
    print("=" * 60)

    # --------------------------------------------------------
    # ขนาดภาพ
    # --------------------------------------------------------
    print("\n[1] ขนาดภาพ")
    print(
        f"Original Image : "
        f"{original_array.shape[1]} x {original_array.shape[0]}"
    )

    print(
        f"Resize Image   : "
        f"{resized_array.shape[1]} x {resized_array.shape[0]}"
    )

    print(
        f"Denoised Image : "
        f"{denoised_array.shape[1]} x {denoised_array.shape[0]}"
    )

    # --------------------------------------------------------
    # ค่าพิกเซลก่อน Normalization
    # --------------------------------------------------------
    print("\n[2] Normalization")

    pixel_before = denoised_array[0, 0]
    pixel_after = normalized_array[0, 0]

    print(f"Pixel ก่อน Normalization : {pixel_before}")
    print(f"Pixel หลัง Normalization : {pixel_after}")
    print("\nช่วงค่าพิกเซลก่อน Normalize:")
    print(
        f"Min = {denoised_array.min():.2f}, "
        f"Max = {denoised_array.max():.2f}"
    )

    print("\nช่วงค่าพิกเซลหลัง Normalize:")
    print(
        f"Min = {normalized_array.min():.4f}, "
        f"Max = {normalized_array.max():.4f}"
    )

    print("\nตัวอย่างการคำนวณ:")
    print(
        f"{pixel_before[0]:.0f} / 255 "
        f"= {pixel_after[0]:.4f}"
    )

# ============================================================
# 4. แสดง Before / After
# ============================================================
def show_before_after_pipeline(images):
    """
    แสดงภาพ Before/After ของทุกขั้นตอน
    """

    original_img = images["original"]
    resized_img = images["resized"]
    denoised_img = images["denoised"]
    normalized_array = images["normalized"]
    flipped_img = images["flipped"]
    bright_img = images["bright"]

    # --------------------------------------------------------
    # สร้าง Figure
    # --------------------------------------------------------
    plt.figure(figsize=(18, 12))

    # ========================================================
    # 1. Original
    # ========================================================
    plt.subplot(2, 3, 1)
    plt.imshow(original_img)
    plt.title(
        "1. Original Image\n"
        "ภาพต้นฉบับ\n"
        "เหตุผล: ข้อมูลดิบก่อนประมวลผล",
        fontsize=12
    )
    plt.axis("off")

    # ========================================================
    # 2. Resize
    # ========================================================
    plt.subplot(2, 3, 2)
    plt.imshow(resized_img)
    plt.title(
        f"2. Resize\n"
        f"{TARGET_SIZE[0]} × {TARGET_SIZE[1]} pixels\n"
        "เหตุผล: ทำให้ Input มีขนาดเท่ากัน",
        fontsize=12
    )
    plt.axis("off")

    # ========================================================
    # 3. Denoising
    # ========================================================
    plt.subplot(2, 3, 3)
    plt.imshow(denoised_img)
    plt.title(
        "3. Noise Reduction / Denoising\n"
        "เหตุผล: ลดสัญญาณรบกวนในภาพ",
        fontsize=12
    )
    plt.axis("off")

    # ========================================================
    # 4. Normalization
    # ========================================================
    plt.subplot(2, 3, 4)
    plt.imshow(normalized_array)
    plt.title(
        "4. Normalization\n"
        "Pixel Scale: 0–1\n"
        "เหตุผล: ทำให้ค่าพิกเซลอยู่ในช่วงเดียวกัน",
        fontsize=12
    )
    plt.axis("off")

    # ========================================================
    # 5. Horizontal Flip
    # ========================================================
    plt.subplot(2, 3, 5)
    plt.imshow(flipped_img)
    plt.title(
        "5. Data Augmentation\n"
        "Horizontal Flip\n"
        "เหตุผล: เพิ่มความหลากหลายของ Dataset",
        fontsize=12
    )
    plt.axis("off")

    # ========================================================
    # 6. Brightness
    # ========================================================
    plt.subplot(2, 3, 6)
    plt.imshow(bright_img)
    plt.title(
        "6. Data Augmentation\n"
        "Brightness +30%\n"
        "เหตุผล: จำลองสภาพแสงที่แตกต่าง",
        fontsize=12
    )
    plt.axis("off")

    # ========================================================
    # จัด Layout
    # ========================================================
    plt.suptitle(
        "4.4 Image Processing - Before / After",
        fontsize=18,
        fontweight="bold",
        y=0.98  # ปรับตัวเลขลงมานิดหน่อยไม่ให้ชื่อกราฟตกขอบบน
    )
    
    # เพิ่ม h_pad=4.0 เพื่อเว้นระยะบรรทัดบน-ล่างให้ห่างขึ้น ป้องกันข้อความทับรูป
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.0, w_pad=2.0)
    plt.show()

# ============================================================
# 5. แสดง Before / After ของ Resize โดยเฉพาะ
# ============================================================
def show_resize_comparison(images):
    """
    แสดงการเปรียบเทียบ Original และ Resize
    """

    plt.figure(figsize=(12, 5))

    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(images["original"])
    original_size = images["original"].size
    plt.title(
        f"Before: Original\n"
        f"Size = {original_size[0]} × {original_size[1]}",
        fontsize=12
    )
    plt.axis("off")

    # Resize
    plt.subplot(1, 2, 2)
    plt.imshow(images["resized"])
    plt.title(
        f"After: Resize\n"
        f"Size = {TARGET_SIZE[0]} × {TARGET_SIZE[1]}",
        fontsize=12
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ============================================================
# 6. แสดง Before / After ของ Denoising
# ============================================================
def show_denoising_comparison(images):
    """
    แสดงการเปรียบเทียบก่อนและหลัง Denoising
    """

    plt.figure(figsize=(12, 5))

    # Before
    plt.subplot(1, 2, 1)
    plt.imshow(images["resized"])
    plt.title(
        "Before: Original หลัง Resize",
        fontsize=12
    )
    plt.axis("off")

    # After
    plt.subplot(1, 2, 2)
    plt.imshow(images["denoised"])
    plt.title(
        "After: Denoising / Noise Reduction",
        fontsize=12
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ============================================================
# 7. แสดง Before / After ของ Data Augmentation
# ============================================================
def show_augmentation_comparison(images):
    """
    แสดงผล Data Augmentation
    """

    plt.figure(figsize=(15, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(images["denoised"])
    plt.title(
        "Original",
        fontsize=12
    )
    plt.axis("off")

    # Flip
    plt.subplot(1, 3, 2)
    plt.imshow(images["flipped"])
    plt.title(
        "Horizontal Flip",
        fontsize=12
    )
    plt.axis("off")

    # Brightness
    plt.subplot(1, 3, 3)
    plt.imshow(images["bright"])
    plt.title(
        "Brightness +30%",
        fontsize=12
    )
    plt.axis("off")

    plt.suptitle(
        "Data Augmentation - Before / After",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

# ============================================================
# 8. แสดงค่าพิกเซล Before / After Normalization
# ============================================================
def show_normalization_information(images):
    """
    แสดงตัวอย่างค่าพิกเซลก่อนและหลัง Normalization
    """

    denoised_array = images["denoised_array"]
    normalized_array = images["normalized"]

    # เลือก Pixel ตัวอย่าง
    pixel_y = 0
    pixel_x = 0

    before = denoised_array[pixel_y, pixel_x]
    after = normalized_array[pixel_y, pixel_x]

    print("\n")
    print("=" * 60)
    print("NORMALIZATION BEFORE / AFTER")
    print("=" * 60)

    print("\nตำแหน่ง Pixel:")
    print(f"Y = {pixel_y}, X = {pixel_x}")

    print("\nBefore Normalization:")
    print(f"RGB = {before}")

    print("\nAfter Normalization:")
    print(f"RGB = {after}")

    print("\nตัวอย่างการคำนวณ:")

    for channel, value in zip(
        ["R", "G", "B"],
        before
    ):
        normalized_value = value / 255.0

        print(
            f"{channel}: "
            f"{value:.0f} / 255 "
            f"= {normalized_value:.4f}"
        )

# ============================================================
# 9. Main Program
# ============================================================
def main():

    print("\n")
    print("=" * 60)
    print("4.4 IMAGE PROCESSING")
    print("Cats and Dogs Image Classification")
    print("=" * 60)

    # --------------------------------------------------------
    # ดาวน์โหลด Dataset
    # --------------------------------------------------------
    sample_path = get_sample_image()

    if sample_path is None:
        print("\nไม่สามารถดำเนินการต่อได้")
        return

    # --------------------------------------------------------
    # ประมวลผลภาพ
    # --------------------------------------------------------
    print("\n")
    print("=" * 60)
    print("กำลังประมวลผลภาพ...")
    print("=" * 60)

    images = process_image(sample_path)
    print("ประมวลผลเสร็จเรียบร้อยแล้ว")

    # --------------------------------------------------------
    # แสดงข้อมูล
    # --------------------------------------------------------
    print_image_information(images)

    # --------------------------------------------------------
    # แสดง Normalization
    # --------------------------------------------------------
    show_normalization_information(images)

    # --------------------------------------------------------
    # แสดงภาพ Pipeline
    # --------------------------------------------------------
    print("\nกำลังแสดงผล Before / After...")
    show_before_after_pipeline(images)

    # --------------------------------------------------------
    # แสดง Resize Comparison
    # --------------------------------------------------------
    show_resize_comparison(images)

    # --------------------------------------------------------
    # แสดง Denoising Comparison
    # --------------------------------------------------------
    show_denoising_comparison(images)

    # --------------------------------------------------------
    # แสดง Data Augmentation
    # --------------------------------------------------------
    show_augmentation_comparison(images)

    print("\n")
    print("=" * 60)
    print("เสร็จสิ้นกระบวนการ Image Processing")
    print("=" * 60)

# ============================================================
# เรียกใช้งานโปรแกรม
# ============================================================
if __name__ == "__main__":
    main()