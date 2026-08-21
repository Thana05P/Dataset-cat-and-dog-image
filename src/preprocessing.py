import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageFile, ImageEnhance
import imagehash
import numpy as np
import matplotlib.pyplot as plt

# อนุญาตให้โหลดภาพที่ไม่สมบูรณ์เพื่อป้องกัน Error
ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_step_by_step_plot(original, resized, enhanced, augmented, save_path):
    """ฟังก์ชันยืนยันผลลัพธ์: บันทึกภาพ Before/After ทุกขั้นตอน"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(original)
    axes[0].set_title("1. Original\n(Raw Image)")
    axes[0].axis('off')

    axes[1].imshow(resized)
    axes[1].set_title("2. Resized & Denoised\n(224x224, LANCZOS)")
    axes[1].axis('off')

    axes[2].imshow(enhanced)
    axes[2].set_title("3. Augmented\n(Flip / Brightness)")
    axes[2].axis('off')

    axes[3].imshow(augmented)
    axes[3].set_title("4. Normalized\n(0-1 Scale Visually)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def comprehensive_image_processing():
    current_project_dir = Path(os.path.abspath(__file__)).parent.parent
    data_path = current_project_dir / "Dataset-cat-and-dog-image"
    output_dir = current_project_dir / "data" / "processed"
    report_dir = current_project_dir / "reports" / "figures"

    if not data_path.exists():
        print(f"❌ ไม่พบโฟลเดอร์ Dataset ที่พาธ: {data_path}")
        return

    # ล้างโฟลเดอร์ผลลัพธ์เก่าทิ้งเพื่อป้องกันไฟล์ตกค้าง
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    removed_corrupt = 0
    removed_duplicates = 0
    skipped_not_train = 0
    hashes = {}
    saved_counts = {}
    sample_saved = {'cats': False, 'dogs': False}

    print("[*] เริ่มกระบวนการ Data Preprocessing และ Image Processing...\n")

    for img_path in data_path.glob("**/*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            if img_path.parent.name in ["Dataset-cat-and-dog-image"]:
                continue

            # --- ดึงเฉพาะรูปที่อยู่ใต้โฟลเดอร์ Train เท่านั้น (ไม่สนตัวพิมพ์เล็ก-ใหญ่) ---
            rel_parts = [p.lower() for p in img_path.relative_to(data_path).parts]
            if "train" not in rel_parts:
                skipped_not_train += 1
                continue

            class_name = img_path.parent.name
            class_out_dir = output_dir / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 1. ลบไฟล์ที่เสียหาย
                with Image.open(img_path) as img:
                    img.verify()

                with Image.open(img_path) as img:
                    original_img = img.copy()
                    # 2. แปลง Color Space
                    img = img.convert('RGB')

                    # 3. Resize รูปภาพ
                    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)

                    # 4. ตรวจจับรูปภาพซ้ำ (Duplicate Detection)
                    img_hash = imagehash.average_hash(img_resized)
                    if img_hash in hashes:
                        removed_duplicates += 1
                        continue
                    else:
                        hashes[img_hash] = img_path

                    # 5. Noise Reduction / Denoising
                    enhancer_sharp = ImageEnhance.Sharpness(img_resized)
                    img_denoised = enhancer_sharp.enhance(1.3)

                    # 6. Data Augmentation (หมุน/ปรับแสง เฉพาะรูปที่จะเซฟลง data เท่านั้น)
                    img_augmented = img_denoised.copy()

                    if np.random.rand() > 0.5:
                        angle = int(np.random.choice([0, 45, 90, 135, 180, 225, 270, 315, 360]))
                        img_augmented = img_augmented.rotate(angle)

                    if np.random.rand() > 0.5:
                        enhancer_bright = ImageEnhance.Brightness(img_augmented)
                        img_augmented = enhancer_bright.enhance(np.random.uniform(0.8, 1.2))

                    # 7. Normalization (สำหรับรูปที่จะเซฟลง data)
                    img_array = np.array(img_augmented, dtype=np.float32) / 255.0
                    img_normalized = Image.fromarray((img_array * 255).astype(np.uint8))

                    # --- เตรียมภาพสำหรับ Report โดยเฉพาะ: ไม่หมุน ใช้แค่ Flip เท่านั้น ---
                    if class_name in sample_saved and not sample_saved[class_name]:
                        img_report_aug = img_denoised.copy()
                        if np.random.rand() > 0.5:
                            img_report_aug = img_report_aug.transpose(Image.FLIP_LEFT_RIGHT)

                        img_report_arr = np.array(img_report_aug, dtype=np.float32) / 255.0
                        img_report_norm = Image.fromarray((img_report_arr * 255).astype(np.uint8))

                        plot_path = report_dir / f"step_by_step_{class_name}.png"
                        save_step_by_step_plot(original_img, img_denoised, img_report_aug, img_report_norm, plot_path)
                        print(f"📸 บันทึกภาพยืนยันผลลัพธ์ของ '{class_name}' ไว้ที่: {plot_path}")
                        sample_saved[class_name] = True

                    # บันทึกไฟล์ลงโฟลเดอร์ processed (เซฟรูปที่หมุนแล้ว)
                    out_file = class_out_dir / img_path.name
                    img_normalized.save(out_file)
                    saved_counts[class_name] = saved_counts.get(class_name, 0) + 1

            except Exception as e:
                # ถ้ามี Error ซ่อนอยู่มันจะปรินต์บอกตรงนี้
                # print(f"Error file {img_path.name}: {e}")
                removed_corrupt += 1

    # ========================================================
    # ส่วนแสดงผลรายงาน (Report Logging)
    # ========================================================
    print("\n" + "="*60)
    print("📋 [REPORT] สรุปผลการทำงานทุกขั้นตอน (Data Pipeline Checklist)")
    print("="*60)
    print(f"[0] 🗂️  FILTER: ข้ามไฟล์ที่ไม่ได้อยู่ในโฟลเดอร์ Train ไปจำนวน {skipped_not_train} รูป")
    print(f"[1] 🧹 CLEAN: ลบไฟล์ที่เสียหาย (Corrupted) ไปจำนวน {removed_corrupt} รูป")
    print(f"[2] ✂️ DEDUPLICATE: ตรวจพบและลบรูปภาพซ้ำกันไปจำนวน {removed_duplicates} รูป")
    print(f"[3] 🎨 FORMAT: บังคับแปลงภาพทั้งหมดให้เป็นมาตรฐาน Color Space (RGB)")
    print(f"[4] 📏 RESIZE: ย่อขนาดรูปภาพทุกใบให้เป็น 224x224 Pixel")
    print(f"[5] ✨ DENOISE: ปรับความคมชัดเพื่อลด Noise ในภาพเบื้องต้น")
    print(f"[6] 🔄 AUGMENT: สุ่มหมุนภาพ 360 องศา และสุ่มปรับแสง (เฉพาะรูปที่เซฟลง data)")
    print(f"[7] 🔢 NORMALIZE: ปรับสเกลค่าพิกเซลให้เป็น 0.0 - 1.0 เพื่อเข้าโมเดล")
    print("-" * 60)

    if len(saved_counts) > 0:
        print("[8] ⚖️ BALANCE (Undersampling): เริ่มปรับสมดุล Class Imbalance...")
        min_count = min(saved_counts.values())

        for cls, count in saved_counts.items():
            if count > min_count:
                excess = count - min_count
                print(f"    - คลาส '{cls}' มี {count} รูป (เกินมา {excess} รูป) -> สุ่มลบทิ้ง...")

                class_out_dir = output_dir / cls
                all_files = list(class_out_dir.glob("*.*"))
                files_to_delete = random.sample(all_files, excess)
                for f in files_to_delete:
                    f.unlink()
                saved_counts[cls] = min_count
            else:
                print(f"    - คลาส '{cls}' มี {count} รูป (เป็นฐานขั้นต่ำ พอดีแล้ว)")

        print("\n✅ อัปเดตจำนวนรูปภาพล่าสุด (พร้อมนำไป Train/Test):")
        for cls, count in saved_counts.items():
            print(f"    -> Class '{cls}': {count} รูปภาพ")

    print("="*60)
    print("🎉 การเตรียมข้อมูลเสร็จสมบูรณ์ 100%!")
    # ========================================================

if __name__ == "__main__":
    comprehensive_image_processing()