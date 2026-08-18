import os
import cv2
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# ==========================================
# 0. CONFIG & SETUP PATHS
# ==========================================
DATA_DIR = DATA_DIR = r"C:\Users\ohmde\.cache\kagglehub\datasets\bhavikjikadara\dog-and-cat-classification-dataset\versions\1\PetImages"  # โฟลเดอร์เก็บ Dataset ที่โหลดมาจาก Kaggle
FIGURES_DIR = "reports/figures"  # โฟลเดอร์สำหรับเซฟรูปกราฟสรุปผล

os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# ==========================================
# 1. QUANTITATIVE EDA: METADATA EXTRACTION
# ==========================================
def extract_metadata(data_dir):
    """ดึงข้อมูลพื้นฐานของรูปภาพทั้งหมดใน Dataset"""
    data = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    print("[*] Starting metadata extraction...")

    for root, _, files in os.walk(data_dir):
        class_name = os.path.basename(root)
        if root == data_dir:
            continue  # ข้าม root folder

        for file in files:
            file_path = os.path.join(root, file)

            # ตรวจสอบนามสกุลไฟล์เบื้องต้น
            if not file.lower().endswith(valid_extensions):
                continue

            file_size_kb = os.path.getsize(file_path) / 1024.0

            # ตรวจสอบไฟล์เสีย (Corrupted File)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # ยืนยันโครงสร้างไฟล์
                with Image.open(
                    file_path
                ) as img:  # เปิดอีกรอบเพื่ออ่านค่า Image properties
                    width, height = img.size
                    mode = img.mode
                    channels = len(img.getbands())
                    is_corrupted = False
            except Exception:
                width, height, mode, channels = None, None, None, None
                is_corrupted = True

            data.append(
                {
                    "file_path": file_path,
                    "filename": file,
                    "class": class_name,
                    "file_size_kb": file_size_kb,
                    "width": width,
                    "height": height,
                    "aspect_ratio": (
                        (width / height) if (width and height) else None
                    ),
                    "mode": mode,
                    "channels": channels,
                    "is_corrupted": is_corrupted,
                }
            )

    df = pd.DataFrame(data)
    print(f"[*] Processed {len(df)} images.")
    return df


# ==========================================
# 2. ANOMALY DETECTION (DUPLICATES & CORRUPTED)
# ==========================================
def calculate_md5(file_path):
    """คำนวณ MD5 Hash เพื่อเช็กรูปซ้ำแบบ Byte-by-Byte"""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def detect_anomalies(df):
    """ตรวจหาไฟล์เสีย ไฟล์ซ้ำ และไฟล์สีผิดปกติ (Grayscale ใน RGB)"""
    print("[*] Detecting anomalies...")

    # ตรวจหาไฟล์ซ้ำ (Duplicate Detection)
    valid_df = df[~df["is_corrupted"]].copy()
    valid_df["file_hash"] = valid_df["file_path"].apply(calculate_md5)

    duplicate_hashes = valid_df[
        valid_df.duplicated(subset=["file_hash"], keep=False)
    ]
    duplicate_count = duplicate_hashes["file_hash"].nunique()

    # สรุปผล
    corrupted_count = df["is_corrupted"].sum()
    grayscale_in_rgb = len(df[df["channels"] == 1])

    print("\n--- ANOMALY REPORT ---")
    print(f"Total Images: {len(df)}")
    print(f"Corrupted Images: {corrupted_count}")
    print(f"Duplicate Image Groups: {duplicate_count}")
    print(f"Grayscale Images (Single Channel): {grayscale_in_rgb}")
    print("----------------------\n")


# ==========================================
# 3. PLOTTING QUANTITATIVE CHARTS
# ==========================================
def plot_class_distribution(df):
    """พล็อตและบันทึกกราฟจำนวนรูปภาพต่อ Class"""
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(
        data=df,
        x="class",
        order=df["class"].value_counts().index,
        palette="viridis",
    )
    plt.title("Class Distribution (Check Class Imbalance)", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)

    # แสดงตัวเลขบนแถบ Bar
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=300)
    plt.close()


def plot_image_dimensions(df):
    """พล็อตการกระจายขนาดภาพ Width, Height และ File Size"""
    valid_df = df[~df["is_corrupted"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Width vs Height Scatter Plot
    sns.scatterplot(
        data=valid_df,
        x="width",
        y="height",
        hue="class",
        alpha=0.6,
        ax=axes[0],
    )
    axes[0].set_title("Width vs Height Distribution")

    # Aspect Ratio Histogram
    sns.histplot(valid_df["aspect_ratio"], kde=True, bins=20, ax=axes[1])
    axes[1].set_title("Aspect Ratio Distribution")

    # File Size Histogram
    sns.histplot(valid_df["file_size_kb"], kde=True, bins=20, ax=axes[2])
    axes[2].set_title("File Size Distribution (KB)")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "image_dimensions.png"), dpi=300)
    plt.close()


def plot_pixel_intensity_histogram(df, num_samples=100):
    """พล็อต Histogram ของค่าพิกเซลเฉลี่ยแยกตาม Channel (R, G, B)"""
    valid_df = df[~df["is_corrupted"]]
    sample_df = valid_df.sample(min(num_samples, len(valid_df)), random_state=42)

    r_vals, g_vals, b_vals = [], [], []

    for path in sample_df["file_path"]:
        img = cv2.imread(path)
        if img is not None and len(img.shape) == 3:
            # OpenCV อ่านภาพเป็น BGR
            b_vals.extend(img[:, :, 0].ravel())
            g_vals.extend(img[:, :, 1].ravel())
            r_vals.extend(img[:, :, 2].ravel())

    plt.figure(figsize=(10, 5))
    plt.hist(r_vals, bins=256, color="red", alpha=0.4, label="Red Channel")
    plt.hist(g_vals, bins=256, color="green", alpha=0.4, label="Green Channel")
    plt.hist(b_vals, bins=256, color="blue", alpha=0.4, label="Blue Channel")

    plt.title("Pixel Intensity Distribution (RGB Histogram)")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "pixel_intensity_histogram.png"), dpi=300
    )
    plt.close()


# ==========================================
# 4. QUALITATIVE EDA: VISUAL SAMPLE GRID
# ==========================================
def plot_sample_grid(df, samples_per_class=3):
    """สุ่มดึงภาพจากแต่ละ Class มาแสดงผลแบบ Grid"""
    valid_df = df[~df["is_corrupted"]]
    classes = valid_df["class"].unique()

    fig, axes = plt.subplots(
        len(classes),
        samples_per_class,
        figsize=(samples_per_class * 3, len(classes) * 3),
    )

    for i, cls in enumerate(classes):
        cls_df = valid_df[valid_df["class"] == cls]
        sampled_paths = cls_df.sample(
            min(samples_per_class, len(cls_df)), random_state=42
        )["file_path"].values

        for j in range(samples_per_class):
            ax = axes[i, j] if len(classes) > 1 else axes[j]

            if j < len(sampled_paths):
                img = Image.open(sampled_paths[j])
                ax.imshow(img)
                ax.set_title(f"{cls}\n({img.size[0]}x{img.size[1]})", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "qualitative_sample_grid.png"), dpi=300)
    plt.close()


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(DATA_DIR):
        print(
            f"[!] Path {DATA_DIR} does not exist. Please run Data Collection script first!"
        )
        return

    # 1. Extract Metadata
    df = extract_metadata(DATA_DIR)

    # 2. Detect Anomalies
    detect_anomalies(df)

    # 3. Generate Quantitative Figures
    print("[*] Generating Quantitative plots...")
    plot_class_distribution(df)
    plot_image_dimensions(df)
    plot_pixel_intensity_histogram(df)

    # 4. Generate Qualitative Figures
    print("[*] Generating Qualitative sample grid...")
    plot_sample_grid(df)

    print(
        f"\n[✓] EDA Analysis complete! Figures saved in '{FIGURES_DIR}' folder."
    )


if __name__ == "__main__":
    main()