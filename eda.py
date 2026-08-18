import hashlib
import os
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageFile

# ==========================================
# CONFIGURATION & WARNING SETTINGS
# ==========================================
# อนุญาตให้โหลดไฟล์ภาพที่ไม่สมบูรณ์ (Fix: Truncated File Warning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ปิดการแสดงผล Warning บน Terminal
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = r"C:\Users\ohmde\AppData\Local\Programs\Microsoft VS Code"
FIGURES_DIR = "reports/figures"
SUMMARY_FILE = "reports/eda_summary.md"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
sns.set_theme(style="whitegrid")


# ==========================================
# 1. METADATA EXTRACTION & ANOMALY DETECTION
# ==========================================
def calculate_md5(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def extract_and_clean_metadata(data_dir):
    print("[*] Starting Data Extraction & Anomaly Detection...")
    data = []
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for root, _, files in os.walk(data_dir):
        if root == data_dir:
            continue
        cls = os.path.basename(root)

        for f in files:
            file_path = os.path.join(root, f)
            if not f.lower().endswith(valid_exts):
                continue

            file_size_kb = os.path.getsize(file_path) / 1024.0
            is_corrupted = False
            w, h, channels = None, None, None

            # Check Corrupted File & Extract Info
            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    img.load()  # บังคับอ่านข้อมูลภาพเพื่อดักจับไฟล์ที่เสียจริง
                    w, h = img.size
                    channels = len(img.getbands())
            except Exception:
                is_corrupted = True

            data.append(
                {
                    "file_path": file_path,
                    "filename": f,
                    "class": cls,
                    "file_size_kb": file_size_kb,
                    "width": w,
                    "height": h,
                    "aspect_ratio": (w / h) if (w and h) else None,
                    "channels": channels,
                    "is_corrupted": is_corrupted,
                }
            )

    df = pd.DataFrame(data)

    # Check Duplicate Files via MD5
    valid_df = df[~df["is_corrupted"]].copy()
    valid_df["file_hash"] = valid_df["file_path"].apply(calculate_md5)
    dup_hashes = set(
        valid_df[valid_df.duplicated(subset=["file_hash"], keep=False)][
            "file_hash"
        ]
    )
    df["is_duplicate"] = df["file_path"].apply(
        lambda x: valid_df.loc[
            valid_df["file_path"] == x, "file_hash"
        ].values[0]
        in dup_hashes
        if x in valid_df["file_path"].values
        else False
    )

    # Check Grayscale in RGB
    df["is_grayscale"] = df["channels"] == 1

    return df


# ==========================================
# 2. QUANTITATIVE & QUALITATIVE PLOTS
# ==========================================
def generate_plots(df):
    print("[*] Generating Figures...")
    valid_df = df[~df["is_corrupted"]]

    # 1. Class Distribution
    # แนะนำให้ปรับความกว้างเพิ่มขึ้นเล็กน้อย เช่น (10, 5) หรือ (12, 6)
    plt.figure(figsize=(10, 5)) 
    ax = sns.countplot(data=df, x="class", hue="class", palette="viridis")
    plt.title("1. Class Distribution (Class Imbalance Check)")

    # === เพิ่มบรรทัดนี้เพื่อหมุนตัวหนังสือแกน X ===
    plt.xticks(rotation=45, ha='right')
    # ======================================

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/class_distribution.png", dpi=300)
    plt.close()

    # 2. Dimensions & File Size Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(
        valid_df, x="width", kde=True, ax=axes[0], color="blue", bins=30
    )
    axes[0].set_title("Width Distribution")

    sns.histplot(
        valid_df, x="aspect_ratio", kde=True, ax=axes[1], color="green", bins=30
    )
    axes[1].set_title("Aspect Ratio Distribution")

    sns.histplot(
        valid_df,
        x="file_size_kb",
        kde=True,
        ax=axes[2],
        color="orange",
        bins=30,
    )
    axes[2].set_title("File Size (KB) Distribution")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/dimension_distributions.png", dpi=300)
    plt.close()

    # 3. Channel Pixel Intensity Histogram & Stats
    sample_df = valid_df.sample(min(300, len(valid_df)), random_state=42)
    r_list, g_list, b_list = [], [], []

    for path in sample_df["file_path"]:
        img = cv2.imread(path)
        if img is not None and len(img.shape) == 3:
            b_list.extend(img[:, :, 0].flatten())
            g_list.extend(img[:, :, 1].flatten())
            r_list.extend(img[:, :, 2].flatten())

    plt.figure(figsize=(8, 4))
    sns.kdeplot(r_list, color="red", label="Red Channel")
    sns.kdeplot(g_list, color="green", label="Green Channel")
    sns.kdeplot(b_list, color="blue", label="Blue Channel")
    plt.title("Pixel Intensity Histogram per Channel")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/channel_histogram.png", dpi=300)
    plt.close()

    # Calculate Channel Statistics
    stats = {
        "Channel": ["Red", "Green", "Blue"],
        "Mean": [np.mean(r_list), np.mean(g_list), np.mean(b_list)],
        "Std": [np.std(r_list), np.std(g_list), np.std(b_list)],
        "Min": [np.min(r_list), np.min(g_list), np.min(b_list)],
        "Max": [np.max(r_list), np.max(g_list), np.max(b_list)],
    }
    stats_df = pd.DataFrame(stats)

    # 4. Qualitative Grid Sample
    classes = valid_df["class"].unique()
    fig, axes = plt.subplots(len(classes), 4, figsize=(12, 3 * len(classes)))
    for i, cls in enumerate(classes):
        samples = valid_df[valid_df["class"] == cls].sample(
            4, random_state=42
        )["file_path"].values
        for j, path in enumerate(samples):
            ax = axes[i, j]
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(f"{cls}\n{img.size[0]}x{img.size[1]}")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/sample_grid.png", dpi=300)
    plt.close()

    return stats_df


# ==========================================
# 3. GENERATE MD SUMMARY REPORT & INSIGHTS
# ==========================================
def write_summary_report(df, stats_df):
    print("[*] Writing Markdown Summary Report...")
    total_imgs = len(df)
    class_counts = df["class"].value_counts().to_dict()
    corrupted_cnt = df["is_corrupted"].sum()
    duplicate_cnt = df["is_duplicate"].sum()
    grayscale_cnt = df["is_grayscale"].sum()

    md_content = f"""# Exploratory Data Analysis (EDA) Report

## 1. Quantitative EDA
* **Total Images:** {total_imgs}
* **Class Counts:** {class_counts}
* **Channel Pixel Statistics:**

{stats_df.to_markdown(index=False)}

## 2. Anomaly Detection Report
* **Corrupted Files:** {corrupted_cnt}
* **Duplicate Images:** {duplicate_cnt}
* **Grayscale Images in RGB Dataset:** {grayscale_cnt}

## 3. Qualitative Assessment & Content Issues
จากการสุ่มตรวจ Sample Grid พบปัญหาเชิงเนื้อหาดังนี้:
1. **Aspect Ratio Variation:** ภาพมีความกว้างและยาวไม่เท่ากัน
2. **Lighting Conditions:** แสงสว่างมีความแตกต่างกันอย่างมากในแต่ละรูป
3. **Complex Backgrounds:** มีสิ่งกีดขวางและ Background ที่ซับซ้อน เช่น กรง หญ้า เฟอร์นิเจอร์

## 4. Insights & Impact on Model Training
* **Class Balance:** จำนวนข้อมูลระหว่างคลาสมีความสมดุล ทำให้ไม่ต้องปรับ Weight ใน Loss Function
* **Preprocessing Needs:** 
  * จำเป็นต้องลบไฟล์ที่ **Corrupted** ออกก่อนส่งเข้า Pipeline เพื่อป้องกัน Error
  * ต้องทำ **Resize** ภาพให้เป็นขนาดมาตรฐาน (เช่น 224x224)
  * ต้องทำ **Normalization** ค่าพิกเซลตาม Mean และ Std ของแต่ละ Channel
* **Data Augmentation:** ควรใช้ Random Flip, Rotation และ Brightness Adjustment เพื่อลดปัญหา Overfitting จาก Background และสภาวะแสงที่ต่างกัน
"""
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(md_content)


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    df = extract_and_clean_metadata(DATA_DIR)
    stats_df = generate_plots(df)
    write_summary_report(df, stats_df)
    print(f"[✓] EDA Complete! Results saved to '{SUMMARY_FILE}' and '{FIGURES_DIR}/'")


if __name__ == "__main__":
    main()