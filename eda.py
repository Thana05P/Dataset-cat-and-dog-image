import hashlib
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# ==========================================
# 0. CONFIG & SETUP PATHS
# ==========================================
DATA_DIR = r"C:\Users\ohmde\.cache\kagglehub\datasets\bhavikjikadara\dog-and-cat-classification-dataset\versions\1\PetImages"
FIGURES_DIR = "reports/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# ==========================================
# 1. QUANTITATIVE EDA: METADATA EXTRACTION
# ==========================================
def extract_metadata(data_dir):
    data = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    print("[*] Starting metadata extraction...")

    for root, _, files in os.walk(data_dir):
        class_name = os.path.basename(root)
        if root == data_dir:
            continue

        for file in files:
            file_path = os.path.join(root, file)

            if not file.lower().endswith(valid_extensions):
                continue

            file_size_kb = os.path.getsize(file_path) / 1024.0

            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
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
                    "resolution": (
                        f"{width}x{height}" if (width and height) else "Unknown"
                    ),
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
# 2. ANOMALY DETECTION
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


def detect_anomalies(df):
    print("[*] Detecting anomalies...")
    valid_df = df[~df["is_corrupted"]].copy()
    valid_df["file_hash"] = valid_df["file_path"].apply(calculate_md5)

    duplicate_hashes = valid_df[
        valid_df.duplicated(subset=["file_hash"], keep=False)
    ]
    duplicate_count = duplicate_hashes["file_hash"].nunique()

    corrupted_count = df["is_corrupted"].sum()
    grayscale_in_rgb = len(df[df["channels"] == 1])

    print("\n--- ANOMALY REPORT ---")
    print(f"Total Images: {len(df)}")
    print(f"Corrupted Images: {corrupted_count}")
    print(f"Duplicate Image Groups: {duplicate_count}")
    print(f"Grayscale Images (Single Channel): {grayscale_in_rgb}")
    print("----------------------\n")


# ==========================================
# 3. EXISTING & NEW PLOTS
# ==========================================
def plot_class_distribution(df):
    """1. กราฟกระจายตัวตาม Class"""
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df,
        x="class",
        order=df["class"].value_counts().index,
        hue="class",
        palette="viridis",
        legend=False,
    )
    plt.title("Class Distribution (Check Class Imbalance)", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)

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


def plot_top_resolutions(df, top_n=10):
    """2. [ใหม่] กราฟความละเอียดภาพที่พบบ่อยที่สุด Top N"""
    valid_df = df[~df["is_corrupted"]]
    top_res = valid_df["resolution"].value_counts().head(top_n).reset_index()
    top_res.columns = ["Resolution", "Count"]

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=top_res,
        x="Count",
        y="Resolution",
        hue="Resolution",
        palette="mako",
        legend=False,
    )
    plt.title(f"Top {top_n} Most Common Image Resolutions", fontsize=14)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Resolution (WxH)", fontsize=12)

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_width())}",
            (p.get_width(), p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "top_resolutions.png"), dpi=300)
    plt.close()


def plot_aspect_vs_filesize(df):
    """3. [ใหม่] กราฟ Aspect Ratio เปรียบเทียบกับ File Size"""
    valid_df = df[~df["is_corrupted"]]

    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=valid_df,
        x="aspect_ratio",
        y="file_size_kb",
        hue="class",
        alpha=0.5,
        palette="Set1",
    )
    plt.title("Aspect Ratio vs File Size (KB)", fontsize=14)
    plt.xlabel("Aspect Ratio (Width / Height)", fontsize=12)
    plt.ylabel("File Size (KB)", fontsize=12)
    plt.axvline(
        x=1.0, color="gray", linestyle="--", label="Square (1:1)"
    )  # เส้นแบ่งสัดส่วนจัตุรัส

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "aspect_vs_filesize.png"), dpi=300
    )
    plt.close()


def plot_brightness_distribution(df, num_samples=300):
    """4. [ใหม่] กราฟวิเคราะห์ความสว่าง (Luminance) ของภาพ"""
    valid_df = df[~df["is_corrupted"]]
    sample_df = valid_df.sample(
        min(num_samples, len(valid_df)), random_state=42
    )

    brightness_list = []

    for _, row in sample_df.iterrows():
        img = cv2.imread(row["file_path"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # คำนวณค่าความสว่างเฉลี่ยของรูปนั้นๆ
            mean_brightness = np.mean(img)
            brightness_list.append(
                {"class": row["class"], "brightness": mean_brightness}
            )

    bright_df = pd.DataFrame(brightness_list)

    plt.figure(figsize=(9, 5))
    sns.kdeplot(
        data=bright_df,
        x="brightness",
        hue="class",
        common_norm=False,
        fill=True,
        alpha=0.4,
    )
    plt.title("Image Brightness Distribution (0=Dark, 255=Bright)", fontsize=14)
    plt.xlabel("Mean Pixel Intensity (Brightness)", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "brightness_distribution.png"), dpi=300
    )
    plt.close()


def plot_channel_boxplot(df, num_samples=300):
    """5. [ใหม่] Boxplot เปรียบเทียบค่าเฉลี่ยของแต่ละช่องสี (R, G, B)"""
    valid_df = df[~df["is_corrupted"]]
    sample_df = valid_df.sample(
        min(num_samples, len(valid_df)), random_state=42
    )

    channel_data = []

    for path in sample_df["file_path"]:
        img = cv2.imread(path)
        if img is not None and len(img.shape) == 3:
            # OpenCV อ่านแบบ BGR -> แปลงเป็น RGB
            b, g, r = (
                np.mean(img[:, :, 0]),
                np.mean(img[:, :, 1]),
                np.mean(img[:, :, 2]),
            )
            channel_data.append({"Red": r, "Green": g, "Blue": b})

    ch_df = pd.DataFrame(channel_data)

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=ch_df, palette=["red", "green", "blue"], boxprops=dict(alpha=0.6)
    )
    plt.title("Average RGB Intensity Comparison Across Dataset", fontsize=14)
    plt.ylabel("Pixel Value (0-255)", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "channel_boxplot.png"), dpi=300)
    plt.close()


def plot_sample_grid(df, samples_per_class=3):
    """6. ตัวอย่างรูปภาพ Grid"""
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
                ax.set_title(
                    f"{cls}\n({img.size[0]}x{img.size[1]})", fontsize=10
                )
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "qualitative_sample_grid.png"), dpi=300
    )
    plt.close()


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(DATA_DIR):
        print(f"[!] Path {DATA_DIR} does not exist!")
        return

    df = extract_metadata(DATA_DIR)
    detect_anomalies(df)

    print("[*] Generating All Quantitative & Qualitative plots...")
    plot_class_distribution(df)
    plot_top_resolutions(df)
    plot_aspect_vs_filesize(df)
    plot_brightness_distribution(df)
    plot_channel_boxplot(df)
    plot_sample_grid(df)

    print(
        f"\n[✓] All figures successfully generated and saved in '{FIGURES_DIR}'!"
    )


if __name__ == "__main__":
    main()