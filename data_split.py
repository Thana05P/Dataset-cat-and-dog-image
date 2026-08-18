import os
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub

print("กำลังตรวจสอบและดึง Path ของ Dataset อัตโนมัติ...")
# 1. ใช้ kagglehub ดึง Path อัตโนมัติ (ถ้าเครื่องเพื่อนยังไม่เคยโหลด มันจะโหลดให้เลย)
base_dataset_dir = kagglehub.dataset_download("samuelcortinhas/cats-and-dogs-image-classification")
print(f"พบ Dataset ที่: {base_dataset_dir}")

filepaths = []
labels = []

print("กำลังค้นหาและรวบรวมรูปภาพจากทั้งโฟลเดอร์ train และ test...")

# 2. ลูปเข้าไปในโฟลเดอร์ทั้ง 'train' และ 'test'
for split_folder in ['train', 'test']:
    split_dir = os.path.join(base_dataset_dir, split_folder)
    
    if os.path.exists(split_dir):
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        filepaths.append(os.path.join(class_dir, img_name))
                        labels.append(class_name)

df = pd.DataFrame({
    'filepath': filepaths,
    'label': labels
})

print(f"รวบรวมข้อมูลสำเร็จ! พบรูปภาพทั้งหมด: {len(df)} รูป\n")

if len(df) == 0:
    print("ไม่พบรูปภาพ กรุณาตรวจสอบการดาวน์โหลด")
else:
    X = df['filepath']
    y = df['label']

    # 3. แบ่งข้อมูล 80% Train, 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y 
    )

    # 4. แบ่ง Temp เป็น 10% Val, 10% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.50, 
        random_state=42, 
        stratify=y_temp
    )

    train_df = pd.DataFrame({'filepath': X_train, 'label': y_train, 'split': 'train'})
    val_df = pd.DataFrame({'filepath': X_val, 'label': y_val, 'split': 'val'})
    test_df = pd.DataFrame({'filepath': X_test, 'label': y_test, 'split': 'test'})

    manifest_df = pd.concat([train_df, val_df, test_df])

    # 5. หา Path ปัจจุบันของโฟลเดอร์โปรเจกต์ และสร้างไฟล์ CSV
    # (รันเครื่องไหน ก็จะเซฟลงโฟลเดอร์ที่รันโค้ดนั้นเสมอ)
    current_project_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(current_project_dir, 'dataset_manifest.csv')
    
    manifest_df.to_csv(manifest_path, index=False)

    print("=== สรุปผลการแบ่งชุดข้อมูล (80/10/10) ===")
    print(f"Train (80%):      {len(X_train)} รูป")
    print(f"Validation (10%): {len(X_val)} รูป")
    print(f"Test (10%):       {len(X_test)} รูป")
    print("========================================")
    print(f"บันทึกไฟล์ Manifest เรียบร้อยแล้วที่:\n{manifest_path}")