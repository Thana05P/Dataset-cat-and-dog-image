import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. ชี้ไปที่โฟลเดอร์หลักของ Dataset (ไม่ต้องใส่ \train หรือ \test ต่อท้าย)
base_dataset_dir = r"C:\Users\Monster\.cache\kagglehub\datasets\samuelcortinhas\cats-and-dogs-image-classification\versions\4"

filepaths = []
labels = []

print("กำลังค้นหาและรวบรวมรูปภาพจากทั้งโฟลเดอร์ train และ test...")

# 2. ลูปเข้าไปในโฟลเดอร์ทั้ง 'train' และ 'test'
for split_folder in ['train', 'test']:
    split_dir = os.path.join(base_dataset_dir, split_folder)
    
    # เช็คว่ามีโฟลเดอร์นี้อยู่จริงไหม
    if os.path.exists(split_dir):
        # ลูปเข้าไปในโฟลเดอร์คลาส (cats, dogs) ที่อยู่ข้างในอีกที
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        filepaths.append(os.path.join(class_dir, img_name))
                        labels.append(class_name)

# สร้าง DataFrame รวมรูปทั้งหมด
df = pd.DataFrame({
    'filepath': filepaths,
    'label': labels
})

print(f"รวบรวมข้อมูลสำเร็จ! พบรูปภาพทั้งหมด: {len(df)} รูป\n")

if len(df) == 0:
    print("ไม่พบรูปภาพ กรุณาตรวจสอบ Path อีกครั้ง")
else:
    # 3. กำหนด Features และ Target
    X = df['filepath']
    y = df['label']

    # 4. แบ่งข้อมูล 80% Train, 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y 
    )

    # 5. แบ่ง Temp เป็น 10% Val, 10% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.50, 
        random_state=42, 
        stratify=y_temp
    )

    # 6. สร้าง DataFrame สำหรับแต่ละชุด
    train_df = pd.DataFrame({'filepath': X_train, 'label': y_train, 'split': 'train'})
    val_df = pd.DataFrame({'filepath': X_val, 'label': y_val, 'split': 'val'})
    test_df = pd.DataFrame({'filepath': X_test, 'label': y_test, 'split': 'test'})

    # รวมเป็น Manifest ไฟล์เดียว
    manifest_df = pd.concat([train_df, val_df, test_df])

    # 7. บันทึกทับลงในโฟลเดอร์โปรเจกต์
    manifest_path = r'C:\Users\Monster\Documents\GitHub\Dataset-cat-and-dog-image\dataset_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)

    print("=== สรุปผลการแบ่งชุดข้อมูล (80/10/10) ===")
    print(f"Train (80%):      {len(X_train)} รูป")
    print(f"Validation (10%): {len(X_val)} รูป")
    print(f"Test (10%):       {len(X_test)} รูป")
    print("========================================")
    print(f"บันทึกไฟล์ Manifest เรียบร้อยแล้วที่:\n{manifest_path}")