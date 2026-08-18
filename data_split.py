import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. กำหนด Path ที่เก็บรูปภาพที่ Clean แล้ว (ใส่ r ข้างหน้าเพื่อป้องกัน Error จาก Backslash ใน Windows)
base_dir = r"C:\Users\Monster\.cache\kagglehub\datasets\samuelcortinhas\cats-and-dogs-image-classification\versions\4\test"

filepaths = []
labels = []

# 2. ลูปเข้าไปอ่านโฟลเดอร์ย่อย (เช่น cats, dogs) และดึงรายชื่อไฟล์รูปภาพ
print("กำลังค้นหารูปภาพ...")
for class_name in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, class_name)
    
    # เช็คว่าเป็นโฟลเดอร์จริงๆ
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            # กรองเอาเฉพาะไฟล์รูปภาพ
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # เก็บที่อยู่ไฟล์แบบเต็ม (Absolute Path) และชื่อคลาส
                filepaths.append(os.path.join(class_dir, img_name))
                labels.append(class_name)

# นำข้อมูลทั้งหมดมาสร้างเป็น DataFrame
df = pd.DataFrame({
    'filepath': filepaths,
    'label': labels
})

print(f"ดึงข้อมูลสำเร็จ! พบรูปภาพทั้งหมด: {len(df)} รูป\n")

if len(df) == 0:
    print("ไม่พบรูปภาพ กรุณาตรวจสอบว่ามีรูปภาพอยู่ในโฟลเดอร์ย่อยหรือไม่")
else:
    # 3. กำหนด Features (X) เป็นที่อยู่ไฟล์ และ Target (y) เป็นคลาสสัตว์
    X = df['filepath']
    y = df['label']

    # 4. แบ่งข้อมูลรอบที่ 1: Train 80% / Temp 20%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y 
    )

    # 5. แบ่งข้อมูลรอบที่ 2: Temp 20% แบ่งครึ่งเป็น Validation 10% / Test 10%
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

    # นำข้อมูลทั้ง 3 ชุดมารวมกันเป็นไฟล์เดียว (Manifest)
    manifest_df = pd.concat([train_df, val_df, test_df])

    # 7. เซฟลงในโฟลเดอร์ GitHub ของโปรเจกต์คุณ
    manifest_path = r'C:\Users\Monster\Documents\GitHub\Dataset-cat-and-dog-image\dataset_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)

    print("=== สรุปผลการแบ่งชุดข้อมูล (80/10/10) ===")
    print(f"Train (80%):      {len(X_train)} รูป")
    print(f"Validation (10%): {len(X_val)} รูป")
    print(f"Test (10%):       {len(X_test)} รูป")
    print("========================================")
    print(f"บันทึกไฟล์ Manifest เรียบร้อยแล้วที่:\n{manifest_path}")