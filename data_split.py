import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. หาตำแหน่งโฟลเดอร์โปรเจกต์หลัก (ดึงตำแหน่งอ้างอิงจากไฟล์นี้)
current_project_dir = os.path.dirname(os.path.abspath(__file__))

# 2. ชี้ไปที่โฟลเดอร์ที่เพื่อนทำ Preprocessing เอาไว้
# *** สำคัญ: ต้องถามเพื่อนว่าโค้ดของเขาเซฟรูปลงในโฟลเดอร์ชื่ออะไร ***
# (สมมติว่าเพื่อนจัดเก็บไว้ในโฟลเดอร์ 'data/processed' ภายในโฟลเดอร์โปรเจกต์)
processed_dir = os.path.join(current_project_dir, 'data', 'processed')

print(f"กำลังค้นหารูปภาพที่ผ่านการทำ Preprocessing จาก:\n{processed_dir}")

filepaths = []
labels = []

# เช็คว่ามีโฟลเดอร์ที่เพื่อนทำไว้หรือยัง
if not os.path.exists(processed_dir):
    print(f"\n❌ ไม่พบโฟลเดอร์: {processed_dir}")
    print("กรุณาตรวจสอบว่าได้รันไฟล์ preprocessing.py ของเพื่อนก่อน และตั้งชื่อโฟลเดอร์ตรงกัน")
else:
    # 3. ลูปอ่านโฟลเดอร์ย่อย (เช่น cats, dogs) 
    for class_name in os.listdir(processed_dir):
        class_dir = os.path.join(processed_dir, class_name)
        
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # สร้าง Full Path สำหรับตัวแปลผล
                    full_path = os.path.join(class_dir, img_name)
                    
                    # *** ทีเด็ด: แปลงเป็น Relative Path ป้องกัน Error เวลารันบนคอมคนอื่น ***
                    rel_path = os.path.relpath(full_path, current_project_dir)
                    # rel_path จะออกมาหน้าตาแบบนี้: data\processed\cats\cat_01.jpg
                    
                    filepaths.append(rel_path)
                    labels.append(class_name)

    df = pd.DataFrame({
        'filepath': filepaths,
        'label': labels
    })

    print(f"รวบรวมข้อมูลสำเร็จ! พบรูปภาพทั้งหมด: {len(df)} รูป\n")

    if len(df) == 0:
        print("❌ ไม่พบรูปภาพในโฟลเดอร์ กรุณาตรวจสอบโค้ด Preprocessing ของเพื่อน")
    else:
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

        train_df = pd.DataFrame({'filepath': X_train, 'label': y_train, 'split': 'train'})
        val_df = pd.DataFrame({'filepath': X_val, 'label': y_val, 'split': 'val'})
        test_df = pd.DataFrame({'filepath': X_test, 'label': y_test, 'split': 'test'})

        manifest_df = pd.concat([train_df, val_df, test_df])

        # 6. บันทึกไฟล์ Manifest ลงโฟลเดอร์โปรเจกต์
        manifest_path = os.path.join(current_project_dir, 'dataset_manifest.csv')
        
        # เซฟเป็น CSV (ใช้เครื่องหมายทับ / แบบมาตรฐานสากล แทน \ ของ Windows)
        manifest_df['filepath'] = manifest_df['filepath'].str.replace('\\', '/')
        manifest_df.to_csv(manifest_path, index=False)

        print("=== สรุปผลการแบ่งชุดข้อมูล (80/10/10) ===")
        print(f"Train (80%):      {len(X_train)} รูป")
        print(f"Validation (10%): {len(X_val)} รูป")
        print(f"Test (10%):       {len(X_test)} รูป")
        print("========================================")
        print(f"บันทึกไฟล์ Manifest เรียบร้อยแล้วที่:\n{manifest_path}")