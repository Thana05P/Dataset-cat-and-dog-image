import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. หาตำแหน่งโฟลเดอร์โปรเจกต์หลัก (ถอยออกจาก src/ มา 1 ระดับ)
current_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 2. กำหนด Path ต้นทาง และ ปลายทาง
processed_dir = os.path.join(current_project_dir, 'data', 'processed')

# กำหนดปลายทางให้อยู่ที่ Root ของโปรเจกต์ (อยู่นอกโฟลเดอร์ data)
output_base_dir = current_project_dir 

print(f"กำลังค้นหารูปภาพที่ผ่านการทำ Preprocessing จาก:\n{processed_dir}")

filepaths = []
labels = []

# เช็คว่ามีโฟลเดอร์ processed หรือยัง
if not os.path.exists(processed_dir):
    print(f"\n❌ ไม่พบโฟลเดอร์: {processed_dir}")
    print("กรุณาตรวจสอบว่าได้รันไฟล์ preprocessing.py แล้ว และมีโฟลเดอร์ data/processed อยู่ในโปรเจกต์")
else:
    # 3. ลูปอ่านโฟลเดอร์ย่อย (เช่น cats, dogs)
    for class_name in os.listdir(processed_dir):
        class_dir = os.path.join(processed_dir, class_name)
        
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(class_dir, img_name)
                    filepaths.append(full_path)
                    labels.append(class_name)

    df = pd.DataFrame({
        'src_path': filepaths,
        'label': labels
    })

    print(f"รวบรวมข้อมูลสำเร็จ! พบรูปภาพทั้งหมด: {len(df)} รูป\n")

    if len(df) == 0:
        print("❌ ไม่พบรูปภาพในโฟลเดอร์ กรุณาตรวจสอบโค้ด Preprocessing อีกครั้ง")
    else:
        # 4. แบ่งข้อมูล 80% Train, 20% Temp (Val + Test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.20, 
            random_state=42, 
            stratify=df['label']
        )

        # 5. แบ่ง Temp เป็น 10% Val, 10% Test
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.50, 
            random_state=42, 
            stratify=temp_df['label']
        )

        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        # 6. คัดลอกไฟล์รูปภาพแยกไปยังโฟลเดอร์ train/, val/, test/ (อยู่นอก data)
        print("กำลังคัดลอกรูปภาพไปยังโฟลเดอร์ train, val, test...")
        manifest_records = []

        for split_name, split_data in splits.items():
            for _, row in split_data.iterrows():
                src_file = row['src_path']
                class_name = row['label']
                img_name = os.path.basename(src_file)

                # ปลายทาง: project_root/{split_name}/{class_name}/ (อยู่นอก data)
                dest_folder = os.path.join(output_base_dir, split_name, class_name)
                os.makedirs(dest_folder, exist_ok=True)

                dest_file = os.path.join(dest_folder, img_name)
                
                # คัดลอกไฟล์รูปภาพ
                shutil.copy2(src_file, dest_file)

                # บันทึก relative path ลง Manifest
                rel_path = os.path.relpath(dest_file, current_project_dir)
                manifest_records.append({
                    'filepath': rel_path.replace('\\', '/'),
                    'label': class_name,
                    'split': split_name
                })

        # 7. บันทึกไฟล์ Manifest CSV
        manifest_df = pd.DataFrame(manifest_records)
        manifest_path = os.path.join(current_project_dir, 'dataset_manifest.csv')
        manifest_df.to_csv(manifest_path, index=False)

        print("\n=== สรุปผลการแบ่งชุดข้อมูล (80/10/10) ===")
        print(f"Train (80%):       {len(train_df)} รูป")
        print(f"Validation (10%):  {len(val_df)} รูป")
        print(f"Test (10%):        {len(test_df)} รูป")
        print("========================================")
        print(f"✅ บันทึกโฟลเดอร์ train, val, test สำเร็จที่:\n{current_project_dir}")
        print(f"✅ บันทึกไฟล์ Manifest เรียบร้อยแล้วที่:\n{manifest_path}")