import pandas as pd
from sklearn.model_selection import train_test_split

# 1. สมมติว่าคุณมีข้อมูลรายชื่อไฟล์และ Label ที่ผ่านการ Clean เรียบร้อยแล้ว
# (ในโปรเจกต์จริง คุณอาจจะดึงชื่อไฟล์มาจากโฟลเดอร์หลังทำ Preprocessing)
# ตัวอย่างจำลองข้อมูล:
data = {
    'filepath': ['img_001.jpg', 'img_002.jpg', 'img_003.jpg', 'img_004.jpg', 'img_005.jpg'],
    'label': ['cat', 'dog', 'cat', 'dog', 'cat']
}
df = pd.DataFrame(data)

# กำหนด Features (X) เป็นชื่อไฟล์ และ Target (y) เป็นคลาสสัตว์
X = df['filepath']
y = df['label']

# 2. แบ่งข้อมูลรอบที่ 1: ดึง Train (80%) ออกมาก่อน และเหลือ Temp ไว้ (20% สำหรับ Val+Test)
# ใช้ stratify=y เพื่อรักษาสัดส่วนคลาส และ random_state=42 (ตั้ง Seed)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y 
)

# 3. แบ่งข้อมูลรอบที่ 2: แบ่ง Temp (20%) เป็น Validation (10%) และ Test (10%)
# test_size=0.5 คือแบ่งครึ่งจาก 20% ที่เหลือ จะได้ฝั่งละ 10% พอดี
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.50, 
    random_state=42, 
    stratify=y_temp
)

# 4. สร้าง DataFrame สำหรับแต่ละชุดเพื่อรวมเป็น Manifest
train_df = pd.DataFrame({'filepath': X_train, 'label': y_train, 'split': 'train'})
val_df = pd.DataFrame({'filepath': X_val, 'label': y_val, 'split': 'val'})
test_df = pd.DataFrame({'filepath': X_test, 'label': y_test, 'split': 'test'})

# นำข้อมูลทั้ง 3 ชุดมาต่อกัน (Concatenate)
manifest_df = pd.concat([train_df, val_df, test_df])

# 5. บันทึกลงไฟล์ CSV (Manifest) เพื่อให้ตรวจสอบย้อนหลังได้
manifest_path = 'dataset_manifest.csv'
manifest_df.to_csv(manifest_path, index=False)

print(f"แบ่งข้อมูลสำเร็จ! บันทึกไฟล์ Manifest ไว้ที่: {manifest_path}")
print(f"สัดส่วนข้อมูล: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")