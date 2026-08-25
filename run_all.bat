@echo off
chcp 65001 > nul
title Cat and Dog Classification Pipeline

echo =====================================================================
echo  🐱🐶 CAT VS DOG CLASSIFICATION: AUTOMATED PIPELINE
echo =====================================================================

:: 1. ติดตั้ง Dependencies ทั้งหมดจาก requirements.txt
echo [Step 1/8] Installing / Verifying requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b %errorlevel%
)

:: 2. รวบรวมข้อมูลดิบ (Data Collection)
echo.
echo [Step 2/8] Running Data Collection...
python src/data_collection.py

:: 3. ทำความสะอาดและแปลงข้อมูล (Preprocessing)
echo.
echo [Step 3/8] Running Data Preprocessing...
python src/preprocessing.py

:: 4. ประมวลผลภาพ (Image Processing)
echo.
echo [Step 4/8] Running Image Processing...
python src/image_processing.py

:: 5. วิเคราะห์ข้อมูลเชิงสำรวจ (EDA)
echo.
echo [Step 5/8] Running Exploratory Data Analysis (EDA)...
python src/eda.py

:: 6. แบ่งชุดข้อมูล Train, Val, Test (Data Split)
echo.
echo [Step 6/8] Running Data Split into Dataset folder...
python src/data_split.py

:: 7. ฝึกสอนโมเดล (Train Model)
echo.
echo [Step 7/8] Training YOLO Classification Model...
python src/train_classification.py
if %errorlevel% neq 0 (
    echo [ERROR] Training failed.
    pause
    exit /b %errorlevel%
)

:: 8. ทดสอบการทำนายผลลัพธ์ภาพเดี่ยว (Predict & Annotate)
echo.
echo [Step 8/8] Testing Prediction on Test Set...
python src/predict_classification.py

:: 9. เปิดหน้าต่าง Web Application (Gradio)
echo.
echo =====================================================================
echo  🚀 LAUNCHING WEB APPLICATION (app.py)
echo =====================================================================
python src/app.py

pause