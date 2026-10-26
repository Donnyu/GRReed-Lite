import sys
import os
from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pytesseract
import cv2
import numpy as np
import uvicorn
from docx import Document
from docx.shared import Pt
import io

# --- ХИТРОСТЬ ДЛЯ РАБОТЫ EXE ---
# Определяем базовую директорию (папка с EXE или папка со скриптом)
if getattr(sys, 'frozen', False):
    # Если запущено как EXE (PyInstaller распаковывает всё в sys._MEIPASS)
    base_dir = sys._MEIPASS
else:
    # Если запущено как обычный скрипт
    base_dir = os.path.dirname(os.path.abspath(__file__))

# --- НАСТРОЙКА TESSERACT И TESSDATA ---
# 1. Ищем tesseract.exe внутри папки 'tesseract', которая лежит рядом (или внутри EXE)
local_tesseract = os.path.join(base_dir, 'tesseract', 'tesseract.exe')

if os.path.exists(local_tesseract):
    pytesseract.pytesseract.tesseract_cmd = local_tesseract
    # Если exe внутри папки tesseract, то tessdata тоже там
    tessdata_dir = os.path.join(base_dir, 'tesseract', 'tessdata')
else:
    # Если локальной папки нет, используем стандартный путь Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    tessdata_dir = r'C:\Program Files\Tesseract-OCR\tessdata'

# Формируем конфиг для Tesseract с указанием пути к tessdata
custom_config = r'--oem 3 --psm 3 --tessdata-dir "' + tessdata_dir + '"'

app = FastAPI(title="GRReed Lite API")
templates = Jinja2Templates(directory=base_dir)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# Функция для общей предобработки картинки (с Deskew)
def preprocess_image(contents: bytes):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Перевод в ч/б
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Убираем шум
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 3. Увеличение (если картинка маленькая)
    if max(gray.shape) < 1500:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 4. ВЫРАВНИВАНИЕ (DESKEW)
    gray_inv = cv2.bitwise_not(gray)
    thresh_otsu = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    coords = np.column_stack(np.where(thresh_otsu > 0))
    
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        if abs(angle) > 0.5:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 5. Инверсия (если фон темный)
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # 6. Бинаризация для Tesseract
    thresh_final = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh_final

@app.post("/api/ocr")
async def process_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)

        text = pytesseract.image_to_string(processed_img, lang='rus+eng', config=custom_config)

        return JSONResponse(content={"status": "success", "text": text.strip()})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/ocr-word")
async def process_image_to_word(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)

        text = pytesseract.image_to_string(processed_img, lang='rus+eng', config=custom_config)

        doc = Document()
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Times New Roman'
        font.size = Pt(12)
        
        for paragraph in text.split('\n'):
            if paragraph.strip():
                doc.add_paragraph(paragraph.strip())
        
        file_stream = io.BytesIO()
        doc.save(file_stream)
        file_stream.seek(0)

        return Response(
            content=file_stream.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": "attachment; filename=GRReed_Lite.docx"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    print("Запуск GRReed Lite сервера на http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)