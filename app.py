import os
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from rembg import remove
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import uuid

app = FastAPI()
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(file: UploadFile = File(...)):

    # 1. Load image
    img = Image.open(file.file).convert("RGB")
    img.thumbnail((900, 900))

    # 2. Remove background
    fg = remove(img, session=None).convert("L")

    # 3. Convert to OpenCV
    gray = np.array(fg)

    # 4. Adaptive threshold (THIS IS KEY)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    # 5. Edge detection (anchors face)
    edges = cv2.Canny(gray, 80, 160)

    # 6. Combine edges + darkness
    portrait_map = np.minimum(bw, 255 - edges)

    h, w = portrait_map.shape

    # 7. Prepare canvas
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    # 8. Load words
    with open("words.txt", "r", encoding="utf-8") as f:
        words = f.read().split()

    # 9. Fonts
    try:
        small = ImageFont.truetype("DejaVuSans.ttf", 8)
        medium = ImageFont.truetype("DejaVuSans.ttf", 12)
    except:
        small = medium = ImageFont.load_default()

    word_i = 0

    # 10. Row-wise drawing (CRITICAL)
    for y in range(0, h, 9):
        darkness = np.mean(portrait_map[y:y+3, :])

        # Skip mostly white rows
        if darkness > 240:
            continue

        font = small if darkness < 180 else medium
        x = 0

        while x < w:
            if portrait_map[y, x] < 200:
                word = words[word_i % len(words)]
                word_i += 1

                draw.text((x, y), word, fill=(0, 0, 0), font=font)
                x += draw.textlength(word, font=font) + 6
            else:
                x += 12

    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    canvas.save(path)

    return FileResponse(path, media_type="image/png")
