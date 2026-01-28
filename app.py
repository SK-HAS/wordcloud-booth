import os
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from rembg import remove
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np
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

    # 3. Improve contrast
    fg = ImageOps.autocontrast(fg)
    fg = fg.filter(ImageFilter.GaussianBlur(radius=1))

    fg_np = np.array(fg)
    h, w = fg_np.shape

    # 4. Prepare canvas
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    # 5. Load words
    with open("words.txt", "r", encoding="utf-8") as f:
        words = f.read().split()

    # 6. Load fonts
    try:
        font_small = ImageFont.truetype("DejaVuSans.ttf", 9)
        font_big = ImageFont.truetype("DejaVuSans.ttf", 13)
    except:
        font_small = font_big = ImageFont.load_default()

    word_i = 0

    # 7. Row-based drawing (FIXED)
    for y in range(0, h, 10):
        row = fg_np[y]
        darkness = int(np.mean(row))

        # Skip mostly white rows
        if darkness > 210:
            continue

        font = font_small if darkness < 150 else font_big
        x = 0

        while x < w:
            xi = int(x)  # 🔑 CRITICAL FIX

            if row[xi] < 180:
                word = words[word_i % len(words)]
                word_i += 1

                draw.text((xi, y), word, fill=(0, 0, 0), font=font)

                word_width = int(draw.textlength(word, font=font))
                x += word_width + 6
            else:
                x += 10

    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    canvas.save(path)

    return FileResponse(path, media_type="image/png")
