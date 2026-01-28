import os
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from rembg import remove
from PIL import Image, ImageDraw, ImageFont
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
    # Load image
    img = Image.open(file.file).convert("RGB")
    img.thumbnail((900, 900))

    # Remove background and convert to grayscale
    fg = remove(img, session=None).convert("L")

    # Increase contrast (face = black, background = white)
    fg = fg.point(lambda x: 0 if x < 120 else 255)

    fg_np = np.array(fg)
    width, height = fg.size

    # Create white canvas
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    # Load words
    with open("words.txt", "r", encoding="utf-8") as f:
        words = f.read().split()

    # Load font (Railway safe)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()

    word_index = 0

    # Draw words ONLY on dark pixels
    for y in range(0, height, 10):
        for x in range(0, width, 60):
            if fg_np[y, x] == 0:
                word = words[word_index % len(words)]
                word_index += 1
                draw.text((x, y), word, fill=(0, 0, 0), font=font)

    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    canvas.save(path)

    return FileResponse(path, media_type="image/png")
