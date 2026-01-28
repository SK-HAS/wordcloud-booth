import os
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from rembg import remove
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
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
    input_image = Image.open(file.file).convert("RGB")
    input_image.thumbnail((900, 900))

    # Remove background
    no_bg = remove(input_image, session=None).convert("RGBA")

    # Convert to grayscale (this carries facial details)
    gray = no_bg.convert("L")

    # Convert grayscale to numpy
    gray_array = np.array(gray)

    # Invert grayscale: dark areas get more words
    mask_array = 255 - gray_array

    # Threshold background to pure white
    mask_array[gray_array < 10] = 255

    with open("words.txt", "r", encoding="utf-8") as f:
        text = f.read()

    wc = WordCloud(
        background_color="white",
        max_words=3000,
        mask=mask_array,
        contour_width=0,
        collocations=False,
        prefer_horizontal=0.85,
        min_font_size=6,
        max_font_size=120,
        relative_scaling=0.6
    )

    wc.generate(text)

    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    wc.to_file(path)

    return FileResponse(path, media_type="image/png")



