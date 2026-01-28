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
    input_image = Image.open(file.file).convert("RGB")
    input_image.thumbnail((1024, 1024))

    no_bg = remove(input_image, session=None).convert("RGBA")
    alpha = np.array(no_bg.split()[3])

    mask_array = np.where(alpha > 20, 255, 0).astype(np.uint8)
    mask_array = 255 - mask_array  # invert for WordCloud

    with open("words.txt", "r", encoding="utf-8") as f:
        text = f.read()

    wc = WordCloud(
        background_color="white",
        max_words=1200,
        mask=mask_array,
        collocations=False,
        prefer_horizontal=0.9
    )

    wc.generate(text)
    wc.recolor(color_func=ImageColorGenerator(np.array(input_image)))

    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    wc.to_file(path)

    return FileResponse(path, media_type="image/png")


