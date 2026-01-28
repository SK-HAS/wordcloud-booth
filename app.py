python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from rembg import remove
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
import uuid
import os

app = FastAPI()

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    input_image = Image.open(file.file).convert("RGB")

    no_bg = remove(input_image)
    mask = no_bg.split()[-1]
    mask = mask.resize(input_image.size)

    mask_array = np.array(mask)
    mask_array = np.where(mask_array > 10, 255, 0)

    with open("words.txt", "r") as f:
        text = f.read()

    wc = WordCloud(
        background_color="white",
        max_words=1200,
        mask=mask_array,
        collocations=False,
        prefer_horizontal=1.0
    )

    wc.generate(text)

    color_func = ImageColorGenerator(np.array(input_image))
    wc.recolor(color_func=color_func)

    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    wc.to_file(path)

    return FileResponse(path, media_type="image/png")