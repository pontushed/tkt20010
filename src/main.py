from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import base64
import numpy as np
from PIL import Image, ImageOps
from neuroverkko import Neuroverkko
from tensorflow.keras.datasets import mnist

nv = Neuroverkko(None)
try:
    nv.lataa("neuroverkko.pkl")
except:
    print("Neuroverkkoa ei löytynyt. Aja ensin 'poetry run invoke kouluta'.")
    exit()

_, (X_testi, y_testi) = mnist.load_data()
X_testi = X_testi.reshape((10000, 28 * 28))
X_testi = X_testi.astype("float32") / 255

app = FastAPI()

app.mount("/static", StaticFiles(directory="src/app/static"), name="static")
templates = Jinja2Templates(directory="src/app/templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/randomkuva")
def randomkuva():
    i = np.random.randint(0, len(X_testi))
    img = Image.fromarray(X_testi[i].reshape(28, 28) * 255).convert("L")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    prediction = nv.ennusta([X_testi[i]], todennakoisyydet=True)[0]
    number = np.argmax(prediction)
    certainty = prediction[number] * 100
    other_candidates = ", ".join(np.argsort(prediction)[:3].astype(str)) if certainty < 99 else "-"
    return {
        "kuva": base64.b64encode(buffered.getvalue()),
        "tunnistus": f"{number} (varmuus {certainty:.2f}%)\nMuut vaihtoehdot: {other_candidates}\n",
        "arvo": int(y_testi[i]),
    }


@app.post("/predict")
def predict(data: str = Body()):
    image = str.encode(data[22:])
    base64bytes = base64.b64decode(image)
    bytesObj = io.BytesIO(base64bytes)
    pil_image = ImageOps.grayscale(Image.open(bytesObj)).resize((28, 28))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    np_image = np.asarray(pil_image).reshape(
        784,
    )
    prediction = nv.ennusta([np_image])[0]
    return {"numero": str(prediction), "kuva": base64.b64encode(buffered.getvalue())}
