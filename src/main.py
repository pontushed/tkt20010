from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import base64
import numpy as np
from PIL import Image, ImageOps
from verkko.neuroverkko import Neuroverkko
import utils.utils as utils

try:
    nv = Neuroverkko(utils.lataa_malli("neuroverkko.pkl"))
except:
    print("Neuroverkkoa ei löytynyt. Aja ensin 'poetry run invoke kouluta'.")
    exit()

_, (X_testi, y_testi) = utils.lataa_mnist()

app = FastAPI()

app.mount("/static", StaticFiles(directory="src/app/static"), name="static")
templates = Jinja2Templates(directory="src/app/templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/randomkuva")
def randomkuva():
    kuva, numero, varmuus, muut_vaihtoehdot, oikea_arvo = ennusta_satunnainen()
    return {
        "kuva": kuva,
        "tunnistus": f"{numero} (varmuus {varmuus:.2f}%)\nMuut vaihtoehdot: {muut_vaihtoehdot}\n",
        "arvo": oikea_arvo,
    }


@app.post("/predict")
def predict(data: str = Body()):
    np_image, base64image = kasittele_syote(data)
    ennuste = nv.ennusta([np_image])[0]
    return {"numero": str(ennuste), "kuva": base64image}


def ennusta_satunnainen():
    """Ennustaa satunnaisen kuvan ja palauttaa kuvan sekä tulokset."""
    i = np.random.randint(0, len(X_testi))
    img = Image.fromarray(X_testi[i].reshape(28, 28) * 255).convert("L")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    oikea_arvo = int(y_testi[i])
    kuva = base64.b64encode(buffered.getvalue())
    ennuste = nv.ennusta([X_testi[i]], todennakoisyydet=True)[0]
    numero = np.argmax(ennuste)
    varmuus = ennuste[numero] * 100
    muut_vaihtoehdot = ", ".join(np.argsort(ennuste)[:3].astype(str)) if varmuus < 99 else "-"
    return kuva, numero, varmuus, muut_vaihtoehdot, oikea_arvo


def kasittele_syote(data: str):
    """Käsittelee käyttäjän syötteen ja palauttaa np.arrayn sekä base64-koodatun kuvan."""
    image = str.encode(data[22:])
    base64bytes = base64.b64decode(image)
    bytesObj = io.BytesIO(base64bytes)
    pil_image = ImageOps.grayscale(Image.open(bytesObj)).resize((28, 28))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    np_image = np.asarray(pil_image).reshape(
        784,
    )
    base64image = base64.b64encode(buffered.getvalue())
    return np_image, base64image
