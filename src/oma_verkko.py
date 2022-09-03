# Oma toteutus numpy-kirjaston avulla
from tensorflow.keras.datasets import mnist
from time import perf_counter
from verkko.neuroverkko import Neuroverkko, Tihea, ReLU
import matplotlib.pyplot as plt
import utils.utils as utils

utils.tulosta_otsake("Oma toteutus")

# Ladataan MNIST-datasetti, tyyppi numpy.ndarray
print("[1/4] Ladataan MNIST-datasetti...")
(X_koulutus, y_koulutus), (X_testi, y_testi) = utils.lataa_mnist()

# Neuroverkon määrittely
print("[2/4] Määritellään neuroverkko...")
VERKKO = [Tihea(X_koulutus.shape[1], 512), ReLU(), Tihea(512, 10)]

# Neuroverkon rakennus
neuroverkko = Neuroverkko(VERKKO)

# Neuroverkon kouluttaminen
print("[3/4] Koulutetaan neuroverkkoa...")
start = perf_counter()
historia = neuroverkko.sovita(
    X_koulutus,
    y_koulutus,
    epookit=10,
    alijoukon_koko=100,
    validaatiodata=(X_testi, y_testi),
)
end = perf_counter()
print(f"Koulutus kesti {end - start:.5f} sekuntia.")

# Neuroverkon testaus
print("[4/4] Testataan neuroverkkoa...")
tarkkuus = neuroverkko.evaluoi(X_testi, y_testi)
print()
print(f"Neuroverkon tarkkuus testidatalla (n={len(y_testi)}): {tarkkuus*100:.2f}%")

utils.piirra_kaaviot_oma(historia)

print("Tallennetaan verkko...")
utils.tallenna_malli(neuroverkko.verkko, "neuroverkko.pkl")
