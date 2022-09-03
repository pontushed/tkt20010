# Vertailu referenssitason ja oman toteutuksen välillä
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential, layers
from time import perf_counter
import matplotlib.pyplot as plt
from verkko.neuroverkko import Neuroverkko, Tihea, ReLU
import utils.utils as utils

utils.tulosta_otsake("Vertailu oman toteutuksen ja TensorFlown välillä")

# Parametrit
epookit = 10
alijoukon_koko = 100

# Ladataan MNIST-datasetti
print("[1/4] Ladataan MNIST-datasetti...")
(X_koulutus, y_koulutus), (X_testi, y_testi) = utils.lataa_mnist()

# Neuroverkkojen määrittely
print("[2/4] Määritellään neuroverkot...")
malli = Sequential(
    [
        layers.Dense(512, input_shape=(784,), activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
VERKKO = [Tihea(X_koulutus.shape[1], 512), ReLU(), Tihea(512, 10)]

# Neuroverkon rakennus
malli.compile(optimizer="RMSprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
neuroverkko = Neuroverkko(VERKKO)


# Neuroverkon kouluttaminen
print("[3/4] Koulutetaan neuroverkot...")
alku = perf_counter()
historia_tf = malli.fit(
    X_koulutus,
    y_koulutus,
    epookit=epookit,
    alijoukon_koko=alijoukon_koko,
    validation_data=(X_testi, y_testi),
)
loppu = perf_counter()
print(f"Tensorflown koulutus kesti {loppu - alku:.3f} sekuntia.")
testihukka, testitarkkuus = malli.evaluate(X_testi, y_testi)
print(f"\nTF-Neuroverkon tarkkuus testidatalla (n={len(X_testi)}): {testitarkkuus*100:.2f}%")

alku = perf_counter()
historia_oma = neuroverkko.sovita(
    X_koulutus,
    y_koulutus,
    epookit=epookit,
    alijoukon_koko=alijoukon_koko,
    validaatiodata=(X_testi, y_testi),
)
loppu = perf_counter()
print(f"Oman neuroverkon koulutus kesti {loppu - alku:.5f} sekuntia.")
tarkkuus = neuroverkko.evaluoi(X_testi, y_testi)
print(f"\nOman neuroverkon tarkkuus testidatalla (n={len(X_testi)}): {tarkkuus*100:.2f}%")

# Tulostetaan yhteenveto
print("[4/4] Tulostetaan yhteenveto...")

utils.piirra_kaaviot(historia_tf, historia_oma)
