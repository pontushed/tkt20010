# Vertailu referenssitason ja oman toteutuksen välillä
# käyttäen ristiinvalidointia
from tensorflow.keras import Sequential, layers
from time import perf_counter
from verkko.neuroverkko import Neuroverkko, Tihea, ReLU
import numpy as np
import utils.utils as utils

print("*" * 70)
print("*" + " MNIST-numerontunnistus ".center(68) + "*")
print("*" + " Vertailu oman toteutuksen ja TensorFlown välillä ".center(68) + "*")
print("*" + " Metodi: Ristiinvalidointi ".center(68) + "*")
print("*" * 70 + "\n")

# Parametrit
epookit = 10
alijoukon_koko = 100
k_ristiinvalidoinnit = 5

# Ladataan MNIST-datasetti
print("[1/4] Ladataan MNIST-datasetti...")
(X_koulutus, y_koulutus), (X_tsti, y_tsti) = utils.lataa_mnist()
X = np.concatenate((X_koulutus, X_tsti))
y = np.concatenate((y_koulutus, y_tsti))

# Neuroverkkojen määrittely
print("[2/4] Määritellään neuroverkot...")
VERKKO = [Tihea(X_koulutus.shape[1], 512), ReLU(), Tihea(512, 10)]

# Neuroverkon kouluttaminen
print("[3/4] Koulutetaan neuroverkot...")


def k_ristiinvalidointi(X_koul, y_koul, X_tst, y_tst):
    alku = perf_counter()
    malli = Sequential(
        [
            layers.Dense(512, input_shape=(784,), activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    malli.compile(optimizer="RMSprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    malli.fit(
        X_koul,
        y_koul,
        epookit=epookit,
        alijoukon_koko=alijoukon_koko,
        # validation_data=(X_tst, y_tst),
    )
    loppu = perf_counter()
    print(f"Tensorflown koulutus kesti {loppu - alku:.3f} sekuntia.")
    _, test_acc = malli.evaluate(X_tst, y_tst)
    print(f"\nTF-Neuroverkon tarkkuus testidatalla (n={len(y_tst)}): {test_acc*100:.2f}%")

    alku = perf_counter()
    VERKKO = [Tihea(X_koulutus.shape[1], 512), ReLU(), Tihea(512, 10)]
    neuroverkko = Neuroverkko(VERKKO)
    neuroverkko.sovita(
        X_koul,
        y_koul,
        epookit=epookit,
        alijoukon_koko=alijoukon_koko,
        # validaatiodata=(X_tst, y_tst),
    )
    loppu = perf_counter()
    print(f"Oman neuroverkon koulutus kesti {loppu - alku:.5f} sekuntia.")
    tarkkuus = neuroverkko.evaluoi(X_tst, y_tst)
    print(f"\nOman neuroverkon tarkkuus testidatalla (n={len(y_tst)}): {tarkkuus*100:.2f}%")
    return test_acc, tarkkuus


tulokset = []
for iteraatio in range(k_ristiinvalidoinnit):
    print(f"\nIteraatio {iteraatio + 1}/{k_ristiinvalidoinnit}")
    # Otetaan koko datasetistä osa koulutukseen ja osa testaamiseen
    # Ensimmäisellä kerralla jako on 60000-10000
    if iteraatio == 0:
        X_koul = X[:60000]
        X_tst = X[60000:]
        y_koul = y[:60000]
        y_tst = y[60000:]
    else:
        split = 60000 - iteraatio * 10000
        X_koul = np.concatenate((X[:split], X[split + 10000 :]))
        X_tst = X[split : split + 10000]
        y_koul = np.concatenate((y[:split], y[split + 10000 :]))
        y_tst = y[split : split + 10000]
    tulos = k_ristiinvalidointi(X_koul, y_koul, X_tst, y_tst)
    tulokset.append(tulos)

# Tulostetaan yhteenveto
print("[4/4] Tulostetaan yhteenveto...")
for i, tulos in enumerate(tulokset):
    print(f"Iteraatio {i} - TensorFlow: {tulos[0]*100:.2f}% Oma: {tulos[1]*100:.2f}%")
print("Keskimääräinen tarkkuus:")
print(f"Tensorflown tarkkuus: {sum([t[0] for t in tulokset])/k_ristiinvalidoinnit*100:.2f}%")
print(f"Oman neuroverkon tarkkuus: {sum([t[1] for t in tulokset])/k_ristiinvalidoinnit*100:.2f}%")
