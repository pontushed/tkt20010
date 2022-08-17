# Vertailu referenssitason ja oman toteutuksen välillä
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential, layers
from time import perf_counter
from neuroverkko import Neuroverkko, Tihea, ReLU
import numpy as np

print("*" * 70)
print("*" + " MNIST-numerontunnistus ".center(68) + "*")
print("*" + " Vertailu oman toteutuksen ja TensorFlown välillä ".center(68) + "*")
print("*" + " Metodi: Ristiinvalidointi ".center(68) + "*")
print("*" * 70 + "\n")

print("TensorFlow versio:", tf.__version__)

# Parametrit
epochs = 10
batch_size = 100
k_folds = 5

# Ladataan MNIST-datasetti
print("[1/4] Ladataan MNIST-datasetti...")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
X = np.concatenate((train_images, test_images))
y = np.concatenate((train_labels, test_labels))

# Neuroverkkojen määrittely
print("[2/4] Määritellään neuroverkot...")
VERKKO = [Tihea(train_images.shape[1], 512), ReLU(), Tihea(512, 10)]

# Neuroverkon kouluttaminen
print("[3/4] Koulutetaan neuroverkot...")


def k_fold_validointi(X_train, y_train, X_test, y_test):
    start = perf_counter()
    model = Sequential(
        [
            layers.Dense(512, input_shape=(784,), activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="RMSprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        # validation_data=(X_test, y_test),
    )
    end = perf_counter()
    print(f"Tensorflown koulutus kesti {end - start:.3f} sekuntia.")
    _, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTF-Neuroverkon tarkkuus testidatalla (n={len(y_test)}): {test_acc*100:.2f}%")

    start = perf_counter()
    VERKKO = [Tihea(train_images.shape[1], 512), ReLU(), Tihea(512, 10)]
    neuroverkko = Neuroverkko(VERKKO)
    neuroverkko.sovita(
        X_train,
        y_train,
        epookit=epochs,
        alijoukon_koko=batch_size,
        # validaatiodata=(X_test, y_test),
    )
    end = perf_counter()
    print(f"Oman neuroverkon koulutus kesti {end - start:.5f} sekuntia.")
    tarkkuus = neuroverkko.evaluoi(X_test, y_test)
    print(f"\nOman neuroverkon tarkkuus testidatalla (n={len(y_test)}): {tarkkuus*100:.2f}%")
    return test_acc, tarkkuus


tulokset = []
for iteraatio in range(k_folds):
    print(f"\nIteraatio {iteraatio + 1}/{k_folds}")
    if iteraatio == 0:
        X_train = X[:60000]
        X_test = X[60000:]
        y_train = y[:60000]
        y_test = y[60000:]
    else:
        split = 60000 - iteraatio * 10000
        X_train = np.concatenate((X[:split], X[split + 10000 :]))
        X_test = X[split : split + 10000]
        y_train = np.concatenate((y[:split], y[split + 10000 :]))
        y_test = y[split : split + 10000]
    tulos = k_fold_validointi(X_train, y_train, X_test, y_test)
    tulokset.append(tulos)

# Tulostetaan yhteenveto
print("[4/4] Tulostetaan yhteenveto...")
for i, tulos in enumerate(tulokset):
    print(f"Iteraatio {i} - TensorFlow: {tulos[0]*100:.2f}% Oma: {tulos[1]*100:.2f}%")
print("Keskimääräinen tarkkuus:")
print(f"Tensorflown tarkkuus: {sum([t[0] for t in tulokset])/k_folds*100:.2f}%")
print(f"Oman neuroverkon tarkkuus: {sum([t[1] for t in tulokset])/k_folds*100:.2f}%")
