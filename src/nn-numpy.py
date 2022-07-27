# Oma toteutus numpy-kirjaston avulla
from tensorflow.keras.datasets import mnist
import numpy as np
from time import perf_counter
from neuroverkko import Neuroverkko, Tihea, ReLU
import matplotlib.pyplot as plt

print("*" * 70)
print("*" + " MNIST-numerontunnistus ".center(68) + "*")
print("*" + " Oma toteutus ".center(68) + "*")
print("*" * 70 + "\n")

# Ladataan MNIST-datasetti, tyyppi numpy.ndarray
print("[1/4] Ladataan MNIST-datasetti...")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Neuroverkon määrittely
print("[2/4] Määritellään neuroverkko...")
VERKKO = [Tihea(train_images.shape[1], 512), ReLU(), Tihea(512, 10)]

# Neuroverkon rakennus
neuroverkko = Neuroverkko(VERKKO)

# Neuroverkon kouluttaminen
print("[3/4] Koulutetaan neuroverkkoa...")
start = perf_counter()
history = neuroverkko.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=100,
    validation_data=(test_images, test_labels),
)
end = perf_counter()
print(f"Koulutus kesti {end - start:.5f} sekuntia.")

# Neuroverkon testaus
print("[4/4] Testataan neuroverkkoa...")
test_digits = test_images[0:10]
predictions = neuroverkko.predict(test_digits)
assert predictions[0] == test_labels[0]  # 7
test_acc = neuroverkko.evaluate(test_images, test_labels)
print(
    f"\nNeuroverkon tarkkuus testidatalla (n={test_labels.size}): {test_acc*100:.2f}%"
)
# summarize history for accuracy
plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
# summarize history for loss
plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
