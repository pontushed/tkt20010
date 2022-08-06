# Oma toteutus numpy-kirjaston avulla
from tensorflow.keras.datasets import mnist
from time import perf_counter
from neuroverkko import Neuroverkko, Tihea, ReLU
import matplotlib.pyplot as plt

print("*" * 70)
print("*" + " MNIST-numerontunnistus ".center(68) + "*")
print("*" + " Oma toteutus ".center(68) + "*")
print("*" * 70 + "\n")

# Ladataan MNIST-datasetti, tyyppi numpy.ndarray
print("[1/4] Ladataan MNIST-datasetti...")
(X_koulutus, y_koulutus), (X_testi, y_testi) = mnist.load_data()
X_koulutus = X_koulutus.reshape((60000, 28 * 28))
X_koulutus = X_koulutus.astype("float32") / 255
X_testi = X_testi.reshape((10000, 28 * 28))
X_testi = X_testi.astype("float32") / 255

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
    epookit=5,
    alijoukon_koko=100,
    validaatiodata=(X_testi, y_testi),
)
end = perf_counter()
print(f"Koulutus kesti {end - start:.5f} sekuntia.")

# Neuroverkon testaus
print("[4/4] Testataan neuroverkkoa...")
test_digits = X_testi[0:10]
y_ennuste = neuroverkko.ennusta(test_digits)
assert y_ennuste[0] == y_testi[0]  # 7
tarkkuus = neuroverkko.evaluoi(X_testi, y_testi)
print()
print(f"Neuroverkon tarkkuus testidatalla (n={len(y_testi)}): {tarkkuus*100:.2f}%")
# Tee kaavio tarkkuuden kehittymisestä
plt.plot(historia["tarkkuus"])
plt.plot(historia["validointitarkkuus"])
plt.title("Mallin tarkkuus")
plt.ylabel("tarkkuus")
plt.xlabel("epookki")
plt.legend(["koulutus", "testi"], loc="upper left")
plt.show()
# Tee kaavio hukka-arvon kehittymisestä
plt.plot(historia["hukka"])
plt.plot(historia["validointihukka"])
plt.title("Mallin hukka-arvo")
plt.ylabel("hukka")
plt.xlabel("epookki")
plt.legend(["koulutus", "testi"], loc="upper left")
plt.show()

print("Tallennetaan verkko...")
neuroverkko.tallenna("neuroverkko.pkl")
