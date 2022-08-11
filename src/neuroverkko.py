from typing import Dict, Iterator, List, Tuple, TypedDict
import numpy as np
from numpy.typing import NDArray
import pickle

from tqdm import trange  # Näyttää hienon edistysmittarin

np.random.seed(99)


class Historia(TypedDict):
    """Neuroverkko.sovita() palauttaa tämän dict-olion"""

    hukka: List[float]
    tarkkuus: List[float]
    validointihukka: List[float]
    validointitarkkuus: List[float]


class Kerros:
    """Abstrakti luokka neuroverkon kerrokselle"""

    def __init__(self):
        pass

    def eteenpain(self, data):
        """Suorittaa matriisilaskun Wx+b kerroksen läpi"""
        pass

    def taaksepain(self, data, gradientti_ulos):
        """Suorittaa gradienttien päivityksen ja palauttaa gradientit seuraavalle kerrokselle"""
        pass


class Tihea(Kerros):  # Vastaa TensorFlow/Keras-kirjastosta kerrosta "Dense"
    """Tiheä kerros RMSProp-optimoijalla"""

    def __init__(self, yksikot_sisaan: NDArray, yksikot_ulos: NDArray) -> None:
        # RMSProp-optimoijan parametrit
        self.rho = 0.9  # Painotetun keskiarvon vaimennuskerroin
        self.epsilon = 1e-7  # Pieni arvo jolla varmistetaan, ettei jaeta nollalla
        self.oppimisvauhti = 0.001
        # RMSProp-kerääjämatriisit painoille ja vakioille
        self.keraajamatriisi_painot = np.zeros(shape=(yksikot_sisaan, yksikot_ulos))
        self.keraajamatriisi_vakiot = np.zeros(yksikot_ulos)

        # Painokertoimien alustus
        # https://wandb.ai/sayakpaul/weight-initialization-tb/reports/Effects-of-Weight-Initialization-on-Neural-verkkos--Vmlldzo2ODY0NA
        # Edellisen mukaan paras arvo on asettaa ne jakauman mukaan,
        # jossa arvot ovat välillä -y,y ja y = 1/sqrt(yksikot_sisaan)
        _y = 1 / np.sqrt(yksikot_sisaan)
        self.painot = np.random.uniform(low=-_y, high=_y, size=(yksikot_sisaan, yksikot_ulos))
        self.vakiot = np.zeros(yksikot_ulos)

    def __str__(self) -> str:
        return f"Tiheä kerros ({self.keraajamatriisi_painot.shape[1]})"

    def eteenpain(self, data: NDArray) -> NDArray:
        """Laske kerroksen tulos

        Parametrit
        ==========
        data : NDArray
            Kerrokselle syötettävä data [n,x]

        Returns:
            NDArray: Kerroksen tulos [n, k], jossa k = Wx + b
        """
        return np.matmul(data, self.painot) + self.vakiot

    def taaksepain(self, data: NDArray, gradientti_ulos: NDArray) -> NDArray:
        """Vastavirta-algoritmin osio kerrokselle


        Args:
            data (NDArray): Kerrokselle syötettävä data [n,x]
            gradientti_ulos (NDArray): Kerrokselle syötettävä gradientti [n,k]

        Returns:
            NDArray: Kerroksen gradientti [n,x], jota käytetään syötteenä edeltävälle kerrokselle
        """
        # Laske gradientit edeltävälle kerrokselle
        grad_data = np.dot(gradientti_ulos, np.transpose(self.painot))

        # Laske painokertoimien gradientit
        grad_painot = np.transpose(np.dot(np.transpose(gradientti_ulos), data))
        grad_vakiot = np.sum(gradientti_ulos, axis=0)

        # Päivitä RMSprop-optimoijan kerääjämatriisit
        self.keraajamatriisi_painot = self.rho * self.keraajamatriisi_painot + (1 - self.rho) * grad_painot**2.0
        self.keraajamatriisi_vakiot = self.rho * self.keraajamatriisi_vakiot + (1 - self.rho) * grad_vakiot**2.0

        # Päivitä tämän kerroksen painot ja vakiot
        self.painot = self.painot - (
            self.oppimisvauhti * grad_painot / (np.sqrt(self.keraajamatriisi_painot) + self.epsilon)
        )
        self.vakiot = self.vakiot - (
            self.oppimisvauhti * grad_vakiot / (np.sqrt(self.keraajamatriisi_vakiot) + self.epsilon)
        )

        # Palauta gradientit edeltävälle kerrokselle
        return grad_data


class ReLU(Kerros):
    """ReLU-kerros suorittaa ReLU-aktivointifunktion syötteelle."""

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return "ReLU-kerros"

    def eteenpain(self, data: NDArray) -> NDArray:
        """Suorita ReLU-funktio alkioittain syötematriisille"""
        return np.maximum(0, data)

    def taaksepain(self, data: NDArray, gradientti_ulos: NDArray) -> NDArray:
        """Laske gradientti ReLU-kerrokselle

        Args:
            data (NDArray): matriisi muotoa [alijoukko, muuttujat]
            gradientti_ulos (NDArray): matriisi muotoa [alijoukko, muuttujat]

        Returns:
            NDArray: gradientit matriisille [alijoukko, muuttujat], johon on laskettu ReLU-funktion gradientti
        """
        relu_grad = data > 0
        return gradientti_ulos * relu_grad


def softmax(y):
    """Softmax-funktio"""
    e_y = np.exp(y - np.max(y, axis=-1, keepdims=True))
    return e_y / e_y.sum(axis=-1, keepdims=True)


def softmax_ristientropia(y_true: NDArray, y_ulos: NDArray) -> NDArray:
    """Laske ristientropia softmax-funktioon"""
    logits = y_ulos[np.arange(len(y_ulos)), y_true]
    ristientropia = -logits + np.log(np.sum(np.exp(y_ulos), axis=-1))
    return ristientropia


def grad_softmax_ristientropia(y_true: NDArray, y_ulos: NDArray) -> NDArray:
    """Laske gradientti softmax-ristientropia-funktioon"""

    # Tee apumatriisi, jossa oikean vastauksen kohdalla on 1, muuten 0
    vastausmatriisi = np.zeros_like(y_ulos)  # (n,10)
    vastausmatriisi[np.arange(len(y_ulos)), y_true] = 1

    # Laske gradientti softmax-ristientropia-funktioon
    return (-vastausmatriisi + softmax(y_ulos)) / y_ulos.shape[0]


class Neuroverkko:
    """Neuroverkko RMSProp-optimoijalla

    Attributes:
        verkko(list): Neuroverkon arkkitehtuuri
    """

    def __init__(self, verkko: list[Kerros] | None) -> None:
        """Konstruktori

        Args:
            verkko (list[Kerros]): Neuroverkon arkkitehtuuri
        """
        self.verkko = verkko

    def __str__(self):
        return " -> ".join([str(kerros) for kerros in self.verkko])

    def lataa(self, tiedosto: str) -> None:
        """Lataa kerrokset tiedostosta"""
        with open(tiedosto, "rb") as f:
            self.verkko = pickle.load(f)

    def tallenna(self, tiedosto: str) -> None:
        """Tallenna neuroverkko tiedostoon"""
        with open(tiedosto, "wb") as f:
            pickle.dump(self.verkko, f)

    def eteenpain(self, X: NDArray) -> List[NDArray]:
        """
        Laske kaikkien kerrosten aktivoinnit järjestyksessä.
        Palauttaa listan kerroksien aktivoituja dataa.
        """
        aktivoinnit_lista = []
        aktivoinnit = X
        for i in range(len(self.verkko)):
            aktivoinnit = self.verkko[i].eteenpain(aktivoinnit)
            aktivoinnit_lista.append(aktivoinnit)

        assert len(aktivoinnit_lista) == len(self.verkko)
        return aktivoinnit_lista

    def ennusta(self, X: NDArray, todennakoisyydet: bool = False) -> NDArray:
        """
        Laske neuroverkon päättelemät arvot datalle.

        Args:
            X (NDArray): Data, jonka päättelemät arvot lasketaan.
        """
        logits = self.eteenpain(X)[-1]
        if todennakoisyydet:
            return softmax(logits)
        return logits.argmax(axis=-1)

    def kouluta(self, X: NDArray, y: NDArray) -> float:
        """
        Kouluttaa verkkoa alijoukolla X ja vastauksilla y.
        """
        assert len(X) == len(y)

        # Laske kerrosten aktivoinnit
        kerrosten_aktivoinnit = self.eteenpain(X)
        y_ulos = kerrosten_aktivoinnit[-1]

        # Compute the hukka and the initial gradient
        hukka = softmax_ristientropia(y, y_ulos)
        hukka_grad = grad_softmax_ristientropia(y, y_ulos)

        for i in range(len(self.verkko), 1, -1):
            hukka_grad = self.verkko[i - 1].taaksepain(kerrosten_aktivoinnit[i - 2], hukka_grad)
        self.verkko[0].taaksepain(X, hukka_grad)

        return np.mean(hukka)

    def sovita(
        self,
        X_koulutus: NDArray,
        y_koulutus: NDArray,
        epookit=10,
        alijoukon_koko=32,
        validaatiodata=None,
    ) -> Historia:
        """
        Kouluta neuroverkko.

        Args:
            X_koulutus (NDArray): Data, jonka päättelemät arvot lasketaan.
            y_koulutus (NDArray): X_koulutus:n vastaukset.
            epookit (int): Kuinka monta kertaa koulutus suoritetaan.
            alijoukon_koko (int): Kuinka monta dataa koulutetaan kerralla.

            validaatiodata (tuple) [optional]: Tuple (X_val, y_val), jossa on validaatiodataa. Tätä käytetään koulutusprosessin seurantaan.

        Returns:
            historia (Historia): Koulutushistoria.
        """

        def iteroi_alijoukot(
            data: NDArray, vastaukset: NDArray, alijoukon_koko: int, sekoita=False
        ) -> Iterator[Tuple[NDArray, NDArray]]:
            """Iteroija joka tuottaa alijoukot

            Args:
                data (NDArray): data
                vastaukset (NDArray): vastaukset
                alijoukon_koko (int): alijoukon koko
                sekoita (bool, optional): Tehdäänkö satunnaisotanta. Defaults to False.

            Yields:
                Iterator[Tuple[NDArray, NDArray]]: Alijoukko (Data, vastaukset)
            """
            assert len(data) == len(vastaukset)
            if sekoita:
                indeksit = np.random.permutation(len(data))
            for start_idx in trange(0, len(data) - alijoukon_koko + 1, alijoukon_koko):
                if sekoita:
                    otos = indeksit[start_idx : start_idx + alijoukon_koko]
                else:
                    otos = slice(start_idx, start_idx + alijoukon_koko)
                yield data[otos], vastaukset[otos]

        historia = {
            "hukka": [],
            "tarkkuus": [],
            "validointihukka": [],
            "validointitarkkuus": [],
        }

        for epookki in range(epookit):
            print(f"Epookki {epookki+1}/{epookit}")
            hukka_arvot = []
            for x_alijoukko, y_alijoukko in iteroi_alijoukot(
                X_koulutus, y_koulutus, alijoukon_koko=alijoukon_koko, sekoita=True
            ):
                hukka = self.kouluta(x_alijoukko, y_alijoukko)
                hukka_arvot.append(hukka)

            hukka_keskiarvo = np.mean(hukka_arvot)
            hukka_maksimi = np.max(hukka_arvot)
            hukka_minimi = np.min(hukka_arvot)
            historia["hukka"].append(hukka_keskiarvo)
            historia["tarkkuus"].append(np.mean(self.ennusta(X_koulutus) == y_koulutus))
            if validaatiodata is not None:
                validointitarkkuus, validointihukka = self.evaluoi(
                    validaatiodata[0], validaatiodata[1], return_dict=True
                ).values()
                historia["validointitarkkuus"].append(validointitarkkuus)
                historia["validointihukka"].append(validointihukka)
            print(
                f"Minimi hukka: {hukka_minimi:.4f}, Maksimi hukka: {hukka_maksimi:.4f}, Keskimääräinen hukka: {hukka_keskiarvo:.4f}, Tarkkuus: {historia['tarkkuus'][-1]:.4f}"
            )
        return historia

    def evaluoi(self, X, y, return_dict=False):
        """
        Evaluoi mallia käyttäen annettua dataa.
        """
        if return_dict:
            y_ulos = self.eteenpain(X)[-1]
            hukka = softmax_ristientropia(y, y_ulos)
            return {
                "tarkkuus": np.mean(y_ulos.argmax(axis=-1) == y),
                "hukka": np.mean(hukka),
            }
        else:
            return np.mean(self.ennusta(X) == y)
