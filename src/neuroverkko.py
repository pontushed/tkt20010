from typing import Dict, Iterator, List, Tuple, TypedDict
import numpy as np
from numpy.typing import NDArray

from tqdm import trange  # Näyttää hienon edistysmittarin

np.random.seed(99)


class Historia(TypedDict):
    """malli.sovita() palauttaa tämän olion"""

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

    def __init__(self, yksikot_data: NDArray, yksikot_ulos: NDArray) -> None:
        self.rho = 0.9
        self.epsilon = 1e-7
        self.keraajamatriisi_painot = np.zeros(shape=(yksikot_data, yksikot_ulos))
        self.keraajamatriisi_vakiot = np.zeros(yksikot_ulos)
        self.oppimisvauhti = 0.001

        # Painokertoimien alustus
        # https://wandb.ai/sayakpaul/weight-initialization-tb/reports/Effects-of-Weight-Initialization-on-Neural-verkkos--Vmlldzo2ODY0NA
        # Edellisen mukaan paras arvo on asettaa ne jakauman mukaan,
        # jossa arvot ovat välillä -y,y ja y = 1/sqrt(yksikot_data)
        _y = 1 / np.sqrt(yksikot_data)
        self.painot = np.random.uniform(low=-_y, high=_y, size=(yksikot_data, yksikot_ulos))
        self.vakiot = np.zeros(yksikot_ulos)

    def eteenpain(self, data: NDArray) -> NDArray:
        """Laske kerroksen tulos

        Args:
            data (NDArray): Kerrokselle syötettävä data

        Returns:
            NDArray: Kerroksen tulos Wx + b
        """
        return np.matmul(data, self.painot) + self.vakiot

    def taaksepain(self, data: NDArray, gradientti_ulos: NDArray) -> NDArray:
        # Laske gradientit edeltävälle kerrokselle
        grad_data = np.dot(gradientti_ulos, np.transpose(self.painot))

        # Laske painokertoimien gradientit
        grad_painot = np.transpose(np.dot(np.transpose(gradientti_ulos), data))
        grad_vakiot = np.sum(gradientti_ulos, axis=0)

        # Päivitä RMSprop-optimoijan kerääjämatriisit
        self.keraajamatriisi_painot = 0.9 * self.keraajamatriisi_painot + 0.1 * grad_painot**2
        self.keraajamatriisi_vakiot = 0.9 * self.keraajamatriisi_vakiot + 0.1 * grad_vakiot**2

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

    def eteenpain(self, data: NDArray) -> NDArray:

        """Suorita ReLU-funktio alkioittain matriisille
        [alijoukko, muuttujat]"""
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


def softmax_ristientropia(y_true: NDArray, y_ulos: NDArray) -> NDArray:
    """Laske ristientropia softmax-funktioon"""
    logits_for_answers = y_ulos[np.arange(len(y_ulos)), y_true]
    ristientropia = -logits_for_answers + np.log(np.sum(np.exp(y_ulos), axis=-1))
    return ristientropia


def grad_softmax_ristientropia(y_true: NDArray, y_ulos: NDArray) -> NDArray:
    """Laske gradientti softmax-ristientropia-funktioon"""
    ones_for_answers = np.zeros_like(y_ulos)
    ones_for_answers[np.arange(len(y_ulos)), y_true] = 1

    softmax = np.exp(y_ulos) / np.exp(y_ulos).sum(axis=-1, keepdims=True)

    return (-ones_for_answers + softmax) / y_ulos.shape[0]


class Neuroverkko:
    """Neuroverkko RMSProp-optimoijalla

    Attributes:
        verkko(list): Neuroverkon arkkitehtuuri
    """

    def __init__(self, verkko: list[Kerros]) -> None:
        """Konstruktori

        Args:
            verkko (list[Kerros]): Neuroverkon arkkitehtuuri
        """
        self.verkko = verkko

    def eteenpain(self, X: NDArray) -> List[NDArray]:
        """
        Laske kaikkien kerrosten aktivoinnit järjestyksessä.
        Palauttaa listan kerroksien aktivoituja dataa.
        """
        aktivoinnit = []
        for i in range(len(self.verkko)):
            aktivoinnit.append(self.verkko[i].eteenpain(X))
            X = self.verkko[i].eteenpain(X)

        assert len(aktivoinnit) == len(self.verkko)
        return aktivoinnit

    def ennusta(self, X: NDArray) -> NDArray:
        """
        Laske neuroverkon päättelemät arvot datalle.

        Args:
            X (NDArray): Data, jonka päättelemät arvot lasketaan.
        """
        logits = self.eteenpain(X)[-1]
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

        for i in range(1, len(self.verkko)):
            hukka_grad = self.verkko[len(self.verkko) - i].taaksepain(
                kerrosten_aktivoinnit[len(self.verkko) - i - 1], hukka_grad
            )

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

            validaatiodata (tuple): Tuple (X_val, y_val), jossa on validaatiodataa.

        Returns:
            historia (dict): Koulutushistoria.
                  Tämä on muotoa:
                  {
                    'hukka': [],
                    'accuracy': [],
                    'validointihukka': [],
                    'validointitarkkuusuracy': []
                }
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
            hukka_mini = np.min(hukka_arvot)
            historia["hukka"].append(hukka_keskiarvo)
            historia["tarkkuus"].append(np.mean(self.ennusta(X_koulutus) == y_koulutus))
            if validaatiodata is not None:
                validointitarkkuus, validointihukka = self.evaluoi(
                    validaatiodata[0], validaatiodata[1], return_dict=True
                ).values()
                historia["validointitarkkuus"].append(validointitarkkuus)
                historia["validointihukka"].append(validointihukka)
            print(
                f"Minimi hukka: {hukka_mini:.4f}, Maksimi hukka: {hukka_maksimi:.4f}, Keskimääräinen hukka: {hukka_keskiarvo:.4f}, Tarkkuus: {historia['tarkkuus'][-1]:.4f}"
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
