# Viikkoraportti 5

| Mitä tein                        | Aika   |
| -------------------------------- | ------ |
| numpy.einsum-testaus             | 2t     |
| Neuroverkoon optimointi          | 1t     |
| Testausdokumentin kirjoittaminen | 1t     |
| Ristiinvalidoinnin toteutus      | 1t     |
| **Yhteensä**                     | **5t** |

_Mitä olen tehnyt tällä viikolla?_

Löysin mielenkiintoista tietoa numpyn np.einsum-metodista, jolla matriisilaskut saattavat olla jopa 10x nopeampia. Valitettavasti testauksen jälkeen tässä sovelluksessa np.einsum on hitaampi. Se on tarkoitettu lähinnä tensorilaskuihin, jossa matriisit ovat todella isoja ja ulottuvuudet ovat yli 2.

Tein ohjaajan vinkin perusteella ristiinvalidoinnin. Tulos on samaa luokkaa, eli tarkkuus pysyy 98% paikkeilla sekä Tensorflowlla ja omalla toteutuksella.

_Miten ohjelma on edistynyt?_

Sain optimoitua koulutusalgoritmia hieman korvaamalla np.dot-operaatiot np.matmul-operaatioilla.

_Mitä opin tällä viikolla / tänään?_

np.einsumin olemassaolon. Pidän tämän mielessä, kun tulevaisuudessa tulee enemmän deep learning-asioita eteen.

_Mikä jäi epäselväksi tai tuottanut vaikeuksia?_

_Mitä teen seuraavaksi?_

Odotan palautetta vertaisarvoinnista ja teen tarvittavat muutokset sen perusteella.
