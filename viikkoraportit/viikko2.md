# Viikkoraportti 2

| Mitä tein                         | Aika    |
| --------------------------------- | ------- |
| Testauksen valmistelu             | 2t      |
| Aiheeseen tutustuminen            | 10t     |
| Toteutusdokumentin kirjoittaminen | 3t      |
| Algoritmin toteutusta             | 8t      |
| Algoritmin optimointia            | 4t      |
| **Yhteensä**                      | **27t** |

_Mitä olen tehnyt tällä viikolla?_

Olen lukenut kirjallisuutta neuroverkoista ja tutkinut keras-kirjaston toteutusta. Olen kokeillut eri vaihtoehtoja itse koodaamalla. Koska neuroverkossa tehdään paljon matriisikertolaskuja, niin toiminnan hahmottaminen on melko vaikeaa. Olen lukenut myös paljon numpy-kirjaston käytön nopeusoptimoinnista. Pääasia on, että yrittää vektoroida laskuoperaatiot, jolloin nopeus voi olla moninkertainen silmukoilla toteutettuihin operaatioihin nähden. Suurilla matriiseilla kannattaa lisäksi suosia numpyn `np.matmul`-operaatiota `np.dot`-operaation sijasta.

_Miten ohjelma on edistynyt?_

Ohjelma on edistynyt ihan hyvin. Haluan kokeilla osa-alueita esim. Jupyter Notebook-ympäristössä ennenkuin kirjoitan ne ns. puhtaaksi koodiksi, jotta voin varmistua toiminnasta. Olen jo päässyt vaiheeseen, jossa neuroverkko oppii, mutta ei riittävän hyvin. Riittävän hyvin oppiva verkko ei sitten taas ole nopea, joten joudun vielä varmistamaan, että mistä kiikastaa. Kun referenssitoteutus TensorFlow/Keras-kirjastolla pääsee viiden epookin aikana tarkkuuteen 98,xx%, niin omat toteutukseni pääsevät 98,xx% ja 94,xx%. Tavoite on päästä 0,5 prosenttiyksikön päähän referenssitoteutuksesta, ja sitten kouluttaa neuroverkko 100 epookilla.

_Mitä opin tällä viikolla / tänään?_

Neuroverkkojen kerrosten painoarvojen alustamisen tavalla on merkitystä! Erään artikkelin mukaan painoarvot kannattaa alustaa satunnaisluvuilla, jotka ovat väliltä [$-y$, $y$], jossa $y=\frac{1}{\sqrt{n}}$ ja $n$ on kerroksen sisääntulon vektorin pituus.

Opin myös, että Github tukee latexia markdownissa!

Opin RMSprop-optimoijan periaatteen ja suositeltavat arvot oppimisvauhdille sekä vaimennuskertoimelle: $\eta=0,001$ ja $\rho=0,9$.

Sparse Cross Entropy-laskua voi nopeuttaa tuplasti jättämällä jokaisen todennäköisyyden laskennan pois ja valita vain oikean luokan todennäköisyyden.

_Mikä jäi epäselväksi tai tuottanut vaikeuksia?_

Neuroverkon koulutusprosessin yksityiskohtien saaminen kuntoon ja "pullonkaulojen" etsiminen. Pitää tehdä pala kerrallaan ja käyttää mm. pythonin `timeit`-moduulia.

_Mitä teen seuraavaksi?_

Jatkan neuroverkon koulutusalgoritmin parissa. Kun koulutus toimii riittävän hyvin, teen keinon jolla koulutetun mallin voi tallentaa ja ladata erikseen päättelytehtävää varten.
