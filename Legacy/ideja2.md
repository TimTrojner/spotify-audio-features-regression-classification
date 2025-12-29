Super ideja – **“osebni Spotify Wrapped, ampak podatkovno-modelski”** je pravzaprav *idealno* ujemanje z navodili seminarske naloge. Spodaj ti bom pomagal **izoblikovati koncept**, ne kodo. Cilj je, da na koncu dobiš:

* **jasno zgodbo procesa** (kot da bi bil to realen poslovni problem),
* **en regresijski + en klasifikacijski problem**, oba smiselna,
* **modele, ki jih lahko interpretiraš, simuliraš in optimiziraš**,
* **artistično / vizualno predstavitev**, ki je “Spotify Wrapped feel”, a še vedno statistično resna.

Bom strukturiral odgovor v **5 sklopov**, ki se lepo preslikajo na navodila.

---

## 1️⃣ Kakšni podatki so sploh na voljo (in zakaj so idealni)

Spotify dataseti (Kaggle, Spotify API) imajo običajno:

### 🎵 Lastnosti skladb (neodvisne spremenljivke)

Numerične (idealno za linearno/logistično regresijo):

* danceability
* energy
* valence (pozitivnost)
* loudness
* tempo
* acousticness
* instrumentalness
* speechiness
* liveness
* duration_ms

- pogosto:

* popularity (0–100)
* release_year
* genre / artist (lahko odstraniš ali agregiraš)

➡️ **To je sanjski primer “procesnih vhodov”**, kot jih želi Six Sigma / optimizacija.

---

## 2️⃣ “Spotify Wrapped” kot proces (ključno za poročilo)

Namesto “analiza glasbe” si zastaviš **procesno zgodbo**, npr.:

> 🎯 *Proces: Kako zvočne lastnosti skladbe vplivajo na njen uspeh pri poslušalcih.*

Ali še bolje:

> 🎯 *Kako lahko producent optimizira lastnosti skladbe, da poveča verjetnost, da bo postala hit.*

To ti omogoča:

* **simulacijo** (kaj če povečamo energy?),
* **optimizacijo** (kakšna kombinacija lastnosti je najboljša),
* **Six Sigma** interpretacijo (napake = neuspešne skladbe).

---

## 3️⃣ Regresijski primer (numerični Y) – “Spotify Wrapped, ampak napoved”

### 🎯 Regresijski cilj (Y)

Najbolj naravna izbira:

* **popularity (0–100)**

Zakaj je to odlično:

* numerično,
* kontinuirano,
* interpretabilno,
* omogoča simulacije (“če spremenim X, koliko se spremeni Y”).

---

### 📈 Regresijski modeli (ki imajo smisel)

* Linearna regresija → **razlaga vplivov**
* Ridge / Lasso → **feature selection**
* Random Forest / GB → **benchmark**
* (SVR, KNN – če rabiš do 5 modelov)

**Ampak:**
👉 linearna regresija je *glavni interpretativni model*, ne nujno najboljši.

---

### 🔧 Transformacije (zelo pomembno za nalogo)

Tu lahko pokažeš *modelarsko zrelost*:

* log(speechiness), log(instrumentalness)
* interakcije:

    * energy × loudness
    * danceability × tempo
* kompozitne metrike:

    * `mood_score = 0.4*valence + 0.3*energy + 0.3*danceability`

➡️ To vse **direktno povežeš z linearno regresijo**:

> “Koeficient mood_score = +12 pomeni, da bolj ‘pozitivne’ skladbe dosegajo višjo popularnost.”

---

### 🎛️ Simulacija (točka 4 in 6 v navodilih)

To je zlato:

* uporabnik v aplikaciji:

    * spremeni danceability (slider),
    * vidi novo napovedano popularnost,
    * graf “pred–potem”.

To je **učbeniški primer optimizacije procesa**.

---

## 4️⃣ Klasifikacijski primer – “Ali bo skladba HIT?”

Tu narediš jasno binarno odločitev.

### 🎯 Klasifikacijski cilj (Y)

Definiraš:

```text
Is_Hit = 1, če popularity ≥ 70 (ali top 20 %)
Is_Hit = 0 sicer
```

Zakaj je to idealno:

* binarna odvisna spremenljivka ✔️
* poslovno smiselna ✔️
* omogoča logistično regresijo ✔️

---

### 🔮 Klasifikacijski modeli

* Logistična regresija (**obvezno, interpretabilna**)
* Random Forest
* Gradient Boosting
* SVM
* KNN

---

### 📊 Zakaj je logistična regresija tu popolna

Ker lahko rečeš:

* koeficient +1.2 pri danceability →

    * **višja verjetnost hita**
* odds ratio:

    * “10 % več energy → 25 % večja verjetnost hita”

To je *točno* to, kar profesorji radi vidijo.

---

### 🔁 Simulacija verjetnosti

V aplikaciji:

* uporabnik spreminja lastnosti skladbe,
* vidi:

    * **P(hit)**,
    * kdaj preseže npr. 0.8.

To je **optimizacija odločitve**, ne samo napoved.

---

## 5️⃣ “Spotify Wrapped” – artistična in vizualna plast 🎨

To ni samo lep dodatek – to ti pomaga pri **interpretaciji**.

### 🎧 Za posameznega uporabnika / umetnika

Če imaš user-level podatke:

* povprečni mood_score uporabnika,
* “tvoj glasbeni DNK” (radar chart),
* primerjava z globalnim povprečjem.

Če nimaš user-level podatkov:

* “idealna skladba po modelu”
* “tvoja pesem vs. hit povprečje”

---

### 📊 Vizualizacije, ki imajo smisel

* radar chart (audio features),
* “impact plot”:

    * koliko % popularity prispeva posamezna spremenljivka,
* decision boundary (logistična reg.).

To ni samo estetika – to je **model explanation**.

---

## 6️⃣ Kako se to popolnoma ujema z navodili

| Zahteva           | Tvoja ideja                  |
| ----------------- | ---------------------------- |
| Regresija         | popularity                   |
| Klasifikacija     | Is_Hit                       |
| Feature selection | Lasso, RF importance         |
| Simulacija        | spreminjanje audio lastnosti |
| Optimizacija      | max popularity / max P(hit)  |
| Six Sigma         | napake = neuspešne skladbe   |
| Aplikacija        | “mini Spotify Wrapped”       |

---

## 7️⃣ Če želiš, naslednji koraki

Lahko:

1. skupaj **zapiševa točno definicijo procesa** (kot v poročilu),
2. izbereva **konkreten dataset** (enega, ki pokrije vse zahteve),
3. ti pomagam napisati **Executive Summary vnaprej** (kar je redko, ampak močno).

Če želiš, mi samo povej:
👉 *Ali želiš bolj “producentsko” (kako narediti hit) ali bolj “uporabniško” (kakšen je moj glasbeni okus) zgodbo?*
