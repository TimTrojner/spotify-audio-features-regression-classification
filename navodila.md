> t - validation mark by Tim
> z TODO: so označene opombe - za preverit!!!!

# Seminarska naloga – Celovita podatkovna analiza in optimizacija procesa

## Namen seminarske naloge

V seminarski nalogi boste na izbrani javni bazi podatkov izvedli celoten proces podatkovnega raziskovanja, modeliranja in optimizacije, kot ste ga spoznavali tekom vaj. Vaš cilj je razumeti delovanje izbranega procesa, identificirati pomembne dejavnike, razviti najboljši napovedni model ter pripraviti priporočila za optimizacijo procesa na podlagi rezultatov modeliranja.

Nalogo izvajata **dva študenta**:

* **En primer mora biti regresijski** (odvisna spremenljivka je **numerična**).
* **En primer mora biti klasifikacijski** (odvisna spremenljivka je **binarna**).

Od točke 1.4. dalje torej naredite vse ločeno za a) regresijski primer in b) klasifikacijski primer.

Nalogo oddate v roku v spletno učilnico

## 1. PRVI DEL: Priprava podatkov in osnovna analiza

**1.1 Izbor podatkov**

t• Izberite javno dostopno bazo podatkov.

t• Zastavite namen in cilje analize.

t• Opišite vse spremenljivke (ime, pomen, tip).

**1.2 Pregled in čiščenje podatkov**

t• Preverite manjkajoče vrednosti, podvojene zapise, ekstremne vrednosti.

t• Po potrebi izvedite nadomeščanje manjkajočih vrednosti in utemeljite postopek.

t• Jasno utemeljite vse sprejete odločitve. // TODO kake odločitve wtf???

**1.3 Deskriptivna statistika z grafi**

Za vsako spremenljivko:

t• pripravite osnovne statistike (mean ± SD ali mediana (Q1–Q3), min–max oz. n (%)), // TOOD: tu da majnakajo min, in n (%)

t• pripravite grafični prikaz (histogram, boxplot, barplot …),

t• interpretirajte opažanja (porazdelitve, odstopanja, posebnosti). // TODO: tega ni - majnka besedilo

t Izpišite tabelo z opisno statistiko za vse spremenljivke. // TODO: katero tabelo misli to - konkretno.

**1.4 Bivariatna analiza**

t• Preverite povezave med vsako spremenljivko in odvisno spremenljivko.

t• Uporabite ustrezne statistične teste (korelacije, χ², t-test, Mann–Whitney, ANOVA …).

t• Dodajte grafične prikaze.

t• Interpretirajte rezultate v kontekstu izbranega procesa.

t* Rezultate predstavite v skupni tabeli

**1.5 Izbor spremenljivk (Feature Selection)**

tUporabite eno ali več metod za izbiro pomembnih spremenljivk:

t• random forest importance,

t• LASSO / elastic net,

t• RFE,..

t• Na koncu določite, katere spremenljivke boste vključili v modele.

## 2. DRUGI DEL: Gradnja in ocenjevanje modelov

**2.1 Priprava podatkov za modeliranje**

t• Podatke razdelite na učno množico (80%) in testno množico (20%).

t• Če so podatki časovno odvisni → zadnjih 20 % uporabite kot testno množico.

**2.2 Gradnja modelov**

t• Zgradite vsaj 5 modelov s pomočjo različnih algoritmov.

t• Eden med temi 5 modeli mora biti linearni ali logistični regresijski model.

t• Pri vseh modelih nastavite hiperparametre (ne uporabljajte privzetih nastavitev).

t• Uporabite 10-fold cross-validacijo na učni množici za testiranje uspešnosti algoritmov (pazite na seme).

**2.3 Metrike napovedne uspešnosti**

* Regresija: R², RMSE, MAE, MAPE.
t* Klasifikacija: AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, F1.
t* Rezultate predstavite v tabeli (mean ± SD čez 10-fold CV) in grafično. // TODO: tu ne pozabit da more bit 10-fold CV boxplot!!

### Tabela: Primerjava modelov (validacijska množica)
t
|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | Tip | Parametri | Metrike (mean±SD) | AIC/BIC | Komentar | Izbor |
|  |  |  |  |  |  |  |

## 3. Izbor najboljših modelov in testiranje

t• Izberite 3 najboljše modele glede na validacijske rezultate.

t• Zgradite končne modele na celotni učni množici (80%).

t• Preizkusite jih na testni množici.

t• Izračunajte vse pomembne metrike in rezultate predstavite v tabeli in grafično.

### Tabela: Primerjava modelov (validacijska množica)
t
|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | Tip | Parametri | Metrike | AIC/BIC | Komentar | Izbor |
|  |  |  |  |  |  |  |

## 4. Interaktivna aplikacija (Simulacija in optimizacija)

**Aplikacija mora omogočati:**

**4.1 Izbiro modela**

t* Uporabnik izbere enega izmed 3 najboljših modelov.

**4.2 Napoved za posamezen ali skupinski vzorec**

t* vnos vrednosti spremenljivk (v polje, spreminjanje z drsnikom, izbor vrednosti…),
t* izračun napovedi Y,
t* grafični prikaz rezultatov.

**4.3 Simulacija sprememb**

Uporabnik lahko:

t* spreminja vrednosti neodvisnih spremenljivk,
t* opazuje **grafični prikaz vpliva spremembe posamezne neodvisne spremenljivke**,
t* vidi **novo napovedano vrednost Y**,
t* izvozi rezultate za izbrani vzorec v CSV.

To je ključni del optimizacije procesa. Aplikacija mora omogočati simulacijo optimizacije procesa.

## 5. Povzetek ugotovitev in priporočila za optimizacijo

Izberite najboljši model glede na rezultate na testni množici.

V zaključku opišite:

t• katere spremenljivke imajo največji vpliv,

t• ali je njihov vpliv pozitiven ali negativen,

t• katere spremembe so procesno smiselne,

t• kaj priporočate vodstvu.

Povzetek mora biti utemeljen in razumljiv vodstvu podjetja.

## 6. Testiranje globalnih sprememb (Simulacije)

t• Iz izbranih modelov definirati **globalne spremembe** (npr. zmanjšanje časa, povečanje kakovosti).

t• Pripravite novo različico testne množice z uvedenimi spremembami.

t• Izračunajte napovedi pred in po spremembah.

t• Primerjajte metrike in razložite učinke.

**Tabela: Pred–potem**
t
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | Metrika | Pred | Po | Razlika | Interpretacija |
|  |  |  |  |  |  |

## 7. Six Sigma analiza – PREJ in POTEM

t• Izračunajte DPMO in sigma nivo pred spremembami.

- regresijski: napake definirate preko tolerance,

t- klasifikacijski: napake = napačne klasifikacije.

t• Izračunajte DPMO in sigma nivo po spremembah.

t• Pripravite tabelo in analizo izboljšave procesa.

Dodajte kratko interpretacijo:

t* kako se je proces izboljšal,
t* ali je sprememba statistično in procesno pomembna,
t* ali upravičuje implementacijo v praksi.

Tabela: Primerjava sigma stopnje
t
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | DPMO PREJ | Sigma PREJ | DPMO POTEM | Sigma POTEM | Izboljšava |
|  |  |  |  |  |  |

## 8. Končni povzetek (Executive Summary)

Pripravite ½–1 stran povzetka:

t• kaj je bil problem,

t• kateri modeli so najboljši in zakaj,

t• katere spremenljivke je smiselno optimizirati,

t• kakšni so učinki sprememb,

t• kako se je sigma stopnja izboljšala,

t• priporočilo za implementacijo sprememb.

## 9. Oddaja naloge

Oddati morate:

1. **Python Notebook (.ipynb)** ali **R skripto (.Rmd)** – z vso kodo in rezultati,
2. **Končno poročilo (.docx ali .pdf)** z razlago in grafi,
3. **Delujočo aplikacijo** (npr. ShinyApp, Streamlit ali lokalno izvedbo z navodili za zagon).