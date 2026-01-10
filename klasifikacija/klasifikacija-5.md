## 5. Povzetek ugotovitev in priporočila za optimizacijo

Na podlagi celovite analize in testiranja več različnih algoritmov strojnega učenja smo kot najustreznejši model za napovedovanje uspešnosti glasbenih del izbrali **metodo stopnjevanja gradienta (XGBoost)**. Model je na testni množici dosegel najvišjo natančnost (**81,16 %**) in se izkazal za najzanesljivejšega pri ločevanju med bodočimi uspešnicami in manj uspešnimi skladbami (AUC = 0,8969).

V nadaljevanju predstavljamo ključne ugotovitve in konkretne korake za optimizacijo glasbene produkcije.

### 5.1 Analiza vplivnih dejavnikov: Kaj ustvari uspešnico?

Naša analiza je razkrila, da glasbeni trg ne deluje po načelu "več je bolje", temveč zahteva iskanje **optimalnega ravnovesja**. Identificirali smo naslednje zakonitosti:

* **Čustvena pozitivnost (Valence):** To je statistično najmočnejši napovednik uspeha. Obstaja neposredna in močna povezava: bolj ko skladba zveni optimistično, veselo in vedro, večja je verjetnost, da postane plesna uspešnica. Melanholija v tem žanru zmanjšuje možnost za uspeh.
* **Energija in Glasnost (Energy & Loudness):** Pri teh dveh parametrih smo odkrili točko nasičenja.
* Največjo verjetnost za uspeh imajo skladbe z energijo okoli vrednosti **0,7** in glasnostjo okoli **-4 dB**.
* Preseganje teh vrednosti (torej preveč agresivna ali preglasna produkcija) deluje kontraproduktivno in znižuje verjetnost, da bo skladba poslušalcem všeč.


* **Akustičnost (Acousticness):** Trg zavrača ekstreme. Najbolje se odrežejo skladbe, ki niso niti popolnoma sintetične niti povsem akustične, temveč kombinirajo oba svetova (vrednosti med **0,3 in 0,65**).
* **Vokalna izraznost (Speechiness):** Analiza kaže na obstoj spodnjega praga (**0,2**). Skladbe morajo vsebovati dovolj vokalnih elementov ali ritmičnega govora, da pritegnejo pozornost. Čisti instrumentali imajo drastično manjše možnosti za uspeh.
* **Občutek žive izvedbe (Liveness):** Tu velja pravilo: "manj je več". Poslušalci plesne glasbe preferirajo studijsko popolnost. Z večanjem občutka, da gre za posnetek v živo (šum publike, odmev dvorane), verjetnost za uspeh strmo pada.
* **Hitrost ritma (Tempo):** Presenetljivo se je izkazalo, da hitrost skladbe nima statistično značilnega vpliva na uspeh. To pomeni, da je uspešnica lahko tako počasnejša kot hitra pesem, če so ostali elementi (pozitivnost, energija) pravilno nastavljeni.

### 5.2 Predlogi za optimizacijo procesa produkcije

Glede na rezultate modela predlagamo uvedbo naslednjih standardov v ustvarjalnem procesu:

1. **Ciljno uravnavanje energije:** V fazi aranžiranja in mešanja zvoka ne smemo težiti k maksimalni možni energiji. Cilj producentov mora biti ohranjanje dinamike na ravni 0,7, kar omogoča dolgotrajnejše poslušanje brez utrujenosti poslušalca.
2. **Vokalna prioriteta:** Vsaka instrumentalna podlaga mora biti nadgrajena z vokalnim vložkom ali izrazitim ritmičnim govorom, ki presega prag 0,2.
3. **Hibridna produkcija:** V elektronske skladbe je treba načrtno vključevati posnete akustične inštrumente (npr. kitara, klavir, tolkala), da dosežemo optimalno stopnjo akustičnosti.

### 5.3 Strateška priporočila vodstvu

Za povečanje tržnega deleža in optimizacijo vlaganj predlagamo vodstvu podjetja tri ključne ukrepe:

* **Uvedba sistema za predhodno preverjanje (Pre-screening):**
Preden se potrdijo sredstva za drago promocijo in videospote, naj se demo posnetki analizirajo z našo razvito aplikacijo. Skladbe, ki jim model napove manj kot 50-odstotno verjetnost uspeha, naj se vrnejo v studio na dodelavo (povečanje pozitivnosti, prilagoditev energije).
* **Preusmeritev vlaganj v vsebino:**
Podatki kažejo, da so posnetki v živo (koncertne verzije) tržno manj zanimivi za širšo publiko. Predlagamo, da se proračun, namenjen snemanju koncertov, preusmeri v studijsko produkcijo, ki zagotavlja višjo kakovost zvoka (nizka stopnja parametra "Liveness").
* **Ustvarjalna svoboda pri ritmu:**
Umetnikom in producentom se lahko dovoli popolna svoboda pri izbiri tempa (BPM), saj ta ne vpliva na uspeh. Namesto ukvarjanja s hitrostjo naj se fokusirajo na **čustveni naboj skladbe (pozitivnost)**, saj je to dejavnik, ki dejansko prodaja glasbo.
