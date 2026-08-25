# Detekcija napada iz zapisa vatrozida

Ovaj projekt sadrzi eksperimentalni kod za diplomski rad u kojem se zapisi vatrozida koriste za treniranje i provjeru modela za klasifikaciju mreznog prometa. Skup podataka sadrzi normalan promet i tri klase napada: `Port Scan`, `DDoS` i `Brute Force`.

## Struktura projekta

```text
eksperiment_model/
├── firewall_logs_unlabeled/   # nelabelirani train, test i eval CSV zapisi
├── firewall_logs_labeled/     # labelirani train, test i eval CSV zapisi
├── model_artifacts/           # spremljeni modeli, vokabular i scaler
├── labeliranje_logova.py      # pravilo-bazirano labeliranje zapisa
├── 1_text_cnn_trening.py      # treniranje TextCNN modela
├── 2_text_cnn_test.py         # testiranje TextCNN modela
├── 3_text_cnn_eval.py         # evaluacija TextCNN modela
├── 1_biLSTM_trening.py        # stara/alternativna biLSTM skripta
├── 1_mlp_train.py             # stara/alternativna MLP skripta
└── requirements.txt           # Python ovisnosti
```

## Podaci

Ulazni zapisi nalaze se u `firewall_logs_unlabeled`, a labelirani zapisi u `firewall_logs_labeled`. Podaci su podijeljeni na:

```text
train  - koristi se za treniranje modela
test   - koristi se za zavrsnu provjeru modela
eval   - koristi se za dodatnu evaluaciju modela
```

Labelirani CSV zapisi sadrze firewall atribute kao sto su zone, sucelja, IP adrese, portovi, protokol, akcija vatrozida, status sesije i zavrsna klasa prometa u stupcu `label`.

## Labeliranje

Skripta `labeliranje_logova.py` automatski labelira zapise prema pravilima nad vremenskim prozorom od jedne minute. Svi zapisi se pocetno oznacavaju kao `Normal`, a zatim se prema obrascima prometa preoznacavaju u `Port Scan`, `DDoS` ili `Brute Force`.

```powershell
python labeliranje_logova.py
```

## Treniranje TextCNN modela

Skripta `1_text_cnn_trening.py` ucitava labelirani train skup, pretvara zapise vatrozida u tekstualni oblik, tokenizira zapise, gradi vokabular, radi padding sekvenci i normalizira numericke znacajke. Model kombinira tekstualne znacajke iz TextCNN dijela i normalizirane numericke znacajke.

```powershell
python 1_text_cnn_trening.py
```

Nakon treniranja spremaju se:

```text
model_artifacts/textcnn_model_trained.pth
model_artifacts/vocab_trained.pth
model_artifacts/label_encoder_trained.pth
model_artifacts/numeric_scaler_trained.pth
```

## Instalacija

Preporuka je koristiti virtualno okruzenje u kratkoj Windows putanji kako bi se izbjegao problem s predugim putanjama kod instalacije PyTorcha.

```powershell
python -m venv C:\venv-tfpu
C:\venv-tfpu\Scripts\activate
pip install --upgrade pip
pip install numpy pandas scikit-learn torch
```

Zatim pokrenuti skripte iz foldera projekta:

```powershell
cd "C:\Users\Administrator\Desktop\TFPU\2. Godina\Diplomski\eksperiment_model"
python 1_text_cnn_trening.py
```

## Napomena

CSV datoteke, virtualna okruzenja, cache direktoriji i spremljeni modeli smatraju se generiranim ili lokalnim artefaktima te nisu namijenjeni za verzioniranje u Git repozitoriju.
