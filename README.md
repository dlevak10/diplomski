# Detekcija napada iz zapisa vatrozida

Ovaj projekt sadrzi eksperimentalni kod za diplomski rad u kojem se zapisi vatrozida koriste za treniranje i provjeru modela za klasifikaciju mreznog prometa. Skup podataka sadrzi normalan promet i tri klase napada: `Port Scan`, `DDoS` i `Brute Force`.

## Struktura projekta

```text
eksperiment_model/
├── firewall_logs_unlabeled/   # nelabelirani train, eval i test CSV zapisi
├── firewall_logs_labeled/     # labelirani train, eval i test CSV zapisi
├── model_artifacts/           # spremljeni modeli, vokabular i scaler
├── eval_results/              # rezultati podesavanja i validacije po modelu
├── test_results/              # rezultati zavrsnog testiranja po modelu
├── scripts/
│   └── labeliranje_logova.py  # pravilo-bazirano labeliranje zapisa
├── 1_text_cnn_train.py        # treniranje TextCNN modela
├── 2_text_cnn_eval.py         # podesavanje i validacija TextCNN modela
├── 3_text_cnn_test.py         # zavrsno testiranje TextCNN modela
├── 1_biLSTM_train.py          # treniranje BiLSTM modela
├── 2_biLSTM_eval.py           # podesavanje i validacija BiLSTM modela
├── 3_biLSTM_test.py           # zavrsno testiranje BiLSTM modela
├── 1_mlp_train.py             # treniranje MLP modela
├── 2_mlp_eval.py              # podesavanje i validacija MLP modela
├── 3_mlp_test.py              # zavrsno testiranje MLP modela
└── requirements.txt           # Python ovisnosti
```

## Podaci

Ulazni zapisi nalaze se u `firewall_logs_unlabeled`, a labelirani zapisi u `firewall_logs_labeled`. Podaci su podijeljeni na:

```text
train - koristi se za treniranje modela
eval  - koristi se za podesavanje i validaciju modela
test  - koristi se za zavrsnu evaluaciju modela
```

Labelirani CSV zapisi sadrze firewall atribute kao sto su zone, sucelja, IP adrese, portovi, protokol, akcija vatrozida, status sesije i zavrsna klasa prometa u stupcu `label`.

## Labeliranje

Skripta `scripts/labeliranje_logova.py` automatski labelira zapise prema pravilima nad vremenskim prozorom od jedne minute. Svi zapisi se pocetno oznacavaju kao `Normal`, a zatim se prema obrascima prometa preoznacavaju u `Port Scan`, `DDoS` ili `Brute Force`.

```powershell
python scripts/labeliranje_logova.py
```

## Treniranje TextCNN modela

Skripta `1_text_cnn_train.py` ucitava labelirani train skup, pretvara zapise vatrozida u tekstualni oblik, tokenizira zapise, gradi vokabular, radi padding sekvenci i normalizira numericke znacajke. Model kombinira tekstualne znacajke iz TextCNN dijela i normalizirane numericke znacajke.

```powershell
python 1_text_cnn_train.py
```

Nakon treniranja spremaju se:

```text
model_artifacts/textcnn_model_trained.pth
model_artifacts/textcnn_vocab_trained.pth
model_artifacts/textcnn_label_encoder_trained.pth
model_artifacts/textcnn_numeric_scaler_trained.pth
```

BiLSTM i MLP artefakti spremaju se istim obrascem: `ime_modela_vrsta_artefakta_trained.pth`, npr. `bilstm_vocab_trained.pth` ili `mlp_feature_scaler_trained.pth`.

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
python 1_text_cnn_train.py
```
