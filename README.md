# Detekcija napada iz zapisa vatrozida

Ovaj projekt sadrzi eksperimentalni kod za diplomski rad u kojem se zapisi vatrozida koriste za treniranje i provjeru modela za klasifikaciju mreznog prometa. Skup podataka sadrzi normalan promet i tri klase napada: `Port Scan`, `DDoS` i `Brute Force`.

## Struktura projekta

```text
eksperiment_model/
├── firewall_logs_unlabeled/   #nelabelirani train, eval i test CSV zapisi
├── firewall_logs_labeled/     #labelirani train, eval i test CSV zapisi
├── model_artifacts/           #spremljeni modeli i pomocni artefakti
├── eval_results/              #rezultati evaluacije po modelu
├── test_results/              #rezultati zavrsnog testiranja po modelu
├── scripts/
│   └── labeliranje_logova.py  #pravilo-bazirano labeliranje zapisa
├── 1_text_cnn_train.py        #treniranje TextCNN modela
├── 2_text_cnn_eval.py         #evaluacija TextCNN modela
├── 3_text_cnn_test.py         #zavrsno testiranje TextCNN modela
├── 1_biLSTM_train.py          #treniranje BiLSTM modela
├── 2_biLSTM_eval.py           #evaluacija BiLSTM modela
├── 3_biLSTM_test.py           #zavrsno testiranje BiLSTM modela
├── 1_mlp_train.py             #treniranje MLP modela
├── 2_mlp_eval.py              #evaluacija MLP modela
├── 3_mlp_test.py              #zavrsno testiranje MLP modela
└── requirements.txt           #python dependency
```

## Podaci

Ulazni zapisi nalaze se u `firewall_logs_unlabeled`, a labelirani zapisi u `firewall_logs_labeled`. Podaci su podijeljeni na:

```text
train - koristi se za treniranje modela
eval  - koristi se za provjeru modela na nevidenim podacima tijekom razvoja
test  - koristi se za zavrsnu procjenu modela na novom skupu podataka
```

Labelirani CSV zapisi sadrze firewall atribute kao sto su zone, sucelja, IP adrese, portovi, protokol, akcija vatrozida, status sesije i zavrsna klasa prometa u stupcu `label`.

## Labeliranje

Skripta `scripts/labeliranje_logova.py` automatski labelira zapise prema pravilima nad vremenskim prozorom od jedne minute. Svi zapisi se pocetno oznacavaju kao `Normal`, a zatim se prema obrascima prometa preoznacavaju u `Port Scan`, `DDoS` ili `Brute Force`.

```powershell
python scripts/labeliranje_logova.py
```

## Modeli

U eksperimentu su koristena tri modela:

```text
TextCNN - koristi tokenizirani tekstualni prikaz zapisa i dodatne numericke znacajke
BiLSTM  - koristi tokenizirani tekstualni prikaz zapisa i obradu slijeda u oba smjera
MLP     - koristi pripremljene numericke i kategoricke znacajke
```

TextCNN skripta pretvara odabrane atribute zapisa vatrozida u tekstualni oblik, tokenizira zapise, gradi vokabular, radi padding sekvenci i normalizira numericke znacajke. Model zatim kombinira tekstualne znacajke izdvojene konvolucijskim slojem i normalizirane numericke znacajke.

BiLSTM koristi slicnu tekstualnu pripremu podataka, ali umjesto konvolucijskih filtara koristi dvosmjerni LSTM sloj koji obraduje slijed tokena s lijeve i desne strane.

MLP model ne koristi tekstualni slijed, nego tablicni prikaz podataka. IP adrese se pretvaraju u numericke vrijednosti, numericke znacajke se skaliraju, a kategoricke znacajke enkodiraju. Model koristi dva skrivena sloja s ReLU aktivacijom i dropout regularizacijom.

## Instalacija

Preporuka je koristiti Python virtual env


```powershell
python -m venv C:\venv-tfpu
C:\venv-tfpu\Scripts\activate
cd "C:\Users\Administrator\Desktop\TFPU\2. Godina\Diplomski\eksperiment_model"
pip install --upgrade pip
pip install -r requirements.txt
```

## Pokretanje eksperimenata

Skripte se pokrecu iz foldera projekta redom: prvo treniranje, zatim evaluacija, a na kraju testiranje.

```powershell
cd "C:\Users\Administrator\Desktop\TFPU\2. Godina\Diplomski\eksperiment_model"

# TextCNN
python 1_text_cnn_train.py
python 2_text_cnn_eval.py
python 3_text_cnn_test.py

# BiLSTM
python 1_biLSTM_train.py
python 2_biLSTM_eval.py
python 3_biLSTM_test.py

# MLP
python 1_mlp_train.py
python 2_mlp_eval.py
python 3_mlp_test.py
```

Nakon treniranja spremaju se naucene tezine modela i pomocni artefakti potrebni za evaluaciju i testiranje:

```text
model_artifacts/textcnn_model_trained.pth
model_artifacts/textcnn_vocab_trained.pth
model_artifacts/textcnn_label_encoder_trained.pth
model_artifacts/textcnn_numeric_scaler_trained.pth
```

BiLSTM i MLP artefakti spremaju se istim obrascem: `ime_modela_vrsta_artefakta_trained.pth`, npr. `bilstm_vocab_trained.pth` ili `mlp_feature_scaler_trained.pth`.

## Rezultati

Rezultati evaluacije spremaju se u `eval_results`, a rezultati zavrsnog testiranja u `test_results`. Za svaki model spremaju se:

```text
*_metrics.txt              # accuracy, loss i classification report
*_confusion_matrix.csv     # matrica zabune
*_predictions.csv          # stvarne i predvidene klase po zapisu
```

Zavrsni rezultati na testnom skupu:

| Model | Accuracy | Loss |
|---|---:|---:|
| TextCNN | 90.14% | 0.3039 |
| BiLSTM | 92.09% | 0.2511 |
| MLP | 99.44% | 0.0218 |

TextCNN i BiLSTM ostvaruju dobre ukupne rezultate, ali kod oba modela najveci pad je vidljiv kod klase `Port Scan`. MLP model ostvaruje najvisu tocnost na testnom skupu, sto pokazuje da su numericke i kategoricke znacajke vrlo korisne za ovaj skup podataka.


