# Begleitender Code zur Masterarbeit
Dieses Repository enthält die im Rahmen der Masterarbeit

**„Korngrößenkartierung in fluvialen Sedimenten aus Drohnen-Orthophotos mittels Deep Learning: Kombination aus Bildsegmentierung und CNN-basierter Regressionsmodellierung“**

von Fabian Wild entwickelten Datenaufbereitungsskripte und
projektspezifischen Anpassungen an GRAINet und SediNet.


## Originalimplementierungen

### GRAINet

Original repository:

https://github.com/langnico/GRAINet

nach Lang, Nico, Andrea Irniger, Agnieszka Rozniak, Roni Hunziker, Jan Dirk Wegner, and Konrad Schindler. "GRAINet: mapping grain size distributions in river beds from UAV images with convolutional neural networks." Hydrology and Earth System Sciences 25, no. 5 (2021): 2567-2597.

Reference commit:

`d33ae686e770ab95f99387ce5e70ab353607d7a9`

### SediNet

Original repository:

https://github.com/DigitalGrainSize/SediNet

nach Buscombe, D. (2019). SediNet: a configurable deep learning model for mixed qualitative and quantitative optical granulometry. Earth Surface Processes and Landforms 45 (3), 638-651.

Reference commit:

`666ffaa5edc9b83d860aecaab309b12fc55600e9`

## Zielgröße

Beide Modellarchitekturen wurden auf die Vorhersage des
kachelbezogenen 50. Perzentils der durch PebbleCounts
ermittelten b-Achsen angewandt.

Der Zielwert entspricht:

`P50,b = median(b1, b2, ..., bn)`

Bei GRAINet wurde der Wert in Zentimetern im technisch als
`dm` bezeichneten NPZ-Feld gespeichert. Er entspricht nicht
dem ursprünglich von Lang et al. (2021) verwendeten mittleren
Korndurchmesser dm.

Bei SediNet wurde derselbe Zielwert in Millimetern in der
Spalte `P50` gespeichert.

## Skripts zur Datenaufbereitung (kein Bestandteil von SediNet/GRAINet)

`preprocessing/CSV_to_NPZ.py` erzeugt aus einem Orthofoto und
georeferenzierten PebbleCounts-Messungen den GRAINet-kompatiblen
NPZ-Datensatz.

Verwendete Einstellungen:

- Modus: `per_window_d50`
- Fenstergröße: 224 × 224 Pixel
- Stride: 64 Pixel
- Mindestzahl erkannter Steine: 10
- Zielgröße: P50 der b-Achsen

`preprocessing/SediNet_csv_create.py` erzeugt die von SediNet
benötigten Trainings- und Testtabellen.

## GRAINet

Die ursprüngliche Netzwerkarchitektur wurde nicht verändert.
Anpassungen betreffen hauptsächlich TensorFlow-/Keras-
Kompatibilität, GPU-Speicherverwaltung, und batchweise Inferenz aufgrund von Speichermanagement auf schwächeren Systemen

# GRAINet: Änderungen, Datenaufbereitung und Auswirkungen


| Datei | Zeilen | Art | Bedeutung |
|---|---:|---|---|
| `helper.py` | 54–55 | Konfiguration | Standardgröße der Bildkacheln geändert |
| `train_test.py` | 15–28, 37, 104, 115–116 | technische Anpassung | Mixed Precision, GPU-Speicherverwaltung, kleinere Validierungs-Batches, aktuelle Optimizer-Syntax |
| `resnet_architecture.py` | 1–4, 166–179 | Kompatibilität | TensorFlow-2-/Keras-Kompatibilität; alte manuelle Gewichtsinitialisierung ersetzt |
| `inference_bank.py` | 30, 78, 86–93, 112–127 | Inferenzanpassung | Kachelgröße, GSD, feste Modellgröße und speicherschonendere Vorhersage |



## Unveränderte GRAINet-Dateien


- `plots.py`
- `loss_functions.py`
- `preprocessing.py`
- `test_vis.py`


## SediNet

Änderungen betreffen vor allem Erweiterung der Inferenz zur räumlichen Rasterausgabe

# SediNet: Änderungen, Erweiterungen und Auswirkungen

| Datei | Zeilen | Art | Bedeutung |
|---|---:|---|---|
| `defaults.py` | 19–22, 37 | Hyperparameter | Ensemble-Batchgrößen und Epochenzahl geändert |
| `defaults-global.py` | 14–22, 37 | Hyperparameter/Eingabe | Bildgröße, Ensemble-Batchgrößen und Epochenzahl geändert |
| `sedinet_eval.py` | 12–14, 218–267, 271–286, 294–330, 349–350 | technische/funktionale Anpassung | große Bilder, Gewichtsdateien, Pfadkorrektur und Ergebnisnormalisierung |
| `sedinet_predict1image.py` | aktiver neuer Code 1–351; altes Original auskommentiert 354–545 | wesentliche Erweiterung | gekachelte GeoTIFF-Inferenz und räumliche Rasterausgabe |

## Unveränderte SediNet-Dateien

- `sedinet_train.py`
- `imports.py`
- `sedinet_predict.py`
- `sedinet_predictfolder.py`
- `sedinet_models.py`
- `sedinet_utils.py`
- `train_all.sh`
- `sedinet_infer.py`

