# Begleitender Code zur Masterarbeit
Dieses Repository enthält die im Rahmen der Masterarbeit

**„Korngrößenkartierung in fluvialen Sedimenten aus Drohnen-Orthophotos mittels Deep Learning: Kombination aus Bildsegmentierung und CNN-basierter Regressionsmodellierung“**

von Fabian Wild entwickelten Datenaufbereitungsskripte und
projektspezifischen Anpassungen an GRAINet und SediNet.


## Originalimplementierungen

### GRAINet

Original repository:

https://github.com/langnico/GRAINet

Reference commit:

`d33ae686e770ab95f99387ce5e70ab353607d7a9`

### SediNet

Original repository:

https://github.com/DigitalGrainSize/SediNet

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

## Datenaufbereitung

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
Kompatibilität, GPU-Speicherverwaltung, batchweise Inferenz
und experimentabhängige Eingabeparameter.

## SediNet

SediNet wurde erweitert um:

- projektspezifische CSV-Erstellung aus PebbleCountsAuto;
- gekachelte Verarbeitung großer Orthofotos;
- GeoTIFF-Ausgabe;
- GPU-Speicherverwaltung;
- flexiblere Verarbeitung der Modellgewichte.

## Daten

Orthofotos, PebbleCounts-Rohdaten, Modellgewichte und erzeugte
Ergebnisdateien sind nicht Bestandteil dieses Repositories.
