# DEL Mini-Challenge: Hyperparameter Tuning und Model Evaluation

Mini-Challenge 1 der Kompetenz *Deep Learning (DEL)*, BSc Data Science FHNW, HS26.

Klassifikation von Sentinel-2-Satellitenbildern (EuroSAT, 10 Landnutzungsklassen) mit einem selbst
gebauten CNN. Untersucht werden Lernrate und Batchsize, anschliessend eine Reihe von Hypothesen zu
Modelltiefe, Breite, Conv-Settings, Regularisierung, BatchNorm, Initialisierung, Optimizern und
Transfer Learning.

## Abgabe

Der Bericht ist **ein einziges Notebook**:
[`notebooks/mini_challenge_del.ipynb`](notebooks/mini_challenge_del.ipynb). Es enthält alle Schritte,
Resultate und Interpretationen. Der Code unter `src/delmc/` ist reine Infrastruktur — man muss keine
`.py`-Datei öffnen, um zu verstehen *was* gemacht wurde.

Abgegeben wird in zwei Etappen, je als eingefrorene git-Version:

| Tag | Inhalt |
|---|---|
| `stage-1` | Basismodell, Lernrate und Batchsize evaluiert, statistischer Fehler |
| `stage-2` | Alle Hypothesen, bestes Modell, finale Evaluation |

## Aufbau

```
notebooks/   Der Bericht (ein Notebook)
src/delmc/   Infrastruktur: Daten, Training, Probes, Tracking, Caching, Plots
results/     Trainingsverläufe als JSON — das Notebook zeichnet daraus, ohne neu zu trainieren
reports/     Proposal für den Fachcoach, Q&A-Notizen, Präsentation
data/        EuroSAT (nicht versioniert, wird beim ersten Lauf geladen)
```

Die Resultate liegen als JSON im Repository. Das Notebook lässt sich damit vollständig von oben nach
unten ausführen, ohne ein einziges Training neu zu starten.

## Setup

```bash
conda env create -f environment.yml
conda activate ml-bachelor-del
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

PyTorch kommt aus einem eigenen Index und wird deshalb separat installiert.

### Rechenwerk

Trainiert wird auf einer **Intel Arc 140T** über das XPU-Backend von PyTorch (`device="xpu"`). CUDA
gibt es auf dieser Maschine nicht und wird von der Aufgabenstellung auch nicht verlangt — die
Referenzumgebung des Kurses ist selbst eine CPU-Installation. **CPU bleibt die Referenz**: jedes
Resultat muss auch ohne Beschleuniger reproduzierbar sein, und die Gerätewahl steckt ausschliesslich
in `delmc.utils.select_device`.

### Weights & Biases

Getrackt wird nach `University-Projects/DEL-EuroSAT-Land-Classifier-Sentinel-2`.

```bash
wandb login
```

Der API-Key gehört **nicht** ins Repository, nicht ins Notebook und nicht in eine Zelle mit
gespeichertem Output. `wandb login` legt ihn in `~/.netrc` ausserhalb des Projekts ab; alternativ
`WANDB_API_KEY` in der Shell setzen.

## Aufgabenstellung

`SGDS_DEL_MC_v3` (Trainingscenter DEL, Stand 2026-02-25).
