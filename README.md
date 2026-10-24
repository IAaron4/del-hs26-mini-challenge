# DEL Mini-Challenge: Hyperparameter Tuning und Model Evaluation

Mini-Challenge 1 der Kompetenz *Deep Learning (DEL)*, BSc Data Science FHNW, HS26.

Klassifikation von Sentinel-2-Satellitenbildern (EuroSAT, 10 Landnutzungsklassen) mit einem selbst
gebauten CNN. Untersucht werden Lernrate und Batchsize, anschliessend eine Reihe von Hypothesen zu
Modellkomplexität, Conv-Settings, Regularisierung, BatchNorm, Initialisierung, Optimizern und Transfer
Learning.

## Bericht

Der Bericht ist ein einziges Notebook:
[`notebooks/mini_challenge_del.ipynb`](notebooks/mini_challenge_del.ipynb). Es enthält alle Schritte,
Resultate und Interpretationen, die Outputs sind gespeichert. Wer nichts installieren will, kann
dieselbe Version als [`reports/mini_challenge_del.html`](reports/mini_challenge_del.html) im Browser
öffnen.

Der Code unter `src/delmc/` ist Infrastruktur, die sonst das Notebook mit Boilerplate füllen würde.
Was gemacht wurde, steht im Notebook, die `.py`-Dateien zeigen nur, wie es umgesetzt ist.

## Aufbau

```
notebooks/   Der Bericht (ein Notebook)
src/delmc/   Daten, Modell, Trainingsschleife, Gradienten-Messung, Tracking, Caching, Plots
results/     Trainingsverläufe als JSON, das Notebook zeichnet daraus, ohne neu zu trainieren
reports/     HTML-Export des Notebooks
data/        EuroSAT (nicht versioniert, wird beim ersten Lauf von Hugging Face geladen)
```

Weil die Resultate als JSON vorliegen, läuft das Notebook von oben nach unten durch, ohne dass ein
Training wiederholt wird. Neu gerechnet wird nur der kurze Overfitting-Test in Abschnitt 5.2.

## Setup

```bash
conda env create -f environment.yml
conda activate ml-bachelor-del
```

Verwendet wurden Python 3.13, PyTorch 2.14 und torchvision 0.29.

### Rechenwerk

Alle Läufe sind auf der CPU gerechnet. Die Maschine hat zwar eine Intel-GPU, das XPU-Backend von
PyTorch hat sie unter WSL aber nicht erkannt. Die Aufgabenstellung verlangt keine GPU, und der Code
wählt das Gerät an einer einzigen Stelle (`delmc.utils.select_device`), mit einer GPU läuft er also
ohne Änderung.

### Weights & Biases

Die Trainingsläufe sind in W&B unter `University-Projects/DEL-EuroSAT-Land-Classifier-Sentinel-2`
abgelegt. Sie wurden nach dem Training aus den JSON-Verläufen hochgeladen und sind deshalb mit
`replayed` markiert.

```bash
wandb login
```

Der API-Key gehört nicht ins Repository. `wandb login` legt ihn in `~/.netrc` ausserhalb des Projekts
ab, alternativ lässt sich `WANDB_API_KEY` in der Shell setzen.

## Aufgabenstellung

`SGDS_DEL_MC_v3` (Trainingscenter DEL, Stand 2026-02-25).
