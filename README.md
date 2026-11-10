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
src/delmc/   Daten, Modell, Trainingsschleife, Cross-Validation, Gradienten-Messung, Caching, Plots
scripts/     Startet die Trainingsläufe und lädt die Verläufe nach W&B hoch
results/     Trainingsverläufe als JSON, das Notebook zeichnet daraus, ohne neu zu trainieren
reports/     HTML-Export des Notebooks
data/        EuroSAT (nicht versioniert, wird beim ersten Lauf von Hugging Face geladen)
```

Weil die Resultate als JSON vorliegen, läuft das Notebook von oben nach unten durch, ohne dass ein
Training wiederholt wird. Neu gerechnet werden nur der kurze Overfitting-Test in Abschnitt 5.2, die
Messung direkt nach der Initialisierung in 10.6 und die Auswertung auf dem Testset in Abschnitt 13.
Für das Testset braucht es die Gewichte der beiden finalen Modelle unter `results/final/`, die übrigen
Gewichte sind nicht versioniert.

Trainiert wird auf einem stratifizierten Subset von 6000 der 18'900 Trainingsbilder, weil auf der CPU
sonst jede Epoche gut dreimal so lange dauert. Die Begründung steht im Notebook in Abschnitt 2.4. Nur
das Basismodell und das beste Modell werden am Schluss auf dem ganzen Trainings-Split trainiert
(Abschnitt 12).

Alle Trainingsläufe startet `scripts/run_hypotheses.py`, eine Welle pro Abschnitt oder Hypothese. Für
die erste Etappe zum Beispiel `python scripts/run_hypotheses.py kapazitaet lernrate batchsize crossval`,
für die zweite `python scripts/run_hypotheses.py control h01 best final`. Bereits gerechnete Läufe werden
übersprungen. Nur der frühe Vorlauf unter `results/baseline_selection/`, auf den sich Abschnitt 10.5
bezieht, stammt aus einer älteren Version des Codes.

## Setup

```bash
conda env create -f environment.yml
conda activate ml-bachelor-del
```

Die Versionen in `environment.yml` sind die, mit denen gerechnet wurde. Lokal war der XPU-Build von
PyTorch installiert (`2.14.0+xpu`), gerechnet wurde aber auf der CPU.

### Hardware

Alle Läufe sind auf der CPU gerechnet. Die Maschine hat zwar eine Intel-GPU, das XPU-Backend von
PyTorch hat sie unter WSL aber nicht erkannt. Die Aufgabenstellung verlangt keine GPU, und der Code
wählt das Gerät an einer einzigen Stelle (`delmc.utils.select_device`), mit einer GPU läuft er also
ohne Änderung.

### Weights & Biases

Die Trainingsläufe sind in W&B unter
[`University-Projects/DEL-EuroSAT-Land-Classifier-Sentinel-2`](https://wandb.ai/University-Projects/DEL-EuroSAT-Land-Classifier-Sentinel-2)
abgelegt, eine Gruppe pro Ordner unter `results/`. Sie wurden nach dem Training mit
`scripts/replay_wandb.py` aus den JSON-Verläufen hochgeladen und sind deshalb mit `replayed` markiert.

```bash
wandb login
```

Der API-Key gehört nicht ins Repository. `wandb login` legt ihn in `~/.netrc` ausserhalb des Projekts
ab, alternativ lässt sich `WANDB_API_KEY` in der Shell setzen.

## Aufgabenstellung

`SGDS_DEL_MC_v3` (Trainingscenter DEL, Stand 2026-02-25).
