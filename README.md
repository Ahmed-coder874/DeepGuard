<div align="center">

# DeepGuard

**AI-Based Deepfake Detection**

Project **Deepfake Technology** &middot; Project ID **24CGCS06**
Department of **ECE** &middot; SDG **10** &middot; **Atria Institute of Technology, Bengaluru**
Project Guide: **Prof. Jayanth U**

`EfficientNet-B0` &middot; `PyTorch` &middot; `Streamlit`

**Experiment 1 (frozen baseline) test accuracy: 75.00%** on the 120-image held-out test set
**Experiment 2 test accuracy: 97.56%** on the 900-image held-out test set

These are **two separate experiments on two different datasets**, not one result.
Experiment 2 is **not** a controlled dataset-size-only experiment — see
[Limitations](#limitations).

</div>

---

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [System architecture](#system-architecture)
- [Technology stack](#technology-stack)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model setup](#model-setup)
- [Run the application](#run-the-application)
- [Demo instructions](#demo-instructions)
- [How the prediction works](#how-the-prediction-works)
- [Experimental setup](#experimental-setup)
- [Official results](#official-results)
  - [Experiment 1 — Frozen Baseline](#experiment-1--frozen-baseline)
  - [Experiment 2 — Larger, Differently Distributed Dataset](#experiment-2--larger-differently-distributed-dataset)
  - [Experiment 1 vs Experiment 2](#experiment-1-vs-experiment-2)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Academic project status](#academic-project-status)
- [Testing](#testing)
- [Dataset](#dataset)
- [License and attribution](#license-and-attribution)

---

## Overview

**DeepGuard** is an AI/ML-based **deepfake detection** system. It takes a single
image of a face, runs it through a trained **EfficientNet-B0** convolutional
neural network, and returns a binary classification:

| Class | Meaning |
| --- | --- |
| `REAL` | The image is judged to be an authentic, unmanipulated frame |
| `FAKE` | The image is judged to be manipulated (a "deepfake") |

The model was trained from ImageNet-pretrained weights on frames extracted from
**FaceForensics++**, and the model itself is committed with the repository, so a
fresh `git clone` can run live inference immediately.

**Two checkpoints are committed.** Experiment 1
(`models/deepguard_efficientnet_b0.pt`) is the frozen academic baseline, trained
on FaceForensics++ frames. Experiment 2
(`models/experiment_2_deepguard_efficientnet_b0.pt`) is a separate, larger,
differently distributed experiment and is **the checkpoint the Streamlit app
serves by default**. Neither replaces the other.

> **Responsible AI notice.** Deepfake detection is not a solved problem. A
> prediction produced by this application is an *indication requiring further
> verification*, never proof of authenticity. The served model reaches 97.56%
> test accuracy on its own 900-image Experiment 2 test split, and the Experiment 1
> baseline reaches 75.00% on its own 120-image split. **It can be wrong, and it
> does not generalise to deepfakes it was never trained on.**

---

## Key features

- **Image upload** — JPG, JPEG and PNG through a drag-and-drop Streamlit widget.
- **Input validation** — rejects unsupported extensions, corrupt files and
  images that are too small, with plain-language messages instead of stack traces.
- **Image preprocessing** — decode, BGR→RGB, resize to 224×224, normalise to 0–1,
  batch dimension added for display.
- **Face / deepfake classification** — REAL vs FAKE from the trained model.
- **EfficientNet-B0 backbone** — a compact CNN (≈4.2 M parameters) chosen for
  CPU-only training; see [`docs/model_architecture_research.md`](docs/model_architecture_research.md).
- **Model confidence display** — the softmax probability of the predicted class
  plus the full per-class output, shown exactly as the model produced it.
- **Streamlit interface** — a single-command local web app, no server deployment required.
- **Inference result** — a clear REAL/FAKE verdict, the confidence bar, the model
  identity and the checkpoint that produced the answer.
- **Honest status reporting** — the app never fabricates a prediction. If the
  checkpoint is missing it says so and shows exactly where the file belongs.
- **Full offline pipeline** — dataset download, extraction, preparation, training
  and evaluation scripts are all included for reproducibility.

---

## System architecture

```
┌──────────────────────┐
│      Image Upload    │   JPG / JPEG / PNG via Streamlit
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Input Validation    │   extension, size, decodability, dimensions
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Image Preprocessing  │   decode → RGB → 224×224 → normalise → (1,3,224,224)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   EfficientNet-B0    │   ImageNet-pretrained backbone, fine-tuned (2 classes)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│      Prediction      │   softmax → argmax over {real, fake}
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│    Result Display    │   REAL / FAKE + model confidence + per-class output
└──────────────────────┘
```

All seven stages — input, validation, preprocessing, feature/representation
preparation, model, classification, prediction — are implemented and connected.

---

## Technology stack

Only technologies actually used by this project are listed.

| Technology | Version | Role |
| --- | --- | --- |
| **Python** | 3.14.7 (win_amd64) | Language |
| **PyTorch** (`torch`) | 2.14.0 (CPU) | Model construction, training, inference |
| **Torchvision** | 0.29.0 | ImageNet-pretrained EfficientNet-B0 weights, eval transforms |
| **EfficientNet-B0** | — | Backbone architecture (via `torchvision.models`) |
| **Streamlit** | 1.64.0 | Web user interface |
| **OpenCV** (`opencv-python`) | 5.0.0.93 | Image decode, colour conversion, resize |
| **NumPy** | 2.5.3 | Array representation and normalisation |
| **Pillow** (`Pillow`) | 12.3.0 | Safe image validation, dataset scanning |
| **tqdm** | 4.70.1 | *Optional* — dataset download helper only |

Metrics (accuracy, precision, recall, F1, confusion matrix) are implemented in
plain Python in `src/evaluation.py`, so **no scikit-learn dependency** is needed.

---

## Project structure

```
deepguard/
│
├── app.py                        # Streamlit application (entry point)
├── config.py                     # Central configuration: paths, seed, splits, labels
├── README.md                     # This file
├── .gitignore
│
├── requirements.txt              # APPLICATION dependencies        (venv)
├── requirements-train.txt        # TRAINING/EVALUATION dependencies (venv-train)
├── requirements-data.txt         # OPTIONAL dataset tooling dependencies
│
├── download.py                   # FaceForensics++ download helper
├── extract_dataset_frames.py     # FaceForensics++ videos -> labelled frames
├── prepare_dataset.py            # Deterministic, leak-free train/val/test split
├── train_model.py                # Stage 4: training entry point
├── evaluate_model.py             # Stage 5: official evaluation entry point
│
├── src/
│   ├── __init__.py
│   ├── validation.py             # Upload validation
│   ├── preprocessing.py          # Display preprocessing (0-1 array)
│   ├── dataset.py                # Dataset discovery, splitting, statistics
│   ├── extraction.py             # Deterministic frame extraction + provenance
│   ├── model_interface.py        # Truthful checkpoint status reporting
│   ├── training.py               # Model building, training loop, transforms
│   ├── evaluation.py             # Metrics + test-split evaluation
│   └── inference.py              # Guarded real inference (Stage 6)
│
├── tests/                        # Unit + Streamlit end-to-end tests
│   ├── test_pipeline.py
│   ├── test_dataset.py
│   ├── test_model_interface.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   ├── test_extraction.py
│   └── test_app.py               # Runs the real trained model through AppTest
│
├── docs/                         # Research and methodology documentation
│   ├── dataset_research.md               # Public dataset comparison
│   ├── model_architecture_research.md    # Candidate architectures + selection
│   ├── training_plan.md                  # Stage 4 training design
│   └── evaluation_plan.md                # Stage 5 evaluation design
│
├── models/                      # BOTH checkpoints are committed
│   ├── deepguard_efficientnet_b0.pt           # Experiment 1 - frozen baseline
│   └── experiment_2_deepguard_efficientnet_b0.pt  # Experiment 2 - served by the app
│
├── experiments/
│   ├── experiment_1_baseline/
│   │   └── README.md                           # Experiment 1 methodology summary
│   └── experiment_2_large_dataset/
│       ├── README.md                           # Full Experiment 2 methodology + caveats
│       ├── build_dataset.py                    # Scan, MD5/pHash de-dup, seeded sampling, audit
│       ├── train_experiment_2.py               # Resume-capable trainer
│       ├── generate_plots.py                   # Renders plots from recorded values only
│       ├── dataset_report.json                 # Dataset, quota and audit evidence
│       ├── dataset_manifest.csv                # 6,000 rows: label, split, md5, pHash (no pixels)
│       ├── training_summary.json               # Hyperparameters, epochs, duration, fingerprint
│       ├── train_run_log.txt                   # Header-only; see Experiment 2 README
│       └── reports/
│           ├── evaluation_report.json          # Official Experiment 2 test metrics
│           ├── confusion_matrix.png
│           ├── per_class_metrics.png
│           ├── training_loss_curve.png
│           └── training_accuracy_curve.png
│
├── reports/                      # Frozen experimental evidence
│   ├── DeepGuard_SIP_Final_Technical_Report.md / .pdf
│   ├── DeepGuard_SIP_Presentation.md
│   ├── DeepGuard_Viva_QA.md
│   ├── DeepGuard_Submission_Checklist.md
│   ├── evaluation_report.json            # Experiment 1 official Stage 5 metrics
│   ├── experiment_2_report.md           # Experiment 2 full written report
│   └── stage6/
│       ├── train_run_log.txt             # Full Experiment 1 training log
│       ├── stage6_results_analysis.md
│       ├── confusion_matrix.png
│       ├── per_class_metrics.png
│       ├── training_curves.png
│       └── generate_stage6_plots.py
│
├── assets/
│   └── demo/README.md            # Rules and sources for demonstration images
│
└── data/                         # EXCLUDED FROM GIT - see data/README.md
    ├── README.md
    ├── faceforensics/            # (optional) source videos
    ├── raw/                      # (optional) 300 real + 300 fake frames
    └── processed/                # (optional) train 360 / val 120 / test 120
```

**Not committed to Git.** Everything below is reproducible from the committed
scripts and seeds, but is excluded from the repository:

| Path | Reason |
| --- | --- |
| `data/faceforensics/`, `data/raw/`, `data/processed/` | FaceForensics++ is restricted, research-use-only; ~793 MB of derived frames |
| `experiments/experiment_2_large_dataset/source/` | ~4 GB Kaggle dataset — CC BY-NC-SA 4.0, not redistributable |
| `experiments/experiment_2_large_dataset/processed/` | the 6,000 selected images — rebuild with `build_dataset.py` |
| `models/experiment_2_training_state.pt` | large resume state, meaningless without the dataset |

Both **model checkpoints**, all Experiment 1 and Experiment 2 **evidence files**
(manifests, JSON reports, logs, plots) and all **documentation** listed above *are*
committed.

---

## Requirements

- **Operating system:** Windows (commands below are PowerShell). macOS/Linux also
  work with the equivalent path separator.
- **Python:** **3.14.7** (verified; `win_amd64`). Python 3.11+ is expected to work.
- **Disk space:** ~1.5 GB for `venv` (PyTorch CPU is the bulk), plus ~16 MB for
  the model. The optional `data/` rebuild needs a further ~800 MB.
- **GPU:** not required. Everything runs on CPU.

### Environments

| Environment | Purpose | Requirements file | Needed to run the app? |
| --- | --- | --- | --- |
| `venv` | **Application** — Streamlit UI, preprocessing, live inference | `requirements.txt` | **Yes** |
| `venv-train` | **Training / evaluation only** | `requirements-train.txt` | No |

`venv-train` is never required to launch or demonstrate DeepGuard.

---

## Installation

### Windows PowerShell

```powershell
cd <project folder>
```

```powershell
# 1. Create the application environment
python -m venv venv

# 2. Install the application dependencies
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optionally activate the environment so you can call `python` and `streamlit`
directly:

```powershell
.\venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

### Verify the environment

```powershell
.\venv\Scripts\python.exe --version
.\venv\Scripts\python.exe -m streamlit --version
Test-Path .\models\deepguard_efficientnet_b0.pt
Test-Path .\models\experiment_2_deepguard_efficientnet_b0.pt
```

Expected: Python `3.14.7`, Streamlit `1.64.0`, and `True` for both checkpoints.
The app serves the **Experiment 2** checkpoint by default, so that file is the one
required to run the demo.

---

## Model setup

The models live at **project-relative** paths that are resolved from the
location of `config.py`, so the application works from any clone location on any
machine. No environment variable or absolute path is required.

```
<project root>
└── models
    ├── deepguard_efficientnet_b0.pt            # Experiment 1 (frozen baseline)
    └── experiment_2_deepguard_efficientnet_b0.pt  # Experiment 2 (served by the app)
```

### Two checkpoints, two experiments

| | Experiment 1 — frozen baseline | Experiment 2 |
| --- | --- | --- |
| File | `models/deepguard_efficientnet_b0.pt` | `models/experiment_2_deepguard_efficientnet_b0.pt` |
| Size | 16,341,163 bytes (≈ 16.3 MB) | 16,346,305 bytes (≈ 16.3 MB) |
| Trained on | FaceForensics++ frames | Kaggle *140k Real and Fake Faces* subset |
| Best epoch | 6 | 10 |
| Test accuracy | 75.00% (120 images) | 97.56% (900 images) |
| Status | Frozen. Never retrained or replaced. | Complete. The default checkpoint in `app.py`. |

**Experiment 2 does not replace Experiment 1.** Experiment 1 remains the frozen
academic baseline; its checkpoint, evaluation report, dataset and split are
untouched. Both files are committed and both remain in the repository.

### Storage method: ordinary Git (no Git LFS)

| Property | Value |
| --- | --- |
| File | `models/deepguard_efficientnet_b0.pt` |
| Size | 16,341,163 bytes (**≈ 16.3 MB**) |
| GitHub plain-file limit | 100 MB |
| Repository limit | 1 GB (soft warning), 5 GB (hard block) |
| **Decision** | **Committed normally.** 16.3 MB is well under every GitHub limit, so Git LFS or any external download mechanism is unnecessary. A plain `git clone` gives you a working model with no extra step. |

The file is listed as an explicit exception in `.gitignore` so it can never be
accidentally excluded, and no other checkpoint pattern is tracked.

### If the model is missing

The application does **not** crash and does **not** invent a prediction. It shows:

- a clear error in the sidebar and in *Section 4 — Detection Model Status*,
- the exact absolute path the file is expected at,
- and the model card changes to **"Not found"**.

Upload and preprocessing still work, and the detection result is simply
withheld. The fix is to place the file the app expects at
`models/experiment_2_deepguard_efficientnet_b0.pt`.

---

## Run the application

From the project folder:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```
http://localhost:8501
```

Stop the server with `Ctrl + C`.

> Calling the interpreter explicitly (instead of `streamlit run app.py`) makes
> certain the application always uses the `venv` interpreter and never a
> system-wide or `venv-train` installation.

---

## Demo instructions

> **Important:** demo images are **separate** from the official evaluation
> dataset. Never use the official 120-image held-out test set
> (`data/processed/test/`) as casual demo material — that would contaminate the
> frozen evaluation. See [`assets/demo/README.md`](assets/demo/README.md).

1. Start the app (command above) and open `http://localhost:8501`.
2. In **1. Upload an Image**, choose a JPG, JPEG or PNG file. The app validates
   it and shows the **Original Image** with its metadata.
3. Click **Analyze Image**. The trained EfficientNet-B0 runs under `torch.no_grad()`.
4. The app shows the **Preprocessed Image** (224×224) and the **Detection
   Result**: a REAL or FAKE verdict, the model confidence, the per-class model
   output, and the checkpoint that produced it.
5. Scroll to **4. Detection Model Status** to confirm the model is loaded, and to
   **5. Dataset Status** to see the real dataset counts (this section is only
   populated if you rebuilt `data/`).

**Suggested demo pair**

| Purpose | Where to get it |
| --- | --- |
| Genuine face | your own photo, or `data/processed/train/real/` |
| Manipulated face | `data/processed/train/fake/` (FaceForensics++ Deepfakes) |

The training split is already part of the model, so it is an honest illustration
rather than evaluation evidence.

**How to present the numbers correctly**

- The app is serving the **Experiment 2** checkpoint, so its official result is
  **97.56% test accuracy on 900 held-out images**. Do not quote 75.00% while the
  Experiment 2 model is loaded — that would misattribute the wrong experiment's
  number to the model on screen.
- Mention the **Experiment 1 frozen baseline (75.00% on 120 held-out images)**
  only as a separate, earlier experiment, and state that Experiment 2 is not a
  controlled dataset-size-only comparison.
- Say a demo prediction is a *live single-image model output*, not part of any
  official evaluation. The two are never mixed.

---

## How the prediction works

Accurately described from the code — no behaviour is invented here.

1. **Upload** — Streamlit returns the file bytes. The widget restricts the
   chooser to `jpg`, `jpeg` and `png`; the server-side check in
   `src/validation.py` independently re-validates extension, MIME type,
   decodability and minimum dimensions, so a renamed or corrupt file is rejected
   with a readable message.
2. **Display preprocessing** — `src/preprocessing.preprocess_image()` decodes the
   bytes with OpenCV (`BGR`), converts to `RGB`, resizes to **224×224**
   (`INTER_AREA`), scales pixels from 0–255 to **0–1** (`float32`) and adds a
   batch dimension, producing a `(1, 224, 224, 3)` array. This array is shown to
   the user; it is **not** what the network sees.
3. **Model input** — `src/inference.preprocess_for_inference()` builds the tensor
   the network actually consumes: the training-time evaluation transforms
   (resize + centre crop, RGB-to-tensor) followed by **ImageNet mean/std
   normalisation**, giving shape **`(1, 3, 224, 224)`**. This is deterministic —
   no random augmentation at inference time.
4. **Model** — the EfficientNet-B0 classifier is rebuilt with
   `src.training.build_model(model_name, pretrained=False)` so that **all
   parameters come exclusively from the checkpoint**; pretrained weights are
   never downloaded at inference time. Weights are loaded with
   `torch.load(..., map_location="cpu", weights_only=True)` and moved to CPU,
   then set to `eval()` mode.
5. **Prediction** — a single forward pass under `torch.no_grad()` produces
   logits; `softmax(dim=1)` gives per-class probabilities; `argmax` gives the
   class. The label mapping is fixed and shared with training
   (`config.py`): **`real = 0`, `fake = 1`**.
6. **Result** — the app shows the predicted class, the **model confidence**
   (the softmax probability of the predicted class), both class probabilities,
   and the checkpoint path. A `st.progress` bar visualises the confidence.

Nothing is threshold-tuned, and no value is rounded before it is calculated.
The class decision is a plain `argmax` at 0.5 by construction.

### Failure handling

| Situation | Behaviour |
| --- | --- |
| No file selected | Friendly warning, analysis does not start |
| Unsupported extension (e.g. `.txt`) | `⚠` error naming the accepted formats |
| Corrupt / undecodable file | `⚠` error stating the file is not a readable image |
| Image too small | `⚠` error stating the minimum size |
| Preprocessing failure | `⚠` error with the underlying reason |
| Checkpoint missing | `⚠` error **plus** the exact path where the file belongs |
| PyTorch missing | `⚠` error explaining that `requirements.txt` must be installed |
| Checkpoint corrupt / wrong architecture | `⚠` error; no prediction is shown |
| Unexpected exception | Generic `⚠` message in the UI; the full traceback stays in the terminal log where it is needed for debugging |

The application never falls back to a random or placeholder prediction.

---

## Experimental setup

> The `Dataset`, `Split` and `Model` subsections below describe **Experiment 1**.
> The setup for the separate Experiment 2 is given in
> [Experiment 2 setup](#experiment-2-setup) at the end of this section.

### Dataset

A **20-video FaceForensics++ experiment** (compression `c23`): 10 original
(`real`) and 10 `Deepfakes`-manipulated (`fake`) videos. **30 deterministically
sampled frames** per video (evenly spaced) give the working set below.

| Class | Count |
| --- | --- |
| Real frames | 300 |
| Fake frames | 300 |
| **Total** | **600** |

### Split

A **leak-free, group-aligned, seeded** split (`--group-aligned`, seed `42`).
Every *identity component* — the connected group of the FaceForensics
target/source graph, e.g. `183`, `183_253`, `253_183` — is kept inside
**exactly one** split across both classes, so no source sequence can leak
between train, validation and test.

| Split | Real | Fake | Total |
| --- | --- | --- | --- |
| **Train** | 180 | 180 | **360** |
| **Validation** | 60 | 60 | **120** |
| **Test** | 60 | 60 | **120** |

The test split was used for **evaluation only** and never during training or
model selection.

### Model

| Setting | Value |
| --- | --- |
| Architecture | **EfficientNet-B0** (via `torchvision.models`) |
| Initialisation | ImageNet-pretrained, transfer learning |
| Framework | PyTorch 2.14.0 (CPU) |
| Loss | Cross-entropy |
| Optimiser | AdamW, lr `3e-4`, weight decay `1e-4` |
| Schedule | `ReduceLROnPlateau` |
| Batch size | 8 |
| Max epochs | 10 (early stopping, patience 3) |
| Seed | 42 |
| Device | CPU |
| Augmentation | Random horizontal flip + colour jitter (train only) |
| Best checkpoint | **Epoch 6** (lowest validation loss) |
| **Training time** | **52 min 28 s** |

### Experiment 2 setup

A separate experiment on an independently sourced dataset. **It changes the
dataset, not just its size** — see
[Limitations](#limitations).

#### Dataset

| Property | Value |
| --- | --- |
| Source | Kaggle `xhlulu/140k-real-and-fake-faces` (v2), 140,000 images |
| Real class | FFHQ photographs (NVIDIA), CC BY-NC-SA 4.0 |
| Fake class | **StyleGAN-generated** faces — *not* FaceForensics++ manipulations |
| Selected subset | **6,000** images (3,000 real + 3,000 fake), 164.7 MB |
| Image properties | all JPEG, RGB, 256×256 |
| Exact duplicates (MD5) | 0 removed |
| **Near-duplicates removed** | **5,069** (64-bit pHash, threshold 3 bits) |
| Unreadable source images | 0 |
| Build audits | **6 of 6 passed** |

#### Split

Deterministic and seeded (`seed: 42`). Experiment 2 inherits the publisher's
official `train`/`valid`/`test` folders, so no image ever crosses that boundary.

| Split | Real | Fake | Total | Per-class quota |
| --- | --- | --- | --- | --- |
| Train | 2,100 | 2,100 | **4,200** | 2,100 |
| Validation | 450 | 450 | **900** | 450 |
| Test | 450 | 450 | **900** | 450 |

Every split is class-balanced. Minimum cross-split pHash distances were **4, 4 and
6 bits** against the 3-bit threshold.

#### Model

| Setting | Value |
| --- | --- |
| Architecture | **EfficientNet-B0** (via `torchvision.models`) |
| Initialisation | ImageNet-pretrained, transfer learning |
| Framework | PyTorch 2.14.0 (CPU) |
| Loss | Cross-entropy |
| Learning rate / weight decay | `3e-4` / `1e-4` |
| Batch size | 8 |
| Epochs | 10 (early stopping, patience 3 — **not triggered**) |
| Seed | 42 |
| Device | CPU |
| Best checkpoint | **Epoch 10** (lowest validation loss) |
| **Training time** | **600.47 minutes** (≈ 10.0 hours) |
| Dataset fingerprint | `cafc0173…4f3d8bb` (SHA-256, verified at training start) |

---

## Official results

This project reports **two separate experiments**. Each has its own dataset, its
own checkpoint and its own held-out test split. They are presented side by side,
and **neither replaces the other**.

| | Experiment 1 | Experiment 2 |
| --- | --- | --- |
| Status | **Frozen academic baseline** | Complete, larger/differently distributed dataset |
| Headline | 75.00% test accuracy | 97.56% test accuracy |

> **Read the comparison with care.** Experiment 2 is **not** a controlled
> dataset-size-only experiment. The two differ in data amount, fake-generation
> method, framing, resolution, compression and test-set size simultaneously, so
> the difference between them **cannot be attributed to dataset size alone**. See
> [Experiment 1 vs Experiment 2](#experiment-1-vs-experiment-2) and
> [Limitations](#limitations).

### Experiment 1 — Frozen Baseline

> These are the **frozen, official experimental results** of this project. They
> are produced by the committed checkpoint on the 120-image held-out test split
> and are preserved unchanged. The machine-readable source of truth is
> [`reports/evaluation_report.json`](reports/evaluation_report.json); the full
> training log is [`reports/stage6/train_run_log.txt`](reports/stage6/train_run_log.txt).

| Metric | Value |
| --- | --- |
| **Best validation accuracy** | **78.33%** (120 images) |
| **Official test accuracy** | **75.00%** (120 images) |
| **Test set size** | **120 images** (60 real + 60 fake) |
| **Best checkpoint** | **Epoch 6** |
| **Training time** | **52 min 28 s** |

#### Test confusion matrix

```
[[60,  0],
 [30, 30]]
```

Rows = true class, columns = predicted class, order `[real, fake]`.

```
                 predicted
                 real    fake
   true real  [   60       0  ]
   true fake  [   30      30  ]
```

#### Derived metrics

| Metric | Value |
| --- | --- |
| **False positives** (real called fake) | **0** |
| **False negatives** (fake called real) | **30** |
| **Fake precision** | **100%** |
| **Fake recall** | **50%** |
| Fake F1 | 0.6667 |
| Real precision | 66.67% |
| Real recall | 100% |
| Real F1 | 0.8000 |
| Macro precision / recall / F1 | 0.8333 / 0.7500 / 0.7333 |

`fake` is treated as the **positive class**, because detecting manipulated
content is the purpose of the project.

#### How to read this result

The model is **conservative**: it flagged **no** genuine frame as fake, so its
precision on the fake class is perfect, but it let **half** of the manipulated
frames (30 of 60) pass as real. This is a real and honest weakness of a small
model trained on only 360 images, and it is reported here rather than hidden.

> A single demo prediction in the running application is **not** an official
> evaluation result and is never presented as one.

### Experiment 2 — Larger, Differently Distributed Dataset

> These are the **official Experiment 2 results**. They come from the committed
> Experiment 2 checkpoint on the **900-image** held-out test split, which was not
> used for training or model selection. Machine-readable sources of truth:
> [`experiments/experiment_2_large_dataset/reports/evaluation_report.json`](experiments/experiment_2_large_dataset/reports/evaluation_report.json),
> [`…/training_summary.json`](experiments/experiment_2_large_dataset/training_summary.json)
> and [`…/dataset_report.json`](experiments/experiment_2_large_dataset/dataset_report.json).
> The full written report is
> [`reports/experiment_2_report.md`](reports/experiment_2_report.md).

| Metric | Value |
| --- | --- |
| **Dataset** | Kaggle `xhlulu/140k-real-and-fake-faces` (v2) — FFHQ photographs vs **StyleGAN-generated** fakes |
| **Number of images** | **6,000** (3,000 real + 3,000 fake), selected from a 140,000-image pool |
| **Train / validation / test split** | **4,200 / 900 / 900** (70% / 15% / 15%, class-balanced in every split) |
| Best validation accuracy | 96.89% (900 images) — *checkpoint-selection signal, **not** the reported result* |
| **Official test accuracy** | **97.56%** (900 images) |
| **Macro F1** | **97.55%** |
| **Test set size** | **900 images** (450 real + 450 fake) |
| **Confusion matrix** | `[[430, 20], [2, 448]]` |
| **Best checkpoint** | **Epoch 10** |
| **Training time** | **600.47 minutes** (≈ 10.0 hours, CPU) |
| **Checkpoint file** | `models/experiment_2_deepguard_efficientnet_b0.pt` |
| De-duplication | MD5 + 64-bit pHash (3-bit threshold); **5,069** near-duplicates removed; **6 of 6** build audits passed |

> **97.56% is Experiment 2's TEST accuracy**, measured on 900 held-out images the
> model never saw during training. It is not a validation figure — the validation
> accuracy was 96.89% and was used only to select the best checkpoint.

#### Test confusion matrix

```
[[430,  20],
 [  2, 448]]
```

Rows = true class, columns = predicted class, order `[real, fake]`.

```
                  predicted
                  real    fake
    true real  [  430      20  ]
    true fake  [    2     448  ]
```

#### Derived metrics

| Metric | Value |
| --- | --- |
| **False positives** (real called fake) | **20** |
| **False negatives** (fake called real) | **2** |
| **Fake precision** | **95.73%** |
| **Fake recall** | **99.56%** |
| Fake F1 | 0.9760 |
| Real precision | 99.54% |
| Real recall | 95.56% |
| Real F1 | 0.9751 |
| Macro precision / recall / F1 | 0.9763 / 0.9756 / 0.9755 |

`fake` is treated as the **positive class**, because detecting manipulated
content is the purpose of the project.

#### How to read this result

The Experiment 2 model is **mildly biased towards calling images fake**: it
misses only 2 of 450 synthetic images, at the cost of 20 false alarms on 450
genuine ones. Experiment 1 failed in the opposite direction — it flagged **zero**
real images as fake and missed **half** its fakes (30 of 60), which is why its
`100%` fake precision is an artifact of barely attempting any fake prediction.

**This is not a perfect detector, and it does not generalise to all deepfakes.**
It was trained and tested on one dataset containing one style of synthetic face.
The result is bounded by the caveats in [Limitations](#limitations).

> A single demo prediction in the running application is **not** an official
> evaluation result and is never presented as one.

### Experiment 1 vs Experiment 2

| Metric (test split) | Experiment 1 — Frozen Baseline | Experiment 2 |
| --- | --- | --- |
| Test images | 120 | 900 |
| Real / fake in test | 60 / 60 | 450 / 450 |
| **Test accuracy** | **75.00%** | **97.56%** |
| Macro F1 | 0.7333 | 0.9755 |
| Confusion matrix | `[[60, 0], [30, 30]]` | `[[430, 20], [2, 448]]` |
| Fake recall | 50% | 99.56% |
| Real recall | 100% | 95.56% |
| Checkpoint | `models/deepguard_efficientnet_b0.pt` | `models/experiment_2_deepguard_efficientnet_b0.pt` |
| Best epoch | 6 | 10 |
| Training time | 52 min 28 s | 600.47 min |

**This table describes two separate results. It is not a controlled experiment,
and the difference between the columns cannot be attributed to dataset size.**

Experiment 1 and Experiment 2 differ in **six dimensions at once**:

| # | Dimension | Experiment 1 | Experiment 2 |
| --- | --- | --- | --- |
| 1 | Data amount | 600 images (360 train) | 6,000 images (4,200 train) |
| 2 | **Fake-generation method** | FaceForensics++ **face-swap manipulation** | **StyleGAN synthesis** |
| 3 | Image framing / distribution | centre-cropped widescreen frames | full square 256×256 images |
| 4 | Source resolution | 1280×720, 1920×1080, 856×480, 656×480 | 256×256 |
| 5 | Compression characteristics | `c23`-compressed video stored as PNG | JPEG compression / quantisation |
| 6 | **Test-set size** | **120 images** | **900 images** |

Dimension 2 matters most. A face-swap inherits the identity, pose and lighting of
a real source photograph, so its artefacts are blending and boundary
inconsistencies layered onto genuine sensor noise. A StyleGAN image is synthesised
end to end, so its artefacts are a completely different signature. A detector's
difficulty can differ sharply between those families **on its own**, regardless of
how much training data it was given.

Dimension 6 matters too: Experiment 1's figure rests on 120 images and
Experiment 2's on 900, so the two accuracies are **not equally precise**.

The dataset build records the same caveat in its own artifact:

> *"Experiment 1 fake class = FaceForensics++ face-swap manipulation. Experiment 2
> fake class = StyleGAN synthesis. Dataset size AND fake-image distribution both
> change, so Experiment 2 is NOT a controlled dataset-size experiment and must
> never be described as one."*

**The correct conclusion, and only this one:**

> Experiment 2 achieved higher test performance than Experiment 1 under its
> specified larger-dataset and independently sourced data conditions, but the
> design does not permit attributing that difference to dataset size alone. It
> demonstrates the performance of this particular larger, independently sourced
> dataset/model setup; it does **not** isolate dataset size as a causal variable.

**Statements that must not be made about this project:**

- ~~"Accuracy improved because the dataset was larger."~~
- ~~"The 10× larger dataset improved detection."~~
- Any wording presenting Experiment 2 as a controlled demonstration of a
  dataset-size effect.

---

## Reproducibility

The following is preserved in this repository so that the experiments can be
audited, verified and reviewed without being re-run. Items 1–7 are **Experiment 1**
evidence; item 8 is **Experiment 2** evidence.

1. **The trained checkpoint** — `models/deepguard_efficientnet_b0.pt` is the
   authoritative, unmodified best-epoch-6 model. It is committed with ordinary
   Git, so `git clone` alone is enough to obtain it.
2. **The full training log** — `reports/stage6/train_run_log.txt` records every
   epoch's training and validation loss and accuracy, the early-stopping trigger,
   the best epoch, and the seed.
3. **The machine-readable evaluation** — `reports/evaluation_report.json` holds
   the checkpoint epoch, the PyTorch version, the device, and the complete
   metrics block (confusion matrix, per-class and macro precision/recall/F1).
4. **Plots** — `reports/stage6/training_curves.png`,
   `reports/stage6/confusion_matrix.png`, `reports/stage6/per_class_metrics.png`,
   regenerable with `reports/stage6/generate_stage6_plots.py`.
5. **The dataset methodology** — `docs/dataset_research.md` documents the dataset
   choice, `docs/evaluation_plan.md` the evaluation design, and
   `docs/training_plan.md` the training design. The split is seeded (`42`) and
   group-aligned, so `data/` is deterministically rebuildable (see
   [`data/README.md`](data/README.md)).
6. **The provenance manifest** — the extraction step writes
   `data/raw/frames_provenance.json` recording which frame came from which video
   and component.
7. **The source videos** — the original 20 FaceForensics++ videos are never
   modified by the pipeline. They are excluded from Git because the dataset is
   restricted and research-use-only.
8. **Experiment 2 evidence** — committed separately from Experiment 1 and stored
   under `experiments/experiment_2_large_dataset/`:
   - `models/experiment_2_deepguard_efficientnet_b0.pt` — the epoch-10
     Experiment 2 checkpoint, committed with ordinary Git.
   - `experiments/experiment_2_large_dataset/dataset_report.json` — source,
     quotas, seed, pHash threshold, de-duplication counts and the six passing
     build audits.
   - `experiments/experiment_2_large_dataset/dataset_manifest.csv` — 6,000 rows
     recording image id, label, split, official split, source file, byte size,
     MD5 and pHash. Contains **no absolute paths and no pixels**.
   - `experiments/experiment_2_large_dataset/training_summary.json` —
     hyperparameters, epochs completed, best epoch, wall-clock duration and the
     **dataset fingerprint** that ties the checkpoint to this exact dataset.
   - `experiments/experiment_2_large_dataset/reports/evaluation_report.json` —
     the official Experiment 2 test metrics block.
   - `experiments/experiment_2_large_dataset/reports/*.png` — confusion matrix,
     per-class metrics and the training loss / accuracy curves.
   - `reports/experiment_2_report.md` — the full written Experiment 2 report,
     including the complete scientific limitation.
   - Not committed: the ~4 GB Kaggle source dataset and the 6,000 selected
     images (CC BY-NC-SA 4.0, not redistributable). They are reproducible via
     `build_dataset.py`.

**Integrity statement.** No retraining, checkpoint replacement, dataset change,
split change, label change, synthetic data, threshold tuning on the test set, or
result fabrication was performed in preparing this repository. The official
numbers above are exactly those produced by the original run.

This statement covers **both** experiments. Experiment 1's artifacts were never
modified when Experiment 2 was added: `reports/evaluation_report.json`,
`models/deepguard_efficientnet_b0.pt`, the Experiment 1 dataset and its
360/120/120 split are byte-identical to their original state, and Experiment 2
writes to its own `experiments/experiment_2_large_dataset/reports/` directory.
Reproducing Experiment 2 additionally requires re-downloading the Kaggle dataset
and roughly ten hours of CPU training.

---

## Limitations

Stated plainly, because a result is only as useful as its honest framing.

### Experiment 1 — Frozen Baseline limitations

- **Small, single-source test set.** The 75.00% figure comes from this project's
  own **120-image held-out test set**, drawn from **20 FaceForensics++ videos**
  with the `c23` compression setting. It must **not** be generalised to all
  real-world deepfake content.
- **A single dataset and a single forgery type.** Only FaceForensics++
  `Deepfakes` manipulations were used. Other generators, techniques,
  compressions and post-processing are untested.
- **Frames, not videos.** Classification is per-frame and independently
  performed; no temporal consistency is exploited, even though deepfakes are
  inherently a video problem.
- **Only JPG/JPEG/PNG input.** No video, no audio, no animated formats.
- **Poor fake recall (50%).** Half of the manipulated frames were classified as
  real. The model errs toward "real", which is unsafe in a forensic context.
- **Fixed 0.5 decision threshold.** Chosen by `argmax` at construction; it was
  never tuned, deliberately, to keep the test set free of contamination.
- **Tiny training set.** 360 images is far below what modern detectors use;
  the number of source identities (10 per class) is the real bottleneck.
- **No calibration.** The displayed confidence is a raw softmax score, not a
  calibrated probability, and must not be read as "97% sure".
- **Not deployed.** CPU-only, single-image, local application. Not production
  software, and not suitable for real-world forensic judgments.
- **CPU-only inference.** Feasible for a prototype demo; not for real-time
  video streams.

### Experiment 2 limitations

These apply **in addition to** the Experiment 1 limitations above, which continue
to constrain the project as a whole.

- **Experiment 2 is NOT a controlled dataset-size-only experiment.** The
  difference between Experiment 1 and Experiment 2 **cannot be attributed solely
  to the increase from 600 to 6,000 images**, because multiple dataset
  characteristics changed simultaneously: data amount; fake-generation method
  (FaceForensics++ manipulated/deepfake faces vs StyleGAN-generated fake faces);
  image framing and distribution; source resolution; compression characteristics;
  and test-set size (120 vs 900). This is the single most important caveat in this
  README.
- **Different forgery family.** Experiment 2's fakes are StyleGAN generations,
  not face-swap manipulations. The model has therefore been evaluated against a
  synthetic-image signature it was built for, and says nothing about performance
  on face-swap, face-reenactment or any other manipulation family.
- **No identity metadata.** The Experiment 2 dataset publishes **no identity
  metadata**, so identity-level grouping is **not possible**. Leakage control
  rests on the publisher's official split plus de-duplication, not on
  person-level separation. Experiment 1 *had* identity-component grouping;
  Experiment 2 does not. **This makes Experiment 1's split stricter, not
  Experiment 2's.**
- **JPEG / compression artefacts.** All Experiment 2 images are JPEG and carry
  JPEG compression and quantisation artefacts, where Experiment 1's frames were
  `c23`-compressed video stored losslessly as PNG. Part of the measured
  difference may reflect compression signatures rather than better modelling.
- **Threshold-relative near-duplicate margins.** The minimum cross-split pHash
  distances were **4, 4 and 6 bits** against a **3-bit** threshold. All audits
  passed, but two of those margins are a single bit, so the no-leakage guarantee
  is *threshold-relative* rather than comfortable. A stricter threshold could
  have flagged borderline pairs.
- **Best checkpoint at the final epoch — no evidence of a ceiling.** The
  epoch-10 checkpoint had the lowest validation loss, and validation loss was
  still falling when the 10-epoch budget ran out. The model was probably **not
  fully converged**, so **97.56% must not be read as a ceiling** on this
  architecture; a longer run could move it in either direction.
- **Larger but still single-source test set.** The 97.56% figure comes from 900
  images drawn from one publisher's test folders. It must **not** be generalised
  to all real-world deepfake content, and it does **not** demonstrate
  generalisation to all deepfakes.
- **Frames, not videos.** As with Experiment 1, classification is per-frame; no
  temporal consistency is exploited.
- **Fixed 0.5 decision threshold.** As with Experiment 1, chosen by `argmax` at
  construction and never tuned on the test set.
- **No calibration.** As with Experiment 1, the displayed confidence is a raw
  softmax score, not a calibrated probability, and must not be read as
  "97% sure".

---

## Future work

Ideas only. **None of these have been implemented**, and none of them were used
to produce the official results above.

The first group follows directly from the Experiment 2 caveats:

- **Evaluation on further independently sourced data** — test the current
  checkpoint against additional datasets it was not built for, to measure how far
  its 97.56% travels beyond the distribution it was trained on.
- **A genuine controlled dataset-size study** — hold the fake-generation method,
  framing, resolution, compression and test-set size **all fixed**, then vary only
  the training sample count. This is the only design that could attribute a
  performance difference to dataset size. Experiment 2 is **not** that study and
  must not be cited as one.
- **Broader manipulation types** — face-swap, face-reenactment, talking-head and
  non-face synthesis, so the model is not assessed only against the one forgery
  family it was trained on.
- **Stronger source and identity metadata** — where a dataset publishes identity
  or subject information, apply identity-level group splitting as Experiment 1
  did, so leakage control does not have to rest on de-duplication alone.
- **Longer training with a proper convergence check** — Experiment 2's best
  checkpoint landed on its final epoch, so the 97.56% figure is not a known
  ceiling. Re-run with a larger epoch budget and an explicit stopping criterion.
- **Distribution-shift reporting** — evaluate across compression levels,
  resolutions and post-processing, and report degradation rather than a single
  headline number.

Then the existing broader roadmap:

- **Enlarge and diversify the training data** — more identities, more forgery
  methods, multiple compression levels; the single largest expected gain.
- **Add stronger backbones** (EfficientNet-B2/B3, ConvNeXt-T, ViT) and compare
  them under the same protocol.
- **Exploit temporal information** — video-level or clip-level models using
  optical flow, landmark jitter and heartbeat/physiological cues.
- **Learn better representations** — self-supervised pretraining on a large
  unlabelled face corpus, then fine-tuning.
- **Frequency-domain and artefact cues** — high-frequency noise residuals, JPEG
  compression traces, blending-boundary detection.
- **Improve the decision threshold and calibration** — chosen on the
  *validation* split, never on the test set, with probability calibration
  (temperature scaling) and an explicit cost-sensitive operating point.
- **Target the false-negative problem** directly, e.g. class-weighted loss,
  focal loss, or hard-negative mining on the 30 misclassified fake frames.
- **Robustness evaluation** — cross-dataset testing, adversarial perturbations,
  compression and rescaling stress tests, and a proper ROC/PR analysis.
- **Extend to audio and video** — lip-sync, voice-clone and temporal artefacts.
- **Add explainability** — saliency/Grad-CAM overlays so a reviewer can see
  *where* the model looked.
- **Deployment** — optional GPU path and a containerised service, if the
  project scope ever allows it.

---

## Academic project status

| Component | Status |
| --- | --- |
| **Experiment 1** | **Complete — frozen academic baseline.** 75.00% test accuracy on 120 held-out images. Checkpoint, evaluation report, dataset and split are unmodified. |
| **Experiment 2** | **Complete — larger, differently distributed dataset experiment.** 97.56% test accuracy on 900 held-out images. Build, training and evaluation artifacts committed. |
| **Streamlit application** | **Working, serving the Experiment 2 checkpoint** by default. Both checkpoints remain in the repository. |
| **Documentation** | **Being updated** to cover both experiments. `reports/experiment_2_report.md` added; this README updated. |

> **Both experiments are complete and their results are frozen.** No retraining,
> re-evaluation, threshold tuning or result regeneration was performed while
> documenting Experiment 2.

Experiment 2 was a **supplementary** experiment: it evaluates the same
architecture under different dataset conditions. It does not invalidate or replace
Experiment 1, which remains the frozen baseline of this project.

**When reporting results, always state which experiment is being quoted**, and
never present the difference between them as a controlled dataset-size effect.

---

## Testing

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

| Test file | Covers |
| --- | --- |
| `tests/test_pipeline.py` | Validation and preprocessing |
| `tests/test_dataset.py` | Dataset scanning, splitting, copying, summary |
| `tests/test_extraction.py` | Deterministic frame extraction and provenance |
| `tests/test_model_interface.py` | Truthful availability reporting; never a prediction |
| `tests/test_training.py` | Dataset readiness; PyTorch tests auto-skip if absent |
| `tests/test_evaluation.py` | Metric maths; PyTorch tests auto-skip if absent |
| `tests/test_inference.py` | Honest refusal without PyTorch/checkpoint; real output when present |
| `tests/test_app.py` | Streamlit end-to-end via `AppTest`, running the **real trained model** |

The tests build their own small images in temporary folders, so they do not
require the real dataset.

---

## Dataset

**No dataset is committed to this repository.**

- `data/faceforensics/` holds the original FaceForensics++ videos. That dataset
  is **restricted and research-use-only** and must not be redistributed here.
- `data/raw/` and `data/processed/` hold ~793 MB of derived frames. They are
  deterministically rebuildable and are not needed for inference.
- `experiments/experiment_2_large_dataset/source/` would hold the ~4 GB Kaggle
  *140k Real and Fake Faces* dataset. Its photographs are **CC BY-NC-SA 4.0**
  (non-commercial, attribution required) and must not be redistributed here.
- `experiments/experiment_2_large_dataset/processed/` holds the 6,000 selected
  Experiment 2 images. They are deterministically rebuildable with
  `build_dataset.py` (seed `42`) and are not needed for inference.

The Streamlit application runs real predictions **without** any of this, because
both trained checkpoints are committed.

See [`data/README.md`](data/README.md) for the expected layout and the exact
rebuild commands, and [`docs/dataset_research.md`](docs/dataset_research.md) for
the dataset comparison and access procedure.

---

## License and attribution

- **FaceForensics++** — the dataset used in Experiment 1, by its original
  authors, under **its own research-only licence and terms of use**. It is *not*
  redistributed in this repository.
- **Kaggle `xhlulu/140k-real-and-fake-faces` (v2)** — the dataset used in
  Experiment 2, distributed under Kaggle's "Other (specified in description)"
  terms.
- **FFHQ (NVIDIA)** — the source of the real photographs in the Experiment 2
  dataset, under **CC BY-NC-SA 4.0**: non-commercial use with attribution. Its
  images are *not* redistributed in this repository.
- **EfficientNet-B0** pretrained weights — from `torchvision` / the original
  authors, under their own licences.
- The project source code, documentation and the trained checkpoints in this
  repository are the work of the DeepGuard team.

> **Academic disclaimer.** DeepGuard is a student prototype developed for
> academic purposes. It is not a production system and must not be used to make
> real-world judgments about the authenticity of media.

---

<div align="center">

**DeepGuard** — Deepfake Technology (24CGCS06) — Atria Institute of Technology

</div>
