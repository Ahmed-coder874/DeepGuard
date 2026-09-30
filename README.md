<div align="center">

# DeepGuard

**AI-Based Deepfake Detection**

Project **Deepfake Technology** &middot; Project ID **24CGCS06**
Department of **ECE** &middot; SDG **10** &middot; **Atria Institute of Technology, Bengaluru**
Project Guide: **Prof. Jayanth U**

`EfficientNet-B0` &middot; `PyTorch` &middot; `Streamlit`

**Official test accuracy: 75.00%** on the 120-image held-out test set

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

> **Responsible AI notice.** Deepfake detection is not a solved problem. A
> prediction produced by this application is an *indication requiring further
> verification*, never proof of authenticity. The system reaches 75.00% accuracy
> on its own 120-image test split and can be wrong.

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
├── models/
│   └── deepguard_efficientnet_b0.pt      # THE authoritative frozen checkpoint
│
├── reports/                      # Frozen experimental evidence
│   ├── DeepGuard_SIP_Final_Technical_Report.md / .pdf
│   ├── DeepGuard_SIP_Presentation.md
│   ├── DeepGuard_Viva_QA.md
│   ├── DeepGuard_Submission_Checklist.md
│   ├── evaluation_report.json            # Official Stage 5 metrics (machine-readable)
│   └── stage6/
│       ├── train_run_log.txt             # Full training log
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
```

Expected: Python `3.14.7`, Streamlit `1.64.0`, and `True`.

---

## Model setup

The model lives at a **project-relative** path that is resolved from the
location of `config.py`, so the application works from any clone location on any
machine. No environment variable or absolute path is required.

```
<project root>
└── models
    └── deepguard_efficientnet_b0.pt
```

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
withheld. The fix is to place the file at
`models/deepguard_efficientnet_b0.pt`.

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

- Say the official result is **75.00% test accuracy on 120 held-out images**.
- Say a demo prediction is a *live single-image model output*, not part of the
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

---

## Official results

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

### Test confusion matrix

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

### Derived metrics

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

### How to read this result

The model is **conservative**: it flagged **no** genuine frame as fake, so its
precision on the fake class is perfect, but it let **half** of the manipulated
frames (30 of 60) pass as real. This is a real and honest weakness of a small
model trained on only 360 images, and it is reported here rather than hidden.

> A single demo prediction in the running application is **not** an official
> evaluation result and is never presented as one.

---

## Reproducibility

The following is preserved in this repository so that the experiment can be
audited, verified and reviewed without being re-run:

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

**Integrity statement.** No retraining, checkpoint replacement, dataset change,
split change, label change, synthetic data, threshold tuning on the test set, or
result fabrication was performed in preparing this repository. The official
numbers above are exactly those produced by the original run.

---

## Limitations

Stated plainly, because a result is only as useful as its honest framing.

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

---

## Future work

Ideas only. **None of these have been implemented**, and none of them were used
to produce the official results above.

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

> **The experiment and the official evaluation are complete.** This repository
> is the **finalised project and demo**. All development stages — dataset
> research, dataset preparation, architecture selection, training, validation,
> official evaluation, inference integration and testing — are finished. No
> further model changes are planned for this submission.

The committed state is the submission state: the frozen checkpoint, the frozen
results, and a working Streamlit demo.

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

The Streamlit application runs real predictions **without** any of this, because
the trained checkpoint is committed.

See [`data/README.md`](data/README.md) for the expected layout and the exact
rebuild commands, and [`docs/dataset_research.md`](docs/dataset_research.md) for
the dataset comparison and access procedure.

---

## License and attribution

- **FaceForensics++** — the dataset used in this project, by its original
  authors, under **its own research-only licence and terms of use**. It is *not*
  redistributed in this repository.
- **EfficientNet-B0** pretrained weights — from `torchvision` / the original
  authors, under their own licences.
- The project source code, documentation and the trained checkpoint in this
  repository are the work of the DeepGuard team.

> **Academic disclaimer.** DeepGuard is a student prototype developed for
> academic purposes. It is not a production system and must not be used to make
> real-world judgments about the authenticity of media.

---

<div align="center">

**DeepGuard** — Deepfake Technology (24CGCS06) — Atria Institute of Technology

</div>
