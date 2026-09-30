# DeepGuard - Training Plan (Stage 4)

Project: **Deepfake Technology** (24CGCS06)
Stage: **Stage 4 - Model Training (planned design)**
Architecture: **EfficientNet-B0** (alternative: ResNet-50)

> **This document is a PLAN, not a report of results.**
> No model has been trained. Every value below is a **planned, configurable
> setting**, not an experimentally optimised one. No accuracy, loss, precision,
> recall, F1-score or confusion matrix is claimed anywhere. The actual values
> can only be produced by running `train_model.py` on a real prepared dataset.

---

## 1. Goal

Fine-tune an ImageNet-pretrained EfficientNet-B0 to classify a single face
image as **real** or **fake**, using the dataset prepared by the Stage 2 tools,
without touching the Streamlit application environment.

---

## 2. Data

### 2.1 Source
The processed dataset produced by `prepare_dataset.py`:

```
data/processed/
├── train/{real,fake}
├── validation/{real,fake}
└── test/{real,fake}
```

Training reads **only** `train` and `validation`. The `test` split is
**reserved for Stage 5 evaluation** and is never used to tune the model.

### 2.2 Classes and label mapping
Two classes, with a fixed order so that training and future inference agree:

| Class | Index |
| --- | --- |
| `real` | 0 |
| `fake` | 1 |

This mapping is defined once in `config.py` (`LABEL_TO_INDEX`) and is applied
explicitly. TorchVision's `ImageFolder` is deliberately **not** used to decide
labels, because it sorts folder names alphabetically (`fake` before `real`)
and would silently swap the indices.

### 2.3 Input size
224 x 224 x 3, matching the Stage 1 preprocessing and the native EfficientNet-B0
input.

### 2.4 Normalisation - an important consistency point
- Stage 1 (`src/preprocessing.py`) scales pixels to **0-1** for display.
- ImageNet-pretrained weights expect **ImageNet mean/std** normalisation
  (`mean = (0.485, 0.456, 0.406)`, `std = (0.229, 0.224, 0.225)`).

Therefore the training pipeline normalises with the ImageNet statistics. When
inference is integrated in Stage 6, the **same** normalisation must be applied
there, otherwise the model receives a differently scaled input than it was
trained on. This requirement is recorded here so it cannot be forgotten.

### 2.5 Augmentation (training split only)
Kept deliberately **modest**, because aggressive augmentation can destroy the
subtle forensic artifacts that distinguish real from fake faces:

| Transform | Setting | Why |
| --- | --- | --- |
| `RandomResizedCrop` | 224, scale 0.8-1.0 | small scale/position variation |
| `RandomHorizontalFlip` | p = 0.5 | faces are roughly left-right symmetric |
| `RandomRotation` | +/- 10 degrees | small orientation variation |

Deliberately **not** used by default: strong colour jitter, blur, noise
injection, cutout/erasing, and vertical flips. These can erase or fabricate
manipulation traces and are not justified for a first prototype.

The **validation** and **test** splits receive only deterministic processing
(resize / centre-crop to 224, tensor conversion, ImageNet normalisation) and
**no random augmentation**.

---

## 3. Model

- **Backbone:** EfficientNet-B0 with ImageNet-pretrained weights
  (`torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1`).
- **Head:** the final `Linear(1280 -> 1000)` is replaced with
  `Linear(1280 -> 2)`.
- **Output:** two logits, interpreted as class `real` (index 0) and `fake`
  (index 1).
- **Alternative:** ResNet-50 (`models.resnet50`, replace `model.fc`), selectable
  from the command line for comparison.
- **Fallback:** `--no-pretrained` builds the same architecture with random
  initialisation (useful for tests and for offline environments).

The model is **not** described as a trained deepfake detector until real
training has actually happened.

---

## 4. Training configuration (planned, configurable)

All values live in `config.py` and can be overridden from the command line.
They are **planned defaults, not optimised values**.

| Setting | Planned value | Note |
| --- | --- | --- |
| `IMAGE_SIZE` | (224, 224) | matches Stage 1 |
| `NUM_CLASSES` | 2 | `real`, `fake` |
| `MODEL_NAME` | `efficientnet_b0` | primary |
| `ALT_MODEL_NAME` | `resnet50` | alternative |
| `BATCH_SIZE` | 16 | conservative for CPU / ~16 GB RAM |
| `LEARNING_RATE` | 3e-4 | AdamW |
| `WEIGHT_DECAY` | 1e-4 | AdamW |
| `EPOCHS` | 10 | configurable; not claimed optimal |
| `EARLY_STOPPING_PATIENCE` | 3 | on validation loss |
| `NUM_WORKERS` | 0 | Windows + CPU: avoid worker-process overhead |
| `RANDOM_SEED` | 42 | reproducibility |
| `CHECKPOINT_DIR` | `models/` | best checkpoint only |

---

## 5. Loss, optimizer and schedule

- **Loss:** `CrossEntropyLoss` over the two logits.
- **Optimizer:** `AdamW(learning_rate=3e-4, weight_decay=1e-4)`.
- **Schedule:** `ReduceLROnPlateau` monitoring **validation loss**
  (`factor = 0.1`, `patience = 2`). This is the schedule recorded in this plan;
  if the implementation uses a different schedule, this document must be
  updated and the reason stated.
- **Class imbalance:** if one class has far more images, the Stage 2
  `--max-per-class` option keeps the classes balanced. If imbalance remains, a
  weighted loss can be enabled later; this is not enabled by default.

---

## 6. Training loop

For every epoch:

1. **Training phase** - forward pass, loss, backward pass, optimizer step;
   accumulate training loss and training accuracy.
2. **Validation phase** - no gradient updates; accumulate validation loss and
   validation accuracy.
3. **Record** the epoch results and print a readable summary:

```
Epoch 1/N
Train Loss: <actual value>
Train Accuracy: <actual value>
Validation Loss: <actual value>
Validation Accuracy: <actual value>
```

The values are produced by execution only. They are never written in advance.

Training accuracy and validation accuracy are recorded as **optimisation
signals**, not as the project's reported performance. Final metrics belong to
Stage 5 and are computed on the untouched test set.

---

## 7. Checkpointing

- The **best** model is saved according to the validation criterion
  (**lowest validation loss**).
- Location: `models/deepguard_efficientnet_b0.pt`.
- The file stores a dictionary: `model_state_dict`, architecture name, number
  of classes, class names, epoch, the validation loss/accuracy at that epoch,
  the seed, and the library versions used.
- **Only real training produces a checkpoint.** No placeholder `.pt`/`.pth`
  file is ever created. `models/` stays empty until training genuinely runs.
- The `test` split plays no part in checkpoint selection.

---

## 8. Early stopping

- **Criterion:** stop when validation loss has not improved for
  `EARLY_STOPPING_PATIENCE` consecutive epochs.
- The best checkpoint is the one with the lowest validation loss seen so far.
- Early stopping is only reported as having occurred if execution actually
  demonstrates it.

---

## 9. Avoiding data leakage

Deepfake datasets contain many related images, so a naive random split leaks
information between splits. The plan:

- **Source-video identity:** all frames from the same source video belong to
  one "group".
- **Frames:** frames of one video are near-duplicates; they must never be split
  across train and test.
- **Subject identity:** the same person may appear in several videos. Some
  public datasets (for example FaceForensics++) share identities between the
  original and manipulated sets, so subject-level leakage cannot always be
  eliminated.
- **Near-duplicate images:** frames close in time are visually almost
  identical; group-based splitting keeps them together.
- **Train / validation / test separation:** the existing Stage 2 group-aware
  splitting (`src/dataset.py`, `group_key`) is reused. It groups images by
  their top-level sub-folder, so a whole group stays in a single split.

**Honest limitation:** this is **not** claimed to be perfect leakage
prevention. It prevents the most obvious and most damaging leakage (identical
frames in train and test) but cannot guarantee the absence of all statistical
relatedness. If images are stored flat with no sub-folders, each image becomes
its own group and the protection is weaker; preserving sub-folders is
therefore recommended.

---

## 10. Reproducibility

- A single seed (`RANDOM_SEED = 42`) is applied to Python `random`, NumPy and
  PyTorch.
- DataLoader shuffling and the train/validation split are seeded.
- Because the system is **CPU-based**, no GPU non-determinism is involved.
  However, **perfect bit-for-bit reproducibility across different
  environments, library versions or CPU instruction sets is not claimed**.

---

## 11. Training environment (CPU-only)

- **Separate environment:** `venv-train`, so the Streamlit application
  environment (`venv`) is never modified. This is required because PyTorch is
  large and the app must stay runnable at all times.
- **Framework:** PyTorch 2.14.0 + TorchVision 0.29.0, CPU wheels
  (verified `cp314` win_amd64 wheels; `torchvision 0.29.0` pins `torch==2.14.0`).
- **No CUDA packages** are installed: the machine has no NVIDIA GPU.
- **CPU training is possible** for a small subset (the Stage 2 default is 500
  images per class) but is expected to be slow. **No training-time estimate is
  given here**, because it depends on the real dataset size and has not been
  measured.
- **Memory:** ~16 GB RAM; batch size 16 at 224 x 224 is comfortably within
  budget. Batch size is kept conservative to avoid memory pressure.
- **Storage:** checkpoints are small (tens of MB); the project disk has ample
  free space. Dataset storage depends on the dataset chosen in Stage 2.
- **Workers:** `NUM_WORKERS = 0` on Windows to avoid process-spawn overhead and
  excessive multiprocessing on a 4-core CPU.

---

## 12. What Stage 4 does NOT do

- Does not download a dataset.
- Does not evaluate the test set (reserved for Stage 5).
- Does not report accuracy, precision, recall, F1-score or a confusion matrix.
- Does not integrate prediction into the Streamlit app (Stage 6).
- Does not create any model file unless real training runs.

---

## 13. Status

Architecture selected and this plan written. **No training has been executed
and no checkpoint exists.** If no prepared dataset is present, `train_model.py`
stops cleanly and states that training cannot be executed.
