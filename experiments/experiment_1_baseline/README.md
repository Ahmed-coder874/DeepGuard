# Experiment 1 - Baseline (FROZEN)

**Status: complete and frozen. Nothing in this experiment may be modified,
retrained or overwritten.**

This folder is a pointer to the official results of the original DeepGuard
experiment. It intentionally contains no data and no copies of the checkpoint -
the authoritative artifacts stay where they have always lived, so there is only
ever one copy of each.

## Question

Can an EfficientNet-B0 classifier trained on a small, video-derived dataset tell
manipulated face images apart from real ones?

## Dataset

| Property | Value |
|---|---|
| Source | FaceForensics++ (`c23` compression), original + `Deepfakes` manipulation |
| Videos | 10 total: 5 original + 5 manipulated versions of the same 5 source videos |
| Frames per video | 30, evenly spaced (deterministic) |
| Raw frames | 300 (150 real + 150 fake) |
| Selected images | **600 (300 real + 300 fake)** |
| Split | train 360 / validation 120 / test 120 |
| Leakage control | group-aligned by identity component, audit run before and after copy |

Identity components (a component is the pair of original + manipulated videos
of the same people, and it never spans two splits):

| Split | Identity components | Images |
|---|---|---|
| train | `469_481`, `585_599`, `672_720` | 360 (180 real / 180 fake) |
| validation | `866_878` | 120 (60 real / 60 fake) |
| test | `183_253` | 120 (60 real / 60 fake) |

## Configuration

Values are pinned to what the run actually used, not to later defaults.

| Setting | Value |
|---|---|
| Architecture | `efficientnet_b0` (ImageNet pretrained) |
| Input | 224 x 224 |
| Batch size | 8 |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4 |
| LR schedule | ReduceLROnPlateau, mode `min`, factor 0.1, patience 2 |
| Max epochs | 10, early-stopping patience 3 on validation loss |
| Seed | 42 |
| Device | CPU (`torch 2.14.0+cpu`, CUDA unavailable) |
| Duration | 52 min 28 s |

## Results

Early stopping fired after epoch 9. Best model: **epoch 6**
(validation loss 0.4908, validation accuracy 0.7833).

Test split (120 images, never seen during training):

| Metric | Value |
|---|---|
| Accuracy | **0.7500** |
| Precision (fake) | 1.0000 |
| Recall (fake) | 0.5000 |
| F1 (fake) | 0.6667 |
| Macro F1 | 0.6667 |

Confusion matrix (rows = actual, columns = predicted):

|  | predicted real | predicted fake |
|---|---|---|
| **actual real** | 60 | 0 |
| **actual fake** | 30 | 30 |

The model never produced a false positive on real images, but it detected only
half of the fake ones (30 of 60). On this dataset that is the headline weakness.

## Authoritative artifacts

| Artifact | Path |
|---|---|
| Checkpoint | `models/deepguard_efficientnet_b0.pt` (16,341,163 bytes) |
| Official evaluation report | `reports/evaluation_report.json` |
| Training log | `reports/stage6/train_run_log.txt` |
| Training curves, confusion matrix, per-class plots | `reports/stage6/*.png` |
| Dataset | `data/raw/`, `data/processed/` |
| Frame provenance | `data/raw/frames_provenance.json` |

## Rules for anyone continuing this project

1. Do not retrain, edit or delete any file listed above.
2. Do not run `evaluate_model.py` without `--reports-dir` while working on
   Experiment 2 - the default `reports/` is this experiment's frozen record.
3. Experiment 2 lives in `experiments/experiment_2_large_dataset/` and uses its
   own dataset, checkpoint and reports.