# DeepGuard - Stage 6 Results Analysis

Project: **Deepfake Technology** (24CGCS06)
Stage: **Stage 6 - Results Analysis, Visualization & Verification**

> This document analyses the actual results produced by Stage 4 training and
> Stage 5 test evaluation. It reports **no new empirical result**. Every number
> below comes from a real project artifact:
>
> - the verbatim Stage 4 training console log (`reports/stage6/train_run_log.txt`),
> - the trained checkpoint `models/deepguard_efficientnet_b0.pt`,
> - the Stage 5 evaluation report `reports/evaluation_report.json`.
>
> **Validation metrics and official test metrics are kept strictly separate
> throughout.** Training/validation values are optimisation signals. The
> official result is the test-set evaluation from Stage 5.

---

## 1. Experiment Overview

| Item | Value |
| --- | --- |
| Model | EfficientNet-B0, ImageNet-pretrained, two-class head (1280 -> 2) |
| Classes | `real` = 0, `fake` = 1 (positive class = fake) |
| Input | 224 x 224 x 3, ImageNet mean/std normalisation |
| Dataset | FaceForensics++ (c23), Deepfakes method subset |
| Raw frames | 300 real + 300 fake = 600 PNG frames |
| Prepared splits | train 180/180, validation 60/60, test 60/60 |
| Training environment | `venv-train` (PyTorch 2.14.0+cpu, torchvision 0.29.0+cpu), CPU-only |
| Checkpoint | `models/deepguard_efficientnet_b0.pt` (16,341,163 bytes) |
| Test evaluation | Stage 5 `evaluate_model.py`, test split only, 120 frames |

## 2. Dataset Verification

Verified from disk (Stage 6 re-check, `data/processed/dataset_summary.json` and
`data/raw/frames_provenance.json`):

- Raw frames on disk: **300 real + 300 fake = 600** PNGs across 5 identity
  components (183_253, 469_481, 585_599, 672_720, 866_878).
- Prepared processed dataset: train 180/180 = 360; validation 60/60 = 120;
  test 60/60 = 120. Total processed = 600.
- Leakage verification: train/validation/test path sets are disjoint
  (overlap counts 0/0/0). Frames are grouped by identity component, and no
  component appears in more than one split.
- Test split: component **183_253** only (originals 183.mp4, 253.mp4 and
  manipulated 183_253.mp4, 253_183.mp4). Whole group 183_253 is **absent**
  from train and validation.

**Dataset documentation discrepancy (recorded, not silently corrected):**
The earlier project documentation/header claimed 600 real / 600 fake /
1,200 total. Repeated disk verification shows the verified dataset contains
**300 real + 300 fake = 600 frames** and a processed dataset of 600 images as
above. No second 600-image batch was ever generated. This is a provenance
documentation limitation of the dataset, not a code defect; all counts in this
report are the verified disk values.

## 3. Training Configuration

Configuration actually used in the recorded Stage 4 run
(`reports/stage6/train_run_log.txt`; planned defaults in `config.py`):

| Setting | Value used |
| --- | --- |
| Architecture | `efficientnet_b0` (ImageNet pretrained) |
| Input size | 224 x 224 (RandomResizedCrop 0.8-1.0, flip p=0.5, rot +/-10 on train) |
| Batch size | **8** (overridden from the planned default 16) |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss |
| LR schedule | ReduceLROnPlateau (factor 0.1, patience 2, monitoring validation loss) |
| Max epochs | 10 |
| Early stopping | patience 3 on validation loss |
| num_workers | 0 |
| Seed | 42 |
| Device | CPU (no CUDA) |
| Validation | deterministic eval transforms, `shuffle=False` |

## 4. Training Results (complete recorded epoch table)

These are **training/validation values from the training run** (verbatim
`train_run_log.txt`). They are **not** test results.

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
| --- | --- | --- | --- | --- |
| 1 | 0.6099 | 0.6361 | 0.6585 | 0.6250 |
| 2 | 0.2206 | 0.9111 | 0.7458 | 0.5500 |
| 3 | 0.1317 | 0.9611 | 0.6652 | 0.5333 |
| 4 | 0.1489 | 0.9500 | 0.5997 | 0.8083 |
| 5 | 0.0261 | 0.9944 | 0.6748 | 0.5417 |
| 6 | 0.0362 | 0.9917 | 0.4908 | 0.7833 |
| 7 | 0.0469 | 0.9833 | 0.5771 | 0.6500 |
| 8 | 0.1215 | 0.9583 | 0.6341 | 0.6167 |
| 9 | 0.0335 | 0.9889 | 0.6448 | 0.6083 |

Best checkpoint (lowest validation loss): **epoch 6** (val loss 0.4908,
val accuracy 0.7833), as recorded in the checkpoint metadata. Early stopping
triggered after epoch 9 (no validation-loss improvement at epochs 7, 8, 9).
Training completed 9 of 10 requested epochs.

## 5. Validation Analysis (validation-only)

- **Training loss** fell from 0.6099 (epoch 1) to roughly 0.03-0.12 for
  epochs 5-9; the observations are consistent with the model fitting the
  training split very closely from epoch 3 onward.
- **Training accuracy** reached >=0.95 by epoch 3 and stayed there (peak
  0.9944 at epoch 5). Recorded history is consistent with a training split
  that was almost completely memorized/fitted by the end of training.
- **Validation loss** never dropped below ~0.49 (best 0.4908, epoch 6) and did
  not consistently track the training loss. The validation loss fluctuates
  (0.49 - 0.75) across epochs.
- **Validation accuracy** fluctuated substantially (0.5333 - 0.8083) with no
  monotonic trend; the best validation accuracy (0.8083, epoch 4) and the
  best-checkpoint epoch (6, by lowest validation loss) do not coincide.
- The gap between near-perfect training metrics and the fluctuating validation
  metrics is consistent with a persistent train/validation performance gap.
  This is an observation about recorded values on a small validation split and
  is not claimed to be definitive proof of overfitting.
- Caveat on variance: the validation split contains **only one identity
  component (866_878)**, so a single frame changing label moves accuracy by
  ~0.008; this small, single-source split partly explains the epoch-to-epoch
  swings.

## 6. Official Test Results (test-set only, from Stage 5)

`reports/evaluation_report.json`, produced by `evaluate_model.py` on the
untouched test split (component 183_253, 60 real + 60 fake = 120 frames).
These are the **official Stage 5 test-set results**:

| Metric | real | fake |
| --- | --- | --- |
| Precision | 0.6667 | 1.0000 |
| Recall | 1.0000 | 0.5000 |
| F1-score | 0.8000 | 0.6667 |
| Support | 60 | 60 |

- **Accuracy = 0.7500** (90 / 120 correct; 30 / 120 incorrect)
- Macro averages: precision 0.8333, recall 0.7500, F1 0.7333

Confusion matrix (rows = actual, columns = predicted):

```
          predicted
          real   fake
actual
  real     60      0
  fake     30     30
```

## 7. Error Analysis (test set)

- **False positives (real predicted as fake): 0.** No real frame was flagged as
  fake; the model never produced a false alarm on this test set.
- **False negatives (fake predicted as real): 30.** Half of the 60 fake frames
  were missed and classified as real. This is the dominant failure.
- Class behaviour:
  - `Fake precision = 1.0000`: every frame the model called "fake" was fake
    (30/30). High trust when it raises an alarm.
  - `Fake recall = 0.5000`: of 60 fake frames, only 30 were caught. Half of
    the manipulated frames went undetected.
  - `Precision` and `recall` measure different properties and must not be
    conflated: a model can have perfect fake precision (never a wrong alarm)
    while missing half the fakes (low fake recall) - exactly the recorded
    behaviour.
- Practical meaning: the detector is conservative (silent rather than false-
  alarming), but the missed fakes (30) are the safety-relevant direction for a
  deepfake detector. These statements describe only the 120-frame test set.

## 8. Validation vs Test Separation

- Stage 4 best validation: accuracy 0.7833, loss 0.4908, epoch 6, on the
  **validation split (component 866_878)**.
- Stage 5 official test: accuracy 0.7500 on the **test split (component
  183_253)**, untouched during training.
- The two numbers come from different data splits (different identity
  components, different images) and serve different purposes (model selection
  vs independent estimate). No claim of "improvement" or "degradation" is made
  from their difference, and no model change is proposed from the test result.
- Validation metrics are never presented as test results anywhere in this
  report or in `evaluation_report.json` (the report stores the Stage 4 best as
  `checkpoint_validation_*` fields only).

## 9. Limitations

- The verified raw dataset contains **600 frames** (300/300), not the 1,200
  originally documented (see Section 2).
- The test set contains only **120 frames** from a **single identity
  component (183_253)**, which makes all test metrics noisy and single-subject
  biased.
- Results are specific to this prepared dataset/split and to the FaceForensics++
  Deepfakes method at c23 compression. They must **not** be generalised to all
  deepfake videos or to other manipulation methods.
- Evaluation is at **frame level**; it does not by itself demonstrate
  video-level detection performance (temporal information was not used).
- The prototype's test result (75.0%) is an estimate with a small sample, to
  be read with that limitation in mind.

## 10. Reproducibility

| Item | Value |
| --- | --- |
| Python | 3.14.7 (both `venv-train` and app `venv`) |
| PyTorch | 2.14.0+cpu (CPU-only; CUDA unavailable) |
| TorchVision | 0.29.0+cpu |
| Pillow | 12.3.0 (both environments) |
| NumPy | 2.5.3 (venv-train) |
| Seed | 42 |
| Checkpoint | `models/deepguard_efficientnet_b0.pt` (records model, seed, epoch 6, torch version) |

Commands/artifacts used:

```
.\venv-train\Scripts\python.exe extract_dataset_frames.py     # Stage 2 (600 frames + manifest)
.\venv-train\Scripts\python.exe prepare_dataset.py            # Stage 2 (group-aware splits)
.\venv-train\Scripts\python.exe train_model.py --batch-size 8  # Stage 4 (9/10 epochs, best epoch 6)
.\venv-train\Scripts\python.exe evaluate_model.py              # Stage 5 (test split, 120 images)
.\venv-train\Scripts\python.exe reports\stage6\generate_stage6_plots.py  # Stage 6 plots
```

Artifacts: `reports/stage6/train_run_log.txt` (verbatim Stage 4 output),
`reports/evaluation_report.json` (Stage 5), `reports/stage6/*.png` (plots
generated from the values above). Evaluation itself is deterministic (fixed
transforms, no shuffle, CPU). Perfect bit-for-bit reproducibility across
environments is not claimed.

## 11. Conclusion

The trained EfficientNet-B0 (best checkpoint epoch 6, validation loss 0.4908)
was evaluated once on the untouched 120-frame test split: **accuracy 75.0%**,
no false alarms (fake precision 1.0000), but only half the fake frames caught
(fake recall 0.5000), so macro F1 = 0.7333. The training history is consistent
with a close fit to the training split that does not fully transfer to the held-
out splits. These are the results of one prototype experiment on a small,
single-component test set; the model is **not** described as optimal, state of
the art, or production-ready, and the numbers do not generalise beyond the
prepared dataset/split.

---

*Stage 4 training completed. Stage 5 official test evaluation completed.
Stage 6 = analysis and documentation only (no new training or test
experiments performed).*