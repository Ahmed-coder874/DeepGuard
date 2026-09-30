# DeepGuard - SIP Presentation

## AI-Based Deepfake Detection Prototype

**Project:** Deepfake Technology - **Project ID:** 24CGCS06
**Department:** ECE - **Institution:** Atria Institute of Technology, Bengaluru
**Project Guide:** Prof. Jayanth U - **SDG:** 10 (Reduced Inequalities)

> **Built from verified project artifacts only.** No new experimental claim is
> introduced in this presentation. Every value traces to the final technical
> report, the checkpoint, `reports/evaluation_report.json`, the training log and
> `data/processed/dataset_summary.json` (see Appendix A of the final report).
>
> **Presenter placeholders:** [Student Name], [Register Number] - to be filled
> in before presenting.

---

## Slide 1 - Title

**DeepGuard: AI-Based Deepfake Detection Prototype**

- Project ID 24CGCS06 | Department of ECE | Atria Institute of Technology, Bengaluru
- Project Guide: Prof. Jayanth U | SDG 10 - Reduced Inequalities
- [Student Name] | [Register Number]
- An end-to-end, reproducible image-based deepfake detection prototype

---

## Slide 2 - Problem Statement

- Deepfakes are AI-generated or manipulated facial media that are increasingly
  hard to distinguish by eye.
- Concerns identified in the project field visit: deepfake-related cybercrime,
  identity fraud, misinformation, difficulty of verifying digital content, and
  limited public awareness.
- Manual verification is unreliable; professional detectors need large datasets
  and heavy compute.
- DeepGuard builds a small, CPU-trainable, honest binary classifier
  (real vs. fake) whose claims match its evidence.

---

## Slide 3 - Objectives

1. Research publicly available deepfake datasets and select a realistic one.
2. Prepare a small, balanced, leakage-free train/validation/test dataset.
3. Select an architecture that fits 224 x 224 input and a CPU-only machine.
4. Implement and run a documented training pipeline.
5. Evaluate the trained model **once** on the held-out test split.
6. Trace every reported value to a real project artifact.

---

## Slide 4 - Deepfake Detection Background

- Deepfakes produced by face-swapping and face-reenactment pipelines
  (non-learned and GAN-based methods).
- Standard benchmark: **FaceForensics++** (FF++) - original sequences as
  "real", manipulated versions as "fake" (Deepfakes, Face2Face, FaceSwap,
  NeuralTextures; FaceShifter added later), at raw/c23/c40 compression.
- Detection framed as binary image classification; EfficientNet backbones are
  used in published deepfake-detection work.
- Published ImageNet numbers were used **only for architecture selection** -
  they are not DeepGuard benchmark results.

---

## Slide 5 - Dataset

**FaceForensics++ (Deepfakes method, c23), frame-extracted on this machine**

- 20 MP4 videos: 10 original (real) + 10 manipulated (fake), 10 identities.
- **30 deterministic, evenly-spaced frames per video** (no randomness):
  - 300 real frames + 300 fake frames = **600 raw frames** (verified on disk).
  - Processed: train 180 + 180, validation 60 + 60, test 60 + 60 = 600.
- 0 corrupt / 0 unsupported files; provenance manifest records every frame.
- Video sources under `data/faceforensics` kept **intact** (never modified).

**Documentation note:** earlier headers stated 600 real + 600 fake (1,200);
the verified disk state is 300 + 300 = 600, which is the dataset used.

---

## Slide 6 - Dataset Splitting and Leakage Prevention

- Frames grouped by **identity component** (connected component of the
  target/source graph of manipulated videos).
- The **same component id** exists in both class folders, so a whole identity
  stays inside exactly one split.
- Ratios 70% / 15% / 15%, **seed 42**, balanced per class
  (`--max-per-class`).
- Verified leakage audit: no component appears in more than one split; the test
  component is absent from train and validation.

| Split | Components | Count |
| --- | --- | --- |
| Train | 469_481, 585_599, 672_720 | 360 |
| Validation | 866_878 | 120 |
| Test | 183_253 | 120 |

---

## Slide 7 - System Architecture

```
Input image/video frame
  -> Validation (extension, size, image integrity)
  -> Preprocessing (BGR->RGB, resize 224x224, 0-1 normalisation)
  -> Model input tensor (1, 3, 224, 224), ImageNet-normalised
  -> EfficientNet-B0 (2-class head: real=0, fake=1)
  -> Classifier logits (1, 2) -> prediction + model confidence
```

- Modular code: `src/{validation, preprocessing, dataset, extraction,
  training, evaluation, inference, model_interface}.py` + CLI scripts.
- Separate environments: `venv-train` (PyTorch) vs. app `venv` (Streamlit).

---

## Slide 8 - EfficientNet-B0 Methodology

- EfficientNet-B0: convolutional network of MBConv blocks with
  squeeze-and-excitation, scaled by a compound coefficient (baseline model).
- **5.3M parameters, 0.39B FLOPs** - roughly 10x lighter than ResNet-50:
  trainable on the available CPU (Intel Core i5-8250U, no GPU).
- **Native 224 x 224 input** - exact match with the existing pipeline.
- Official ImageNet pretrained weights (TorchVision transfer learning);
  final layer replaced: Linear(1280 -> 2).
- Documented alternative: ResNet-50 (not trained in this project).

---

## Slide 9 - Training Configuration

| Setting | Value |
| --- | --- |
| Architecture / pretraining | EfficientNet-B0, ImageNet-pretrained |
| Batch size / LR / optimizer | 8 / 3e-4 / AdamW |
| Weight decay / loss | 1e-4 / CrossEntropyLoss |
| LR schedule | ReduceLROnPlateau (factor 0.1, patience 2) |
| Max / actual epochs | 10 requested / 9 completed (early stopping, patience 3) |
| Seed / device | 42 / CPU only (no CUDA) |
| Validation | deterministic eval transforms, no shuffle |
| Duration | 52 min 28 s |

> First reproducible prototype experiment; conservative defaults, **not**
> claimed optimal (no hyperparameter tuning).

---

## Slide 10 - Training/Validation Results

Complete recorded epoch history (training/validation only - not test).

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | --- | --- | --- | --- |
| 1 | 0.6099 | 0.6361 | 0.6585 | 0.6250 |
| 2 | 0.2206 | 0.9111 | 0.7458 | 0.5500 |
| 3 | 0.1317 | 0.9611 | 0.6652 | 0.5333 |
| 4 | 0.1489 | 0.9500 | 0.5997 | 0.8083 |
| 5 | 0.0261 | 0.9944 | 0.6748 | 0.5417 |
| **6** | **0.0362** | **0.9917** | **0.4908** | **0.7833** |
| 7 | 0.0469 | 0.9833 | 0.5771 | 0.6500 |
| 8 | 0.1215 | 0.9583 | 0.6341 | 0.6167 |
| 9 | 0.0335 | 0.9889 | 0.6448 | 0.6083 |

**Best checkpoint = epoch 6 (lowest validation loss).**

- **Validation result: 78.33% accuracy at best epoch 6** (validation split =
  component 866_878, 120 frames). Note: best validation *accuracy* overall was
  80.83% at epoch 4; the checkpoint is selected by lowest validation *loss*.
- Training accuracy >= 95% from epoch 3; validation fluctuates (53.33% -
  80.83%) - a train/validation performance gap is observed.

![Training curves](stage6/training_curves.png)

---

## Slide 11 - Official Test Results

One evaluation on the **held-out test split** (component 183_253, 60 real + 60
fake = 120 frames), untouched during training.

| Metric | Value |
| --- | --- |
| Accuracy | **75.00%** (90 / 120 correct) |
| Incorrect | 30 / 120 |
| real precision / recall / F1 | 66.67% / 100.00% / 80.00% |
| fake precision / recall / F1 | 100.00% / 50.00% / 66.67% |
| Macro precision / recall / F1 | 83.33% / 75.00% / 73.33% |

**Official test result: 75.00% accuracy on 120 test frames**
(from `reports/evaluation_report.json`).

---

## Slide 12 - Confusion Matrix and Error Analysis

**Confusion matrix:** [[60, 0], [30, 30]]  (rows = actual, columns = predicted)

| | Predicted real | Predicted fake |
| --- | --- | --- |
| **Actual real** | 60 | 0 |
| **Actual fake** | 30 | 30 |

- **60 / 60 real frames correctly detected** (no false alarms).
- **30 / 60 fake frames correctly detected**.
- **30 fake frames incorrectly classified as real** (false negatives).
- **0 real frames incorrectly classified as fake** (no false positives).

Meaning: the detector is conservative - high trust when it raises an alarm
(fake precision 100.00%), but half the fakes are missed (fake recall 50.00%),
which is the safety-relevant direction.

![Confusion matrix](stage6/confusion_matrix.png)
![Per-class metrics](stage6/per_class_metrics.png)

---

## Slide 13 - Limitations (must be presented)

- Verified dataset is **600 frames** (300 + 300), not the 1,200 originally
  documented; all reported counts use the verified disk state.
- Test set is **120 frames from a single identity component (183_253)** -
  every metric is noisy and single-subject biased.
- **Frame-level only** - no temporal / video-level detection.
- Results are specific to FF++ Deepfakes at c23; **they do not generalise** to
  other methods, datasets or compression levels.
- No hyperparameter tuning - first-prototype configuration.
- Model is **not** production-ready and no state-of-the-art claim is made.

---

## Slide 14 - Conclusion

DeepGuard delivered an honest, reproducible image-based deepfake-detection
workflow:

- 600-frame FF++ (Deepfakes, c23) dataset extracted deterministically and
  split 360/120/120 with verified identity-leakage control.
- ImageNet-pretrained EfficientNet-B0 fine-tuned on CPU (52 min 28 s), best
  checkpoint epoch 6 (validation loss 0.4908, validation accuracy 78.33%).
- **One** held-out test evaluation: **75.00% accuracy**, 0 false alarms,
  fake recall 50.00%, macro F1 73.33%.
- Training history is consistent with a close fit to the training split that
  does not fully transfer to held-out splits.

The experiment is finished; results are reported with their limitations.

---

## Slide 15 - Future Work

1. Integrate inference into the Streamlit UI (real prediction + confidence).
2. Enlarge and diversify the dataset (more identities, methods, compression
   levels; cross-dataset tests such as Celeb-DF).
3. Video-level detection using temporal information.
4. Hyperparameter tuning and comparison with the documented ResNet-50
   alternative (reported only after a real evaluation).
5. Robustness / generalisation studies on unseen generators and "in the wild"
   media - documented with the same evidence-traceability discipline.

---

*Presentation prepared from verified project artifacts (Stage 8). See the full
evidence tables in `reports/DeepGuard_SIP_Final_Technical_Report.md`.*