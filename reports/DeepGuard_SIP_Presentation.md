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
> **Two experiments are presented.** Slides up to *Limitations* cover
> **Experiment 1**, the frozen academic baseline. A clearly separated
> **Experiment 2** section follows it, sourced from
> `experiments/experiment_2_large_dataset/dataset_report.json`,
> `training_summary.json`, `reports/evaluation_report.json` and
> `reports/experiment_2_report.md`. **Experiment 2 does not replace Experiment 1.**
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
7. (Experiment 2) Repeat the same honest protocol on a **larger, independently
   sourced dataset with a different fake class**, and report the outcome **without**
   claiming that dataset size caused any change.

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

## Slide 13 - Experiment 1 Limitations (must be presented)

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

# Part 2 - Experiment 2 (Supplementary)

> **One-sentence framing for the presenter.** Experiment 1 is the frozen
> baseline. Experiment 2 asks: what does the same architecture do on a much
> larger dataset whose synthetic images are made by a **different generator**?
> It is a **second evaluation under different conditions - not a controlled
> dataset-size experiment.**

## Slide 14 - Why a Second Experiment?

Experiment 1's headline weakness was **fake recall of only 50.00%** - half the
fakes were missed. Two possible causes:

1. The training set was tiny (only **360** training images).
2. The **forgery type** (face-swap) is intrinsically harder.

Experiment 2 was designed to move both variables at once and see what happened.

- **Dataset:** Kaggle *140k Real and Fake Faces* (v2) - 140,000 images.
- **Fake class changed completely:** FF++ **face-swap** (Experiment 1) ->
  **StyleGAN synthesis** (Experiment 2).
- Same architecture, same honest "evaluate once" protocol, same CPU-only setup.

## Slide 15 - Experiment 2: Dataset and Integrity Controls

- **6,000 images** selected from the 140,000-image pool (3,000 real / 3,000 fake).
- **Deterministic, seeded selection** (`seed 42`), per-class quotas.

| Split | Real | Fake | Total | Share |
| --- | --- | --- | --- | --- |
| Train | 2,100 | 2,100 | 4,200 | 70% |
| Validation | 450 | 450 | 900 | 15% |
| Test | 450 | 450 | 900 | 15% |

**Duplicate / leakage controls - 6 of 6 audits passed**

| Control | Result |
| --- | --- |
| Exact duplicates (MD5) | **0** removed |
| **Near-duplicates removed (64-bit pHash, threshold 3)** | **5,069** |
| Unreadable / corrupt images | **0** |
| Min cross-split pHash distance | **4 / 4 / 6 bits** (threshold 3) |
| Cross-split image leakage | **none** (publisher's official splits inherited) |

- Dataset pinned by a **SHA-256 fingerprint**, re-verified at training start.
- Copies **nothing** until every audit passes; validator exits non-zero on failure.

## Slide 16 - Experiment 2: Training and Test Results

**Training** - EfficientNet-B0, ImageNet-pretrained, CPU only

| Setting | Value |
| --- | --- |
| Epochs requested / completed | 10 / 10 (**early stopping not triggered**) |
| Best epoch | **10** (lowest validation loss 0.0684) |
| Batch / LR / weight decay | 8 / 3e-4 / 1e-4 |
| Duration | **600.47 minutes** (~10 hours, CPU) |
| Checkpoint | `experiment_2_deepguard_efficientnet_b0.pt` (16.3 MB) |

**Official test result** - one evaluation on the untouched **900-image** test split

| Metric | Value |
| --- | --- |
| **Accuracy** | **97.56%** (878 / 900 correct) |
| real precision / recall / F1 | 99.54% / 95.56% / 0.9751 |
| fake precision / recall / F1 | 95.73% / 99.56% / 0.9760 |
| Macro precision / recall / F1 | 97.63% / 97.56% / **97.55%** |

**Confusion matrix:** [[430, 20], [2, 448]] (rows = actual, columns = predicted)

| | Predicted real | Predicted fake |
| --- | --- | --- |
| **Actual real** | 430 | 20 |
| **Actual fake** | 2 | 448 |

- **Only 2 of 450 fakes missed** (was 30 of 60 in Experiment 1).
- **20 real frames raised false alarms** (was 0).

![Experiment 2 confusion matrix](experiment_2_large_dataset/reports/confusion_matrix.png)

## Slide 17 - Experiment 2: Comparison and Interpretation

| Test metric | Experiment 1 (frozen baseline) | Experiment 2 |
| --- | --- | --- |
| Test images | 120 | 900 |
| **Test accuracy** | **75.00%** | **97.56%** |
| Macro F1 | 73.33% | 97.55% |
| False negatives (missed fakes) | 30 | **2** |
| False positives (false alarms) | 0 | 20 |
| Training time | 52 min 28 s | 600.47 min |

> **The mandatory caveat - say this out loud.**
> Experiment 2 is **NOT** a controlled dataset-size experiment. Six things
> changed at once:
>
> 1. Data amount (600 -> 6,000 images)
> 2. **Fake generator: face-swap -> StyleGAN synthesis**
> 3. Image framing (video crops vs. full square images)
> 4. Source resolution (3.2x vs. 1.14x downsampling)
> 5. Compression (c23 video vs. JPEG)
> 6. Test-set size (120 vs. 900 images)

**Why this matters most:** a face-swap inherits a real person's pose and
lighting, so its artefacts are blending errors layered onto genuine sensor
noise. A StyleGAN image is synthesised end to end - a completely different
artefact signature. **A large part of the gap may be an easier fake source, not
a better-trained model.**

**Correct claim:** Experiment 2 achieved higher test performance under its
larger-dataset, independently-sourced conditions, and the design does not permit
attributing that to dataset size alone.

**Incorrect claims (do not say these):** "accuracy improved because the dataset
was bigger", "the 10x larger dataset improved detection".

## Slide 18 - Experiment 2 Limitations (must be presented)

- **Not a dataset-size-controlled experiment** (previous slide).
- **Different forgery family** - StyleGAN only; says nothing about face-swap.
- **No identity metadata** in this dataset, so identity-level grouping is
  impossible. Leakage control rests on the publisher's splits + de-duplication.
  (Experiment 1's identity-component split was the stricter one.)
- **Best checkpoint is the final epoch** - validation loss was still falling, so
  the model was **not fully converged**. 97.56% is **not a ceiling**.
- Near-duplicate margins are **tight**: two cross-split margins are only **1
  bit** above the threshold.
- Single-source test set - **does not generalise** to all real-world deepfakes.
- Frame-level only; displayed confidence is **uncalibrated**.
- 20 false alarms would matter more under a real-world prior where most images
  are genuine.

---

## Slide 19 - Conclusion

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

**Experiment 2 (supplementary).** The same architecture on a **larger,
differently distributed** dataset (6,000 Kaggle 140k images, de-duplicated, 6/6
audits passed, 600.47 min) reached **97.56% test accuracy** and **97.55% macro
F1** on a 900-image test split, with **2** missed fakes instead of 30. The
change from Experiment 1 is **not attributable to dataset size alone**, because
data amount, forgery family, framing, resolution, compression and test-set size
all changed together.

Both experiments are reported with their limitations. Neither is claimed to be
optimal, state of the art, production-ready, or generalisable beyond its own
dataset.

---

## Slide 20 - Future Work

1. ~~Integrate inference into the Streamlit UI.~~ **Done** - the app now serves
   the Experiment 2 checkpoint and returns a real prediction with confidence.
2. Enlarge and diversify the Experiment 1 dataset (more identities, methods,
   compression levels; cross-dataset tests such as Celeb-DF).
3. Video-level detection using temporal information.
4. Hyperparameter tuning and comparison with the documented ResNet-50
   alternative (reported only after a real evaluation).
5. Robustness / generalisation studies on unseen generators and "in the wild"
   media - documented with the same evidence-traceability discipline.
6. **Run Experiment 2 to convergence** - validation loss was still falling.
7. **Run the missing controlled experiment** - same dataset at two sizes, or
   both datasets at matched size, to isolate the dataset-size variable.
8. **Cross-forgery transfer test** - each checkpoint on the forgery family it
   never saw.

---

*Presentation prepared from verified project artifacts (Stage 8), covering
Experiment 1 (frozen baseline) and Experiment 2 (supplementary). See the full
evidence tables in `reports/DeepGuard_SIP_Final_Technical_Report.md` and
`reports/experiment_2_report.md`.*