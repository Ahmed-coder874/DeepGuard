# Experiment 2 Report — DeepGuard

**Project:** AI-Based Deepfake Detection ("Deepfake Technology"), Project ID 24CGCS06
**Experiment:** 2 — Larger, differently distributed dataset
**Status:** Complete. Results frozen.
**Report scope:** Documentation of an experiment that has already been built, trained and evaluated. No training, evaluation or metric generation was performed while writing this report.

---

## Authoritative sources

Every numerical value in this report is traceable to one of three committed, machine-readable artifacts:

| Source | Role |
|---|---|
| `experiments/experiment_2_large_dataset/dataset_report.json` | Dataset construction, quotas, de-duplication, audit results |
| `experiments/experiment_2_large_dataset/training_summary.json` | Training configuration, epochs, duration, fingerprint |
| `experiments/experiment_2_large_dataset/reports/evaluation_report.json` | Official Experiment 2 **test** metrics |

Methodology and framing follow `experiments/experiment_2_large_dataset/README.md`, which documents the completed run.

No value in this report was newly measured, estimated, projected or recalculated from raw data. Where a figure is a restatement of a recorded value at a different precision or unit, the recorded value is shown alongside it.

---

## 1. Experiment 2 overview

Experiment 2 evaluates DeepGuard on a **substantially larger, independently sourced** subset of the Kaggle *140k Real and Fake Faces* dataset, and measures it on its own held-out test split.

**Experiment 2 is a separate experiment. It does not replace, overwrite or supersede Experiment 1.** Experiment 1 remains the frozen academic baseline of this project, with its own checkpoint, its own dataset and its own recorded test result. Both experiments are reported side by side in this document and both sets of artifacts are retained.

Experiment 2 was carried out to answer a different question from Experiment 1. Experiment 1 established what a small detector trained on a minimal dataset can achieve, and where it fails. Experiment 2 asks what the same architecture achieves on a dataset an order of magnitude larger, drawn from a different publisher with a different kind of synthetic imagery.

The headline result of Experiment 2 is a **test accuracy of `0.9756` (97.56%)** on a 900-image held-out test split.

> **This number is the Experiment 2 TEST accuracy.** It is not validation accuracy, not a training-set figure, and not a live application output. Section 6 and Section 7 keep these strictly separate.

---

## 2. Dataset

### 2.1 Source

| Property | Value |
|---|---|
| Dataset | Kaggle: `xhlulu/140k-real-and-fake-faces` (v2) |
| URL | `https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces` |
| Licence | Kaggle: "Other (specified in description)". Underlying FFHQ photographs are CC BY-NC-SA 4.0 (NVIDIA) — non-commercial use, attribution required. **No images are redistributed by this repository.** |
| Pool root on the build machine | `E:\SIP\140k_dataset\real_vs_fake\real-vs-fake` |
| Build recorded at | `2026-10-03T14:53:56+00:00` |
| Source pool scanned | 140,000 images |

### 2.2 Official split sizes in the source pool

The publisher already partitions the pool into `train`, `valid` and `test` folders. Experiment 2 inherits those boundaries rather than re-partitioning the data:

| Official split | Real | Fake | Total |
|---|---|---|---|
| `train` | 50,000 | 50,000 | 100,000 |
| `validation` | 10,000 | 10,000 | 20,000 |
| `test` | 10,000 | 10,000 | 20,000 |

### 2.3 Selected subset

**6,000 images were selected: 3,000 real and 3,000 fake.**

| Measure | Value |
|---|---|
| Selected images | 6,000 |
| Real | 3,000 |
| Fake | 3,000 |
| Size on disk | 172,652,155 bytes (164.7 MB) |

### 2.4 Selection methodology

The builder (`build_dataset.py`) runs five stages. **Nothing is copied until every check has passed.**

1. **Scan** — the six official folders, 140,000 images.
2. **Exact-duplicate removal (MD5)** — byte-identical images collapse to one. Copies in the same class and split reduce to one; copies of the same class across different splits keep the copy in the higher-priority split (`test` > `validation` > `train`), which removes any leakage the source dataset introduced; copies carrying **different** class labels are dropped entirely, because the label contradicts itself.
3. **Near-duplicate removal (pHash)** — a 64-bit DCT perceptual hash per image; see Section 4.
4. **Deterministic subsampling** to the per-class quotas, using a per-pool seed derived from `--seed 42`.
5. **Audit** — see Section 4.

**Determinism.** Selection is seeded (`seed: 42`, `phash_jobs: 8`). The seed governs subsampling, so the selected subset is reproducible rather than arbitrary. The independence of the whole build is pinned by a **dataset fingerprint** — a SHA-256 computed over the manifest's `split/label/md5` rows:

```
cafc017380d10484326d668e2893f3582ddeb8c5e23c278569804d7e44f3d8bb
```

`train_experiment_2.py` recomputes this fingerprint at startup and refuses to resume against a changed dataset. The recorded fingerprint therefore proves that the recorded training run used exactly this dataset.

### 2.5 Source directory handling

`--source-dir` is **not** hard-coded to one machine or folder. It accepts the dataset's parent directory, and the builder locates the pool root underneath it automatically by searching three levels deep, so all of the following work:

```
--source-dir "E:\SIP\140k_dataset"                              # parent of real_vs_fake
--source-dir "E:\SIP\140k_dataset\real_vs_fake"                  # the real_vs_fake folder itself
--source-dir "E:\SIP\140k_dataset\real_vs_fake\real-vs-fake"     # the pool root itself
```

The path recorded above is simply where the dataset sat on the machine where this experiment was prepared; it is an illustration, not a requirement. The default, if `--source-dir` is omitted, is `experiments/experiment_2_large_dataset/source`, so the dataset may also live inside the project — though keeping it outside is recommended, because it is roughly 4 GB and its licence forbids redistribution.

---

## 3. Dataset split

| Split | Real | Fake | Total | Share | Drawn only from |
|---|---|---|---|---|---|
| `train` | 2,100 | 2,100 | 4,200 | 70.0% | `train_real`, `train_fake` |
| `validation` | 450 | 450 | 900 | 15.0% | `valid_real`, `valid_fake` |
| `test` | 450 | 450 | 900 | 15.0% | `test_real`, `test_fake` |
| **Total** | **3,000** | **3,000** | **6,000** | **100%** | 140,000-image source pool |

- **Every split is class-balanced** — the real and fake counts are equal in all three splits. This is confirmed by the audit check `classes_balanced_in_every_split`, which passed.
- **Split policy:** *"Experiment 2 splits inherit the publisher's official train/valid/test folders; no image ever crosses that boundary."* This is confirmed by the audit check `experiment_split_maps_to_one_official_split`, which passed.
- The **test split was not used for tuning.** It was untouched from the build until the single recorded evaluation.

### Pool sizes after de-duplication

The de-duplication stage reduced the available pool unevenly, which is worth recording because it explains the selection margins:

| Official split | Real after de-dupe | Fake after de-dupe |
|---|---|---|
| `train` | 49,981 | 45,421 |
| `validation` | 10,000 | 9,649 |
| `test` | 10,000 | 9,880 |

The fake pool lost far more images than the real pool — 5,069 near-duplicates were removed in total, and they came predominantly from the synthetic side. The quotas (2,100 / 450 / 450 per class) were nonetheless met exactly in every split.

---

## 4. Duplicate and leakage controls

### 4.1 Exact-duplicate detection (MD5)

Every image is hashed with MD5 over its full contents. Byte-identical images are collapsed according to the rules in Section 2.4 step 2. Recorded outcome:

| Category | Count |
|---|---|
| Exact duplicates removed, same class + same split | 0 |
| Exact duplicates removed, cross-split | 0 |
| Exact duplicates removed, cross-class | 0 |

**No exact duplicates were found anywhere in the 140,000-image pool.** The zero cross-split figure is the meaningful one for leakage: the publisher's own folders contain no repeated file.

### 4.2 Near-duplicate detection (pHash)

A 64-bit DCT perceptual hash is computed per image. **Two images within `3` bits of each other (Hamming distance ≤ 3) are treated as near-duplicates and the later one is dropped.**

| Measure | Value |
|---|---|
| pHash threshold | **3 bits** |
| pHash worker jobs | 8 |
| Images scanned | 140,000 |
| **Near-duplicates removed** | **5,069** |
| Total removed (exact + near) | 5,069 |
| Unreadable / corrupt images encountered | **0** |

Candidate pairs are found by cutting the hash into `threshold + 1` disjoint bands (multi-index hashing). This construction is *exact* for Hamming distances up to the threshold — it has no false negatives, so the removal step cannot silently miss a near-duplicate pair.

**The threshold is 3 bits and the observed minimum cross-split distances are 4, 4 and 6 bits.** Because the audit criterion is a minimum cross-split distance *greater than* 3, all three pairs pass:

| Pair | Minimum cross-split pHash distance | Against threshold of 3 |
|---|---|---|
| `train` ↔ `validation` | **4 bits** | passes, margin of 1 bit |
| `train` ↔ `test` | **4 bits** | passes, margin of 1 bit |
| `validation` ↔ `test` | **6 bits** | passes, margin of 3 bits |

> **Honest qualification.** Two of these three margins are a single bit. The no-leakage guarantee is *threshold-relative* rather than comfortable. A stricter threshold could have flagged borderline pairs. This is a property of the recorded build and travels with the result.

### 4.3 Corrupted / unreadable image handling

| Check | Result |
|---|---|
| Unreadable images found while scanning the 140,000-image pool | **0** |

Every one of the 140,000 source images decoded successfully.

### 4.4 Audit results

**All six required build audits passed** (`audit_passed: true`):

| Audit check | Meaning | Result |
|---|---|---|
| `exact_counts_match_quota` | every split has exactly the requested number of images | **passed** |
| `classes_balanced_in_every_split` | real count equals fake count in each split | **passed** |
| `no_exact_duplicate_across_splits` | no MD5 appears in two different splits | **passed** |
| `no_image_labelled_both_real_and_fake` | no image carries contradictory labels | **passed** |
| `experiment_split_maps_to_one_official_split` | no image crosses the publisher's official split boundary | **passed** |
| `no_near_duplicate_across_splits` | minimum cross-split pHash distance greater than 3 bits | **passed** |

A further independent post-build re-verification confirmed all 6,000 copied files against the manifest. The builder and the validator both exit non-zero if any check fails.

---

## 5. Training configuration

All values below are read directly from `training_summary.json`.

| Setting | Value |
|---|---|
| Experiment | `experiment_2_large_dataset` |
| Architecture | `efficientnet_b0` (EfficientNet-B0) |
| ImageNet pretrained | **yes** (`pretrained: true`) |
| Device | **CPU** |
| Random seed | **42** |
| Epochs requested | **10** |
| Epochs completed | **10** |
| Best epoch | **10** |
| Early stopping enabled | yes, `patience: 3` |
| Early stopping triggered | **no** (`stopped_early: false`) |
| Batch size | **8** |
| Learning rate | **0.0003** |
| Weight decay | **0.0001** |
| Training images seen per epoch | 4,200 |
| Validation images | 900 |
| **Wall-clock duration** | **600.47 minutes** (≈ 10.0 hours) |
| Dataset fingerprint | `cafc017380d10484326d668e2893f3582ddeb8c5e23c278569804d7e44f3d8bb` |

**The model was trained on CPU.** This is recorded explicitly in the artifact and explains the ten-hour duration: 10 epochs over 4,200 images at 224 × 224 with no GPU acceleration.

### Checkpoint information

| Property | Value |
|---|---|
| Checkpoint | `models/experiment_2_deepguard_efficientnet_b0.pt` |
| Recorded path in `training_summary.json` | `E:\SIP\Prototype\deepguard\models\experiment_2_deepguard_efficientnet_b0.pt` |
| Saved from epoch | **10** — the epoch with the lowest validation loss |
| Resumed/continued from an earlier checkpoint | no; 10 of 10 epochs completed in the recorded run |

**Checkpoint integrity.** `train_experiment_2.py` recomputes the dataset fingerprint at startup and refuses to resume against a changed dataset, so the fingerprint in the table above ties this checkpoint to exactly this 6,000-image dataset.

---

## 6. Training results

### Validation metrics — recorded

| Metric | Recorded value |
|---|---|
| Best validation loss | **0.0684** (`0.06835601629805751`) |
| Best validation accuracy | **0.9689** (`0.9688888888888889`) |
| Epoch at which the best checkpoint was saved | **10 of 10** |
| Early stopping triggered | **no** |

### Validation metrics are NOT the reported result

The model was permitted to learn from the validation split: it is the signal used to select the best checkpoint. `0.9689` is therefore **not** the reported result of Experiment 2.

> **Experiment 2's final, reported result is the test accuracy `0.9756` (97.56%).**

### A note on convergence

The best checkpoint fell at the **final** epoch (10 of 10) and validation loss was still falling when the epoch budget was exhausted. The model was therefore probably **not fully converged**. `0.9756` should not be read as a ceiling on this architecture; a longer run could move it in either direction, since more epochs also carry an overfitting risk.

### Test-set metrics are not produced by the trainer

The evaluator does not emit a test loss. Loss figures in this report are validation loss, and are labelled as such throughout.

---

## 7. Test evaluation

**Authoritative source:** `experiments/experiment_2_large_dataset/reports/evaluation_report.json`
**Evaluated:** `2026-10-04T07:37:44`, on CPU, `torch 2.14.0+cpu`, using the epoch-10 checkpoint.

The evaluation was run against the **untouched 900-image Experiment 2 test split** (450 real + 450 fake) with `--reports-dir` pointed at the Experiment 2 folder, so Experiment 1's frozen `reports/` directory was never a write target.

### 7.1 Official Experiment 2 test result

| Metric | Recorded value | Percentage |
|---|---|---|
| **Test accuracy** | **`0.9756`** (`0.9755555555555555`) | **97.56%** |
| **Macro F1** | **`0.9755`** (`0.9755457738651017`) | **97.55%** |
| Macro precision | `0.9763` | 97.63% |
| Macro recall | `0.9756` | 97.56% |
| Test samples | 900 | — |

### 7.2 Confusion matrix

Rows are **actual** `[real, fake]`; columns are **predicted** `[real, fake]`.

```
[[430,  20],
 [  2, 448]]
```

| | predicted `real` | predicted `fake` | total |
|---|---|---|---|
| **actual `real`** | 430 | 20 | 450 |
| **actual `fake`** | 2 | 448 | 450 |
| **total** | 432 | 468 | 900 |

Derived directly from the matrix above:

- **True negatives** (real called real): **430**
- **False positives** (real wrongly called fake): **20**
- **False negatives** (fake wrongly called real): **2**
- **True positives** (fake called fake): **448**
- **Correct: 430 + 448 = 878 of 900**

### 7.3 Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `real` | `0.9954` | `0.9556` | `0.9751` | 450 |
| `fake` | `0.9573` | `0.9956` | `0.9760` | 450 |
| **macro** | `0.9763` | `0.9756` | `0.9755` | 900 |

`fake` is treated as the **positive class**, because detecting manipulated content is the purpose of the project. The positive-class figures equal the `fake` row by construction.

### 7.4 Behavioural reading

Experiment 2's detector is **mildly biased towards calling images fake**: fake recall `0.9956` against real recall `0.9556`. It misses only 2 of 450 synthetic images, at the cost of 20 false alarms on 450 genuine images.

On this balanced test set that bias costs almost nothing in accuracy. Under a different real/fake prior — for example a deployment where most incoming images are genuine — it would matter, and the operating threshold would need revisiting. The threshold was fixed at `argmax` and was never tuned, deliberately, to keep the test set free of contamination.

### 7.5 Official evaluation versus live application output

| | Official evaluation | Live application output |
|---|---|---|
| Produced by | `evaluate_model.py` over the frozen 900-image test split | The Streamlit prototype, one uploaded image at a time |
| Reported in this report | **yes — this is `0.9756`** | no |
| Meaning | A measured, reproducible aggregate | A single demo inference |

**A prediction shown in the running application is not an official result** and must never be quoted as one. The application itself carries this warning on every prediction.

---

## 8. Interpretation

### 8.1 What Experiment 2 shows

On its own terms, Experiment 2 is a clear success. A single EfficientNet-B0 classifier, trained on CPU for 10 epochs over 4,200 images, reached **97.56% accuracy on a 900-image held-out test split** it never saw during training, with both classes balanced and all near-duplicate leakage controls satisfied. The confusion matrix shows the failure modes are asymmetric but small: 2 missed synthetic images and 20 false alarms.

### 8.2 The mandatory scientific limitation

> ### Experiment 2 is NOT a controlled dataset-size-only experiment.
>
> **Any difference in performance between Experiment 1 and Experiment 2 cannot be attributed solely to the increase in dataset size.**

Experiment 1 and Experiment 2 differ in **six dimensions simultaneously**, not one:

| # | Dimension | Experiment 1 | Experiment 2 |
|---|---|---|---|
| 1 | **Data amount** | 600 images (360 train) | 6,000 images (4,200 train) |
| 2 | **Fake-image distribution / generation method** | FaceForensics++ **face-swap manipulation** — real manipulated faces from real videos | **StyleGAN synthesis** — entirely generated faces from a different generative family |
| 3 | **Image framing / distribution** | centre-cropped frames from widescreen / 4:3 video | full square 256 × 256 images, background included |
| 4 | **Source resolution**, and therefore downsampling to the 224 model input | measured examples 1280 × 720, 1920 × 1080, 856 × 480, 656 × 480 → ≈ 3.2× downscale before cropping | 256 × 256 → ≈ 1.14× downscale |
| 5 | **Compression characteristics** | `c23`-compressed video content, stored losslessly as PNG | JPEG compression / quantisation artefacts |
| 6 | **Test-set size** | **120 images** | **900 images** |

The dataset build records this caveat in its own artifact, verbatim:

> *"Experiment 1 fake class = FaceForensics++ face-swap manipulation. Experiment 2 fake class = StyleGAN synthesis. Dataset size AND fake-image distribution both change, so Experiment 2 is NOT a controlled dataset-size experiment and must never be described as one."*

**Why the fake-generation change matters so much.** These are not two versions of the same problem. A face-swap manipulation inherits the identity, pose and lighting of a real source photograph, so its artefacts are blending and boundary inconsistencies layered onto genuine sensor noise. A StyleGAN image is synthesised end to end, so its artefacts are a wholly different signature — generator upsampling and spectral regularities. A detector's difficulty can differ sharply between those two families **on its own**, independent of how much training data it was given. A meaningful part, potentially most, of the observed gap may reflect an easier fake source rather than a better-trained model.

**Why the test-set size change matters.** Experiment 1's figure rests on 120 images; Experiment 2's rests on 900. The larger split makes Experiment 2's estimate markedly less sensitive to any individual example. The two accuracies are therefore **not equally precise**, and this must be stated whenever they are placed next to each other.

### 8.3 The correct conclusion, and only this one

> Experiment 2 achieved higher test performance than Experiment 1 under its specified larger-dataset and independently sourced data conditions, but the design does not permit attributing that difference to dataset size alone.
>
> Experiment 1 and Experiment 2 differ simultaneously in dataset amount (600 vs 6,000), fake-generation method and family (face-swap vs StyleGAN), image framing (centre-cropped widescreen vs full square), source resolution and downsampling characteristics, compression and file format (PNG over `c23` video vs JPEG), and test-set size (120 vs 900). This experiment demonstrates the performance of this particular larger, independently sourced dataset/model setup; it does **not** isolate dataset size as a causal variable.

The two experiments should be read as **evaluations under different dataset conditions**, not as a controlled ablation of dataset size. A genuine size experiment would hold every other variable fixed and vary only the sample count — which, here, would require additional FaceForensics++ source videos, on the order of a 30 GB download. That was evaluated and rejected for this iteration.

**Statements that must not be made about Experiment 2:**

- ~~"Accuracy increased because the dataset became larger."~~
- ~~"The 10× larger dataset improved detection."~~
- Any wording that presents Experiment 2 as a controlled demonstration of a dataset-size effect.

---

## 9. Experiment 1 versus Experiment 2

Experiment 1's column is its own frozen, authoritative evaluation report (`reports/evaluation_report.json`, generated `2026-09-22T20:19:05`). **It was not read into, written to, or otherwise touched by Experiment 2.**

| Metric (test split) | Experiment 1 — frozen baseline | Experiment 2 |
|---|---|---|
| Test images | 120 | 900 |
| Real / fake in test | 60 / 60 | 450 / 450 |
| **Accuracy** | **`0.7500` (75.00%)** | **`0.9756` (97.56%)** |
| Macro precision | `0.8333` | `0.9763` |
| Macro recall | `0.7500` | `0.9756` |
| Macro F1 | `0.7333` | `0.9755` |
| Precision (`fake`) | `1.0000` | `0.9573` |
| Recall (`fake`) | `0.5000` | `0.9956` |
| F1 (`fake`) | `0.6667` | `0.9760` |
| Precision (`real`) | `0.6667` | `0.9954` |
| Recall (`real`) | `1.0000` | `0.9556` |
| F1 (`real`) | `0.8000` | `0.9751` |
| Confusion matrix | `[[60, 0], [30, 30]]` | `[[430, 20], [2, 448]]` |
| Checkpoint epoch | 6 | 10 |
| Checkpoint validation accuracy | `0.7833` | `0.9689` |
| Device | CPU | CPU |

**This table is a description of two separate results. It is not evidence of a causal effect of dataset size.** Section 8.2 states why.

### How to read Experiment 1's numbers

Experiment 1's pattern is degenerate, and this is worth stating plainly rather than presenting `0.7500` and `0.9756` as a clean before/after. Experiment 1 flagged **zero** real images as fake. Its perfect `1.0000` fake precision is therefore an artifact of barely attempting any fake prediction — it **missed half of its fake images (30 of 60)**. Its `0.6667` real precision reflects the same behaviour: of the 90 images it called real, 60 were genuinely real.

Experiment 2's improvement is driven almost entirely by **fake recall** (`0.5000` → `0.9956`), not by precision, and it pays for that gain with 20 false alarms. The two models fail in genuinely different directions: Experiment 1 was dangerously permissive about fakes, Experiment 2 is mildly permissive about reals.

Even so, neither figure describes a general-purpose deepfake detector. Section 8.2's distribution caveat applies to the whole comparison, and Section 6's convergence caveat applies to Experiment 2's absolute number.

---

## 10. Reproducibility

### 10.1 Committed artifacts

| Artifact | Path | Contents |
|---|---|---|
| Dataset build report | `experiments/experiment_2_large_dataset/dataset_report.json` | source, quotas, seed, pHash threshold, de-duplication counts, audit results |
| Dataset manifest | `experiments/experiment_2_large_dataset/dataset_manifest.csv` | one row per selected image: image id, label, split, official split, source file, bytes, MD5, pHash — 6,000 data rows, no absolute paths, no pixels |
| Training summary | `experiments/experiment_2_large_dataset/training_summary.json` | hyperparameters, epochs, best epoch, duration, dataset fingerprint |
| Test evaluation report | `experiments/experiment_2_large_dataset/reports/evaluation_report.json` | official Experiment 2 test metrics |
| Experiment 2 checkpoint | `models/experiment_2_deepguard_efficientnet_b0.pt` | epoch-10 weights |
| Dataset builder | `experiments/experiment_2_large_dataset/build_dataset.py` | scan, MD5/pHash de-duplication, deterministic subsampling, audit |
| Training script | `experiments/experiment_2_large_dataset/train_experiment_2.py` | resume-capable trainer; recomputes the dataset fingerprint at startup |
| Plot generator | `experiments/experiment_2_large_dataset/generate_plots.py` | renders the four plots from recorded values only |
| Experiment README | `experiments/experiment_2_large_dataset/README.md` | full methodology, commands, caveats |
| Training run log | `experiments/experiment_2_large_dataset/train_run_log.txt` | historical record — see 10.3 |
| Test confusion matrix | `experiments/experiment_2_large_dataset/reports/confusion_matrix.png` | rendered from the evaluation report |
| Per-class metrics plot | `experiments/experiment_2_large_dataset/reports/per_class_metrics.png` | test precision / recall / F1 per class |
| Training loss curve | `experiments/experiment_2_large_dataset/reports/training_loss_curve.png` | train vs validation loss by epoch |
| Training accuracy curve | `experiments/experiment_2_large_dataset/reports/training_accuracy_curve.png` | train vs validation accuracy by epoch |

The shared evaluation entry point is `evaluate_model.py`, which accepts `--reports-dir` so a second experiment cannot overwrite the first experiment's frozen report.

### 10.2 Not committed, and why

| Path | Reason |
|---|---|
| `experiments/experiment_2_large_dataset/source/` | the ≈ 4 GB Kaggle dataset — CC BY-NC-SA 4.0, not redistributable |
| `experiments/experiment_2_large_dataset/processed/` | the 6,000 selected images — reproducible via `build_dataset.py` |
| `models/experiment_2_training_state.pt` | large resume state, meaningless without the dataset |

The images are therefore reproducible but not redistributable. `dataset_report.json` and `dataset_manifest.csv` carry the counts, hashes and audit results needed to verify a rebuild without shipping pixels.

### 10.3 Known record gaps, stated rather than papered over

- **`dataset_validation.json` is not present.** It would be produced by a `--validate-only` re-audit, which was not run for this experiment. The audit results quoted in this report come from `dataset_report.json` and from an independent post-build re-verification of all 6,000 copied files.
- **`train_run_log.txt` is header-only (170 bytes).** The completed run left the file containing only its "Run started" header, because the `Tee` object created in `train_experiment_2.py` was never passed to the `log()` calls, so per-epoch lines went to stdout only. No data was lost — the per-epoch history was recovered from the resume state and the captured console output, and every value in this report comes from those committed artifacts. The logging wiring has since been corrected so future runs record their epochs; the historical file was deliberately **not** rewritten, since fabricating a fuller log after the fact would misrepresent the run. **No retraining or re-evaluation was performed as part of that fix.**

### 10.4 Reproducing the build

```powershell
# Build the dataset (slowest step: hashing 140,000 images).
# Replace the path with wherever YOU extracted the dataset.
venv\Scripts\python.exe experiments\experiment_2_large_dataset\build_dataset.py `
    --source-dir "E:\SIP\140k_dataset"

# Recommended first pass: report what would be selected, write nothing
venv\Scripts\python.exe experiments\experiment_2_large_dataset\build_dataset.py `
    --source-dir "E:\SIP\140k_dataset" --dry-run --jobs 8

# Re-audit an existing build at any time
venv\Scripts\python.exe experiments\experiment_2_large_dataset\build_dataset.py `
    --validate-only
```

The dataset build needs only numpy and Pillow, so it runs in the app environment. **Training must use `venv-train`.** Reproducing Experiment 2 requires re-downloading the source dataset, re-running the build, and approximately ten hours of CPU training.

---

## 11. Conclusion

Experiment 2 evaluated DeepGuard on a 6,000-image, class-balanced subset of the Kaggle *140k Real and Fake Faces* dataset — 3,000 real and 3,000 synthetic, selected deterministically at seed 42, de-duplicated by MD5 and 64-bit perceptual hash, and verified by six passing build audits with no cross-split leakage.

On the 900-image held-out test split it never saw during training, the EfficientNet-B0 model reached a **test accuracy of `0.9756` (97.56%)** with a **macro F1 of `0.9755`**, a confusion matrix of `[[430, 20], [2, 448]]`, fake recall of `0.9956` and real recall of `0.9556`. All of these figures are recorded in committed, machine-readable artifacts and none were regenerated for this report.

**What this result does and does not establish.** It establishes that this architecture, trained under this protocol on this dataset, reaches 97.56% test accuracy on this test split — with a balanced test set, no duplicate leakage, and the test split excluded from all model selection. It does **not** establish that a larger dataset causes better deepfake detection. Experiment 2 changed six variables at once relative to Experiment 1, including the generative family that produced its fake images: Experiment 1's fakes are FaceForensics++ face-swaps, Experiment 2's are StyleGAN generations. Part or most of the gap from Experiment 1's 75.00% may reflect that easier fake source, or the different framing, resolution and compression characteristics, rather than the tenfold increase in data.

Experiment 2 also has two honest limits of its own: the best checkpoint landed on the final epoch with validation loss still falling, so the model was probably not fully converged and 97.56% is not a ceiling; and two of the three cross-split near-duplicate margins are a single bit, making the no-leakage guarantee threshold-relative rather than comfortable.

The correct reading is therefore comparative but bounded: **Experiment 2 outperformed the frozen Experiment 1 baseline under its own dataset conditions, and the different dataset distribution prevents attributing that difference to dataset size alone.** Establishing a causal dataset-size effect would require a further experiment in which the fake-generation method, framing, resolution and compression are held fixed and only the sample count varies.

---

*Prepared from the frozen Experiment 2 artifacts. No training, evaluation, threshold tuning or metric generation was performed in producing this report. Experiment 1 remains the frozen academic baseline and is unmodified.*
