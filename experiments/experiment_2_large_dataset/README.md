# Experiment 2 - Larger Dataset (6,000 images)

**Status: COMPLETE.** Dataset built, model trained, test split evaluated.
All timestamps below are from the completed run on **2026-10-04**.

**Headline result - test accuracy `0.9756`** on the untouched 900-image
Experiment 2 test split (macro F1 `0.9755`). Experiment 1's frozen test
accuracy is `0.7500`.

> **Read this before quoting any number from this folder.**
>
> 1. The reported result is the **test** accuracy `0.9756`, **not** the
>    validation accuracy `0.9689`. Validation accuracy is not the final result.
> 2. **This is not a dataset-size result.** Experiment 2 did not "improve
>    accuracy because the dataset was larger." The two experiments differ in six
>    dimensions simultaneously and this design cannot isolate any one of them.
>    See the next section.

Experiment 2 trains a second EfficientNet-B0 classifier on a 6,000-image dataset
(3,000 real + 3,000 fake) and evaluates it on its own held-out test split.

---

## READ THIS FIRST: what Experiment 2 can and cannot prove

> **Experiment 2 is NOT a pure "10x more data" experiment.**
>
> | | Experiment 1 | Experiment 2 |
> |---|---|---|
> | Images | 600 | 6,000 |
> | Dataset | FaceForensics++ (10 videos, 30 frames each) | Kaggle "140k Real and Fake Faces" |
> | Source frames | video frames from `c23`-compressed videos | independent still images |
> | Source dimensions | widescreen / 4:3 (e.g. 1280x720, 1920x1080, 856x480, 656x480) | 256x256 square |
> | "real" means | original video frames | FFHQ / Flickr photographs |
> | "fake" means | FaceForensics **face-swap** manipulation | **StyleGAN** synthesis |
> | File format on disk | PNG | JPEG |
> | Compression artefacts | compressed `c23` video content, stored losslessly as PNG | JPEG compression / quantization artefacts |
> | **Test-set size** | **120 images (60 real + 60 fake)** | **900 images (450 real + 450 fake)** |
>
> **Six things change at once, not one:**
>
> 1. the amount of data (600 -> 6,000 images)
> 2. the fake-generation method (face-swap -> StyleGAN synthesis)
> 3. the image framing (widescreen centre-cropped frame -> full square image)
> 4. the source resolution and therefore the amount of downsampling
> 5. the compression / file-format characteristics
> 6. the size of the test set the result is measured on (120 -> 900 images)
>
> Any difference in the test metrics is caused by *some combination* of these.
> None of them can be isolated with the evidence this iteration produces.
>
> **The report must NOT claim "accuracy improved because the dataset was
> larger."** That statement is not supported by this design.
>
> The wording actually used for the completed run is:
>
> > *Experiment 2 achieved higher test performance than Experiment 1 under its
> > specified larger-dataset and independently sourced data conditions, but the
> > design does not permit attributing that difference to dataset size alone.*
>
> Note in particular that Experiment 1's fakes are **face-swap** artefacts while
> Experiment 2's are **StyleGAN generations** - a different generative family
> with a different artefact signature. A detector's difficulty can differ
> sharply between the two families on its own, so part or most of the observed
> gap may reflect *an easier fake source*, not a better-trained model.
>
> The correct framing is:
>
> > *Experiment 2 evaluates the model using a substantially larger and
> > differently distributed dataset, with differences in fake-generation
> > method, image framing, source resolution, and compression
> > characteristics.*
>
> A controlled size experiment would hold every other variable fixed and change
> only the sample count. That would need more FaceForensics++ videos (see
> `docs/dataset_research.md`), which requires dataset access and roughly a 30 GB
> download. It was evaluated and rejected for this iteration.

A second limitation: this dataset publishes **no identity metadata**, so
**identity-level grouping is not possible**. Leakage control therefore rests on
the publisher's official split plus de-duplication, not on person-level
separation. Experiment 1 had identity-component grouping; Experiment 2 does
not. This makes Experiment 1's split stricter, not Experiment 2's.

---

## Completed run results

Everything in this section is a recorded measurement from the completed build,
training and evaluation runs of 2026-10-04. Nothing here is projected or
estimated.

### Final dataset composition

| Split | Real | Fake | Total | Drawn from |
|---|---|---|---|---|
| `train` | 2,100 | 2,100 | 4,200 | publisher's `train` folders |
| `validation` | 450 | 450 | 900 | publisher's `valid` folders |
| `test` | 450 | 450 | 900 | publisher's `test` folders |
| **Total** | **3,000** | **3,000** | **6,000** | 140,000-image source pool |

All images are JPEG, RGB, 256x256. Seed `42`. Every split is class-balanced.

### Dataset integrity identifiers

| Identifier | Value |
|---|---|
| Dataset fingerprint (SHA-256 over the manifest's `split/label/md5` rows) | `cafc017380d10484326d668e2893f3582ddeb8c5e23c278569804d7e44f3d8bb` |
| Manifest SHA-256 (`dataset_manifest.csv`) | `374445e3a27e87942e17f6b0335dfc59dfc38afe37e1596f5b6232fd98c00922` |
| Manifest data rows | 6,000 |

`train_experiment_2.py` recomputes the fingerprint at startup and refuses to
resume against a changed dataset, so the recorded fingerprint proves the
training run used exactly this dataset.

### Build and de-duplication outcome

| Measure | Value |
|---|---|
| Source pool scanned | 140,000 images (70,000 real + 70,000 fake) |
| pHash threshold | 3 bits |
| **Near-duplicates removed** | **5,069** |
| Exact duplicates removed, same class + same split | 0 |
| Exact duplicates removed, cross-split | 0 |
| Exact duplicates removed, cross-class | 0 |
| Min cross-split pHash distance, `train` <-> `validation` | **4 bits** |
| Min cross-split pHash distance, `train` <-> `test` | **4 bits** |
| Min cross-split pHash distance, `validation` <-> `test` | 6 bits |
| Cross-split near-duplicate audit | **passed** |

All six build audits passed (`exact_counts_match_quota`,
`classes_balanced_in_every_split`, `no_exact_duplicate_across_splits`,
`no_image_labelled_both_real_and_fake`,
`experiment_split_maps_to_one_official_split`,
`no_near_duplicate_across_splits`).

### Training outcome

| Setting | Value |
|---|---|
| Architecture | `efficientnet_b0`, ImageNet-pretrained |
| Device | CPU |
| Batch size | 8 |
| Optimiser | AdamW, lr `3e-4`, weight decay `1e-4` |
| LR schedule | ReduceLROnPlateau (`min`, factor 0.1, patience 2) |
| Seed | 42 |
| Epochs requested / completed | 10 / **10** |
| **Best epoch** | **10** |
| Best validation loss | `0.0684` |
| Best validation accuracy | `0.9689` |
| Final training loss | `0.0934` |
| Final training accuracy | `0.9676` |
| Early stopping | **no** (never triggered) |
| Learning rate across all 10 epochs | `3.00e-04` - **never reduced** |
| Wall-clock duration | **~600.5 minutes (10.0 h)** |

### Checkpoint

| Property | Value |
|---|---|
| Path | `models\experiment_2_deepguard_efficientnet_b0.pt` |
| Size | 16,346,305 bytes |
| SHA-256 | `163D6B5BBDBF5EE034863E09DC4FB6AEC3E9F29DA0D066785785D37011B7FD9C` |
| Saved from epoch | 10 (lowest validation loss) |

### FINAL TEST RESULT - the number to quote

Measured by `evaluate_model.py` on the **untouched 900-image test split**
(450 real + 450 fake). This is the reported result of Experiment 2.

| Metric | Value |
|---|---|
| Test images | 900 |
| **Accuracy** | **`0.9756`** |
| Macro precision | `0.9763` |
| Macro recall | `0.9756` |
| Macro F1 | `0.9755` |
| Precision (fake) | `0.9573` |
| Recall (fake) | `0.9956` |
| F1 (fake) | `0.9760` |
| Precision (real) | `0.9954` |
| Recall (real) | `0.9556` |
| F1 (real) | `0.9751` |
| Report | `experiments\experiment_2_large_dataset\reports\evaluation_report.json` |
| Report SHA-256 | `AD67BE142B54E4DF6CDE9E464F7B438A9E6021D3278A0BFF9C63150FB5453930` |

### Confusion matrix - test split

Rows are **actual** `[real, fake]`, columns are **predicted** `[real, fake]`:

| | predicted real | predicted fake |
|---|---|---|
| **actual real** | 430 | 20 |
| **actual fake** | 2 | 448 |

- **False positives: 20** real images wrongly called fake.
- **False negatives: 2** fake images called real.
- Correct: 430 + 448 = **878 of 900** (0.9756).

The detector is mildly biased towards calling images fake: fake recall `0.9956`
against real recall `0.9556`. On this balanced test set that costs almost
nothing, but it would matter under a different real/fake prior.

### Validation vs test - do not mix these up

| | Validation | Test |
|---|---|---|
| Images | 900 | 900 |
| Accuracy | `0.9689` | **`0.9756`** |
| Loss | `0.0684` | not produced by the evaluator |

The model was allowed to learn from the validation split (it selected the best
epoch by validation loss), so `0.9689` is **not** the reported result.
**Experiment 2's final result is the test accuracy `0.9756`.**

Experiment 2's validation accuracy is also **not directly comparable** to
Experiment 1's test accuracy: different split, different data, different build.

### Experiment 1 vs Experiment 2 - test-split comparison

Experiment 1's column is the frozen authoritative result from
`reports\evaluation_report.json` and was not touched by this experiment.

| Metric | Experiment 1 (test) | Experiment 2 (test) |
|---|---|---|
| Test images | 120 | 900 |
| **Accuracy** | `0.7500` | **`0.9756`** |
| Macro precision | `0.8333` | `0.9763` |
| Macro recall | `0.7500` | `0.9756` |
| Macro F1 | `0.7333` | `0.9755` |
| Precision (fake) | `1.0000` | `0.9573` |
| Recall (fake) | `0.5000` | `0.9956` |
| F1 (fake) | `0.6667` | `0.9760` |
| Precision (real) | `0.6667` | `0.9954` |
| Recall (real) | `1.0000` | `0.9556` |
| F1 (real) | `0.8000` | `0.9751` |
| Confusion matrix | `[[60, 0], [30, 30]]` | `[[430, 20], [2, 448]]` |

**How to read Experiment 1's numbers.** Its pattern is degenerate: it flagged
**zero** real images as fake, so its `1.0000` fake precision is an artifact of
barely attempting any fake prediction - it **missed half its fakes** (30 of 60).
Its `0.6667` real precision reflects the same behaviour. Experiment 2's gap is
driven almost entirely by **fake recall** (`0.5000` -> `0.9956`), not by
precision, and it pays for that with 20 false alarms.

**The correct conclusion, and only this one:**

> Experiment 2 achieved higher test performance than Experiment 1 under its
> specified larger-dataset and independently sourced data conditions, but the
> design does not permit attributing that difference to dataset size alone.
>
> Experiment 1 and Experiment 2 differ simultaneously in dataset amount
> (600 vs 6,000), fake-generation method and family (face-swap vs StyleGAN),
> image framing (centre-cropped widescreen vs full square), source
> resolution / downsampling characteristics, compression and file format
> (PNG over `c23` video vs JPEG), and test-set size (120 vs 900). This
> experiment demonstrates the performance of this particular larger,
> independently sourced dataset/model setup; it does **not** isolate dataset
> size as a causal variable.

---

## Caveats on the completed run

These are properties of the finished experiment and must travel with its result.

1. **Experiment 2's fake pool contained substantially more near-duplicates
   before de-duplication.** 5,069 near-duplicates were removed during the build.
   The surviving fake pool is de-duplicated and the leakage audit passed, but the
   raw pool's redundancy is itself a difference from Experiment 1's process.

2. **The minimum cross-split pHash distance was only 4 bits against a threshold
   of 3 bits** (for `train` <-> `validation` and `train` <-> `test`). The audit
   passed because no pair fell inside the 3-bit threshold, but the margin is a
   single bit. The no-leakage guarantee is *threshold-relative*, not
   comfortable, and a stricter threshold could have flagged borderline pairs.

3. **The 900-image test set is much larger than Experiment 1's 120 images**, so
   its estimate is less sensitive to any individual example. The two accuracies
   are not equally precise, and this must be stated whenever they are compared.
   (See *Statistical uncertainty* below.)

4. **The best checkpoint occurred at the final epoch (10 of 10).** Validation
   loss was still falling when the epoch budget ran out, so **the model was
   probably not fully converged** and `0.9756` should not be read as a ceiling.
   More epochs might change it - in either direction, since a longer run also
   risks overfitting.

5. **Validation accuracy is not the reported result** - see
   *Validation vs test* above. Quote `0.9756`.

6. **The historical `train_run_log.txt` is header-only, and that is now
   explained rather than papered over.** The completed run left that file at
   170 bytes containing only the "Run started" header, because the `Tee` object
   created in `train_experiment_2.py` was never passed to the `log()` calls, so
   per-epoch lines went to stdout only. **No data was lost** - the full
   per-epoch history was recovered from `models\experiment_2_training_state.pt`
   and from the captured console output, and every number in this README comes
   from those sources.

   **Status of the fix (2026-10-04):** the source defect has since been
   corrected, so **future** runs will write their per-epoch lines to the run
   log. The change was confined to the logging wiring: `log()` now falls back
   to the run log created by `main()` when no explicit handle is passed, instead
   of silently writing to the console only.

   What this fix deliberately does **not** do:

   - it does **not** rewrite the historical 170-byte `train_run_log.txt`, which
     remains exactly as the completed run left it. That file is a truthful
     record of what was written at the time, and fabricating a fuller log after
     the fact would misrepresent the run;
   - it does **not** alter any Experiment 2 measurement, artifact or conclusion
     in this document - the checkpoint, evaluation report, dataset manifest,
     resume state and training summary are all byte-identical to before;
   - it does **not** mean the model was retrained. **No training, evaluation or
     dataset rebuild was performed after the fix.** The fix was verified in
     isolation, without running a training job.

---

## Source dataset

| Property | Value |
|---|---|
| Name | 140k Real and Fake Faces |
| Publisher | xhlulu, Kaggle |
| URL | <https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces> |
| Size | ~4 GB (version 2) |
| Content | 70,000 real faces (NVIDIA FFHQ / Flickr photographs) + 70,000 StyleGAN-generated fake faces, all resized to 256 px |
| Verified image properties | 140,000 files, all JPEG, all RGB, all 256x256, all readable |
| Official split | the publisher ships its own `train` / `valid` / `test` folders |
| License | "Other (specified in description)". Underlying FFHQ images are **CC BY-NC-SA 4.0** (NVIDIA): non-commercial use, attribution required |

**Non-commercial, no redistribution.** This repository therefore commits the
manifest and the audit reports, never the images.

### Verified on-disk layout

The dataset is kept **outside** this repository. After extraction it looks like
this (this is the layout that was actually inspected and verified):

```
<dataset-root>\
├── train.csv                 metadata (not read by build_dataset.py)
├── valid.csv
├── test.csv
└── real_vs_fake\
    └── real-vs-fake\         <- the "pool root"
        ├── train\
        │   ├── real\         50,000 images
        │   └── fake\        50,000 images
        ├── valid\
        │   ├── real\         10,000 images
        │   └── fake\        10,000 images
        └── test\
            ├── real\         10,000 images
            └── fake\        10,000 images
```

Total: **140,000 images** (70,000 real + 70,000 fake).

`build_dataset.py` supports this nested layout (its "Layout B"). It also
supports a flat alternative ("Layout A"), because different releases of the
dataset ship different shapes:

| | Layout A (flat) | Layout B (nested, verified here) |
|---|---|---|
| train | `<root>/train_real/`, `<root>/train_fake/` | `<root>/train/real/`, `<root>/train/fake/` |
| valid | `<root>/valid_real/`, `<root>/valid_fake/` | `<root>/valid/real/`, `<root>/valid/fake/` |
| test | `<root>/test_real/`, `<root>/test_fake/` | `<root>/test/real/`, `<root>/test/fake/` |

A layout is accepted only when **all six** split x class folders are present, so
a partially-extracted download is rejected rather than silently used.

### File names do not encode the class

Verified examples from the dataset:

| Class | Example file names |
|---|---|
| real | `00000.jpg`, `00002.jpg` |
| fake | `001DDU0NI4.jpg`, `002KDWZBHU.jpg` |

The real and fake files use **different naming conventions** (real files are
numbered, fake files carry hash-like names), but **neither name encodes the
class**. A file name is never treated as evidence of its label.

**The containing folder name is the authoritative class label.**
`build_dataset.py` reads the label from `.../train/real/` vs `.../train/fake/`
and treats the folder as authoritative. It additionally counts any file whose
name *starts* with `real`/`fake` but sits in the opposite folder and reports
that count as a warning; with this dataset's naming that check simply never
fires.

### Compression and file format

| | Experiment 1 | Experiment 2 |
|---|---|---|
| On-disk format | PNG | JPEG |
| Where the artefacts come from | the visual content is `c23`-compressed H.263 video; the PNG wrapper adds none of its own | the source files are JPEG, so they carry JPEG compression and quantization artefacts |
| Recompressed by the pipeline? | no | no |

**Experiment 2 images are not artefact-free.** `build_dataset.py` copies the
selected source files byte-for-byte (`shutil.copy2`) and never re-encodes them,
so whatever compression signature the publisher's files carry is preserved
exactly.

### ⚠️ Compression-domain confound in the comparison

Because Experiment 1 and Experiment 2 differ in file format *and* in the origin
of their compression artefacts, the comparison may contain a
**compression-domain confound**:

- Experiment 1: compressed **video** content stored in **PNG**
- Experiment 2: **JPEG** images with **JPEG** artefacts

A classifier can potentially learn "which compression pipeline produced this
image" as a shortcut to the label, independently of any manipulation trace.
With only two datasets and no format control, that shortcut cannot be ruled
out from this evidence.

This is a **limitation of the experimental comparison**. It is disclosed here
rather than hidden, and it is restated in the Experiment 2 report.

---

## How the dataset is built

`build_dataset.py` runs five steps. Nothing is copied until every check passes.

1. **Scan** the six official folders (140,000 images).
2. **Exact-duplicate removal (MD5).** Byte-identical images collapse to one.
   - same class + same split -> keep one
   - same class + different splits -> keep the copy in the higher-priority
     split (`test` > `validation` > `train`) and drop the others, which removes
     any leakage the source dataset introduced
   - **different classes -> drop every copy**, because the label contradicts
     itself
3. **Near-duplicate removal (pHash).** A 64-bit DCT perceptual hash per image.
   Two images within `--phash-threshold` bits (default 3) are near-duplicates;
   the later one is dropped. Candidate pairs are found by cutting the hash into
   `threshold + 1` disjoint bands (multi-index hashing), which is *exact* for
   Hamming distances up to the threshold - no false negatives.
4. **Deterministic subsampling** to 2,100 / 450 / 450 per class using a
   per-pool seed derived from `--seed 42`.
5. **Audit** (below).

### Split policy

Experiment 2 never crosses the publisher's official split boundaries:

| Experiment 2 split | Drawn only from | Images | Per class |
|---|---|---|---|
| `train` | `train_real`, `train_fake` | 4,200 | 2,100 |
| `validation` | `valid_real`, `valid_fake` | 900 | 450 |
| `test` | `test_real`, `test_fake` | 900 | 450 |

Classes are balanced in every split.

### Audit checks

| Check | Meaning |
|---|---|
| `exact_counts_match_quota` | every split has exactly the requested images |
| `classes_balanced_in_every_split` | real count == fake count in each split |
| `no_exact_duplicate_across_splits` | no MD5 appears in two splits |
| `no_image_labelled_both_real_and_fake` | no contradictory labels |
| `experiment_split_maps_to_one_official_split` | no image crosses the official boundary |
| `no_near_duplicate_across_splits` | minimum cross-split pHash distance > 3 bits |
| `no_missing_files` | every manifest row exists on disk |
| `no_md5_mismatch` | file contents match the manifest (no corruption/tampering) |
| `no_unreadable_files` | every selected image decodes |
| `no_unexpected_extra_files` | no unlisted image in the processed folders |
| `unlisted_files_do_not_duplicate_manifest_images` | any unlisted file was proven not to duplicate a manifest image |

The build and the validator both exit non-zero if any check fails.

---

## Commands

Run from the project root. The dataset build needs only numpy + Pillow, so it
uses the app environment; training must use `venv-train`.

### Pointing `--source-dir` at your copy of the dataset

`--source-dir` is **not** fixed to any particular machine or folder. It accepts
the dataset's **parent directory** (the folder that *contains* `real_vs_fake`),
and `build_dataset.py` locates the pool root underneath it automatically by
searching three levels deep. So any of these work:

```
--source-dir "E:\SIP\140k_dataset"                       # parent of real_vs_fake
--source-dir "E:\SIP\140k_dataset\real_vs_fake"           # the real_vs_fake folder itself
--source-dir "E:\SIP\140k_dataset\real_vs_fake\real-vs-fake"  # the pool root itself
```

**Replace the example path below with your own local dataset location.** The
path above is simply the location used on the machine where this experiment was
prepared; it is an illustration, not a requirement. The default, if you omit
`--source-dir` entirely, is `experiments/experiment_2_large_dataset/source`, so
the dataset may also live inside the project if you prefer - though keeping it
outside is recommended, because it is ~4 GB and its licence forbids
redistribution.

```powershell
# 1. Build the dataset (slowest step: hashing 140,000 images)
#    Replace the path with wherever YOU extracted the dataset.
venv\Scripts\python.exe experiments\experiment_2_large_dataset\build_dataset.py `
    --source-dir "E:\SIP\140k_dataset"

# 1a. Recommended first pass: report what would be selected, write nothing
venv\Scripts\python.exe experiments\experiment_2_large_dataset\build_dataset.py `
    --source-dir "E:\SIP\140k_dataset" --dry-run --jobs 8

# 2. Re-audit an existing build at any time
venv\Scripts\python.exe experiments\experiment_2_large_dataset\build_dataset.py `
    --validate-only

# 3. Train (expect roughly 7-10 hours on CPU)
venv-train\Scripts\python.exe experiments\experiment_2_large_dataset\train_experiment_2.py

# 3b. If the run is interrupted, continue instead of restarting
venv-train\Scripts\python.exe experiments\experiment_2_large_dataset\train_experiment_2.py `
    --resume

# 4. Evaluate on the test split into this experiment's OWN reports folder
venv-train\Scripts\python.exe evaluate_model.py `
    --processed-dir experiments\experiment_2_large_dataset\processed `
    --checkpoint models\experiment_2_deepguard_efficientnet_b0.pt `
    --reports-dir experiments\experiment_2_large_dataset\reports
```

`--dry-run` reports what would be selected without writing anything.
`--overwrite` rebuilds `processed/` from scratch.

---

## Training configuration

Hyperparameters are pinned to what Experiment 1's run **actually used** (see
`reports/stage6/train_run_log.txt`), so that the training recipe is held
constant and the dataset is the intended variable. Note that this is *not* the
same as matching `config.py`'s defaults - Experiment 1 was trained with batch 8
even though `config.BATCH_SIZE` is 16.

| Setting | Value | Matches Experiment 1 run |
|---|---|---|
| Architecture | `efficientnet_b0` | yes |
| ImageNet pretrained | yes | yes |
| Model input | 224 x 224 | yes |
| Batch size | 8 (matches the run, not `config.BATCH_SIZE` = 16) | yes |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4 | yes |
| LR schedule | ReduceLROnPlateau, mode `min`, factor 0.1, patience 2 | yes |
| Max epochs | 10 | yes |
| Early-stopping patience | 3, on validation loss | yes |
| Seed | 42 | yes |
| Device | CPU | yes |

### Preprocessing: what is shared, and what is not

**The model-side preprocessing is identical in both experiments.** Neither
experiment defines its own transforms. `train_experiment_2.py` calls
`training.build_dataloaders()`, which supplies Experiment 1's own transform
functions unchanged, and Experiment 2's evaluation reuses
`training.get_eval_transforms()` in exactly the same way.

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Model input size | 224 x 224 | 224 x 224 |
| Normalisation | ImageNet mean/std | ImageNet mean/std (same values, same code) |
| Training transforms | `RandomResizedCrop(224, 0.8-1.0)`, flip 0.5, rotate ±10° | identical |
| Validation / test transforms | `Resize(224)` + `CenterCrop(224)` | identical |

So the transform *code* and the tensor *dimensions* are the same.

**However, the source images are not the same**, and that changes what the
network actually sees:

| | Experiment 1 source | Experiment 2 source |
|---|---|---|
| Aspect | widescreen / 4:3 | square |
| Measured examples | 1280x720, 1920x1080, 856x480, 656x480 | 256x256 |
| After `Resize(224)` | shorter side -> 224, aspect preserved, e.g. 398x224 | 224x224 |
| After `CenterCrop(224)` | crops the sides away | effectively a no-op |
| **Fraction of the source frame the model sees** | **about 56%-73%** | **100%** |

Two consequences:

1. **Different field of view.** Experiment 1's model input is a centre-cropped,
   effectively zoomed-in patch with the surrounding context removed.
   Experiment 2's model input is the whole square image, background included.
   Forgery cues that live near the face boundary or in the background are
   therefore available to one experiment and not the other.
2. **Different amount of source downsampling.** Experiment 1 downscales by
   roughly 3.2x before cropping; Experiment 2 downscales by about 1.14x. Since
   fake-image cues are largely high-frequency (resampling fingerprints, noise
   inconsistencies, GAN upsampling artefacts), the amount of high-frequency
   signal surviving into the 224x224 tensor is not comparable.

**Therefore these experiments do not isolate dataset size.** They share a
preprocessing pipeline but not an input distribution. This is stated here so
that it is not discovered later and mistaken for a dataset-size effect.

### Resume (opt-in)

`train_experiment_2.py` reuses the building blocks of `src/training.py` but runs
its own outer loop so it can checkpoint the optimiser state. **Resume is opt-in:
without `--resume` it always starts fresh**, which is the default.

* `models/experiment_2_training_state.pt` is rewritten after every epoch using an
  atomic replace, so an interrupted write cannot corrupt it.
* Resuming refuses to run if the dataset changed since the state was written
  (the manifest is fingerprinted with SHA-256).
* Resuming refuses to start if all requested epochs are already done.

`src/training.py` and `train_model.py` are **not modified**. Experiment 1's code
path is untouched.

### Safety guards

The script refuses to write to `models/deepguard_efficientnet_b0.pt` and refuses
to write anywhere inside `reports/`, so Experiment 1 cannot be damaged by a
mistyped path.

---

## Runtime - estimate vs measured

The pre-run estimate and the measured result:

| | Estimated before the run | Measured |
|---|---|---|
| Per epoch | ~62 min | **54-94 min** (epochs 9 and 10 were slowed to ~94 and ~58 min by other CPU load) |
| Full 10-epoch training | ~10.3 h | **600.5 min = 10.0 h** |
| Early stopping | possible at 6-7 epochs (~7 h) | **did not trigger** - all 10 epochs ran |
| Test evaluation (900 images) | ~11 min | a few minutes |

Experiment 2 has 4,200 + 900 = 5,100 images per epoch, about 11x more than
Experiment 1's 480. Training used only ~2.2 of the machine's 8 logical CPU
cores because the IDE, browser and other tools were competing for the rest; on
an otherwise idle machine it should be faster than the figures above.

Keep the machine plugged in, awake and ventilated. If a run stops, re-run with
`--resume`.

---

## Statistical uncertainty

The two experiments are evaluated on **different-sized test sets**:

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Test images | 120 | 900 |
| Source of the test split | 1 identity component, 60 real + 60 fake | publisher's `test` folders, 450 real + 450 fake |
| Weight of one image | 0.833 percentage points | 0.111 percentage points |

A test set of 120 images yields a much noisier accuracy estimate than a test set
of 900 images, **even when both estimates happen to be similar**. Any accuracy
measured on 120 images carries substantially more sampling uncertainty than one
measured on 900.

**This matters when comparing the two numbers.** Experiment 2's larger test set
gives a larger evaluation sample, so its estimate has **lower sampling
uncertainty** and its accuracy is less sensitive to any individual example.
Experiment 1's estimate is correspondingly more fragile.

In this run the two accuracies (`0.7500` on 120 images vs `0.9756` on 900) are
far apart relative to either estimate's uncertainty, so the *difference* itself
is not in doubt. What remains in doubt is **why** - see the six-dimension
comparison above.

No confidence intervals are quoted here. This project has not performed a
formal statistical calculation for either experiment, so any interval would be
an informal approximation. If intervals are wanted, they must be computed
properly and reported as such - they must not be presented as official results.

---

## Files in this folder

### Committed

| File | Contents |
|---|---|
| `README.md` | this document |
| `build_dataset.py` | dataset builder, de-duplication and auditor |
| `train_experiment_2.py` | resume-capable trainer |
| `generate_plots.py` | renders the four plots below from recorded values only |
| `dataset_manifest.csv` | one row per selected image (generated by the build) |
| `dataset_report.json` | build statistics, duplicate counts, audit results |
| `train_experiment_2.py` | resume-capable trainer; logging now captures per-epoch lines for future runs |
| `generate_plots.py` | renders the four plots below from recorded values only |
| `dataset_manifest.csv` | one row per selected image (generated by the build) |
| `dataset_report.json` | build statistics, duplicate counts, audit results |
| `train_run_log.txt` | training log - **historical, header-only** (170 bytes; completed run as written); full history from state/console/summary |
| `training_summary.json` | hyperparameters, best epoch and dataset fingerprint of the run |
| `training_summary.json` | hyperparameters, best epoch and dataset fingerprint of the run |
| `reports/evaluation_report.json` | Experiment 2's own test metrics |
| `reports/confusion_matrix.png` | test confusion matrix |
| `reports/per_class_metrics.png` | test precision / recall / F1 per class |
| `reports/training_loss_curve.png` | training vs validation loss by epoch |
| `reports/training_accuracy_curve.png` | training vs validation accuracy by epoch |

`dataset_validation.json` is **not present**. It would be produced by a
`--validate-only` re-audit, which was never run for this experiment; the audit
results quoted in this README come from `dataset_report.json` and from an
independent post-build re-verification of all 6,000 copied files.

### Plots

`generate_plots.py` renders the four PNGs with **Pillow only** - `matplotlib` is
not installed in this project and is deliberately not required. It reads only
`training_summary.json` and `reports\evaluation_report.json`, adds no new
empirical result, and writes only into `reports\`.

### Manifest schema

`dataset_manifest.csv` has no absolute paths and no pixels:

| Column | Meaning |
|---|---|
| `image_id` | filename inside `processed/`, prefixed with split and class |
| `label` | `real` or `fake` |
| `split` | `train`, `validation` or `test` |
| `official_split` | publisher's split the image came from |
| `source_file` | original path relative to the pool root |
| `bytes` | file size |
| `md5` | exact-content hash (deduplication + audit) |
| `phash_hex` | 64-bit perceptual hash (near-duplicate detection + audit) |

### Not committed (see `.gitignore`)

The ~4 GB Kaggle dataset is **not stored in this repository at all** - it is kept
outside it and reached via `--source-dir` (see above). These paths are ignored
so that a copy placed inside the project can never be committed by accident.

| Path | Reason |
|---|---|
| `source/` | optional in-project location for the ~4 GB Kaggle dataset, CC BY-NC-SA 4.0, not redistributable |
| `processed/` | the 6,000 selected images, reproduced by `build_dataset.py` |
| `../models/experiment_2_training_state.pt` | large, meaningless without the dataset |

---

## Completion status

This experiment is finished. The steps below were all carried out on 2026-10-04.

1. ✅ Dataset built from the 140,000-image source pool: 6,000 images
   (3,000 real + 3,000 fake), all six audits passed.
2. ✅ Trained for 10 epochs on CPU in ~600.5 minutes. No early stopping.
   Best epoch 10 (validation loss `0.0684`).
3. ✅ Evaluated on the untouched 900-image test split with `--reports-dir`
   pointed at this folder, so Experiment 1's `reports\` was never a target.
4. ✅ Test accuracy `0.9756`, macro F1 `0.9755`, confusion matrix
   `[[430, 20], [2, 448]]`.
5. ✅ Compared against Experiment 1 (`0.7500` accuracy, macro F1 `0.7333`) with
   the six-dimension limitation stated above.
6. ✅ Four plots rendered into `reports\`.
7. ⬜ `reports\experiment_2_report.md` and the root `README.md` have **not** been
   updated. This is deliberately left to a later task.

**For the record:** the outcome was favourable (`0.9756` against Experiment 1's
`0.7500`), but the hypothesised "more data helps" mechanism remains
**unproven**. The result is reported as a property of this particular setup, and
the possibility that a different fake source or framing - not the larger dataset
- accounts for much of the gap is stated explicitly rather than glossed over.