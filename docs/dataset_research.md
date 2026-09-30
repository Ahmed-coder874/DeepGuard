# DeepGuard - Dataset Research

Project: **Deepfake Technology** (24CGCS06)
Stage: **Stage 2 - Dataset selection and preparation**
Compiled from the official dataset pages and papers listed in the references.

> **Purpose of this document.** This is a factual comparison of publicly
> available datasets that could be used for an academic, image-based deepfake
> detection project. It deliberately does **not** declare one dataset as "the
> best". It explains the trade-offs and suggests a practical starting point.
> No dataset is downloaded automatically by this project.

---

## 1. Summary comparison

| Dataset | Media | Real/fake labels | Approx. size | Access / registration | Practical for a student prototype | Good for image classification |
| --- | --- | --- | --- | --- | --- | --- |
| FaceForensics++ (FF++) | Videos (frames + binary masks) | Yes, per manipulation method | 1000 source videos; raw is very large, compressed (c23) much smaller | Google form request; custom Terms of Use | Moderate | Yes (after frame extraction) |
| Celeb-DF (v1/v2) | Videos | Yes (real vs synthesised) | 590 real + 5,639 fake videos (v2) | Google/Tencent form; custom terms | Moderate | Yes (after frame extraction) |
| DFDC | Videos | Yes (real vs fake clips) | Over 100,000 clips from 3,426 actors; very large | Facebook AI page + Kaggle (rules acceptance) | Low (size) | Yes (after frame extraction) |
| DF40 | Images **and** videos | Yes, organised by 40 fake methods | ~50 GB train fake + ~93 GB test fake (plus real) | Google form; CC BY-NC 4.0 | Low for full set; possible for a small subset | Yes (images already provided) |
| WildDeepfake | Images (face sequences from videos) | Yes | 7,314 face sequences from 707 internet videos | Agreement form + academic verification; Hugging Face | Moderate to high (small) | Yes |
| 140k Real and Fake Faces | Images (256x256) | Yes; already split train/val/test | ~4 GB (approx.) | Free Kaggle account | High | Yes |

The exact sizes change over time and depend on the compression level and the
subset you download. Always confirm the current size and terms on the official
page before downloading.

---

## 2. FaceForensics++ (FF++)

- **Media type:** Video. It contains 1,000 original video sequences plus
  manipulated versions created with four automated methods (Deepfakes,
  Face2Face, FaceSwap, NeuralTextures). FaceShifter was added later. Binary
  masks are provided, so the data can be used for image classification,
  video classification and segmentation.
- **Real/fake labels:** Yes. The original sequences are the "real" class and
  each manipulation method is a "fake" class. Using one method at a time
  gives a clean binary real-vs-fake setup.
- **Approximate size:** 1,000 source videos. The raw version is very large
  (hundreds of GB); the commonly used compressed versions (for example c23)
  are far smaller but still tens of GB. Verify on the official page.
- **License / access:** Released under the FaceForensics Terms of Use
  (custom, research-oriented); the code is MIT. Access requires filling in a
  Google form; the download script is sent after approval.
- **Registration required:** Yes.
- **Practical for a student prototype:** Moderate. You must request access,
  download a large archive and extract frames. The compressed version is the
  realistic choice.
- **Suitable for image-based classification:** Yes, after extracting frames
  (for example with OpenCV) and optionally cropping the face.
- **Notes:** This is one of the standard benchmarks in the deepfake detection
  literature, which makes results easy to compare with published work.

## 3. Celeb-DF (v1 / v2)

- **Media type:** Video. Version 2 contains 590 original YouTube videos and
  5,639 corresponding DeepFake videos, plus 300 additional real YouTube
  videos. A testing list of 518 videos is provided.
- **Real/fake labels:** Yes (real vs synthesised).
- **Approximate size:** Roughly 6,000+ videos in total; large (tens to
  hundreds of GB depending on version and quality). Verify on the official page.
- **License / access:** Released under the "Terms to Use Celeb-DF" (custom,
  research). Access requires a Google or Tencent form.
- **Registration required:** Yes.
- **Practical for a student prototype:** Moderate. It is known for higher
  visual quality and is more challenging than FF++, which is good for testing
  but makes high accuracy harder.
- **Suitable for image-based classification:** Yes, after frame extraction.
- **Notes:** Useful as a cross-dataset test set (train on FF++, test on
  Celeb-DF) to show generalisation.

## 4. DFDC (Deepfake Detection Challenge)

- **Media type:** Video. The full dataset contains over 100,000 clips from
  3,426 paid actors, created with several deepfake, GAN-based and non-learned
  methods.
- **Real/fake labels:** Yes, with metadata. A smaller "preview" subset also
  exists.
- **Approximate size:** Very large (hundreds of GB for the full training set).
  A smaller preview subset is available. Verify on the official page.
- **License / access:** DFDC terms of use; the Kaggle competition requires
  accepting the competition rules.
- **Registration required:** Yes.
- **Practical for a student prototype:** Low, mainly because of the size and
  the video processing needed on a laptop.
- **Suitable for image-based classification:** Yes, but only after heavy
  frame extraction and preprocessing.
- **Notes:** Excellent for a serious benchmark; usually too big for a
  semester prototype without substantial storage and compute.

## 5. DF40

- **Media type:** Images and videos. It covers 40 distinct generation methods
  (face swapping, face reenactment, entire-face synthesis, face editing),
  including recent diffusion-based methods.
- **Real/fake labels:** Yes. Fake data is organised by method; real data comes
  from the FF++ and Celeb-DF domains (and FFHQ / CelebA for some methods).
- **Approximate size:** About 50 GB for the training fake images and about
  93 GB for the test fake images, plus the real data. Very large.
- **License / access:** CC BY-NC 4.0 (non-commercial). Download via a Google
  form; data is also hosted on Google Drive and Baidu.
- **Registration required:** Yes (form).
- **Practical for a student prototype:** Low for the full set. However, you
  could download only one method's images to build a small, modern dataset.
- **Suitable for image-based classification:** Yes, images are provided after
  preprocessing (frame extraction and face cropping).
- **Notes:** Very current and diverse; a good choice if you later want to test
  generalisation across many manipulation types.

## 6. WildDeepfake

- **Media type:** Images organised as face sequences extracted from real
  internet deepfake videos (7,314 face sequences from 707 videos).
- **Real/fake labels:** Yes (real train/test and fake train/test folders).
- **Approximate size:** Small compared with the datasets above (face
  sequences rather than full videos).
- **License / access:** Strictly research use. Access requires an agreement
  form, an academic email and verification. It is also available on Hugging
  Face after approval.
- **Registration required:** Yes, with verification.
- **Practical for a student prototype:** Moderate to high because it is small,
  but access can take time and requires academic credentials.
- **Suitable for image-based classification:** Yes.
- **Notes:** It is specifically collected "in the wild", so it is much harder
  than FF++ and is best used as an additional test set rather than the main
  training set.

## 7. 140k Real and Fake Faces

- **Media type:** Images, 256x256, JPG. It combines 70,000 real faces (from
  the Flickr / FFHQ collection) and 70,000 fake faces generated by StyleGAN.
- **Real/fake labels:** Yes. The dataset is already divided into train,
  validation and test folders, and CSV files are included.
- **Approximate size:** About 4 GB (approximately).
- **License / access:** Hosted on Kaggle; free to download with a Kaggle
  account. Note that the underlying real images (FFHQ) are distributed under
  CC BY-NC-SA 4.0, so check the dataset page for the current terms.
- **Registration required:** Yes, a free Kaggle account.
- **Practical for a student prototype:** High. It is small, already split,
  and contains images directly, so no frame extraction is needed.
- **Suitable for image-based classification:** Yes, it is designed for binary
  image classification.
- **Important limitation:** The "fake" class is **entirely StyleGAN-generated
  faces**, not face-swap deepfakes. A model trained only on this dataset
  learns to recognise one generator, and should not be presented as a general
  deepfake detector.

---

## 8. Trade-offs and a practical recommendation

There is no single best dataset; the right choice depends on access, storage,
compute and what the project is trying to show.

- If the goal is **a standard deepfake-detection benchmark** that is
  comparable with published work, **FaceForensics++** is the natural choice.
  The cost is a Google-form access request and a large download, plus frame
  extraction.
- If the goal is **a modern, diverse set of manipulation methods**, **DF40**
  is attractive, but the full dataset is very large and non-commercial only.
- If the goal is **real-world difficulty**, **WildDeepfake** is valuable, but
  it requires an academic agreement and is better as a test set.
- If the goal is **a first, working end-to-end pipeline on a normal laptop**,
  the **140k Real and Fake Faces** dataset is the most practical because it
  is small, already split and image-based. Its limitation (StyleGAN-only
  fakes) must be stated honestly in the report.

**Suggested plan for this project (justified by available access and compute):**

1. Start with a small, image-based dataset so the training and evaluation
   pipeline can be built and tested quickly. The 140k dataset (or a small
   subset of it) is suitable for this first step.
2. In parallel, apply for access to **FaceForensics++** (compressed version).
   Once access is granted, extract frames and repeat the pipeline on FF++,
   which is the standard deepfake benchmark.
3. If time allows, use **Celeb-DF** or **WildDeepfake** as a cross-dataset
   test to show how well the model generalises.

Whichever dataset is chosen, the class labels, the source of the data, the
access conditions and the limitations must be documented in the final report.
The dataset must never be claimed to be larger or more representative than
it actually is.

---

## 9. Practical preparation notes

- **Video datasets need frame extraction.** FF++, Celeb-DF, DFDC and
  WildDeepfake are videos or sequences. Frames can be extracted with OpenCV,
  which is already used by this project.
- **Face cropping** is often applied before training. For a first prototype
  it can be skipped, but it should be documented.
- **Avoid data leakage.** Frames from the same video must not appear in both
  the training and the test set. The `src/dataset.py` module groups images by
  their sub-folder (for example one folder per video) so that a whole group
  stays inside a single split.
- **Keep the classes balanced.** If one class has far more images, the model
  can appear accurate by always predicting the majority class. The
  `--max-per-class` option in `prepare_dataset.py` limits each class to the
  same number.
- **Do not mix compression levels** (for example c23 and c40) between train
  and test, as this changes the task difficulty.

---

## 10. How this maps onto DeepGuard

DeepGuard already contains the input, validation and preprocessing stages.
Stage 2 adds dataset research (this document) and dataset preparation
(`src/dataset.py` and `prepare_dataset.py`). The next stage will be model
architecture selection, training, evaluation and inference integration.

The AI/ML detection model is **not** implemented yet.

---

## References

- FaceForensics++ repository and paper:
  https://github.com/ondyari/FaceForensics (paper: arXiv:1901.08971)
- Celeb-DF repository and paper:
  https://github.com/yuezunli/celeb-deepfakeforensics (paper: arXiv:1909.12962)
- DFDC dataset paper: https://arxiv.org/abs/2006.07397
  and https://ai.facebook.com/datasets/dfdc
- DF40 repository and paper:
  https://github.com/YZY-stack/DF40 (paper: arXiv:2406.13495)
- WildDeepfake repository and paper:
  https://github.com/xingjunm/wild-deepfake (paper: arXiv:2101.01456)
- 140k Real and Fake Faces:
  https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
