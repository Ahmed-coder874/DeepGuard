# DeepGuard - Model Architecture Research

Project: **Deepfake Technology** (24CGCS06)
Stage: **Stage 3 - Model Architecture Selection**
Compiled from the primary research papers and official framework documentation
listed in the references.

> **Purpose of this document.** This is a factual comparison of image
> classification architectures that could be used for an academic, image-based
> deepfake detection project. It does **not** declare one architecture as "the
> best". It explains the trade-offs and then selects **one** architecture for
> this project, based on the project's own requirements.
>
> **Published results vs this project's results.** Every accuracy or
> parameter number quoted below is a **published result reported by the
> original authors** (mostly on ImageNet, which is a general object
> classification task, not deepfake detection). **This project has not trained
> any model and has produced no results of its own.** Nothing in this document
> should be read as a DeepGuard benchmark.

---

## 1. What the project requires

The architecture has to fit the project as it actually exists today:

| Requirement | Current project fact |
| --- | --- |
| Input pipeline | Stage 1 already resizes to **224 x 224**, RGB, so a native 224 model avoids extra resizing |
| Task | Binary image classification: `real` vs `fake` |
| Labels | `real = 0`, `fake = 1` (see `config.py`) |
| Framework | PyTorch (see section 7) |
| Hardware | **CPU only** - Intel Core i5-8250U (4 cores / 8 threads), Intel UHD Graphics 620, ~16 GB RAM, **no NVIDIA GPU / no CUDA** |
| Dataset | A small prototype subset (the Stage 2 default is 500 images per class) |
| Transfer learning | Strongly preferred, because a small dataset cannot train a large model from scratch |
| Deployment | The trained model must eventually be loadable from the existing Streamlit app |
| Demonstration | The architecture must be explainable to a non-specialist examiner |

---

## 2. Candidate comparison

Parameter counts, FLOPs and ImageNet top-1 accuracy are **published values**
from the cited papers. `-` means the value was not verified from an
authoritative source and is therefore not quoted.

| Architecture | Params | FLOPs | Native input | ImageNet top-1 | Source |
| --- | --- | --- | --- | --- | --- |
| **EfficientNet-B0** | 5.3M | 0.39B | 224 x 224 | 77.1% | Tan & Le 2019 (arXiv:1905.11946) |
| EfficientNet-B4 | 19M | 4.2B | 380 x 380 | 82.9% | Tan & Le 2019 |
| MobileNetV3-Large | 5.4M | 0.22B (219M MAdds) | 224 x 224 | 75.2% | Howard et al. 2019 (arXiv:1905.02244) |
| MobileNetV3-Small | 2.5M | 0.06B (56M MAdds) | 224 x 224 | 67.4% | Howard et al. 2019 |
| ResNet-50 | ~26M | 4.1B | 224 x 224 | 76.0% | He et al. 2015 (arXiv:1512.03385); FLOPs/top-1 from Tan & Le 2019 |
| Xception | ~23M | 8.4B | 299 x 299 | 79.0% | Chollet 2017 (arXiv:1610.02357); FLOPs/top-1 from Tan & Le 2019 |
| ViT-B/16 | ~86M | - | 224 x 224 | 77.9% | Dosovitskiy et al. 2020 (arXiv:2010.11929); top-1 as reported in Thing 2023 (arXiv:2304.03698) |

These ImageNet numbers are **not** deepfake-detection numbers. They are only a
rough indicator of how well the backbone learns general visual features before
fine-tuning.

---

## 3. Architecture notes

### 3.1 EfficientNet-B0
- **Principle:** a convolutional network built from mobile inverted bottleneck
  blocks (MBConv) with squeeze-and-excitation, found by neural architecture
  search and then scaled with a compound depth/width/resolution coefficient.
  EfficientNet-B0 is the unscaled baseline.
- **Input:** 224 x 224 natively - an exact match for the Stage 1 pipeline.
- **Complexity:** 5.3M parameters, 0.39B FLOPs - about 10x fewer FLOPs than
  ResNet-50.
- **Transfer learning:** official ImageNet weights are distributed with
  TorchVision (`EfficientNet_B0_Weights`).
- **Binary classification:** replace the final `Linear` layer (1280 -> 1000)
  with 1280 -> 2.
- **Advantages:** very good accuracy per unit of compute; native 224 input;
  small enough to fine-tune on a CPU for a limited dataset; the
  squeeze-and-excitation blocks are easy to describe in a presentation.
- **Limitations:** the "mobile" building blocks are tuned for efficiency, not
  for detecting tiny high-frequency forensic artifacts; ImageNet features are
  not specifically forensic.
- **Deepfake relevance:** EfficientNet backbones appear in deepfake-detection
  research. The 2024 systematisation study (arXiv:2401.04364) reports that
  EfficientNet-based detectors (for example MAT and CADDM) are among the
  strongest on hard and "gray-box" evaluation sets. That is a **published
  finding on those authors' datasets**, not a result of this project.

### 3.2 ResNet-50
- **Principle:** deep residual CNN; identity shortcut connections let gradients
  flow through very deep stacks.
- **Input:** 224 x 224 natively.
- **Complexity:** ~26M parameters, 4.1B FLOPs.
- **Transfer learning:** official ImageNet weights ship with TorchVision.
- **Advantages:** the most widely used, best-understood transfer-learning
  baseline; easy to explain; robust and well documented.
- **Limitations:** roughly 10x the FLOPs of EfficientNet-B0, so CPU
  fine-tuning is noticeably slower; no squeeze-and-excitation.

### 3.3 Xception
- **Principle:** an Inception-style network where the Inception modules are
  replaced by depthwise separable convolutions.
- **Input:** 299 x 299 natively, so it does **not** match the existing 224 x 224
  pipeline (the input would have to be upscaled or the network resized).
- **Complexity:** ~23M parameters, 8.4B FLOPs.
- **Transfer learning:** **not** provided by TorchVision; a third-party
  implementation (for example `timm`) would be required.
- **Deepfake relevance:** Xception is the standard baseline for the
  FaceForensics++ benchmark (Roessler et al. 2019, arXiv:1901.08971), which
  makes it attractive for comparability with published work.
- **Limitations for this project:** 299 input, no official TorchVision weights,
  and heavier than EfficientNet-B0 - all three work against the current
  224 x 224, CPU-only, TorchVision-based setup.

### 3.4 MobileNetV3 (Large / Small)
- **Principle:** hardware-aware NAS plus NetAdapt, with h-swish activations and
  squeeze-and-excitation; designed for mobile CPUs.
- **Input:** 224 x 224 natively.
- **Complexity:** Large 5.4M params / 0.22B FLOPs; Small 2.5M / 0.06B.
- **Transfer learning:** official ImageNet weights ship with TorchVision.
- **Advantages:** the fastest candidates on a CPU; smallest checkpoints.
- **Limitations:** lower ImageNet top-1 than EfficientNet-B0 (75.2% vs 77.1%);
  less common as a backbone in the deepfake-detection literature, so it is
  harder to relate to published work.

### 3.5 Vision Transformer (ViT-B/16)
- **Principle:** splits the image into patches and applies pure self-attention.
- **Input:** 224 x 224 natively.
- **Complexity:** ~86M parameters - by far the largest candidate here.
- **Transfer learning:** available (TorchVision / timm), but ViTs are known to
  need large datasets and longer training to match CNNs.
- **Advantages:** strong at modelling long-range relationships; transformer
  based detectors are competitive in recent literature.
- **Limitations for this project:** large, slow on CPU, data-hungry, and
  harder to explain than a CNN. Not a realistic first prototype on this
  hardware.

---

## 4. Selection for this project

### Primary architecture: **EfficientNet-B0**

Selected **for this project**, based on the current project requirements - it
is not claimed to be universally the best architecture.

Reasons, tied to the verified project facts:

1. **Exact input match.** The Stage 1 pipeline already produces 224 x 224, and
   EfficientNet-B0 is natively 224 x 224, so no resizing or re-tuning of the
   existing preprocessing is required.
2. **Trainable on this hardware.** At 5.3M parameters and 0.39B FLOPs it is
   roughly 10x lighter than ResNet-50 and far lighter than ViT-B/16, which is
   the decisive factor on a CPU-only machine with a small prototype subset.
3. **Transfer learning is officially supported.** ImageNet weights ship with
   TorchVision, so only the final layer has to be replaced for 2 classes.
4. **Relevant to the domain.** EfficientNet backbones are used in published
   deepfake-detection work, so the choice is defensible in a report.
5. **Explainable.** A CNN with MBConv and squeeze-and-excitation blocks is easy
   to describe and to visualise during a demonstration.

### Documented alternative: **ResNet-50**

Kept as the comparison architecture because it is the standard,
best-understood transfer-learning baseline with official TorchVision weights.
If EfficientNet-B0 turns out to be hard to fine-tune, ResNet-50 is the natural
fallback - at the cost of much higher CPU training time.

MobileNetV3-Large is noted as a lighter fallback if CPU training of
EfficientNet-B0 proves too slow.

> **No accuracy claim.** This project does **not** claim that EfficientNet-B0
> will detect deepfakes more accurately than the alternatives. Any comparison
> can only be made after real training and evaluation on the same data
> (Stage 5).

---

## 5. Why architecture selection matters here

Choosing an architecture before training avoids three common mistakes:

1. picking a model too large to train on the available CPU and dataset;
2. building a preprocessing pipeline whose input size does not match the
   model, forcing a rewrite;
3. claiming results without a documented, justified choice.

---

## 6. Framework decision (verified)

The deep-learning framework must work on **Windows + Python 3.14.7 + CPU**.

| Framework | Python 3.14 on Windows? | Evidence | Decision |
| --- | --- | --- | --- |
| **PyTorch** | Yes | `torch` 2.9.0 - 2.14.0 and `torchvision` 0.29.0 publish `cp314` **win_amd64** wheels; `torchvision 0.29.0` pins `torch==2.14.0` | **Selected** |
| TensorFlow | No | TensorFlow stable (2.21) supports Python 3.10-3.13 only; Python 3.14 wheels exist only for `tf-nightly` on Linux/macOS, **not Windows** | Rejected |

Chosen stack: **PyTorch 2.14.0 + TorchVision 0.29.0 (CPU)**. Because the
machine has no NVIDIA GPU, no CUDA packages are installed.

---

## 7. Summary

- **Primary architecture:** EfficientNet-B0 (2-class head: `real = 0`, `fake = 1`).
- **Alternative:** ResNet-50.
- **Framework:** PyTorch 2.14.0 + TorchVision 0.29.0, CPU.
- **Training environment:** a separate `venv-train` so the Streamlit
  application environment is never disturbed.
- **Status:** architecture **selected and documented**. No model has been
  trained, no weights exist, and no performance metric exists.

---

## References

- EfficientNet - Tan, M. & Le, Q. V. (2019), "EfficientNet: Rethinking Model
  Scaling for Convolutional Neural Networks", arXiv:1905.11946.
  https://arxiv.org/abs/1905.11946
- ResNet - He, K. et al. (2015), "Deep Residual Learning for Image
  Recognition", arXiv:1512.03385. https://arxiv.org/abs/1512.03385
- Xception - Chollet, F. (2017), "Xception: Deep Learning with Depthwise
  Separable Convolutions", arXiv:1610.02357.
  https://arxiv.org/abs/1610.02357
- MobileNetV3 - Howard, A. et al. (2019), "Searching for MobileNetV3",
  arXiv:1905.02244. https://arxiv.org/abs/1905.02244
- ViT - Dosovitskiy, A. et al. (2020), "An Image is Worth 16x16 Words:
  Transformers for Image Recognition at Scale", arXiv:2010.11929.
  https://arxiv.org/abs/2010.11929
- FaceForensics++ - Roessler, A. et al. (2019), arXiv:1901.08971.
  https://arxiv.org/abs/1901.08971
- CNN vs Transformers for deepfake detection - Thing, V. L. L. (2023),
  arXiv:2304.03698. https://arxiv.org/abs/2304.03698
- SoK: Benchmarking deepfake detectors - (2024), arXiv:2401.04364.
  https://arxiv.org/abs/2401.04364
- TorchVision model documentation and version table:
  https://github.com/pytorch/vision
- PyTorch release compatibility matrix:
  https://github.com/pytorch/pytorch/blob/main/RELEASE.md
- TensorFlow tested build configurations:
  https://www.tensorflow.org/install/source
