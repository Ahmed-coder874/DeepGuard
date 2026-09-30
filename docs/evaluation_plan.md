# DeepGuard - Evaluation Plan (Stage 5)

Project: **Deepfake Technology** (24CGCS06)
Stage: **Stage 5 - Model Evaluation (planned design)**

> **This document is a PLAN, not a report of results.**
> No model has been trained and no evaluation has been run. **No accuracy,
> precision, recall, F1-score or confusion matrix is claimed anywhere.** The
> metric values can only be produced by running `evaluate_model.py` on a real
> trained checkpoint and a real test split.

---

## 1. Goal

Measure how well the trained DeepGuard classifier separates **real** from
**fake** images, using the **test split** that was never used during training.

---

## 2. Data used

- Source: `data/processed/test/{real,fake}`, produced by `prepare_dataset.py`.
- The test split is **reserved for evaluation** and is never used for training
  or for tuning.
- Images are processed with the deterministic evaluation transforms (resize /
  centre-crop to 224, ImageNet normalisation) - no random augmentation.
- If the test split is empty, evaluation refuses to run.

---

## 3. Metrics

All metrics use the standard definitions. `fake` (index 1) is the **positive**
class, because detecting manipulated content is the purpose of the project.

| Metric | Definition |
| --- | --- |
| Accuracy | (TP + TN) / total |
| Precision (fake) | TP / (TP + FP) |
| Recall (fake) | TP / (TP + FN) |
| F1 (fake) | 2 * precision * recall / (precision + recall) |
| Macro precision/recall/F1 | unweighted mean over the two classes |
| Confusion matrix | rows = actual class, columns = predicted class |

A metric whose denominator is zero (for example a class absent from the test
set) is reported as `0.0`. No value is estimated or invented.

Not computed in this stage: ROC-AUC and precision-recall curves. These need
predicted probabilities and are left as optional future work.

---

## 4. Procedure

1. Confirm that `models/` contains a trained checkpoint.
2. Confirm that the test split contains images in both classes.
3. Rebuild the architecture and load the trained weights from the checkpoint
   (`pretrained=False`, so no internet download is needed).
4. Run the model over the test split in evaluation mode (`model.eval()`,
   `torch.no_grad()`), with no gradient updates.
5. Collect the true labels and the predicted labels.
6. Compute the metrics in section 3.
7. Print the results and save them to
   `reports/evaluation_report.json`.

Command:

```powershell
venv-train\Scripts\activate
python evaluate_model.py
python evaluate_model.py --status          # readiness only
```

---

## 5. What is reported and what is not

**Reported:** the metrics actually computed on the test split, plus the
checkpoint path, the checkpoint's training epoch and validation values, the
device and the library version.

**Not reported:**
- No metric is shown before evaluation actually runs.
- No result is copied from a published paper.
- Training and validation values are **not** presented as test results.

---

## 6. Honest interpretation

- The result describes performance **only on the test data that was evaluated**.
  It does not prove the system works on arbitrary real-world images.
- A small prototype subset gives a noisy estimate; the number should be read
  with that in mind.
- Deepfake detection is an adversarial problem: a detector can be fooled by
  manipulations it was not trained on.
- Any deployment would require much larger, more diverse evaluation.

---

## 7. Reproducibility

- The checkpoint records the seed and library versions used during training.
- Evaluation itself is deterministic: fixed transforms, no shuffling, CPU
  execution.
- The saved report includes a timestamp and the exact checkpoint path.

---

## 8. Status

Evaluation **infrastructure** (`src/evaluation.py`, `evaluate_model.py`,
`tests/test_evaluation.py`) is implemented. **No evaluation has been executed
and no report exists**, because there is no trained checkpoint and no prepared
dataset yet.
