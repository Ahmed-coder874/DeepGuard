# DeepGuard - Viva Preparation (Likely Questions & Answers)

## AI-Based Deepfake Detection Prototype - Project ID 24CGCS06

> Answers are based **only on the actual project** (source code, verified
> artifacts, and the final technical report). Where information does not exist
> in the project artifacts, the answer says so rather than inventing anything.

---

### A. Project and motivation

**Q1. What problem does DeepGuard solve?**
DeepGuard is a prototype for image-based detection of deepfakes: given an image
frame, it classifies it as `real` or `fake`. The project addresses the
difficulty of verifying digital content, and the related risks of
deepfake-related cybercrime, identity fraud and misinformation identified in
the project field visit.

**Q2. Why is deepfake detection difficult?**
Generation quality keeps improving, so manipulated media can look genuine by
eye; manual verification is unreliable; forensic traces are often subtle and
high-frequency; and detectors trained on one generator often fail on unseen
generators or compression levels.

**Q3. What is the societal motivation / which SDG does it relate to?**
The project is registered under **SDG 10 (Reduced Inequalities)**: manipulated
media and the inability to verify digital content disproportionately enable
fraud and disinformation against individuals and vulnerable groups.

**Q4. What does the prototype actually include today?**
Input validation and preprocessing (Streamlit app), dataset research,
deterministic frame extraction, leakage-free dataset preparation, model
architecture selection, a training run, an official test evaluation, and
documentation. The trained model is **not yet wired into the UI**; inference
`src/inference.py` exists as the integration building block.

---

### B. Dataset and splitting

**Q5. What dataset was used and why?**
FaceForensics++ material (Deepfakes method, c23 compression), chosen because it
is a standard, well-documented deepfake benchmark with official real/fake
labels from its directory structure. It was downloaded and kept intact under
`data/faceforensics`.

**Q6. Why is the dataset only 600 frames?**
It is a deliberate small prototype subset: 10 original + 10 manipulated videos,
30 frames each, for a CPU-only student laptop. The verified disk state is
**300 real + 300 fake = 600 frames**.

**Q7. What was the 1,200-frame discrepancy?**
Earlier documentation/headers claimed 600 real + 600 fake = 1,200. Repeated
disk verification shows **300 + 300 = 600**. The report records the discrepancy
transparently (Appendix B) and uses only the verified disk values; no second
600-image batch was ever generated.

**Q8. Why was the dataset split by component?**
Splitting by identity component (union of target/source identities linked by
manipulated videos, e.g. `183_253` and `253_183` link identities 183 and 253)
prevents **identity leakage**: the same person's frames can never appear in two
different splits.

**Q9. How was leakage prevented?**
The same component id is used as the folder name in both class folders, so the
splitter keeps a whole component inside exactly one split. Verified audits show
zero overlap between train/validation/test path sets and no component in more
than one split. Ratios are 70/15/15 with seed 42.

**Q10. Why was the test set kept untouched?**
The test split is the only independent estimate of performance. If it were used
during training, model selection or threshold decisions, the reported accuracy
would be optimistic. It was evaluated exactly once, after training finished.

**Q11. What do the three splits contain?**
Train 180 real + 180 fake = 360; validation 60 real + 60 fake = 120; test
60 real + 60 fake = 120. Components: train = 469_481, 585_599, 672_720;
validation = 866_878; test = 183_253.

**Q12. How were frames chosen from the videos?**
Deterministic, evenly spaced frame indices (30 per video) - no randomness, so
repeated runs pick identical frames. Labels come only from the official FF++
directory structure; nothing synthetic was created.

---

### C. Model and methodology

**Q13. Why EfficientNet-B0?**
Native 224 x 224 input (exact pipeline match), 5.3M parameters / 0.39B FLOPs
(~10x lighter than ResNet-50) so it trains on the available CPU, official
ImageNet pretrained weights in TorchVision, and EfficientNet backbones appear
in published deepfake-detection work. Selected for this project's constraints -
not claimed universally best.

**Q14. Why use pretrained ImageNet weights? (transfer learning)**
ImageNet pretraining gives generic visual features learned on a huge dataset,
so a small dataset (360 training images) can fine-tune effectively instead of
training from scratch. Only the final layer (1280 -> 2) was replaced.

**Q15. What is transfer learning?**
Reusing a network trained on one task (ImageNet classification) as the starting
point for another task (real vs. fake), training only a few final layers (here
the whole model was fine-tuned) instead of learning everything from scratch.

**Q16. Why 224 x 224?**
EfficientNet-B0's native input is 224 x 224 and the Stage 1 app pipeline already
resizes to 224 x 224, so no re-tuning or resizing mismatch was needed.

**Q17. What preprocessing was applied?**
Training split: RandomResizedCrop(224, scale 0.8-1.0), horizontal flip p=0.5,
rotation +/-10 deg, then ImageNet mean/std normalisation - deliberately light so
forensic traces survive. Validation/test: deterministic Resize(224) +
CenterCrop(224) + normalisation. App display path: BGR->RGB, resize, 0-1 norm.

**Q18. Why CrossEntropyLoss?**
It is the standard loss for multi-class classification; with 2 classes (real=0,
fake=1) it compares the model's logits against the integer label and is what the
model's log-softmax output is designed for.

**Q19. Why AdamW?**
AdamW is a robust adaptive optimizer with decoupled weight decay, a common,
well-understood default for fine-tuning CNNs; used with lr 3e-4 and weight
decay 1e-4.

**Q20. Why early stopping?**
To stop training when validation loss stops improving, preventing wasted compute
and overfitting. Patience 3: training stopped after epoch 9 because validation
loss did not improve at epochs 7, 8, 9 (9 of 10 requested epochs completed).

**Q21. Why a separate training environment (`venv-train`)?**
To keep the Streamlit application environment free of PyTorch (Python 3.14
Windows wheels) so the app stays small and stable; training and app concerns
are isolated. It also explains the 3 test-suite errors in `venv-train`
(streamlit/OpenCV tests belong to the app environment).

---

### D. Results and metrics

**Q22. What does the confusion matrix mean?**
Rows are the true class, columns the predicted class, so [[60, 0], [30, 30]]
means: of 60 real frames, 60 predicted real / 0 predicted fake; of 60 fake
frames, 30 predicted real / 30 predicted fake.

**Q23. Why is fake recall only 50.00%?**
Recall = TP/(TP+FN). Only 30 of the 60 fake frames were caught (30 false
negatives), so 30/60 = 50.00%. Half the manipulated frames went undetected.

**Q24. Why is fake precision 100.00%?**
Precision = TP/(TP+FP). Every frame the model labelled "fake" was actually fake
(30/30, zero false positives), so 30/30 = 100.00%. The model is conservative:
when it raises an alarm, it is correct on this test set.

**Q25. What does 75.00% test accuracy mean?**
90 of the 120 test frames were classified correctly (60 real + 30 fake), so
90/120 = 75.00%, measured once on the untouched test split. It is an estimate on
a small, single-component test set - not a claim about general deepfake
detection performance.

**Q26. Why is validation accuracy different from test accuracy?**
They are different data: validation = component 866_878 (used to pick the best
epoch), test = component 183_253 (never seen during training). Validation
78.33% (best checkpoint, epoch 6) is an optimisation signal; test 75.00% is the
official independent result. Different components and different frames - no
"improvement/degradation" claim is made from the difference.

**Q27. What are false positives and false negatives here?**
False positive = a real frame flagged as fake (here: 0). False negative = a fake
frame classified as real (here: 30). For a deepfake detector the false-negative
direction (missed fakes) is the safety-relevant one, and it is the dominant
failure in this run.

**Q28. What is macro F1 and why report it?**
Macro F1 = unweighted mean of the per-class F1 scores
((0.8000 + 0.6667)/2 = 0.7333), so neither class dominates the summary. Reported
alongside per-class values because class behaviour differs strongly.

**Q29. Why was the checkpoint chosen by validation loss, not accuracy?**
The training loop saves the checkpoint with the lowest validation loss
(epoch 6, loss 0.4908). Loss uses full probability information, while accuracy
is a coarse thresholded count; the best validation accuracy (80.83%, epoch 4)
does not coincide with the selected checkpoint.

**Q30. Was there any threshold tuning or re-evaluation?**
No. Evaluation was deterministic: argmax over logits, fixed transforms, no
shuffle, no threshold tuning, one run. The experimental evidence is frozen.

---

### E. Limitations and outlook

**Q31. What are the main limitations?**
Small verified dataset (600 frames); test set is 120 frames from a single
identity component (noisy, single-subject biased); frame-level only (no
temporal modelling); results specific to FF++ Deepfakes at c23; first
experiment with no hyperparameter tuning; model not yet integrated into the UI.

**Q32. Is the model production-ready?**
No. It is explicitly not production-ready and not state of the art. It is a
first prototype whose results must be read with the documented limitations;
any future prediction should be treated as an indication requiring further
verification, never proof of authenticity.

**Q33. Why was ResNet-50 not compared?**
ResNet-50 was documented as the alternative (only one architecture was trained
in this project on CPU). A performance comparison would require training it on
the same split and evaluating it - a real experiment that was not performed, so
no comparative accuracy claim exists.

**Q34. What is the difference between frame-level and video-level detection?**
This project classifies individual image frames; video-level detection
aggregates evidence over time (temporal consistency). Temporal information was
not used, so video-level performance is not demonstrated.

**Q35. How would you evaluate generalisation?**
Train/evaluate on different manipulation methods, compression levels and
datasets (e.g. cross-dataset FF++ -> Celeb-DF), on unseen generators, and "in
the wild" media - reporting every result with its own evidence, as the project
does for its current test result. None of these was performed here.

**Q36. What would you improve next?**
Integrate inference into the UI; enlarge and diversify the dataset; add more
methods/compression levels; explore hyperparameter tuning and calibration;
video-level detection; compare with ResNet-50 on the same split; cross-dataset
generalisation studies.

---

*Viva preparation document (Stage 8). Answers reflect only verified project
facts; no new experimental claim is introduced.*