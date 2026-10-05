# DeepGuard - Viva Preparation (Likely Questions & Answers)

## AI-Based Deepfake Detection Prototype - Project ID 24CGCS06

> Answers are based **only on the actual project** (source code, verified
> artifacts, and the final technical report). Where information does not exist
> in the project artifacts, the answer says so rather than inventing anything.
>
> **Two experiments are covered.** Sections A-E are **Experiment 1**, the frozen
> academic baseline. **Section F** covers **Experiment 2**, a supplementary
> experiment on a larger, differently distributed dataset
> (`reports/experiment_2_report.md`). Experiment 2 does not replace Experiment 1.

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
Integrate inference into the UI (since done); enlarge and diversify the dataset;
add more methods/compression levels; explore hyperparameter tuning and
calibration; video-level detection; compare with ResNet-50 on the same split;
cross-dataset generalisation studies.

---

### F. Experiment 2 (supplementary) - the questions most likely to follow

> **Why this section exists.** Once you present a second, better number, the
> examiner's real question is almost never "what is the accuracy?" It is
> **"why is it better, and how do you know it is the dataset?"** The honest
> answer is that you *cannot* attribute it to dataset size, and saying so
> confidently is worth more than a stronger-sounding wrong answer.

**Q37. You report two experiments. Why?**
Experiment 1 is the frozen academic baseline: 600 FaceForensics++ frames, 75.00%
test accuracy. It exposed a real weakness - fake recall of only 50.00%. I then ran
Experiment 2 on a 6,000-image subset of the Kaggle *140k Real and Fake Faces*
dataset and got 97.56% test accuracy. Both are retained; **Experiment 2 does not
replace Experiment 1**, it is a second evaluation under different conditions.

**Q38. Why not just say the bigger dataset improved accuracy to 97.56%?**

Because it did not isolate that variable. Six things changed at once between
the two experiments:

1. Data amount (600 vs 6,000 images; 360 vs 4,200 training images)
2. **The fake generator** (FaceForensics++ face-swap vs. StyleGAN synthesis)
3. Image framing (centre-cropped video frames vs. full square images)
4. Source resolution (~3.2x vs. ~1.14x downsampling to the 224 input)
5. Compression (c23 video stored as PNG vs. JPEG)
6. Test-set size (120 vs. 900 images)

With six variables moving at once, no single cause is identifiable. Stating
"the dataset was bigger, so accuracy improved" would be an unsupported causal
claim.

**Q39. Then which factor do you think matters most?**

Most likely **the forgery family**, and this is the key scientific point. A
face-swap inherits a real person's identity, pose and lighting, so its artefacts
are blending and boundary inconsistencies layered onto *genuine* sensor noise.
A StyleGAN image is synthesised end to end, so its artefacts are a completely
different signature - generator upsampling and spectral regularities. A detector
can face these two families very differently **on its own**, independent of
training-set size. A large part, possibly most, of the gap may reflect an
**easier fake source** rather than a better-trained model.

**Q40. So what exactly can you claim about Experiment 2?**

That under its specified larger-dataset and independently sourced conditions,
the model reached **97.56% test accuracy** and **97.55% macro F1** on a 900-image
held-out test split, and that Experiment 1's weak fake recall was substantially
relieved (2 missed fakes out of 450, versus 30 out of 60). I explicitly cannot
claim dataset size caused it, and I do not claim generalisation to all
deepfakes.

**Q41. What was Experiment 2's dataset, and is it legitimate to use?**

A 6,000-image class-balanced subset (3,000 real FFHQ photographs, 3,000
StyleGAN fakes) of Kaggle `xhlulu/140k-real-and-fake-faces` (v2), which pools
140,000 images. The underlying FFHQ photographs are CC BY-NC-SA 4.0 -
**non-commercial use with attribution** - and the repository **redistributes no
images**. Licence compliance is documented in the dataset report.

**Q42. How did you select 6,000 images from 140,000 - wasn't that cherry-picking?**

It was **deterministic and seeded** (`seed 42`), not hand-picked. The builder
scans all 140,000 images, removes exact duplicates (MD5) and near-duplicates
(64-bit perceptual hash), then subsamples to fixed per-class quotas using a
seed derived from `42`. Because the seed is fixed, the subset is reproducible -
and the whole thing is pinned by a SHA-256 dataset fingerprint that the training
script re-verifies at startup, refusing to run against a changed dataset.

**Q43. How did you prevent leakage in Experiment 2, given it has no identity labels?**

Four layers:

1. The publisher's **official** `train`/`valid`/`test` folders were inherited, so
   no image ever crosses that boundary.
2. **Exact-duplicate (MD5) removal**, with same-class cross-split duplicates
   resolved by keeping the copy in the higher-priority split.
3. **Near-duplicate removal** by 64-bit pHash at threshold 3 bits - **5,069**
   near-duplicates were removed from the pool.
4. **Six automated build audits**, all passing, including an explicit
   `no_near_duplicate_across_splits` check; the builder and validator both exit
   non-zero on failure.

**Honest caveat if pressed:** the minimum cross-split pHash distances were 4, 4
and 6 bits against a threshold of 3. Two of those margins are a single bit, so
the guarantee is threshold-relative rather than generous, and
**Experiment 1's identity-component split was actually the stricter one.**

**Q44. Experiment 2 has no identity metadata - isn't that a weakness?**

Yes, and I state it as a limitation rather than hiding it. Because no identity
labels are published, identity-level grouping is impossible; leakage control
rests on the publisher's splits plus de-duplication instead. **Experiment 1, with
600 frames, had genuine identity-component grouping and is the more
stringent split of the two.** Experiment 2's defence is scale plus automated
de-duplication, not person-level separation.

**Q45. Experiment 2's best checkpoint is the final epoch - doesn't that mean it
was undertrained?**

Yes, and that is exactly why I would not call 97.56% a ceiling. The epoch-10
checkpoint had the lowest validation loss and validation loss was still falling
when the 10-epoch CPU budget ran out, so the model was likely **not converged**.
Running longer could move the number in either direction. A ceiling claim would
be unjustified.

**Q46. Experiment 2 made 20 false positives. Isn't that worse?**

It is a genuine trade-off and the two experiments fail in **opposite
directions**:

| | Experiment 1 | Experiment 2 |
| --- | --- | --- |
| Missed fakes (false negatives) | 30 | **2** |
| False alarms (false positives) | 0 | 20 |

Under the balanced 900-image test set this costs little accuracy. But under a
real-world prior where most incoming images are genuine, 20 false alarms would
matter more. **Experiment 1's perfect fake precision was itself an artefact of
barely attempting any fake prediction** - it missed half its fakes. Neither
result is simply "better"; they fail in opposite directions.

**Q47. Validation accuracy was 96.89% and test accuracy 97.56% - which do you report?**

**97.56% test accuracy is the reported result.** The 96.89% validation accuracy
(and 0.0684 validation loss) were the checkpoint-selection signals, so they are
reported separately and never quoted as performance. Separately again, a
prediction from the running Streamlit app is a **live demo output** - a single
image, `argmax` threshold, uncalibrated - and forms no part of any official
evaluation.

**Q48. Can you design the experiment that would actually answer "does more data
help?"**

Yes, and it is the top item in my future work. To isolate dataset size you must
hold everything else fixed: take **one** dataset and train at two sizes (for
example 600 and 6,000 images from the same Kaggle pool, same generator, same
framing, same test split). Alternatively, hold size fixed and vary the generator.
Until that is run, the honest position is that Experiment 2 demonstrates
**performance under a larger and differently distributed dataset**, and does not
quantify a dataset-size effect.

**Q49. Why use the same EfficientNet-B0 in both experiments? Wasn't that a
limitation?**

Deliberately, so the architecture is not a confounding variable - that is what
makes the two results comparable at all. It does mean the study shows nothing
about whether another architecture would close the gap. Comparing architectures
would need matched data, which I did not run.

**Q50. Why was Experiment 2 trained on CPU for 10 hours?**

No GPU was available. EfficientNet-B0 was chosen partly because 5.3M parameters
and 0.39B FLOPs make CPU training feasible at all; the 600.47-minute run is the
cost of that constraint on 4,200 images. The practical lesson is that dataset
scale, not model size, is what dominates CPU feasibility here.

---

*Viva preparation document (Stage 8), covering Experiment 1 (frozen baseline)
and Experiment 2 (supplementary). Answers reflect only verified project facts; no
new experimental claim is introduced. The single most important line to
remember: **Experiment 2 is not a controlled dataset-size experiment.***