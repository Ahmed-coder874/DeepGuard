# DeepGuard - Final Submission Checklist

**Project:** Deepfake Technology - **Project ID:** 24CGCS06
**Stage:** 8 - Final Review, Presentation & Submission Preparation

Legend: `[x]` = verified complete in the project; `[ ]` = **requires the
student's personal input / manual action** (deliberately not marked complete -
these values do not exist in the project artifacts).

---

> **Two experiments are documented across these artifacts.** Experiment 1
> (FaceForensics++, 600 frames, 75.00% test accuracy) is the **frozen academic
> baseline**; all of its entries below are unchanged. Experiment 2 (Kaggle 140k,
> 6,000 images, 97.56% test accuracy) is a **separate, supplementary experiment**
> and **does not replace Experiment 1**.

## 1. Core deliverables

- [x] Final technical report: `reports/DeepGuard_SIP_Final_Technical_Report.md`
      (now covering Experiment 1 and, in a separate section, Experiment 2)
- [x] PDF of the report: `reports/DeepGuard_SIP_Final_Technical_Report.pdf`
      (regenerated from the Markdown source and verified - 36 pages, 7 figures;
      see section 12)
- [x] SIP presentation outline (20 slides, including a separate Experiment 2
      sequence): `reports/DeepGuard_SIP_Presentation.md`
- [x] Viva questions and answers (50 Q/A, section F covers Experiment 2):
      `reports/DeepGuard_Viva_QA.md`
- [x] Experiment 2 detailed report: `reports/experiment_2_report.md`
- [x] This submission checklist: `reports/DeepGuard_Submission_Checklist.md`

## 2. Figures and references

- [x] Figure: training curves - `reports/stage6/training_curves.png`
- [x] Figure: confusion matrix - `reports/stage6/confusion_matrix.png`
- [x] Figure: per-class metrics - `reports/stage6/per_class_metrics.png`
- [x] Experiment 2 figures (4):
      `experiments/experiment_2_large_dataset/reports/confusion_matrix.png`,
      `per_class_metrics.png`, `training_loss_curve.png`,
      `training_accuracy_curve.png`
- [x] References section present (16 cited sources, from verified research docs;
      includes the Kaggle 140k dataset as source 13)

## 3. Source code

- [x] Application and pipeline source: `app.py`, `config.py`, `train_model.py`,
      `evaluate_model.py`, `prepare_dataset.py`, `extract_dataset_frames.py`
- [x] Library modules: `src/*.py` (9 modules)
- [x] Tests: `tests/` (8 test modules)
- [x] Unit tests executed: 92 OK / 15 skipped (app venv); core suite 62 OK /
      1 skipped (venv-train); 26 modules compile cleanly

## 4. Requirements / environments

- [x] App requirements: `requirements.txt` (streamlit, opencv-python, numpy, Pillow)
- [x] Training requirements: `requirements-train.txt` (torch 2.14.0, torchvision 0.29.0, CPU)
- [x] Environment separation documented (venv vs venv-train)

## 5. Model checkpoint

- [x] **Experiment 1** checkpoint: `models/deepguard_efficientnet_b0.pt`
      (16,341,163 bytes; epoch 6; seed 42; loads and verified) - **frozen,
      unmodified**
- [x] **Experiment 2** checkpoint: `models/experiment_2_deepguard_efficientnet_b0.pt`
      (16,346,305 bytes; epoch 10; seed 42; loads and verified; separate file,
      Experiment 1 checkpoint untouched)
- [x] Both checkpoints served by the app via a single shared code path; the app
      currently loads the Experiment 2 checkpoint

## 6. Dataset documentation

- [x] Dataset research: `docs/dataset_research.md`
- [x] Provenance manifest: `data/raw/frames_provenance.json`
- [x] Dataset summary: `data/processed/dataset_summary.json`
- [x] Architecture research: `docs/model_architecture_research.md`
- [x] Training plan: `docs/training_plan.md`
- [x] Evaluation plan: `docs/evaluation_plan.md`
- [x] Dataset discrepancy documented (1,200 documented vs 600 verified) -
      Appendix B of the final report

## 7. Experimental results

**Experiment 1 - frozen academic baseline**

- [x] Training log preserved: `reports/stage6/train_run_log.txt` (9 epochs,
      best epoch 6, duration 52 min 28 s)
- [x] Official test evaluation: `reports/evaluation_report.json`
      (75.00% accuracy, 120 samples, confusion matrix [[60,0],[30,30]])
- [x] Stage 6 analysis: `reports/stage6_results_analysis.md`
- [x] Final Stage 7/8 consistency audits passed (50/50 checks)

**Experiment 2 - supplementary, does not replace Experiment 1**

- [x] Dataset report: `experiments/experiment_2_large_dataset/dataset_report.json`
      (6,000 images; 3,000 real / 3,000 fake; split 4,200/900/900; seed 42;
      5,069 near-duplicates removed; min cross-split pHash distance 4/4/6 bits;
      6/6 build audits passed; SHA-256 fingerprint recorded)
- [x] Training summary: `…/training_summary.json`
      (EfficientNet-B0, CPU, batch 8, lr 3e-4, weight decay 1e-4, patience 3,
      10/10 epochs, best epoch 10, 600.47 minutes, early stopping not triggered)
- [x] Official test evaluation: `…/reports/evaluation_report.json`
      (**97.56% accuracy**, 900 samples, macro F1 **97.55%**, confusion matrix
      [[430,20],[2,448]])
- [x] Companion report: `reports/experiment_2_report.md`
- [x] Comparison against Experiment 1 recorded with the mandatory caveat that
      the two experiments differ in six variables at once
- [x] Validation figures kept separate from the reported test result
      (validation accuracy 96.89% is a selection signal, not the result)
- [ ] Two Experiment 2 artifacts do not exist and are stated as such:
      `dataset_validation.json`, and a committed processed/training state
      directory (build output is kept out of Git)

## 8. Limitations & Responsible AI

- [x] **Experiment 1** limitations section present (dataset size,
      single-component test set, frame-level only, no generalisation claim,
      first experiment)
- [x] **Experiment 2** limitations section present, including: not a controlled
      dataset-size experiment, different forgery family, no identity metadata,
      best checkpoint at the final epoch (not converged / not a ceiling),
      tight near-duplicate margins, single-source test set, uncalibrated
      confidence
- [x] Experiment 1 limitations explicitly scoped to Experiment 1 (not presented
      as applying to Experiment 2)
- [x] Responsible AI section present (indications not proof, model confidence
      wording, no unsupported statistics)
- [x] No "state of the art" / "production ready" / "generalizes to all
      deepfakes" claims in the report
- [x] No causal "accuracy improved because the dataset was larger" claim; such
      statements are listed in the report as **prohibited phrasing**
- [x] Demo/UI predictions explicitly distinguished from official evaluation
      results in both experiments
- [x] Application status corrected: the report previously stated the trained
      model was "not yet connected to the app"; the app now performs real
      inference on the Experiment 2 checkpoint (verified HTTP 200, 28/28 UI
      checks, 7/7 warm app tests)

## 9. Personal / institutional fields (student must complete)

- [ ] Student full name - **[Student Name]** placeholder only; real name not in
      project artifacts
- [ ] University / college register number - **[Register Number]** placeholder
- [ ] Date of submission - **[Submission Date]** placeholder
- [ ] Internship / field visit dates (if required by the institute format)
- [ ] Institution confirmation (README states "Atria Institute of Technology,
      Bengaluru" - confirm as printed on the submission cover)
- [ ] Supervisor / project guide confirmation (README states "Prof. Jayanth U" -
      confirm exact title and spelling as required)
- [ ] Signature / sign-off of student and guide
- [ ] Acknowledgements section (not written - requires personal text)
- [ ] Certificate / declaration page (if the institute template requires one -
      not part of the project artifacts)

## 10. Final quality pass (manual)

- [ ] Read the final report end-to-end once for spelling/grammar after filling
      personal fields (automated documentation audit already passed)
- [ ] Render the PDF once in a viewer and check page break / image placement
      visually (regenerated edition: 36 pages, 7 figures - automated checks
      passed, and Edge's headless screenshot mode does work on this machine, so
      machine-generated page images can be produced; a **human** visual pass is
      still outstanding; see section 12)
- [ ] Rehearse the presentation and confirm timing (20 slides)
- [ ] Final artifact backup (copy the whole project folder / zip to external
      storage or version control)

## 11. Integrity statement

- [x] No retraining, no test re-evaluation, no dataset or checkpoint
      modification, no new predictions, no fabricated results were performed in
      Stages 7-8; experimental evidence remains frozen
      (checkpoint mtime 2026-09-22 19:35:20; evaluation report mtime
      2026-09-22 20:19:05; data directories unchanged)
- [x] Phase 4 documentation update was **additive only**: no application,
      training, evaluation or dataset code was modified, no model was retrained,
      no evaluation was re-run, and no metric was recomputed
- [x] Experiment 1 content, values and artifacts were preserved unchanged
      (verified by diff and by re-checking every frozen value)
- [x] Experiment 2 figures were transcribed from the committed JSON artifacts
      and re-verified against them field by field

---

## 12. Outstanding blocker - PDF regeneration

- [x] **The report PDF now includes Experiment 2.** Regenerated from the
      Markdown source via the committed `scripts/md2pdf.py` converter and the
      locally installed Microsoft Edge headless print-to-PDF.
- [x] **The PDF-generation workflow is now committed and reproducible.**
      `scripts/md2pdf.py` (Python standard library only) converts the Markdown
      report to HTML and prints it to PDF through Edge. Documented in
      `scripts/README.md`. No new package was installed and neither project
      virtual environment was modified.
- [x] **Four broken Experiment 2 image paths corrected.** The report referenced
      `experiment_2_large_dataset/reports/*.png`, which resolve to a
      non-existent directory now that the report lives in `reports/`. Corrected
      to `../experiments/experiment_2_large_dataset/reports/*.png`. `md2pdf.py`
      now aborts with a non-zero exit if any referenced image is missing, so a
      broken figure can never silently reach the PDF again.
- [x] **Conversion verified programmatically.** 36 pages, 1,111,375 bytes, all
      29 tables / 30 rules / 7 figures present, all HTML tags balanced, zero
      unrendered Markdown markers, and every required text probe present
      (Experiment 1 and Experiment 2 metrics, both confusion matrices, the
      Experiment 2 interpretive caveats, and all seven figure captions).
      Source-to-PDF correspondence was also checked: 7 Markdown image references
      = 7 images embedded in the PDF, each matching a source PNG that exists,
      and each figure caption appears exactly once.
- [x] **Regeneration is byte-reproducible.** Two consecutive runs from the
      unchanged Markdown produced byte-identical PDFs (1,111,375 bytes each)
      apart from 4 timestamp bytes. Edge writes **both** `/CreationDate` and
      `/ModDate`, and the command in `scripts/README.md` normalises both and
      reports `identical`.
- [x] **Edge headless screenshot mode confirmed working on this machine.** An
      earlier note in section 10 stated that it produced no output here; that
      was incorrect. Use the **new** headless mode - plain `--headless` worked
      once and then silently produced no output, while `--headless=new` proved
      reliable:
      `msedge.exe --headless=new --disable-gpu --no-first-run
      --user-data-dir=<temp-profile> --screenshot=<out.png>
      --window-size=<w>,<h> <file-url>`.
      The full report was captured in one 794x36400 px image and sliced into
      32 page-sized images; no blank page and no content outside the print
      margins was detected. This enables machine-generated page images, not a
      human visual review.
- [ ] **Human visual review of the PDF is still outstanding.** Automated text,
      structure, image-embedding, page-count and reproducibility checks all pass,
      and page images were rendered automatically, but nobody has yet looked
      through the 36 pages in a PDF viewer to confirm page breaks, figure
      placement and typography by eye. The student should do this before
      submission. Note that the machine-generated page images are an
      *approximation* of the printed pagination (they are screen captures of the
      same HTML, sliced at fixed A4 content height, so they do not reproduce the
      real page breaks), and automated figure-placement detection was
      inconclusive - the only trustworthy figure evidence is that all 7 source
      images are embedded in the PDF at their original pixel dimensions.

### 12.1 Regenerating the PDF

```
python scripts/md2pdf.py reports/DeepGuard_SIP_Final_Technical_Report.md reports/DeepGuard_SIP_Final_Technical_Report.pdf
```

Requires Python 3 and Microsoft Edge. See `scripts/README.md`.

---

**Submission status:** all technical artifacts, including the Experiment 2
documentation, are ready in Markdown form. The **PDF has been regenerated and
verified**, and the **personal fields in section 9 and the manual items in
section 10 require the student's input** before final submission.