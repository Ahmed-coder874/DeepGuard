# DeepGuard - Final Submission Checklist

**Project:** Deepfake Technology - **Project ID:** 24CGCS06
**Stage:** 8 - Final Review, Presentation & Submission Preparation

Legend: `[x]` = verified complete in the project; `[ ]` = **requires the
student's personal input / manual action** (deliberately not marked complete -
these values do not exist in the project artifacts).

---

## 1. Core deliverables

- [x] Final technical report: `reports/DeepGuard_SIP_Final_Technical_Report.md`
- [x] PDF generated from the report:
      `reports/DeepGuard_SIP_Final_Technical_Report.pdf` (8 pages, 3 figures
      embedded, generated with existing tools only - no new dependencies)
- [x] SIP presentation outline (15 slides):
      `reports/DeepGuard_SIP_Presentation.md`
- [x] Viva questions and answers (36 Q/A):
      `reports/DeepGuard_Viva_QA.md`
- [x] This submission checklist: `reports/DeepGuard_Submission_Checklist.md`

## 2. Figures and references

- [x] Figure: training curves - `reports/stage6/training_curves.png`
- [x] Figure: confusion matrix - `reports/stage6/confusion_matrix.png`
- [x] Figure: per-class metrics - `reports/stage6/per_class_metrics.png`
- [x] References section present (16 cited sources, from verified research docs)

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

- [x] Trained checkpoint: `models/deepguard_efficientnet_b0.pt`
      (16,341,163 bytes; epoch 6; seed 42; loads and verified)

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

- [x] Training log preserved: `reports/stage6/train_run_log.txt` (9 epochs,
      best epoch 6, duration 52 min 28 s)
- [x] Official test evaluation: `reports/evaluation_report.json`
      (75.00% accuracy, 120 samples, confusion matrix [[60,0],[30,30]])
- [x] Stage 6 analysis: `reports/stage6_results_analysis.md`
- [x] Final Stage 7/8 consistency audits passed (50/50 checks)

## 8. Limitations & Responsible AI

- [x] Limitations section present (dataset size, single-component test set,
      frame-level only, no generalisation claim, first experiment)
- [x] Responsible AI section present (indications not proof, model confidence
      wording, no unsupported statistics)
- [x] No "state of the art" / "production ready" / "generalizes to all
      deepfakes" claims in the report

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
      visually (generated and structurally verified: 8 pages, 3 figures)
- [ ] Rehearse the presentation and confirm timing (15 slides)
- [ ] Final artifact backup (copy the whole project folder / zip to external
      storage or version control)

## 11. Integrity statement

- [x] No retraining, no test re-evaluation, no dataset or checkpoint
      modification, no new predictions, no fabricated results were performed in
      Stages 7-8; experimental evidence remains frozen
      (checkpoint mtime 2026-09-22 19:35:20; evaluation report mtime
      2026-09-22 20:19:05; data directories unchanged)

---

**Submission status:** all technical artifacts are ready; the **personal
fields in section 9 and the manual items in section 10 require the student's
input** before final submission.