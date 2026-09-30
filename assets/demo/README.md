# Demonstration images

This folder is for **demo-only images** used when presenting DeepGuard.

## Important rule

> **Do not place any image from the official 120-image held-out test set
> (`data/processed/test/`) in this folder or in a demo.**

The test split is part of the frozen experimental evidence. Re-using it as
casual demo material would contaminate the official evaluation and would
undermine the reproducibility claim in the report. Demo images must be
**separate** from the official evaluation dataset.

## What to use instead

- Your own photographs.
- Screenshots of any interface that is clearly not part of the experiment.
- Frames from the **training** split (`data/processed/train/`) if you need a
  known genuine/manipulated pair - these are already part of the model, so they
  are honest illustrations rather than evaluation evidence.

## Suggested demo pair

| Purpose | Source |
| --- | --- |
| A genuine face photograph | your own photo, or `data/processed/train/real/` |
| A manipulated frame | `data/processed/train/fake/` (FaceForensics++ Deepfakes) |

> FaceForensics++ is a restricted, research-use-only dataset. These images stay
> on your machine and are **not** committed to this repository.

## Never report a demo prediction as an official result

A prediction shown during a demo is a live single-image model output. The only
official numbers for this project are the frozen evaluation results recorded in
`reports/evaluation_report.json` and reported in the README:

- Best validation accuracy: **78.33%**
- Official test accuracy: **75.00%** on the 120-image held-out test set
