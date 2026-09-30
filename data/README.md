# DeepGuard - `data/` directory

**This directory is excluded from Git.** It is created automatically and must be
rebuilt locally. The DeepGuard Streamlit application does **not** need any part
of it in order to run inference on an uploaded image.

## Why it is not committed

- `data/faceforensics/` contains the original FaceForensics++ source videos.
  FaceForensics++ is a **restricted, research-use-only** dataset obtained through
  a signed access request. It must not be redistributed through this repository.
- `data/raw/` and `data/processed/` contain the derived experiment frames
  (~793 MB). They are **reproducible deterministically** from the source videos
  using the scripts and seed documented below, so committing them would bloat
  the repository without adding evidence.
- Only the dataset *methodology* is committed - see `docs/dataset_research.md`
  and `docs/evaluation_plan.md`.

## Expected layout

```
data/
├── faceforensics/            # (you create this) FaceForensics++ download, kept intact
│   ├── original_sequences/youtube/c23/videos/*.mp4          # real
│   └── manipulated_sequences/Deepfakes/c23/videos/*.mp4    # fake
│
├── raw/                      # created by extract_dataset_frames.py
│   ├── real/                 # 300 real frames   (grouped by identity component)
│   ├── fake/                 # 300 fake frames   (grouped by identity component)
│   └── frames_provenance.json
│
└── processed/                # created by prepare_dataset.py
    ├── train/real, train/fake        -> 180 + 180 = 360
    ├── validation/real, validation/fake -> 60 + 60 = 120
    ├── test/real, test/fake          -> 60 + 60 = 120
    └── dataset_summary.json
```

## How to rebuild it (training / evaluation only)

The frozen official results in `reports/` were produced from exactly this
pipeline. Reproducing it is **optional** - it is only needed if you intend to
retrain or re-evaluate. Do not change the split, the labels or the source
videos if you are reproducing the published numbers.

1. Request and download FaceForensics++ (compression `c23`) and extract the 20
   videos (10 real, 10 Deepfakes) into `data/faceforensics/` exactly as shown
   above. See `docs/dataset_research.md` for the access procedure.
2. Extract the frames:

   ```powershell
   python extract_dataset_frames.py
   ```

   Frames are sampled deterministically (30 evenly spaced frames per video).
   Labels come only from the official directory structure. The original videos
   are never modified.

3. Build the leak-free splits:

   ```powershell
   python prepare_dataset.py --group-aligned --audit-leakage --overwrite
   ```

   `--group-aligned` keeps every identity component inside exactly one of
   train/validation/test across both classes, so no source sequence can appear
   in more than one split. The split is seeded (`RANDOM_SEED = 42` in
   `config.py`) and is therefore reproducible.

4. (Optional, training environment only) retrain or re-evaluate:

   ```powershell
   venv-train\Scripts\python.exe train_model.py
   venv-train\Scripts\python.exe evaluate_model.py
   ```

> **Note.** The checkpoint `models/deepguard_efficientnet_b0.pt` is committed,
> so the application works immediately after cloning. Rebuilding `data/` is only
> necessary to reproduce the training or evaluation stages.
