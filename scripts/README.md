# `scripts/md2pdf.py` - Markdown to PDF for the DeepGuard report

Converts a Markdown file to PDF by rendering it to HTML and printing that HTML
with **Microsoft Edge** in headless mode.

This script exists because the committed report PDF
(`reports/DeepGuard_SIP_Final_Technical_Report.pdf`) had no reproducible
generator. The previous PDF's metadata showed a Chromium-family browser had
printed it, but neither the HTML source nor the command was committed, so the
PDF could not be rebuilt after the Experiment 2 edits.

## Requirements

- **Python 3** - standard library only. No packages to install.
- **Microsoft Edge** - already present on Windows 10/11.

The script does **not** use `venv` or `venv-train`, so it cannot perturb the
frozen experiment environments. It installs nothing and touches no environment
variables beyond an optional read-only override.

## Usage

```bash
# Regenerate the submission PDF
python scripts/md2pdf.py \
    reports/DeepGuard_SIP_Final_Technical_Report.md \
    reports/DeepGuard_SIP_Final_Technical_Report.pdf

# Also keep the intermediate HTML (useful for inspecting rendering)
python scripts/md2pdf.py report.md report.pdf --html report.html

# Report the detected Edge executable and exit
python scripts/md2pdf.py --find-edge
```

## Edge discovery order

1. `DEEPGUARD_EDGE` environment variable, if set.
2. Standard install locations (`Program Files`, `Program Files (x86)`,
   `LOCALAPPDATA`).
3. `msedge.exe` / `msedge` on `PATH`.

No absolute path is hard-coded, so the script stays valid on other machines.
Set `DEEPGUARD_EDGE` if Edge lives somewhere unusual.

## Supported Markdown

Headings (ATX), paragraphs, ordered and unordered lists (including nested),
fenced and inline code, blockquotes, pipe tables, images, links, bold, italic,
strikethrough, horizontal rules, and raw HTML passthrough.

Unicode such as `…` and `→` survives the round trip.

## Behaviour worth knowing

- **Relative image paths are resolved against the Markdown file's directory**
  and rewritten to absolute `file:///` URLs, so the PDF is correct regardless
  of the output directory. If a referenced image does not exist, the script
  **aborts with a non-zero exit and lists the missing files** rather than
  emitting a PDF with silent broken figures.
- **A private temporary Edge profile** is used for each run
  (`--user-data-dir`), so printing does not touch the user's real browser
  profile, session or open tabs.
- **A4 pages** with a print-oriented stylesheet: page breaks avoided inside
  tables, code blocks, figures and list items. Tables are kept whole on one
  page, so a table header never needs to repeat across a page break.
- The output PDF is written directly to the path given. **Generate to a scratch
  path and verify before replacing a tracked PDF.**

## Reproducibility

Output is deterministic apart from metadata. Regenerating from an unchanged
Markdown file produces a byte-identical PDF apart from the two timestamps Edge
writes, `/CreationDate` and `/ModDate` (both carry the same value). Comparing
two runs - this normalises both fields:

```bash
python -c "import re,sys; f=lambda p: re.sub(rb'/(?:Creation|Mod)Date\s*\([^)]*\)',b'',open(p,'rb').read()); print('identical' if f(sys.argv[1])==f(sys.argv[2]) else 'DIFFERENT')" old.pdf new.pdf
```

## Adding a new figure

Put the image where the repository keeps it and reference it relative to the
Markdown file:

```markdown
![Descriptive alt text](../experiments/experiment_2_large_dataset/reports/training_loss_curve.png)
```

Note the leading `../` when referencing anything outside `reports/`. Run the
converter and confirm it exits 0 - a typo in the path is reported explicitly.