#!/usr/bin/env python3
"""Convert a Markdown document to PDF on Windows, using Microsoft Edge headless.

Pipeline: Markdown -> HTML (this script, standard library only) -> PDF (Edge).

No third-party Python packages, no network access, and no project virtual
environment are required. Edge must be installed (it ships with Windows 10/11).

Usage:
    python scripts/md2pdf.py INPUT.md OUTPUT.pdf
    python scripts/md2pdf.py INPUT.md OUTPUT.pdf --html OUT.html
    python scripts/md2pdf.py --find-edge
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile

# --------------------------------------------------------------------------
# Edge discovery
# --------------------------------------------------------------------------

_EDGE_CANDIDATES = (
    r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
    r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
    r"%LocalAppData%\Microsoft\Edge\Application\msedge.exe",
    r"%ProgramFiles(x86)%\Microsoft\Edge Beta\Application\msedge.exe",
)


def find_edge() -> str | None:
    """Locate msedge.exe, or return None."""
    override = os.environ.get("DEEPGUARD_EDGE")
    if override:
        if os.path.isfile(override):
            return override
        # An explicit override that does not exist is almost certainly a typo.
        # Fall back, but say so instead of silently ignoring it.
        print(
            "warning: DEEPGUARD_EDGE is set to {!r}, which is not a file; "
            "falling back to automatic discovery".format(override),
            file=sys.stderr,
        )

    for pattern in _EDGE_CANDIDATES:
        path = os.path.expandvars(pattern)
        if os.path.isfile(path):
            return path

    on_path = shutil.which("msedge") or shutil.which("msedge.exe")
    if on_path:
        return on_path
    return None


# --------------------------------------------------------------------------
# Inline Markdown
# --------------------------------------------------------------------------

_PLACEHOLDER = "\x00CODE{}\x00"


def inline(text: str) -> str:
    """Render inline Markdown to HTML.

    Inline code is lifted out first so that emphasis, links and HTML escaping
    cannot corrupt its contents.
    """
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(2))
        return _PLACEHOLDER.format(len(spans) - 1)

    text = re.sub(r"(`+)(.+?)\1", stash, text, flags=re.DOTALL)
    text = html.escape(text, quote=False)

    # Images before links: both start with '[' but '![' disambiguates them.
    text = re.sub(
        r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)",
        lambda m: '<img alt="{}" src="{}">'.format(m.group(1), m.group(2)),
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)",
        lambda m: '<a href="{}">{}</a>'.format(m.group(2), m.group(1)),
        text,
    )

    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text, flags=re.DOTALL)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"<em>\1</em>", text)
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text, flags=re.DOTALL)

    # Backslash escapes, resolved last so they are not treated as markup.
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|~>])", r"\1", text)

    return re.sub(
        _PLACEHOLDER.format(r"(\d+)"),
        lambda m: "<code>{}</code>".format(html.escape(spans[int(m.group(1))], quote=False)),
        text,
    )


# --------------------------------------------------------------------------
# Block Markdown
# --------------------------------------------------------------------------

_HR = re.compile(r"^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$")
_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_UL = re.compile(r"^(\s*)([-*+])\s+(.*)$")
_OL = re.compile(r"^(\s*)(\d{1,9})[.)]\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_row(line: str) -> list[str]:
    """Split a table row on pipes that are not inside inline code."""
    cells: list[str] = []
    buf: list[str] = []
    in_code = False
    for ch in line.strip():
        if ch == "`":
            in_code = not in_code
            buf.append(ch)
        elif ch == "|" and not in_code:
            cells.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf))

    if cells and not cells[0].strip():
        cells.pop(0)
    if cells and not cells[-1].strip():
        cells.pop()
    return [c.strip() for c in cells]


def _aligns(sep: str) -> list[str]:
    out = []
    for cell in _split_row(sep):
        left, right = cell.startswith(":"), cell.endswith(":")
        out.append("center" if left and right else "right" if right else "left")
    return out


def _collect_fence(lines: list[str], i: int) -> tuple[list[str], int]:
    """Collect a fenced code block at lines[i]; return (code, next_index)."""
    fence = _FENCE.match(lines[i])
    indent = len(lines[i]) - len(lines[i].lstrip())
    i += 1
    code: list[str] = []
    while i < len(lines) and not _FENCE.match(lines[i]):
        line = lines[i]
        code.append(line[indent:] if line[:indent].strip() == "" else line.strip())
        i += 1
    return code, i + 1


def _render_list(lines: list[str], start: int, indent: int, ordered: bool) -> tuple[str, int]:
    """Render one list level beginning at lines[start]; return (html, next_index)."""
    tag = "ol" if ordered else "ul"
    out = [f"<{tag}>"]
    i = start
    li_open = False

    while i < len(lines):
        line = lines[i]

        if not line.strip():
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                break
            nxt = lines[j]
            nm = _OL.match(nxt) or _UL.match(nxt)
            if not nm or len(nm.group(1)) < indent:
                break
            if len(nm.group(1)) == indent and ((_OL.match(nxt) is not None) != ordered):
                break
            i = j
            continue

        m_ul, m_ol = _UL.match(line), _OL.match(line)
        match = m_ol or m_ul
        if not match:
            break

        cur = len(match.group(1))
        if cur > indent:
            sub, i = _render_list(lines, i, cur, _OL.match(line) is not None)
            if not li_open:
                out.append("<li>")
            out.append(sub)
            continue
        if cur < indent or (_OL.match(line) is not None) != ordered:
            break

        if li_open:
            out.append("</li>")

        # Gather the item's whole text first: inline spans such as **bold** or
        # `code` may wrap across the item's continuation lines.
        body = [match.group(3)]
        i += 1

        while i < len(lines) and lines[i].strip():
            if _OL.match(lines[i]) or _UL.match(lines[i]):
                break
            if lines[i].lstrip().startswith(">"):
                break
            if _FENCE.match(lines[i]):
                code, i = _collect_fence(lines, i)
                out.append(
                    f"<li>{inline(' '.join(body))}</li>"
                    f"<li><pre><code>{html.escape(chr(10).join(code))}</code></pre>"
                )
                li_open = True
                body = []
                break
            body.append(lines[i].strip())
            i += 1

        if body:
            out.append(f"<li>{inline(' '.join(body))}")
            li_open = True

    if li_open:
        out.append("</li>")
    out.append(f"</{tag}>")
    return "".join(out), i


def render(lines: list[str]) -> list[str]:
    """Render Markdown block lines to an HTML string."""
    out: list[str] = []
    i, total = 0, len(lines)

    while i < total:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            marker, lang = fence.group(1), fence.group(2)
            i += 1
            code = []
            while i < total and not _FENCE.match(lines[i]):
                code.append(lines[i])
                i += 1
            i += 1
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{html.escape(chr(10).join(code))}</code></pre>")
            continue

        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        if _HR.match(line):
            out.append("<hr>")
            i += 1
            continue

        # Table: a header row followed by a separator row.
        if "|" in line and i + 1 < total and _TABLE_SEP.match(lines[i + 1]):
            header = _split_row(line)
            aligns = _aligns(lines[i + 1])
            i += 2
            body = []
            while i < total and "|" in lines[i] and lines[i].strip():
                body.append(_split_row(lines[i]))
                i += 1

            out.append("<table><thead><tr>")
            for idx, cell in enumerate(header):
                style = f' style="text-align:{aligns[idx]}"' if idx < len(aligns) else ""
                out.append(f"<th{style}>{inline(cell)}</th>")
            out.append("</tr></thead><tbody>")
            for row in body:
                out.append("<tr>")
                for idx, cell in enumerate(row):
                    style = f' style="text-align:{aligns[idx]}"' if idx < len(aligns) else ""
                    out.append(f"<td{style}>{inline(cell)}</td>")
                out.append("</tr>")
            out.append("</tbody></table>")
            continue

        if line.lstrip().startswith(">"):
            inner = []
            while i < total and lines[i].lstrip().startswith(">"):
                inner.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append(f"<blockquote>{''.join(render(inner))}</blockquote>")
            continue

        if _UL.match(line) or _OL.match(line):
            ordered = _OL.match(line) is not None
            base = len((_OL.match(line) or _UL.match(line)).group(1))
            block_html, i = _render_list(lines, i, base, ordered)
            out.append(block_html)
            continue

        # Paragraph: absorb following non-blank, non-block lines.
        para = [stripped]
        i += 1
        while i < total:
            nxt = lines[i]
            if not nxt.strip():
                break
            if (
                _HEADING.match(nxt)
                or _HR.match(nxt)
                or _FENCE.match(nxt)
                or _UL.match(nxt)
                or _OL.match(nxt)
                or nxt.lstrip().startswith(">")
                or ("|" in nxt and i + 1 < total and _TABLE_SEP.match(lines[i + 1]))
            ):
                break
            para.append(nxt.strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")

    return out


# --------------------------------------------------------------------------
# HTML document
# --------------------------------------------------------------------------

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body {
  font-family: "Segoe UI", "Calibri", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #16191d;
  max-width: 178mm; margin: 0 auto; padding: 0 2mm;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 19pt; margin: 0 0 6mm; padding-bottom: 3mm;
     border-bottom: 2px solid #16191d; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 14pt; margin: 7mm 0 3mm; padding-bottom: 1.5mm;
     border-bottom: 1px solid #c3c8ce; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 5mm 0 2mm; page-break-after: avoid; }
p { margin: 0 0 3mm; orphans: 2; widows: 2; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }
blockquote {
  margin: 0 0 4mm; padding: 2.5mm 4mm; background: #f5f7f9;
  border-left: 3px solid #9aa4b0; border-radius: 1mm;
  page-break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }
table {
  width: 100%; border-collapse: collapse; margin: 0 0 4mm;
  font-size: 8.8pt; page-break-inside: avoid;
}
th, td { border: 1px solid #b9c0c8; padding: 1.6mm 2.2mm;
         vertical-align: top; text-align: left; }
th { background: #e8ecf1; font-weight: 600; }
tbody tr:nth-child(even) td { background: #f7f9fb; }
pre {
  background: #f2f4f7; border: 1px solid #ccd3da; border-left: 3px solid #6b7684;
  border-radius: 1mm; padding: 2.5mm 3mm; margin: 0 0 3.5mm;
  font-size: 8.2pt; line-height: 1.42; white-space: pre-wrap;
  word-wrap: break-word; page-break-inside: avoid;
}
code {
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: 0.92em; background: #eef1f4; padding: 0.3mm 1mm;
  border-radius: 0.7mm; word-break: break-word;
}
pre code { background: none; padding: 0; font-size: 1em; }
img { max-width: 100%; height: auto; display: block; margin: 2mm auto 4mm;
      page-break-inside: avoid; }
hr { border: 0; border-top: 1px solid #c3c8ce; margin: 5mm 0; }
a { color: #14507d; word-break: break-all; }
strong { font-weight: 650; }
del { color: #6b7280; }
"""


def build_html(md_path: str) -> str:
    with open(md_path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    # Resolve relative image paths to absolute file:// URLs so the HTML works
    # from any output directory.
    base = os.path.dirname(os.path.abspath(md_path))
    missing: list[str] = []

    def absolutise(match: re.Match[str]) -> str:
        alt, src = match.group(1), match.group(2)
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", src) or src.startswith("#"):
            return f'<img alt="{alt}" src="{src}">'
        local = os.path.normpath(os.path.join(base, src))
        if not os.path.isfile(local):
            missing.append(src)
        resolved = local.replace("\\", "/")
        return f'<img alt="{html.escape(alt, quote=True)}" src="file:///{resolved}">'

    body = "\n".join(render(lines))
    body = re.sub(r'<img alt="([^"]*)" src="((?!file:///|https?:)[^"]+)">', absolutise, body)

    # A missing image would otherwise become a silent broken box in the PDF.
    if missing:
        raise SystemExit(
            "error: {} referenced image(s) do not exist, so the PDF would "
            "contain broken figures:\n  {}".format(
                len(missing), "\n  ".join(sorted(set(missing)))
            )
        )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(os.path.splitext(os.path.basename(md_path))[0])}</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )


# --------------------------------------------------------------------------
# PDF generation
# --------------------------------------------------------------------------


def make_pdf(html_path: str, pdf_path: str, edge: str) -> None:
    url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
    profile = tempfile.mkdtemp(prefix="md2pdf-profile-")
    cmd = [
        edge,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--no-first-run",
        "--no-default-browser-check",
        "--user-data-dir=" + profile,
        "--virtual-time-budget=15000",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={os.path.abspath(pdf_path)}",
        url,
    ]
    try:
        # Chromium logs noise to stderr; only its exit code is meaningful.
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    finally:
        shutil.rmtree(profile, ignore_errors=True)

    if not os.path.isfile(pdf_path) or os.path.getsize(pdf_path) == 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise SystemExit(
            "Edge failed to produce a PDF.\n" + "\n".join(tail)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Markdown to PDF via Microsoft Edge headless.")
    parser.add_argument("input", nargs="?", help="Input Markdown file")
    parser.add_argument("output", nargs="?", help="Output PDF file")
    parser.add_argument("--html", help="Also write the intermediate HTML here")
    parser.add_argument("--find-edge", action="store_true", help="Report the Edge path and exit")
    args = parser.parse_args()

    edge = find_edge()
    if args.find_edge:
        if edge:
            print(edge)
            return 0
        print("Microsoft Edge not found. Install Edge or set DEEPGUARD_EDGE.", file=sys.stderr)
        return 1

    if not args.input or not args.output:
        parser.error("input and output are required (or use --find-edge)")
    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1
    if not edge:
        print(
            "Microsoft Edge not found. Install Edge or set DEEPGUARD_EDGE to its path.",
            file=sys.stderr,
        )
        return 1

    html_path = args.html or os.path.join(
        tempfile.gettempdir(), os.path.splitext(os.path.basename(args.input))[0] + ".html"
    )
    os.makedirs(os.path.dirname(os.path.abspath(html_path)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(build_html(args.input))

    make_pdf(html_path, args.output, edge)

    print(f"Edge:   {edge}")
    print(f"HTML:   {html_path}")
    print(f"PDF:    {args.output} ({os.path.getsize(args.output)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())