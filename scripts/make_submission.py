"""Build the journal submission package.

Writes one self-contained LaTeX file each for the article and the supplement, with
every generated table (\\gen), number (\\num) and outcome-dependent sentence
(\\ifflag) resolved to literal text, so that no build machinery is needed. It copies
the figures they use, the bibliography and the compiled .bbl files, compiles both
flattened files in an empty folder, checks that their text is identical to the
multi-file build, and writes the zip.

  python scripts/make_submission.py [--paper paper] [--out submission] [--name TierFed_IoTJ_submission]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

SECTIONS = ("abstract", "introduction", "system", "method", "setup", "results", "discussion", "conclusion")


def brace_group(s: str, i: int):
    """s[i] == '{': return (content, index after the matching brace). Escaped
    characters and comments do not count."""
    if s[i] != "{":
        raise ValueError(f"expected '{{' at {i}: {s[i:i + 40]!r}")
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "%":
            j = s.find("\n", j)
            if j < 0:
                break
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError(f"unbalanced braces from {i}: {s[i:i + 60]!r}")


def read_macros(*files: Path) -> dict:
    out = {}
    for f in files:
        if not f.exists():
            continue
        s = f.read_text()
        for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}", s):
            out[m.group(1)], _ = brace_group(s, m.end())
    return out


def expand(s: str, macros: dict, gen: Path) -> str:
    """Resolve \\num{key}, \\ifflag{key}{A}{B} and \\gen{name} recursively."""
    out, i = [], 0
    pat = re.compile(r"\\(num|ifflag|gen)\{")
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:m.start()])
        name, j = brace_group(s, m.end() - 1)
        kind = m.group(1)
        if kind == "num":
            if name not in macros:
                raise KeyError(f"number macro {name} is not defined")
            out.append(macros[name])
        elif kind == "ifflag":
            a, j = brace_group(s, j)
            b, j = brace_group(s, j)
            v = macros.get("flag" + name)
            if v not in ("true", "false"):
                raise KeyError(f"flag {name} is not defined")
            out.append(expand(a if v == "true" else b, macros, gen))
        else:
            out.append(expand((gen / f"{name}.tex").read_text(), macros, gen))
        i = j


def flatten_preamble(pre: str) -> str:
    keep = []
    drop_comments = ("Every number in this manuscript comes from", "src/analysis/collect.py writes",
                     "A table whose runs do not exist")
    for line in pre.splitlines():
        if line.startswith("%") and any(t in line for t in drop_comments):
            continue
        if line.startswith(r"\documentclass"):
            keep.append(line)
            keep.append("% Every number, table and figure in this file was generated from the per-seed logs by")
            keep.append("% https://github.com/Abraheem13/tierfed and is written here as plain text.")
            continue
        if re.match(r"\\IfFileExists\{generated/", line) or line.startswith(r"\newcommand{\runreq}") \
                or r"\newcommand{\gen}" in line or line.startswith(r"\newcommand{\num}") \
                or line.startswith(r"\newcommand{\ifflag}"):
            continue
        if line.startswith(r"\newcommand{\fig}") or line.startswith(r"\providecommand{\fig}"):
            line = r"\newcommand{\fig}[2]{\includegraphics[width=#2]{figures/#1.pdf}}"
        keep.append(line)
    return "\n".join(keep) + "\n"


def flatten(tex: Path, paper: Path, macros: dict) -> str:
    s = tex.read_text()
    pre, body = s.split("\\begin{document}", 1)
    for sec in SECTIONS:
        body = body.replace("\\input{sections/%s}" % sec, (paper / "sections" / f"{sec}.tex").read_text())
    body = expand(body, macros, paper / "generated")
    out = flatten_preamble(pre) + "\\begin{document}" + body
    for bad in (r"\num{", r"\ifflag{", r"\gen{", "RUN REQUIRED", r"\input{"):
        if bad in out.split("\\begin{document}", 1)[1]:
            raise ValueError(f"{tex.name}: '{bad}' left after flattening")
    return out


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def build(d: Path, stem: str):
    for c in (["pdflatex", "-interaction=nonstopmode", stem + ".tex"], ["bibtex", stem],
              ["pdflatex", "-interaction=nonstopmode", stem + ".tex"], ["pdflatex", "-interaction=nonstopmode", stem + ".tex"]):
        run(c, d)


def text(pdf: Path) -> str:
    return subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--out", default="submission")
    ap.add_argument("--name", default="TierFed_IoTJ_submission")
    a = ap.parse_args(argv)
    paper = Path(a.paper).resolve()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    pkg = out / a.name
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "figures").mkdir(parents=True)
    macros = read_macros(paper / "generated" / "numbers.tex", paper / "generated" / "numbers_au.tex")

    # 1) multi-file reference build
    build(paper, "main"); build(paper, "supp")
    # 2) flattened sources, the figures they use and the bibliography
    for stem in ("main", "supp"):
        flat = flatten(paper / f"{stem}.tex", paper, macros)
        (pkg / f"{stem}.tex").write_text(flat)
        for f in sorted(set(re.findall(r"\\fig\{([^}]+)\}", flat))):
            shutil.copy(paper / "figures" / f"{f}.pdf", pkg / "figures" / f"{f}.pdf")
    shutil.copy(paper / "references.bib", pkg / "references.bib")
    # 3) compile the package alone, keep the .bbl files, compare the text with the reference build
    build(pkg, "main"); build(pkg, "supp")
    for stem, ref in (("main", "main.pdf"), ("supp", "supp.pdf")):
        log = (pkg / f"{stem}.log").read_text(errors="ignore")
        if "undefined" in log or "Overfull" in log or not (pkg / f"{stem}.pdf").exists():
            raise SystemExit(f"{stem}: the flattened file does not build cleanly")
        if text(pkg / f"{stem}.pdf") != text(paper / ref):
            raise SystemExit(f"{stem}: the flattened build differs from the multi-file build")
    shutil.move(pkg / "main.pdf", pkg / "TierFed_manuscript.pdf")
    shutil.move(pkg / "supp.pdf", pkg / "TierFed_supplementary_material.pdf")
    for f in pkg.iterdir():
        if f.suffix in (".aux", ".log", ".out", ".blg", ".spl"):
            f.unlink()
    letter = paper / "cover_letter.md"
    if letter.exists():
        shutil.copy(letter, pkg / "cover_letter.md")
    (pkg / "README.txt").write_text(README)
    zpath = out / f"{a.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out))
    print("written", zpath)
    for f in sorted(pkg.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}  ({f.stat().st_size // 1024} kB)")


README = """TierFed: Every Round Is a Release -- submission package
=======================================================

Manuscript (IEEEtran, journal mode)
  main.tex                         the article, one self-contained file
  main.bbl                         compiled bibliography (BibTeX not needed)
  TierFed_manuscript.pdf           the compiled article

Supplementary material
  supp.tex, supp.bbl               the supplement, one self-contained file
  TierFed_supplementary_material.pdf

Shared
  figures/                         every figure the two files include (PDF, fonts embedded)
  references.bib                   the bibliography database (for a fresh BibTeX run)
  cover_letter.md                  cover letter; confirm the two bracketed items before sending

Build (TeX Live or MiKTeX with the standard IEEEtran class):
  pdflatex main && pdflatex main          (main.bbl is included)
  or: pdflatex main && bibtex main && pdflatex main && pdflatex main
  and the same for supp.

Every number, table and figure was generated from the per-seed logs by the code at
https://github.com/Abraheem13/tierfed (one command: scripts/reproduce_v6) and is
written here as plain text, so the sources need no generated files.
"""


if __name__ == "__main__":
    main()
