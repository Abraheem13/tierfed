"""Build the TMLCN manuscript and its submission package from the run logs.

  python scripts/tmlcn/build.py [--analysis] [--zip]

The manuscript source (paper/tmlcn/src/main.tex, supp.tex) contains no numbers. It
refers to values that the analysis writes from the per-seed logs:

  \\num{key}            a number                       (generated/numbers*.tex)
  \\ifflag{key}{A}{B}   a sentence that depends on an outcome
  \\claim{key}          a statement the prose makes; the build stops if its flag is false
  \\gen{name}           a generated table              (paper/tmlcn/generated/name.tex)
  \\mref{label}         a label of the article, used in the supplement
  \\sref{label}         a label of the supplement, used in the article

This script resolves all of them into the self-contained files paper/tmlcn/main.tex and
paper/tmlcn/supp.tex, compiles both, checks the result (no unresolved reference or
citation, no placeholder, fonts embedded) and, with --zip, writes the submission
archive. With --analysis it first regenerates every number, table and figure.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TM = ROOT / "paper" / "tmlcn"
SRC = TM / "src"
GEN = TM / "generated"
NAME = "TierFed_TMLCN_submission"


def sh(cmd, cwd=ROOT, check=True, quiet=True, env=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=quiet, text=True, env=env)
    if check and r.returncode != 0:
        sys.stderr.write((r.stdout or "")[-3000:] + (r.stderr or "")[-3000:])
        raise SystemExit(f"failed: {' '.join(map(str, cmd))}")
    return r


def analysis():
    py = sys.executable
    env = {**os.environ, "PYTHONHASHSEED": "0", "SOURCE_DATE_EPOCH": "0", "PYTHONPATH": str(ROOT)}
    for mod, args in (
        ("src.analysis.consistency", ["--results", "results"]),
        ("src.analysis.crossmachine", ["--results", "results"]),
        ("src.analysis.collect", ["--results", "results", "--dataset", "diabetes", "--model", "mlp"]),
        ("src.analysis.tmlcn_tex", ["--results", "results", "--out", str(GEN)]),
        ("src.analysis.audit_figures", ["--results", "results", "--out", str(TM / "figures"), "--dataset", "diabetes"]),
        ("src.analysis.results_figures", ["--results", "results", "--out", str(TM / "figures")]),
    ):
        print("analysis:", mod, flush=True)
        sh([py, "-m", mod, *args], env=env)


# --------------------------------------------------------------------------- #
def brace_group(s: str, i: int):
    """s[i] == '{': return (content, index after the matching brace)."""
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
        s = f.read_text()
        for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}", s):
            out[m.group(1)], _ = brace_group(s, m.end())
    return out


def expand(s: str, macros: dict, failed: list) -> str:
    out, i = [], 0
    pat = re.compile(r"\\(num|ifflag|claim|gen)\{")
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
                raise KeyError(f"number {name} is not defined")
            out.append(macros[name])
        elif kind == "claim":
            v = macros.get("flag" + name)
            if v not in ("true", "false"):
                raise KeyError(f"flag {name} is not defined")
            if v == "false":
                failed.append(name)
        elif kind == "ifflag":
            a, j = brace_group(s, j)
            b, j = brace_group(s, j)
            v = macros.get("flag" + name)
            if v not in ("true", "false"):
                raise KeyError(f"flag {name} is not defined")
            out.append(expand(a if v == "true" else b, macros, failed))
        else:
            f_ = GEN / f"{name}.tex"
            if not f_.exists():                  # tables of the supplementary experiments
                f_ = ROOT / "paper" / "generated" / f"{name}.tex"
            out.append(expand(f_.read_text().rstrip("\n"), macros, failed))
        i = j


SOURCE_ONLY = ("\\input{../", "\\newcommand{\\runreq}", "\\newcommand{\\gen}", "\\newcommand{\\num}",
               "\\newcommand{\\ifflag}", "\\newcommand{\\claim}", "\\makeatletter\\newcommand{\\gen}")


def flatten(stem: str, macros: dict) -> str:
    s = (SRC / f"{stem}.tex").read_text()
    pre, body = s.split("\\begin{document}", 1)
    keep, in_note = [], False
    for line in pre.splitlines():
        if line.startswith("% -----") and not in_note and "Source of the manuscript" in pre.split(line, 1)[1][:200]:
            in_note = True
            continue
        if in_note:
            if line.startswith("% -----"):
                in_note = False
            continue
        if any(line.startswith(t) for t in SOURCE_ONLY):
            continue
        if line.startswith("\\documentclass"):
            keep += [line, "% Every number, table and figure in this file was generated from the per-seed logs by",
                     "% https://github.com/Abraheem13/tierfed and is written here as plain text."]
            continue
        keep.append(line.replace("{../figures/", "{figures/"))
    failed: list = []
    body = expand(body, macros, failed)
    if failed:
        raise SystemExit(f"{stem}: statements not supported by the logs: {', '.join(sorted(set(failed)))}")
    out = "\n".join(keep) + "\n\\begin{document}" + body
    for bad in ("\\num{", "\\ifflag{", "\\gen{", "\\claim{", "RUN REQUIRED", "\\runreq", "TEXT"):
        if bad in out.split("\\begin{document}", 1)[1]:
            raise SystemExit(f"{stem}: '{bad}' left after flattening")
    return out


def labels(aux: Path) -> dict:
    out = {}
    if aux.exists():
        for m in re.finditer(r"\\newlabel\{([^}]+)\}\{", aux.read_text()):
            grp, _ = brace_group(aux.read_text(), m.end() - 1)
            first, _ = brace_group(grp, 0)
            out[m.group(1)] = first
    return out


def cross(text: str, cmd: str, table: dict, missing: set) -> str:
    def rep(m):
        if m.group(1) not in table:
            missing.add(m.group(1))
            return "??"
        return table[m.group(1)]
    return re.sub(r"\\" + cmd + r"\{([^}]+)\}", rep, text)


def latex(stem: str):
    for c in (["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], ["bibtex", stem],
              ["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], ["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"]):
        sh(c, cwd=TM, check=False)


def checks(stem: str) -> list:
    log = (TM / f"{stem}.log").read_text(errors="replace")
    txt = sh(["pdftotext", "-layout", f"{stem}.pdf", "-"], cwd=TM).stdout
    fonts = sh(["pdffonts", f"{stem}.pdf"], cwd=TM).stdout
    pages = re.search(r"Pages:\s+(\d+)", sh(["pdfinfo", f"{stem}.pdf"], cwd=TM).stdout).group(1)
    bad = []
    for pat, what in ((r"LaTeX Warning: Reference", "undefined reference"), (r"LaTeX Warning: Citation", "undefined citation"),
                      (r"^! ", "LaTeX error"), (r"multiply defined", "label defined twice")):
        if re.search(pat, log, flags=re.M):
            bad.append(what)
    if "??" in txt or "RUN REQUIRED" in txt or "CLAIM FAILED" in txt:
        bad.append("placeholder in the PDF")
    if "Type 3" in fonts:
        bad.append("Type 3 font")
    if any(line.split()[-5:-4] == ["no"] for line in fonts.splitlines()[2:] if line.strip()):
        bad.append("font not embedded")
    blg = (TM / f"{stem}.blg").read_text(errors="replace")
    if re.search(r"Warning--", blg):
        bad.append("BibTeX warning")
    over = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log)]
    print(f"{stem}.pdf: {pages} pages; overfull boxes: {len(over)}" + (f" (largest {max(over):.1f} pt)" if over else "")
          + ("; " + ", ".join(bad) if bad else "; checks passed"))
    return bad


def plain(v: str) -> str:
    """A generated LaTeX value as plain text (for the Markdown documents)."""
    v = v.replace("\\ensuremath{-}", "-").replace("\\ensuremath{+}", "+").replace("\\ensuremath{<10^{-3}}", "<0.001")
    v = v.replace("\\tierfed{}", "TierFed").replace("\\%", "%").replace("$", "").replace("\\,", "")
    return v.replace("<10^{-3}", "<0.001")


def markdown(macros: dict):
    """response_to_review.md and cover_letter.md take their numbers from the same values."""
    for name in ("response_to_review.md", "cover_letter.md"):
        src = SRC / f"{name}.in"
        if not src.exists():
            continue
        missing = []

        def rep(m):
            if m.group(1) not in macros:
                missing.append(m.group(1))
                return "??"
            return plain(macros[m.group(1)])
        lab = {"M": labels(TM / "main.aux"), "S": labels(TM / "supp.aux")}

        def ref(m):
            if m.group(2) not in lab[m.group(1)]:
                missing.append(m.group(0))
                return "??"
            return lab[m.group(1)][m.group(2)]
        text = re.sub(r"@@([MS]):([A-Za-z:\-]+)@@", ref, src.read_text())
        text = re.sub(r"@@([A-Za-z]+)@@", rep, text)
        if missing:
            raise SystemExit(f"{name}: numbers not defined: {sorted(set(missing))}")
        (TM / name).write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", action="store_true", help="regenerate numbers, tables and figures first")
    ap.add_argument("--zip", action="store_true", help="write the submission archive")
    a = ap.parse_args(argv)
    if a.analysis:
        analysis()
    macros = read_macros(ROOT / "paper" / "generated" / "numbers.tex", GEN / "numbers_au.tex", GEN / "numbers_tmlcn.tex")
    flat = {stem: flatten(stem, macros) for stem in ("main", "supp")}
    missing: set = set()
    for rnd in range(2):                       # second pass: labels of the other document are known
        lab = {stem: labels(TM / f"{stem}.aux") for stem in flat} if rnd else {"main": {}, "supp": {}}
        missing.clear()
        (TM / "main.tex").write_text(cross(flat["main"], "sref", lab["supp"], missing))
        (TM / "supp.tex").write_text(cross(flat["supp"], "mref", lab["main"], missing))
        for stem in flat:
            latex(stem)
    if missing:
        raise SystemExit(f"cross-document labels not found: {sorted(missing)}")
    bad = [f"{stem}: {b}" for stem in flat for b in checks(stem)]
    if bad:
        raise SystemExit("build checks failed: " + "; ".join(bad))
    markdown(macros)
    if a.zip:
        out = ROOT / "paper" / "tmlcn" / "submission"
        pkg = out / NAME
        if pkg.exists():
            shutil.rmtree(pkg)
        (pkg / "figures").mkdir(parents=True)
        used = set(re.findall(r"\\fig\{([^}]+)\}", (TM / "main.tex").read_text() + (TM / "supp.tex").read_text()))
        for f in sorted(used):
            shutil.copy(TM / "figures" / f"{f}.pdf", pkg / "figures" / f"{f}.pdf")
        for f in ("main.tex", "supp.tex", "references.bib", "main.bbl", "supp.bbl", "main.pdf", "supp.pdf"):
            shutil.copy(TM / f, pkg / f)
        # the package must compile on its own, from the .tex and .bib files only
        test = out / "_compile_test"
        if test.exists():
            shutil.rmtree(test)
        shutil.copytree(pkg, test)
        for f in ("main.pdf", "supp.pdf", "main.bbl", "supp.bbl"):
            (test / f).unlink()
        for stem in ("main", "supp"):
            for c in (["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], ["bibtex", stem],
                      ["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], ["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"]):
                sh(c, cwd=test, check=False)
            a_ = sh(["pdftotext", f"{stem}.pdf", "-"], cwd=test).stdout
            b_ = sh(["pdftotext", f"{stem}.pdf", "-"], cwd=pkg).stdout
            if a_ != b_:
                raise SystemExit(f"{stem}: the package does not reproduce the PDF when compiled alone")
        shutil.rmtree(test)
        z = out / f"{NAME}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(pkg.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(out))
        print("wrote", z.relative_to(ROOT), f"({z.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
