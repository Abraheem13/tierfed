r"""Check that every number in a manuscript appears in the repo's generated values.

Corpus: every number in paper/generated/*.tex (written from the logs by the
analysis code) and in scripts/tmlcn/*.out (written from the logs by the TMLCN
scripts). For each number in the manuscript body (comments, \label, \ref, \cite,
\url and figure paths removed) the script reports whether it occurs in the corpus,
with its context, so that the remainder (definitions, constants, equation
coefficients) can be reviewed by hand.
Usage: python check_numbers.py MANUSCRIPT.tex
"""
import glob, re, sys
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?![\w])")
def norm(x):
    x = x.replace(",", "").lstrip("+")
    return x[1:] if x.startswith("-") else x
corpus = set()
for p in glob.glob("paper/generated/*.tex") + glob.glob("scripts/tmlcn/*.out"):
    for m in NUM.finditer(open(p).read().replace("\\ensuremath{-}", "-").replace("\\ensuremath{+}", "+")):
        corpus.add(norm(m.group()))
t = open(sys.argv[1]).read()
t = re.sub(r"(?<!\\)%.*", "", t)
t = t.split("\\begin{document}", 1)[1]
t = re.sub(r"\\(label|ref|eqref|cite|url|includegraphics|fig|bibliography|bibliographystyle|bstctlcite)\{[^}]*\}(\{[^}]*\})?", " ", t)
t = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", t, flags=re.S)
t = t.replace("\\ensuremath{-}", "-").replace("\\ensuremath{+}", "+").replace("\\,", "")
miss = {}
for m in NUM.finditer(t):
    v = norm(m.group())
    if v in corpus:
        continue
    ctx = t[max(0, m.start() - 50): m.end() + 30].replace("\n", " ")
    miss.setdefault(v, ctx)
print(f"numbers not found in the generated corpus: {len(miss)}")
for v, c in sorted(miss.items(), key=lambda kv: kv[0]):
    print(f"{v:>10s} | ...{c}...")
