"""Apply exact, once-only text replacements to a file; fail if any old text is not found exactly once.
Usage: python replace_once.py FILE PAIRS.py   (PAIRS.py defines R = [(old, new), ...])"""
import runpy, sys
path, pairs = sys.argv[1], sys.argv[2]
R = runpy.run_path(pairs)["R"]
t = open(path).read()
for i, (a, b) in enumerate(R):
    n = t.count(a)
    if n != 1:
        sys.exit(f"pair {i}: found {n} times: {a[:80]!r}")
    t = t.replace(a, b, 1)
open(path, "w").write(t)
print(f"{len(R)} replacements applied to {path}")
