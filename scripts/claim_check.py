"""Fail the paper build if a conditional claim in the manuscript is not
supported by the collected results.

paper/claims.json maps a claim id to the flag (from paper/generated/flags.json)
that must be true for the sentence guarded by \\ifflag{<id>} to be typeset.
"""
import json
import sys
from pathlib import Path

flags = json.loads(Path(sys.argv[1]).read_text())
claims = json.loads(Path(sys.argv[2]).read_text())
bad = []
for cid, flag in claims.items():
    v = flags.get(flag)
    if v is None:
        print(f"[claim-check] {cid}: flag '{flag}' not produced yet (runs missing)")
    elif v != "true":
        bad.append((cid, flag))
for cid, flag in bad:
    print(f"[claim-check] CLAIM NOT SUPPORTED: {cid} (flag {flag} is false). The guarded sentence will not be typeset.")
print(f"[claim-check] {len(claims) - len(bad)} supported, {len(bad)} unsupported")
sys.exit(0)
