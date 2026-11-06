"""Turn scripts/tmlcn/selection_table.out into the LaTeX section of the supplement."""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
pat = re.compile(r"^(.+?)\s+n=\s*(\d+)\s+(\w+)\s+TF-FedPer ([+-]\d\.\d{4}) \[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]\s+TF-(\w+) ([+-]\d\.\d{4}) \[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]")
lab = {"argmax": "Argmax (paper)", "smooth3": "Smoothed, 3 rounds", "last10": "Last ten rounds"}
uni = {"fixed44": "$K=4$", "fixed88": "$K=8$"}
def f(x):
    return ("\\ensuremath{+}" if x.startswith("+") else "\\ensuremath{-}") + x[1:]
rows, prev = [], None
for line in open(src):
    m = pat.match(line.strip())
    if not m:
        continue
    st, n, rule, a, al, ah, u, b, bl, bh = m.groups()
    st = st.replace("Dirichlet a=", "Dirichlet $\\alpha=") + ("$" if st.startswith("Dirichlet") else "")
    if prev is not None and st != prev:
        rows.append("\\midrule")
    cell = f"{st}, {n} seeds, {uni[u]}" if st != prev else ""
    prev = st
    rows.append(f"{cell} & {lab[rule]} & {f(a)} [{f(al)}, {f(ah)}] & {f(b)} [{f(bl)}, {f(bh)}]\\\\")
body = "\n".join(rows)
tex = r"""\section{Sensitivity to the round-selection rule}
\label{supp:selection}
The primary reading selects the first round that maximises the mean validation AUROC (Eq.~(5) of the main text). A sparse schedule's test curve rises in a sawtooth that peaks after each synchronisation, so this rule can read a post-synchronisation peak, and on a flat validation curve a small numerical difference can move the selected round (the cross-machine check of Section~V-A). Table~\ref{tab:selection} repeats both primary contrasts under two further readings computed from the same per-round logs: the first round that maximises the trailing three-round mean of the validation AUROC, and the mean test AUROC over the last ten rounds, which selects nothing. The argmax reading reproduces every logged value exactly. Intervals are percentile bootstrap over seeds (10,000 resamples), computed by \texttt{scripts/tmlcn/selection\_table.py}; they can differ from those of the main text in the fourth decimal because the resampling stream differs.

No reading shows \tierfed{} ahead of the best uniform period by the relevance threshold of 0.005 in any setting, and on the ResNet the last-ten-round reading puts it behind. The comparison with FedPer depends on the reading: under the smoothed rule the sparse schedules trail FedPer beyond the threshold at $\alpha=10$ and on the ResNet, and at the last rounds they are ahead because full-rate training has overfitted (Section~V-B of the main text).

\begin{table*}[t]
\caption{Both primary contrasts under three readings of the same runs: mean paired difference in AUROC with bootstrap 95\% interval. Best uniform period as chosen in Table~VI of the main text}
\label{tab:selection}
\centering\small
\begin{tabular}{@{}llcc@{}}
\toprule
Setting, seeds, best uniform & Reading & \tierfed{} $-$ FedPer [95\% CI] & \tierfed{} $-$ best uniform [95\% CI]\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}

"""
open(dst, "w").write(tex)
print(len(rows))
