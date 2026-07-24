"""AUROC vs cumulative upload budget, all arms, 10 seeds each.
Each arm is truncated at its own total budget (it cannot spend more)."""
import glob, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = {
 'fedper'      : 'results/logs/diabetes_mlp_fedper_R60_C20_s*_history.csv',
 'tierfed_0.25': 'results/logs/diabetes_mlp_tierfed_R60_C20_s*_tf_rho0.25_history.csv',
 'tierfed_0.4' : 'results/logs/diabetes_mlp_tierfed_R60_C20_s*_tf_rho0.4_history.csv',
 'tierfed_0.6' : 'results/logs/diabetes_mlp_tierfed_R60_C20_s*_tf_rho0.6_history.csv',
}
SMOOTH = 5
plt.figure(figsize=(6.5, 4.2))
summary = {}
for name, pat in ARMS.items():
    files = [f for f in glob.glob(pat) if re.search(r'_s\d+_(tf_rho[\d.]+_)?history', f)]
    mb, per_seed = None, []
    for f in files:
        h = pd.read_csv(f).sort_values('round')
        per_seed.append(h.fed_auroc.rolling(SMOOTH, min_periods=1).mean().to_numpy())
        mb = h.upload_mb.to_numpy()
    M = np.vstack(per_seed)
    m, s = M.mean(0), M.std(0)
    summary[name] = pd.DataFrame({'upload_mb': mb, 'auroc_mean': m, 'auroc_std': s})
    plt.plot(mb, m, label=f"{name} (n={len(per_seed)})")
    plt.fill_between(mb, m - s, m + s, alpha=.15)
plt.xlabel("cumulative upload (MB)"); plt.ylabel("federated AUROC")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("results/tables/auroc_vs_budget.png", dpi=200)
pd.concat(summary, names=['arm']).to_csv("results/tables/auroc_vs_budget.csv")

# exact crossover: where fedper overtakes each tierfed arm's final value
fp = summary['fedper']
for name in ['tierfed_0.25', 'tierfed_0.4', 'tierfed_0.6']:
    plateau = summary[name].auroc_mean.iloc[-1]
    over = fp[fp.auroc_mean >= plateau]
    x = f"{over.upload_mb.iloc[0]:.1f} MB" if len(over) else "never (>219 MB)"
    print(f"{name:13s} plateau={plateau:.4f} | fedper reaches it at {x} "
          f"(tierfed spent {summary[name].upload_mb.iloc[-1]:.1f} MB)")
