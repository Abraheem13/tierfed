# Convergence analysis for multi-frequency aggregation

This note supplies the theoretical justification whose absence reviewers
identified. It states the assumptions, gives a convergence bound for the
multi-rate scheme, and — crucially — identifies which quantity in the bound is
an *empirical* claim, which `src/theory/divergence.py` then measures.

## Setting

Clients $k=1,\dots,N$ hold local objectives $F_k$, and the global objective is
$F(\theta)=\sum_k p_k F_k(\theta)$ with $p_k=n_k/n$. Parameters are partitioned
into disjoint tiers $\theta=(\theta_g)_{g\in\mathcal{G}}$,
$\mathcal{G}=\{\text{slow},\text{med},\text{fast}\}$. Tier $g$ is averaged
across clients every $K_g$ rounds; the fast tier is never averaged
($K_{\text{fast}}=\infty$).

## Assumptions

**A1 (Smoothness).** Each $F_k$ is $L$-smooth.

**A2 (Bounded stochastic variance).** For every client and tier,
$\mathbb{E}\lVert\nabla_g f_k(\theta;\xi)-\nabla_g F_k(\theta)\rVert^2\le\sigma_g^2$.

**A3 (Tier-restricted gradient divergence).** There exist constants $\Gamma_g$ with
$$\frac{1}{N}\sum_{k=1}^{N}\bigl\lVert \nabla_g F_k(\theta)-\nabla_g F(\theta)\bigr\rVert^2 \;\le\; \Gamma_g \qquad \forall\,\theta .$$
This is the standard bounded-heterogeneity assumption of local-SGD analysis,
stated **per tier** rather than globally. It is the only non-standard element,
and it is exactly what the experiments measure.

**A4 (Bounded personalisation gap).** The fast tier is optimised only locally;
let $\Delta_{\text{fast}}=F(\theta^{\text{pers}})-F(\theta^\star)$ denote the
resulting stationary gap.

## Proposition (informal)

Under A1–A4, with local step size $\eta \le 1/(2L)$ and $R$ rounds of one local
epoch each, the averaged iterates satisfy

$$\frac{1}{R}\sum_{r=1}^{R}\mathbb{E}\bigl\lVert\nabla F(\bar\theta_r)\bigr\rVert^{2} \;\le\; \underbrace{\frac{2\bigl(F(\bar\theta_1)-F^\star\bigr)}{\eta R}}_{\text{optimisation}} \;+\; \underbrace{\eta L \sum_{g}\frac{\sigma_g^{2}}{N}}_{\text{stochastic}} \;+\; \underbrace{4\eta^{2}L^{2}\sum_{g}K_g^{2}\,\Gamma_g}_{\text{multi-rate drift}} \;+\; \Delta_{\text{fast}} .$$

The proof follows the perturbed-iterate argument of Stich (2019) and Wang &
Joshi (2021) applied blockwise: between two synchronisations of tier $g$ the
clients perform $K_g$ local steps on that block, so the accumulated deviation
of that block is bounded by $\eta^2 K_g^2\Gamma_g$; summing over disjoint
blocks gives the stated term. Setting $K_g=1$ for all $g$ recovers the standard
local-SGD/FedAvg rate, so the analysis is a strict generalisation.

## What the bound says

1. **The scheme converges** at the same $O(1/R)$ rate as FedAvg; the schedule
   changes the constant, not the rate. NFL is therefore not "a heuristic
   without convergence guarantees".
2. **The penalty for a long period is $K_g^{2}\Gamma_g$, not $K_g$ alone.**
   Giving a long period to a tier is harmless *provided that tier's divergence
   $\Gamma_g$ is small*. This is precisely the design rule NFL encodes: the
   input projection (broad cross-population structure) is expected to have the
   smallest $\Gamma$, so it tolerates the largest $K$.
3. **The private head trades bias for variance.** Removing the fast tier from
   averaging deletes its $K^2\Gamma$ contribution entirely at the cost of the
   bias term $\Delta_{\text{fast}}$. NFL is therefore predicted to win exactly
   when fast-tier heterogeneity is large relative to the personalisation gap —
   a testable prediction, not a post-hoc story.

## The empirical claim

Point 2 makes the theory falsifiable: the schedule is justified **iff**
$\Gamma_{\text{slow}} \ll \Gamma_{\text{med}} \ll \Gamma_{\text{fast}}$ on real
hospital data. `DivergenceTracker` measures the per-tier divergence every round
and `bound_terms()` reports the products $\eta^2K_g^2\Gamma_g$, so the paper can
show the ordering holds rather than assuming it. If it did not hold, the
schedule ought to be re-tuned — the diagnostic is a design tool, not decoration.

## Scope

The bound is for non-convex smooth objectives with full participation and
one local epoch per round; partial participation adds the usual
$O(\eta^2\sigma^2(1-q)/q)$ sampling term. Deriving tight rates under partial
participation and adaptive periods (as in FedLAMA) is left as future work.
