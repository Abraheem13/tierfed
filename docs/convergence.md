# Convergence analysis for budget-constrained multi-rate aggregation

This note states the bound that motivates the TierFed controller, with the
assumptions made explicit, and says plainly where the implementation departs
from the analysed algorithm. It replaces the v2 note, which (i) omitted the
local-step count, (ii) used a gradient-divergence constant while the code
measured parameter dispersion, and (iii) was stated for an objective that is
not defined when heads are private.

## Setting

Clients $k=1,\dots,N$ hold objectives $F_k$ on their own data; the shared
objective is $F(\theta)=\sum_k p_k F_k(\theta)$ with $p_k=n_k/n$. The shared
parameters are partitioned into disjoint blocks (tiers) $\theta=(\theta_g)_{g\in\mathcal G}$.
The private head $\theta_f$ is **not** part of $\theta$: the analysis is for the
shared backbone with the heads frozen at their current values, i.e. for
$F_k(\theta)=F_k(\theta;\theta_{f,k})$. The personalisation gap this introduces is
not bounded here; it is measured empirically by the FedPer-versus-FedAvg
contrast and by the 2x2 factorial.

In round $r$ every client runs $\tau_k$ local SGD steps with step size $\eta$
(one epoch; $\tau_k = \lceil n_k / B\rceil$). Tier $g$ is averaged across clients
every $K_g$ rounds, so between two synchronisations of tier $g$ a client takes
$K_g\tau_k \le K_g\tau_{\max}$ local steps on that block.

## Assumptions

* **A1** Each $F_k$ is $L$-smooth.
* **A2** Stochastic gradients are unbiased with blockwise variance $\sigma_g^2$.
* **A3** (per-tier bounded heterogeneity)
  $\frac1N\sum_k\|\nabla_g F_k(\theta)-\nabla_g F(\theta)\|^2\le\Gamma_g$ for all $\theta$.
* **A4** $F$ is bounded below by $F^\star$.

## Proposition

Under A1–A4 there are absolute constants $c_0,c_1,c_2$ and a step-size
threshold $\eta_0 = c_0/(L K_{\max}\tau_{\max})$ such that for $\eta\le\eta_0$,
after $R$ rounds,

$$
\frac1R\sum_{r=1}^{R}\mathbb E\|\nabla F(\bar\theta_r)\|^2
\;\le\;
\frac{c_1\,(F(\bar\theta_1)-F^\star)}{\eta\,\tau_{\min} R}
\;+\;c_1\,\eta L\sum_g\frac{\sigma_g^2}{N}
\;+\;c_2\,\eta^2L^2\sum_g K_g^2\tau_{\max}^2\,\bigl(\Gamma_g+\sigma_g^2\bigr).
$$

The constants are those of Wang and Joshi (2021, Theorem 1) and are not
optimised here; the statement we rely on is the *structure* of the third
term, not its constant.

*Proof.* Standard perturbed-iterate argument for local SGD (Stich 2019; Wang
and Joshi 2021, Thm. 1), applied blockwise. For block $g$, the deviation of a
client's iterate from the virtual average between two synchronisations is a
sum of at most $K_g\tau_{\max}$ stochastic gradient steps, whose squared norm
is bounded by $\eta^2K_g^2\tau_{\max}^2(\Gamma_g+\sigma_g^2)$ up to the constants
of the cited theorem; the blocks are disjoint, so the deviations add. The
descent lemma with the step-size condition absorbs the cross terms. The derivation is
written out in `paper/supp.tex` at the level of the cited theorem. $\square$

With $K_g=1$ and a single block this reduces to the local-SGD bound of
Wang and Joshi with $\tau$ local steps, as it should.

## What the bound says, and what it does not

1. The multi-rate drift term is $\sum_g K_g^2\tau_{\max}^2(\Gamma_g+\sigma_g^2)$:
   quadratic in the interval, linear in heterogeneity. Giving a long
   interval to a tier is cheap only if that tier's $\Gamma_g$ is small. This
   is the quantity the controller minimises, up to the additive
   $\sigma_g^2$ that the controller ignores.
2. $\Gamma_g$ is a *gradient* divergence. The controller cannot observe it;
   it observes the relative parameter dispersion
   $\hat\Gamma_g = \sum_{j\in g}\frac1{|S|}\sum_k\|\theta_{kj}-\bar\theta_j\|^2 / \sum_{j\in g}\|\bar\theta_j\|^2$
   of the slices it receives. After $K_g\tau_k$ local steps the dispersion is
   itself of order $\eta^2K_g^2\tau^2(\Gamma_g+\sigma_g^2)$ by the same argument, so
   $\hat\Gamma_g$ is a (noisy, scale-free) proxy for $K_g^2\Gamma_g$, **not** for
   $\Gamma_g$. The paper states this and reports $\hat\Gamma_g$ against the
   interval that produced it; it does not claim that $\hat\Gamma_g$ estimates
   $\Gamma_g$.
3. The bound is for SGD. The benchmark uses Adam with persistent state for
   every method, because that is what the baselines were tuned with in prior
   work and what practitioners use. An SGD arm (`--optimizer sgd`) is
   provided so the reader can check that the ordering of methods does not
   depend on the solver; the paper reports it in the supplement.
4. Nothing here proves that the integer programme the controller solves is
   the right one, only that its objective is an upper bound on the drift
   term. The controller is evaluated empirically against fixed schedules,
   FedLAMA-style adaptive intervals, AdaComm, and LAG.

## The controller

$$
\min_{K_g\in\{1,\dots,K_{\max}\}}\ \sum_{g\in\mathcal G_{sh}} K_g^2\hat\Gamma_g
\quad\text{s.t.}\quad \sum_{g}\frac{w_g}{K_g}\le\rho .
$$

With $|\mathcal G_{sh}|\le 3$ and $K_{\max}=8$ there are at most 512 candidate
points, so the programme is solved exactly by enumeration every round
(`AdaptiveTierFed._solve_exact`). The v2 greedy doubling rule is retained as
`solver="greedy"`; it only reaches powers of two and is compared with the
exact solution in the budget ablation. The realised payload differs from
$\rho$ because of the warm-up rounds (always full) and the private head
(never sent); the realised fraction is logged and reported.
