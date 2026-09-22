# AI x Billable Hours Model

An economic model of how AI affects billable-hours law firms. Four lines of work
(premium, commodity-elite, mid-market, small-client) run through hours
compression, a make-or-buy decision by clients, endogenous pricing under
competition, and a pass-through parameter. Outputs are long-run percentage
changes in price, book, revenue, hours by tier, leverage, and profit per partner.

All parameter values are illustrative, not calibrated to data.

## Quick start

```bash
pip install -r requirements.txt
python scripts/generate_tables.py     # writes tables/*.csv and tables/*.md
python scripts/generate_charts.py     # writes charts/*.svg (light and dark) and charts/*.png
python -m pytest tests
```

```python
from mmbillhr import run_all, sweep, grid2d
run_all()[["price", "revenue", "assoc_hours", "ppp"]]
sweep("delta", [0, .2, .4, .6, .8])
grid2d("delta", [0, .4, .8], "beta", [0, .5, 1], "ppp", "Premium")
```

Any segment or global parameter can be overridden by keyword:
`run_all(delta=0.6, g=6)`.

## Model specification

Each line of work is solved independently. A baseline matter uses 1 partner
hour and `l0` associate hours; baseline price is `P0 = r_S + r_J*l0`.

**Step 1 — Hours compression.** `phi_i = (1 - a_i) + a_i/g` for tier i.
Blended `phi_bar = (phi_S + l0*phi_J)/(1 + l0)`. In-house compression
`phi_c = 1 - inhouse_adoption*(1 - phi_bar)`: in-house legal captures that share
of the firms' hours savings (1 = the same, 0 = none).

**Step 2 — Make-or-buy.** Matters indexed by complexity/stakes `s ∈ [0,1]`
with density `Beta(s_a, s_b)` and value weight `v(s) = 1 + nu*s`. Costs in
units of in-house cost per baseline hour; `psi = P/P0`.

- Outside: `mu * psi`
- In-house, pre-AI: `1 + kappa*s + rho*s`
- In-house, post-AI: `phi_c*(1 + kappa*(1 - delta(s))*s) + rho*s`,
  with `delta(s) = clip(delta*(1 + omega*(1 - 2s)), 0, 1)`. `delta` is the
  leveling at mid-complexity (s = 0.5); `omega` sets how fast it fades with
  complexity. At `omega = 1` leveling reaches zero on the hardest work (s = 1);
  at `omega = 0` it is the same at every level. Whether AI can close the
  expertise gap on the most specialised work is what `omega` encodes.

Probability a matter is outsourced: `p = 1/(1 + exp(-(C_in - mu*psi)/tau))`.
Share of clients able to insource: `eta0` pre-AI, `eta1 = min(1, eta0*phi_c^(-alpha_F))` post.
Book `B(psi) = (1-eta)*W + eta*∫ f v p ds`; retention `R(psi) = B_post(psi)/B_pre(1)`.

**Step 3 — Pricing.** Baseline insourcing elasticity `e0 = -dlnB_pre/dlnpsi` at 1.
Costs are calibrated so baseline prices are an equilibrium:
`c0 = P0*(1 - 1/(theta + e0))`; non-associate cost `o = c0 - w_J*l0` scales
with partner hours; post-AI cost `c1 = w_J*l0*phi_J + o*phi_S`.
Full-pass-through price solves the symmetric Nash condition
`(psi*P0 - c1)/(psi*P0) * (theta + e_R(psi)) = 1`. Actual price
`psi = psi_full^beta`. Implied `lambda` solves
`r_S*phi_S^(1-lambda) + r_J*l0*phi_J^(1-lambda) = psi*P0`.

**Step 4 — Outcomes.** `Q = R(psi)*psi^(-eps)`; revenue `Q*psi`; associate
hours `Q*phi_J`; partner hours `Q*phi_S`; profit pool
`Q*(psi*P0 - w_J*l0*phi_J)` vs `P0 - w_J*l0`; profit per partner = pool
change / max(1, partner hours). Sustainability check: a single deviating firm
picks `x ≤ psi` to maximise `(x*P0 - c1)*R(x)*(x/psi)^(-theta)`; the table
reports its profit gain and required partner hours.

**Firm mix.** `firm_mix()` combines lines by revenue share; profit is
weighted by share × baseline margin, hours by baseline hours per revenue dollar.

## Parameters

| | Meaning |
|---|---|
| `a_J`, `a_S` | share of associate / partner tasks exposed to AI |
| `g` | speed-up on exposed tasks (global) |
| `l0` | baseline leverage |
| `r_S`, `r_J`, `w_J` | partner rate, associate rate, associate cost |
| `eps` | market-level demand elasticity |
| `theta` | firm-level (cross-firm) elasticity; sets baseline margin `1/(theta+e0)` |
| `beta` | pass-through of cost savings to price |
| `mu` | outside rate / in-house cost per hour |
| `kappa` | in-house expertise penalty slope |
| `rho` | insurance/reputation value of outside counsel |
| `delta` | expertise leveling at mid-complexity |
| `omega` | how fast leveling fades with complexity (1 = gone at the top, 0 = flat) |
| `inhouse_adoption` | share of firms' AI hours savings that in-house legal also captures |
| `eta0` | share of clients able to insource pre-AI |
| `s_a`, `s_b` | Beta shape of matter distribution over `s` |
| `tau`, `nu`, `alpha_F` | choice softness, value weight slope, fixed-cost elasticity (global) |

Defaults are in `mmbillhr/params.py`.

## Known limitations

- Static, long-run only; no adoption path or dynamics.
- `beta` is reduced-form; not derived from a capacity-constrained pricing game.
- No stable equilibrium at low `theta` (roughly < 1.2); reported as `n/a`.
- Leverage and AI exposure do not vary with `s` within a line of work.
- No training pipeline, firm entry/exit, or AI tool costs.

## Layout

```
mmbillhr/params.py   parameter dataclasses and defaults
mmbillhr/model.py    SegmentModel, Results, firm_mix
mmbillhr/grid.py     run_all, sweep, grid2d
scripts/generate_tables.py
scripts/generate_charts.py
tables/             generated CSV + Markdown (see tables/README.md)
charts/             generated SVG (light/dark) + PNG
tests/
```
