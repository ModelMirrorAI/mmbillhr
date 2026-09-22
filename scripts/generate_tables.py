"""Generate the static result tables in tables/ (CSV + Markdown).

Run from the repo root:  python scripts/generate_tables.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmbillhr import DEFAULT_SEGMENTS, run_all, sweep, grid2d, firm_mix, SegmentModel

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables")
os.makedirs(OUT, exist_ok=True)
# Start clean so a renamed or dropped table does not linger in the repo.
for f in os.listdir(OUT):
    if f.endswith((".csv", ".md")):
        os.remove(os.path.join(OUT, f))

KEY = ["price", "book", "revenue", "assoc_hours", "partner_hours", "ppp", "lam", "dev_gain"]


def to_md(df: pd.DataFrame, pct_cols=None, decimals=2) -> str:
    df = df.copy()
    pct_cols = set(pct_cols or [])
    cols = list(df.columns)
    idx_names = [n or "" for n in (df.index.names if isinstance(df.index, pd.MultiIndex) else [df.index.name])]
    header = idx_names + [str(c) for c in cols]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in df.iterrows():
        idx = list(idx) if isinstance(idx, tuple) else [idx]
        cells = [str(i) for i in idx]
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v): cells.append("n/a")
                elif c in pct_cols: cells.append(f"{v*100:+.0f}%")
                else: cells.append(f"{v:.{decimals}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save(name, df, pct_cols=None, note=""):
    df.round(6).to_csv(os.path.join(OUT, name + ".csv"))  # rounded to avoid float-noise diffs
    with open(os.path.join(OUT, name + ".md"), "w") as f:
        f.write(f"# {name}\n\n{note}\n\n" + to_md(df, pct_cols) + "\n")
    print("wrote", name)


PCT = ["price", "book", "volume", "revenue", "assoc_hours", "partner_hours", "profit_pool", "ppp",
       "dev_gain", "dev_partner_hours"]

# 1. Baseline
base = run_all()
save("01_baseline", base[KEY], PCT, "Default parameters. Changes vs pre-AI baseline.")

# 2. Elite firm mix
res = {k: SegmentModel(v).solve() for k, v in DEFAULT_SEGMENTS.items()}
rows = []
for w in [0.2, 0.35, 0.5, 0.65, 0.8]:
    m = firm_mix(res, DEFAULT_SEGMENTS, {"Premium": w, "Commodity-elite": 1 - w})
    m["premium_share"] = w
    rows.append(m)
mix = pd.DataFrame(rows).set_index("premium_share")
save("02_elite_firm_mix", mix, ["revenue", "profit_pool", "ppp", "assoc_hours", "partner_hours"],
     "Elite firm outcomes by share of revenue from premium work (Premium beta=0.25, Commodity-elite beta=1).")

# 3. One-parameter sweeps (all segments)
sweeps = {
    "delta": [0, .1, .2, .3, .4, .5, .6, .7, .8],
    "beta": [0, .25, .5, .75, 1.0],
    "theta_mult": None,  # handled below
    "eps": [.2, .4, .6, .8, 1.0, 1.25, 1.5, 2.0],
    "g": [2, 3, 4, 6, 10],
    "inhouse_adoption": [0, .25, .5, .75, 1.0, 1.25],
    "omega": [0, .25, .5, .75, 1.0, 1.5],
    "rho": [0, 1, 2, 3, 4, 6],
    "kappa": [2, 4, 6, 8],
    "a_J": [.3, .4, .5, .6, .7],
}
for prm, vals in sweeps.items():
    if vals is None:
        continue
    df = sweep(prm, vals)
    save(f"03_sweep_{prm}", df.set_index(["segment", prm])[KEY], PCT,
         f"Sweep of `{prm}` with all other parameters at default.")

# theta as a multiple of each segment's default
rows = []
for m in [0.6, 0.8, 1.0, 1.25, 1.5, 2.0]:
    for k, s in DEFAULT_SEGMENTS.items():
        r = SegmentModel(s.with_(theta=s.theta * m)).solve().as_dict()
        r["theta_mult"] = m; r["theta"] = s.theta * m
        rows.append(r)
df = pd.DataFrame(rows).set_index(["segment", "theta_mult"])
save("03_sweep_theta", df[["theta", "margin0"] + KEY], PCT,
     "theta scaled by a multiple of each segment's default. n/a = no equilibrium.")

# 4. Two-parameter grids, ppp and revenue, per segment
grids = [("omega", [0, .5, 1.0, 1.5], "delta", [0, .2, .4, .6, .8]),
         ("delta", [0, .2, .4, .6, .8], "beta", [0, .25, .5, .75, 1.0]),
         ("delta", [0, .2, .4, .6, .8], "eps", [.2, .5, .8, 1.2, 1.6]),
         ("rho", [0, 1, 2, 4, 6], "delta", [0, .2, .4, .6, .8]),
         ("g", [2, 3, 4, 6, 10], "a_J", [.3, .45, .6, .75])]
for p1, v1, p2, v2 in grids:
    for metric in ["ppp", "revenue", "assoc_hours"]:
        for seg in DEFAULT_SEGMENTS:
            g2 = grid2d(p1, v1, p2, v2, metric, seg)
            tag = seg.lower().replace("-", "_")
            save(f"04_grid_{p1}_x_{p2}_{metric}_{tag}", g2, list(g2.columns),
                 f"{metric} for {seg}: rows = `{p1}`, columns = `{p2}`.")

# 5. Index
names = sorted(f[:-3] for f in os.listdir(OUT) if f.endswith(".md") and f != "README.md")
with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write("# Generated tables\n\n" + "\n".join(f"- [{n}]({n}.md) ([csv]({n}.csv))" for n in names) + "\n")
print("done:", len(names), "tables")
