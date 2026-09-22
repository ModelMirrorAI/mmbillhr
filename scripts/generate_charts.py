"""Generate the site charts in charts/ (SVG in light and dark, plus one PNG).

Run from the repo root:  python scripts/generate_charts.py

Output is deterministic: the same code and parameters produce byte-identical
files, so the workflow only commits when a chart actually changes.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mmbillhr import DEFAULT_SEGMENTS, GlobalParams, run_all
from mmbillhr.grid import _run

OUT = os.path.join(ROOT, "charts")
os.makedirs(OUT, exist_ok=True)

# Colours follow the modelmirror.ai tokens so the charts sit on the page.
THEMES = {
    "light": dict(paper="#f6f5f1", ink="#1c1f26", muted="#5d616a", rule="#cfcdc5", accent="#7f621c"),
    "dark":  dict(paper="#131419", ink="#e6e3da", muted="#9a9ca4", rule="#34363f", accent="#c9a24b"),
}

plt.rcParams.update({
    "svg.hashsalt": "mmbillhr",      # stable element ids
    "font.family": "DejaVu Serif",   # ships with matplotlib, so CI and local match
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
SVG_META = {"Date": None}
PNG_META = {"Software": None}

pct = FuncFormatter(lambda v, _: f"{v*100:+.0f}%" if v else "0%")


def style(ax, t):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["rule"])
    ax.tick_params(colors=t["muted"], labelcolor=t["muted"])
    ax.grid(axis="y", color=t["rule"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])


def save(fig, name, theme, t, png=False):
    fig.savefig(os.path.join(OUT, f"{name}_{theme}.svg"), transparent=True, metadata=SVG_META,
                bbox_inches="tight")
    if png:
        fig.savefig(os.path.join(OUT, f"{name}.png"), facecolor=t["paper"], dpi=200, metadata=PNG_META,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, theme)


# ---- 1. Headline: profit per partner vs expertise leveling, fading vs flat ----
DELTAS = np.round(np.linspace(0, 0.8, 9), 2)
curves = {om: {seg: [] for seg in DEFAULT_SEGMENTS} for om in (1.0, 0.0)}
for om in curves:
    for d in DELTAS:
        r = run_all(delta=float(d), omega=om)
        for seg in DEFAULT_SEGMENTS:
            curves[om][seg].append(r.loc[seg, "ppp"])
lo = min(min(v) for c in curves.values() for v in c.values())
ylim = (min(-0.1, np.floor(lo * 10) / 10 - 0.05), 0.05)

for theme, t in THEMES.items():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
    for ax, seg in zip(axes.flat, DEFAULT_SEGMENTS):
        style(ax, t)
        ax.axhline(0, color=t["rule"], linewidth=1)
        ax.plot(DELTAS, curves[1.0][seg], color=t["muted"], linewidth=1.8,
                label="leveling fades on the hardest work")
        ax.plot(DELTAS, curves[0.0][seg], color=t["accent"], linewidth=2.2,
                label="leveling reaches the hardest work")
        ax.set_title(seg, loc="left", fontsize=10.5, color=t["ink"])
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(pct)
    for ax in axes[1]:
        ax.set_xlabel("expertise leveling (delta)")
    for ax in axes[:, 0]:
        ax.set_ylabel("profit per partner")
    h, l = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "headline_leveling", theme, t, png=(theme == "light"))

# ---- 2. Sensitivity: Premium profit per partner, one parameter at a time ----
SEG = "Premium"
RANGES = {  # (low, high) for each parameter; everything else at default
    "delta": (0, .8), "omega": (0, 1.5), "beta": (0, 1), "g": (2, 10),
    "inhouse_adoption": (.5, 1.2), "rho": (1, 6), "kappa": (2, 8), "eps": (.1, .5),
    "theta": (1.25, 2.0), "a_J": (.25, .6), "a_S": (.05, .3), "eta0": (.8, 1.0), "mu": (3.5, 6),
}
LABELS = {
    "delta": "expertise leveling", "omega": "leveling fade with complexity", "beta": "pass-through to price",
    "g": "AI speed-up", "inhouse_adoption": "in-house AI adoption", "rho": "insurance value of outside counsel",
    "kappa": "in-house expertise penalty", "eps": "market demand elasticity", "theta": "cross-firm elasticity",
    "a_J": "associate task exposure", "a_S": "partner task exposure", "eta0": "clients able to insource",
    "mu": "outside rate vs in-house cost",
}
seg = DEFAULT_SEGMENTS[SEG]
base = _run(seg, GlobalParams(), {}).ppp
rows = []
for k, (a, b) in RANGES.items():
    va, vb = _run(seg, GlobalParams(), {k: a}).ppp, _run(seg, GlobalParams(), {k: b}).ppp
    if np.isnan(va) or np.isnan(vb):
        print(f"skipping {k}: no equilibrium in range")
        continue
    rows.append((k, a, b, va, vb))
rows.sort(key=lambda r: abs(r[4] - r[3]))

for theme, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(rows) + 1.2))
    style(ax, t)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["rule"], linewidth=0.6)
    for i, (k, a, b, va, vb) in enumerate(rows):
        # draw the longer bar first so a shorter one on the same side stays visible
        bars = sorted(((va, t["muted"], "low"), (vb, t["accent"], "high")), key=lambda x: -abs(x[0] - base))
        for v, col, lab in bars:
            ax.barh(i, v - base, left=base, height=0.62, color=col,
                    label=f"parameter at {lab} end of range" if i == 0 else None)
    ax.axvline(base, color=t["ink"], linewidth=1)
    ax.set_yticks(range(len(rows)), [f"{LABELS[r[0]]}  ({r[1]:g} to {r[2]:g})" for r in rows])
    ax.tick_params(axis="y", length=0, labelcolor=t["ink"])
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel(f"{SEG} profit per partner; line at the default ({base*100:+.0f}%)")
    h, l = ax.get_legend_handles_labels()
    order = [l.index("parameter at low end of range"), l.index("parameter at high end of range")]
    leg = ax.legend([h[j] for j in order], [l[j] for j in order], loc="lower left", frameon=False)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    lo_x = min(min(r[3], r[4]) for r in rows); hi_x = max(max(r[3], r[4]) for r in rows)
    pad = 0.08 * (hi_x - lo_x)
    ax.set_xlim(lo_x - pad, hi_x + pad)
    fig.tight_layout()
    save(fig, "sensitivity_premium", theme, t, png=False)

print("done")
