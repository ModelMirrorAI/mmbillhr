"""Parameter sweeps and grids, returned as pandas DataFrames."""
from dataclasses import fields
import pandas as pd

from .params import GlobalParams, SegmentParams, DEFAULT_SEGMENTS
from .model import SegmentModel, Results

RESULT_FIELDS = [f.name for f in fields(Results)]
GLOBAL_FIELDS = {f.name for f in fields(GlobalParams)}


def _run(seg: SegmentParams, glob: GlobalParams, overrides: dict) -> Results:
    g_over = {k: v for k, v in overrides.items() if k in GLOBAL_FIELDS}
    s_over = {k: v for k, v in overrides.items() if k not in GLOBAL_FIELDS}
    if g_over:
        glob = GlobalParams(**{**glob.__dict__, **g_over})
    return SegmentModel(seg.with_(**s_over), glob).solve()


def run_all(segments=None, glob: GlobalParams = GlobalParams(), **overrides) -> pd.DataFrame:
    """Solve every segment once. Keyword overrides apply to all segments."""
    segments = segments or DEFAULT_SEGMENTS
    rows = [_run(s, glob, overrides).as_dict() for s in segments.values()]
    return pd.DataFrame(rows).set_index("segment")


def sweep(param: str, values, segments=None, glob: GlobalParams = GlobalParams(), **fixed) -> pd.DataFrame:
    """Vary one parameter (segment or global) across all segments. Long format."""
    segments = segments or DEFAULT_SEGMENTS
    rows = []
    for v in values:
        for s in segments.values():
            r = _run(s, glob, {**fixed, param: v}).as_dict()
            r[param] = v
            rows.append(r)
    df = pd.DataFrame(rows)
    return df[["segment", param] + RESULT_FIELDS[1:]]


def grid2d(p1: str, v1, p2: str, v2, metric: str, segment: str,
           segments=None, glob: GlobalParams = GlobalParams(), **fixed) -> pd.DataFrame:
    """Two-parameter grid for one segment and one output metric (wide format)."""
    segments = segments or DEFAULT_SEGMENTS
    seg = segments[segment]
    data = {}
    for a in v1:
        data[a] = [getattr(_run(seg, glob, {**fixed, p1: a, p2: b}), metric) for b in v2]
    df = pd.DataFrame(data, index=pd.Index(list(v2), name=p2)).T
    df.index.name = p1
    return df
