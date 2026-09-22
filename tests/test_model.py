import numpy as np
from mmbillhr import DEFAULT_SEGMENTS, SegmentModel, run_all, sweep


def test_no_ai_is_identity():
    # g=1, delta=0, gap=1 means AI does nothing; every change should be ~0
    from mmbillhr import GlobalParams
    for s in DEFAULT_SEGMENTS.values():
        r = SegmentModel(s.with_(beta=1.0, delta=0.0, gap=1.0), GlobalParams(g=1.0)).solve()
        assert abs(r.price) < 1e-3 and abs(r.revenue) < 1e-3 and abs(r.ppp) < 1e-3


def test_baseline_equilibrium_holds():
    # with no insourcing threat (delta=0, gap=1, no eta change), lam* ~ 0 => price falls with cost
    r = run_all()
    assert (r["price"] < 0).all()


def test_sweep_shape():
    df = sweep("delta", [0, .4])
    assert len(df) == 2 * len(DEFAULT_SEGMENTS)
