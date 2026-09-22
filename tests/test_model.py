import numpy as np
import pytest
from mmbillhr import DEFAULT_SEGMENTS, GlobalParams, SegmentModel, run_all, sweep


def test_no_ai_is_identity():
    # No speed-up (g=1) and no leveling (delta=0) means AI does nothing, whatever
    # the other parameters are. Every change should be ~0.
    for s in DEFAULT_SEGMENTS.values():
        r = SegmentModel(s.with_(delta=0.0), GlobalParams(g=1.0)).solve()
        assert abs(r.price) < 1e-3 and abs(r.revenue) < 1e-3 and abs(r.ppp) < 1e-3
        assert r.phi_c == pytest.approx(1.0)


def test_no_inhouse_adoption_means_no_inhouse_speedup():
    s = DEFAULT_SEGMENTS["Mid"].with_(inhouse_adoption=0.0)
    assert SegmentModel(s).phi_c == pytest.approx(1.0)


def test_omega_shapes_leveling():
    s = DEFAULT_SEGMENTS["Premium"]
    flat = SegmentModel(s.with_(omega=0.0))
    assert np.allclose(flat.delta_s, s.delta)                     # omega=0: same leveling everywhere
    fading = SegmentModel(s.with_(omega=1.0))
    assert np.allclose(fading.delta_s, np.clip(2 * s.delta * (1 - fading.S), 0, 1))  # omega=1: original form


def test_flat_leveling_hurts_premium_more():
    # Premium work sits at high complexity, so removing the fade should cost it more.
    fading = SegmentModel(DEFAULT_SEGMENTS["Premium"]).solve()
    flat = SegmentModel(DEFAULT_SEGMENTS["Premium"].with_(omega=0.0)).solve()
    assert flat.ppp < fading.ppp


def test_defaults_lower_prices():
    # At default parameters, AI lowers prices in every line of work.
    r = run_all()
    assert (r["price"] < 0).all()


def test_sweep_shape():
    df = sweep("delta", [0, .4])
    assert len(df) == 2 * len(DEFAULT_SEGMENTS)
