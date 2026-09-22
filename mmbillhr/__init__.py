"""Economic model of AI's impact on billable-hours law firms."""
from .params import GlobalParams, SegmentParams, DEFAULT_SEGMENTS
from .model import SegmentModel, Results, firm_mix
from .grid import run_all, sweep, grid2d, RESULT_FIELDS

__all__ = ["GlobalParams", "SegmentParams", "DEFAULT_SEGMENTS", "SegmentModel",
           "Results", "firm_mix", "run_all", "sweep", "grid2d", "RESULT_FIELDS"]
