"""Core model. See README for the full specification.

All outputs are long-run changes relative to the pre-AI baseline, expressed
as fractions (0.10 = +10%). NaN means no equilibrium was found for those
parameters (typically theta too low).
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.optimize import brentq, minimize_scalar

from .params import GlobalParams, SegmentParams


def phi(a: float, g: float) -> float:
    """Hours compression factor for a tier with exposure a and speed-up g."""
    return (1 - a) + a / g


@dataclass
class Results:
    segment: str
    phi_J: float
    phi_S: float
    phi_c: float
    e0: float            # baseline insourcing elasticity
    margin0: float       # baseline percentage margin implied by theta
    psi_full: float      # full-pass-through equilibrium price ratio
    psi: float           # actual price ratio (psi_full ** beta)
    lam: float           # implied lambda (rate capture)
    price: float         # psi - 1
    book: float          # retention R(psi) - 1  (book lost to insourcing/DIY)
    volume: float        # Q - 1
    revenue: float
    assoc_hours: float
    partner_hours: float
    leverage: float      # post-AI leverage level (not a change)
    profit_pool: float
    ppp: float           # profit per partner
    dev_gain: float      # profit gain to a single firm from undercutting
    dev_partner_hours: float  # partner hours the deviator would need vs today

    def as_dict(self):
        return asdict(self)


class SegmentModel:
    def __init__(self, seg: SegmentParams, glob: GlobalParams = GlobalParams()):
        self.p, self.g = seg, glob
        p, g = seg, glob
        self.S = np.linspace(0.0005, 0.9995, g.n_grid)
        self.phi_J, self.phi_S = phi(p.a_J, g.g), phi(p.a_S, g.g)
        phi_bar = (self.phi_S + p.l0 * self.phi_J) / (1 + p.l0)
        self.phi_c = 1 - p.inhouse_adoption * (1 - phi_bar)
        self.delta_s = np.clip(p.delta * (1 + p.omega * (1 - 2 * self.S)), 0, 1)
        self.f = beta_dist.pdf(self.S, p.s_a, p.s_b)
        self.v = 1 + g.nu * self.S
        self.W = np.trapezoid(self.f * self.v, self.S)
        self.P0 = p.r_S + p.r_J * p.l0
        self.eta1 = min(1.0, p.eta0 * self.phi_c ** (-g.alpha_F))
        self._B0 = self.book(1.0, post=False)

    # ---- Step 2: make-or-buy ----------------------------------------------
    def in_house_cost(self, post: bool) -> np.ndarray:
        p, S = self.p, self.S
        if post:
            return self.phi_c * (1 + p.kappa * (1 - self.delta_s) * S) + p.rho * S
        return 1 + p.kappa * S + p.rho * S

    def book(self, psi: float, post: bool) -> float:
        p = self.p
        c_in = self.in_house_cost(post)
        eta = self.eta1 if post else p.eta0
        pr = 1 / (1 + np.exp(-(c_in - p.mu * psi) / self.g.tau))
        return (1 - eta) * self.W + eta * np.trapezoid(self.f * self.v * pr, self.S)

    def retention(self, psi: float, post: bool = True) -> float:
        return self.book(psi, post) / self._B0

    def _log_elasticity(self, psi: float, post: bool) -> float:
        h = self.g.h
        return -(math.log(self.retention(psi + h, post)) -
                 math.log(self.retention(psi - h, post))) / (2 * h) * psi

    # ---- Step 3: pricing ---------------------------------------------------
    def solve(self) -> Results:
        p, P0 = self.p, self.P0
        e0 = self._log_elasticity(1.0, post=False)
        markup_den = p.theta + e0
        if markup_den <= 1:
            return self._nan_results(e0)
        margin0 = 1 / markup_den
        c0 = P0 * (1 - margin0)
        other = c0 - p.w_J * p.l0                      # non-associate cost, scales with partner hours
        c1 = p.w_J * p.l0 * self.phi_J + other * self.phi_S

        R = lambda x: self.retention(x, True)

        def foc(x):
            return (x * P0 - c1) / (x * P0) * (p.theta + self._log_elasticity(x, True)) - 1

        lo, hi = c1 / P0 + 1e-3, 3.0
        try:
            psi_full = brentq(foc, lo, hi)
        except ValueError:
            return self._nan_results(e0, margin0)
        psi = psi_full ** p.beta

        try:
            lam = brentq(lambda l: (p.r_S * self.phi_S ** (1 - l) + p.r_J * p.l0 * self.phi_J ** (1 - l)) / P0 - psi, -10, 10)
        except ValueError:
            lam = float("nan")

        # ---- Step 4: outcomes ----
        Rpsi = R(psi)
        Q = Rpsi * psi ** (-p.eps)
        pool0 = P0 - p.w_J * p.l0
        pool = Q * (psi * P0 - p.w_J * p.l0 * self.phi_J)
        partner_hours = Q * self.phi_S
        ppp = (pool / pool0) / max(1.0, partner_hours)

        # ---- sustainability check: single-firm deviation ----
        def neg_profit(x):
            return -(x * P0 - c1) * R(x) * (x / psi) ** (-p.theta)
        if psi > lo + 1e-6:
            x_dev = minimize_scalar(neg_profit, bounds=(lo, psi), method="bounded").x
            dev_gain = neg_profit(x_dev) / neg_profit(psi) - 1
            dev_ph = R(x_dev) * psi ** (-p.eps) * (x_dev / psi) ** (-p.theta) * self.phi_S
        else:
            dev_gain, dev_ph = 0.0, partner_hours

        return Results(
            segment=p.name, phi_J=self.phi_J, phi_S=self.phi_S, phi_c=self.phi_c,
            e0=e0, margin0=margin0, psi_full=psi_full, psi=psi, lam=lam,
            price=psi - 1, book=Rpsi - 1, volume=Q - 1, revenue=Q * psi - 1,
            assoc_hours=Q * self.phi_J - 1, partner_hours=partner_hours - 1,
            leverage=p.l0 * self.phi_J / self.phi_S,
            profit_pool=pool / pool0 - 1, ppp=ppp - 1,
            dev_gain=dev_gain, dev_partner_hours=dev_ph - 1,
        )

    def _nan_results(self, e0, margin0=float("nan")):
        n = float("nan")
        return Results(self.p.name, self.phi_J, self.phi_S, self.phi_c, e0, margin0,
                       n, n, n, n, n, n, n, n, n, n, n, n, n, n)


def firm_mix(results: dict, segments: dict, weights: dict) -> dict:
    """Combine segment results for a firm with given revenue shares.

    results: {name: Results}; segments: {name: SegmentParams};
    weights: {name: revenue share}. Profit is weighted by revenue share x
    baseline margin; hours by baseline hours per revenue dollar.
    """
    tot_w = sum(weights.values())
    rev = prof_num = prof_den = ah_num = ah_den = ph_num = ph_den = 0.0
    for k, w in weights.items():
        r, p = results[k], segments[k]
        P0 = p.r_S + p.r_J * p.l0
        pm = (P0 - p.w_J * p.l0) / P0
        w = w / tot_w
        rev += w * r.revenue
        prof_num += w * pm * r.profit_pool; prof_den += w * pm
        ah = w * p.l0 / P0; ph = w / P0
        ah_num += ah * (1 + r.assoc_hours); ah_den += ah
        ph_num += ph * (1 + r.partner_hours); ph_den += ph
    pool = prof_num / prof_den
    ph = ph_num / ph_den
    return {"revenue": rev, "profit_pool": pool, "ppp": (1 + pool) / max(1.0, ph) - 1,
            "assoc_hours": ah_num / ah_den - 1, "partner_hours": ph - 1,
            "leverage0": ah_den / ph_den, "leverage1": ah_num / ph_num}
