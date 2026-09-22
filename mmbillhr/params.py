from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GlobalParams:
    """Parameters shared by every line of work."""
    g: float = 4.0        # speed-up on AI-exposed tasks
    tau: float = 0.4      # softness of the make-or-buy choice (logit scale)
    nu: float = 4.0       # value weight slope: v(s) = 1 + nu*s
    alpha_F: float = 1.0  # elasticity of in-house-capable client share to phi_c
    n_grid: int = 1500    # integration grid over s
    h: float = 1e-4       # step for numerical derivatives


@dataclass(frozen=True)
class SegmentParams:
    """Parameters for one line of work (segment)."""
    name: str
    # Step 1: hours compression
    a_J: float   # share of associate tasks exposed to AI
    a_S: float   # share of partner tasks exposed to AI
    l0: float    # baseline associate hours per partner hour (leverage)
    # Rates and costs ($/hr)
    r_S: float   # partner bill rate
    r_J: float   # associate bill rate
    w_J: float   # associate cost
    # Demand and competition
    eps: float   # market-level price elasticity of demand
    theta: float # firm-level (cross-firm) price elasticity
    beta: float  # pass-through of cost savings to price (1 = full, 0 = none)
    # Step 2: make-or-buy
    mu: float    # outside rate as multiple of in-house cost per hour
    kappa: float # in-house expertise penalty slope (kappa_bar)
    rho: float   # insurance/reputation value of outside counsel slope (rho_bar)
    delta: float # expertise leveling at mid-complexity (s = 0.5)
    inhouse_adoption: float  # share of firms' AI hours savings that in-house legal also captures
                             # (phi_c = 1 - inhouse_adoption*(1 - phi_bar); 1 = same as firms, 0 = none)
    eta0: float  # share of clients able to insource before AI
    s_a: float   # Beta(a, b) shape of matter distribution over s
    s_b: float
    omega: float = 1.0  # how fast leveling fades with complexity:
                        # delta(s) = clip(delta*(1 + omega*(1 - 2s)), 0, 1)
                        # 1 = fades to zero at s = 1 (no leveling on the hardest work); 0 = flat

    def with_(self, **kw) -> "SegmentParams":
        return replace(self, **kw)


DEFAULT_SEGMENTS = {
    "Premium": SegmentParams("Premium", a_J=.40, a_S=.15, l0=2.0, r_S=1800, r_J=1000, w_J=400,
                             eps=.2, theta=1.3, beta=.25, mu=4.5, kappa=4, rho=4, delta=.4,
                             inhouse_adoption=.9, eta0=.95, s_a=12, s_b=2),
    "Commodity-elite": SegmentParams("Commodity-elite", a_J=.55, a_S=.25, l0=4.5, r_S=1300, r_J=850, w_J=400,
                                     eps=.6, theta=2.5, beta=1.0, mu=4, kappa=4, rho=2, delta=.4,
                                     inhouse_adoption=.9, eta0=.95, s_a=6, s_b=4),
    "Mid": SegmentParams("Mid", a_J=.50, a_S=.20, l0=2.0, r_S=800, r_J=500, w_J=250,
                         eps=.8, theta=1.5, beta=1.0, mu=3, kappa=4, rho=1, delta=.4,
                         inhouse_adoption=.9, eta0=.5, s_a=4, s_b=4),
    "Small": SegmentParams("Small", a_J=.55, a_S=.35, l0=0.5, r_S=350, r_J=200, w_J=120,
                           eps=1.5, theta=1.55, beta=1.0, mu=3, kappa=8, rho=2, delta=.5,
                           inhouse_adoption=1.0, eta0=1.0, s_a=2, s_b=4),
}
