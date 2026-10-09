"""Worked numbers for "The Information Domain".

Companion script to the tutorial at
https://artifocial.com/blog/information-domain-2026-oct-8 and to the two
notebooks in this folder. Every value printed below is one the article quotes
in a worked example, and the article cites this file by section number
(for example "worked_numbers.py section 21"), so the numbering here is fixed.

Pure NumPy. Requires NumPy >= 2.0 for ``np.trapezoid``. Run with

    python worked_numbers.py

and compare against ``worked_numbers.out``.
"""

from math import comb, erf, erfc, lgamma, sqrt

import numpy as np

LOG2 = np.log(2)


def H_bits(p):
    """Shannon entropy of a probability vector, in bits. Zero entries are skipped."""
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def H_nats(p):
    """Shannon entropy of a probability vector, in nats."""
    return H_bits(p) * LOG2


def H2(x):
    """Binary entropy function, in bits."""
    return H_bits([x, 1 - x])


# ---------------------------------------------------------------------------
# 1. Label smoothing
# ---------------------------------------------------------------------------
print("== 1. Label-smoothing floor, both conventions ==")
print("  K      eps   mix-with-uniform (PyTorch)      true=1-eps (notebook)")
for K, eps in [(10, 0.1), (1000, 0.1), (128000, 0.05), (128000, 0.1), (128000, 0.2)]:
    pt = np.full(K, eps / K)
    pt[0] += 1 - eps
    nb = np.full(K, eps / (K - 1))
    nb[0] = 1 - eps
    print(
        f"{K:>7} {eps:>6.2f}   {H_bits(pt):.6f} bits / {H_nats(pt):.6f} nats"
        f"   {H_bits(nb):.6f} bits / {H_nats(nb):.6f} nats"
    )

# ---------------------------------------------------------------------------
# 2. AR(1) entropy rate
# ---------------------------------------------------------------------------
print("\n== 2. AR(1): Kolmogorov-Szego rate vs Shannon entropy of the normalised spectrum ==")
w = np.linspace(-np.pi, np.pi, 400001)
print(
    "  phi   marginal var   rate h_bar [nats]   -int p log p [nats]   (white = log 2pi = %.4f)"
    % np.log(2 * np.pi)
)
for phi in [0.0, 0.6, 0.9, 0.99, -0.7]:
    S = 1.0 / np.abs(1 - phi * np.exp(-1j * w)) ** 2  # sigma^2 = 1
    hbar = 0.5 * np.log(2 * np.pi * np.e) + np.trapezoid(np.log(S), w) / (4 * np.pi)
    p = S / np.trapezoid(S, w)
    Hspec = -np.trapezoid(p * np.log(p), w)
    print(f"{phi:>5.2f}   {1 / (1 - phi**2):>11.4f}   {hbar:>17.6f}   {Hspec:>19.6f}")

# ---------------------------------------------------------------------------
# 3. Scaling the kernel
# ---------------------------------------------------------------------------
print(
    "\n== 3. Scaling the kernel by 4: rate +0.5 ln 4 = %.4f nats = %.4f bits; "
    "normalised spectrum unchanged" % (0.5 * np.log(4), 0.5 * np.log2(4))
)

# ---------------------------------------------------------------------------
# 4. Data-processing inequality on a cascade of binary symmetric channels
# ---------------------------------------------------------------------------
print("\n== 4. Worked DPI chain: X ~ Bern(1/2) -> BSC(0.1) -> BSC(0.1) ==")
f = 0.1
f2 = 2 * f * (1 - f)
print(f"  I(X;Y) = 1 - H2(0.10) = {1 - H2(f):.6f} bits")
print(f"  cascade flip prob f' = 2f(1-f) = {f2:.4f};  I(X;Z) = 1 - H2({f2:.2f}) = {1 - H2(f2):.6f} bits")
print(f"  loss through the second stage = {H2(f2) - H2(f):.6f} bits")

# ---------------------------------------------------------------------------
# 5. Quantisation of a unit Gaussian
# ---------------------------------------------------------------------------
print("\n== 5. Quantisation correction for a unit Gaussian ==")
h = 0.5 * np.log2(2 * np.pi * np.e)
for d in [1.0, 0.1, 0.01]:
    print(f"  h(X) = {h:.4f} bits; Delta={d}: H(X^Delta) ~ h - log2(Delta) = {h - np.log2(d):.4f} bits")
# exact check at Delta = 0.1 by direct binning
edges = np.arange(-12, 12.0001, 0.1)
cdf = 0.5 * (1 + np.vectorize(erf)(edges / sqrt(2)))
pm = np.diff(cdf)
print(f"  exact H of the Delta=0.1 binned Gaussian = {H_bits(pm):.4f} bits")

# ---------------------------------------------------------------------------
# 6. Speculative decoding
# ---------------------------------------------------------------------------
print("\n== 6. Speculative-decoding acceptance: p=(.5,.3,.2), q=(.4,.4,.2) ==")
p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])
print(
    f"  sum min(p,q) = {np.minimum(p, q).sum():.2f};  TV = {0.5 * np.abs(p - q).sum():.2f};"
    f"  KL(p||q) = {float((p * np.log2(p / q)).sum()):.4f} bits"
)

# ---------------------------------------------------------------------------
# 7. Recursive Gaussian fit
# ---------------------------------------------------------------------------
print("\n== 7. Recursive Gaussian fit: within + between chain variance (n=200) ==")
n = 200
for k in [0, 50, 100, 200, 300]:
    within = (1 - 1 / n) ** k
    between = 1 - within
    print(f"  k={k:>3}: E[v_k]={within:.4f}  Var(mu_k)={between:.4f}  sum={within + between:.4f}")

# ---------------------------------------------------------------------------
# 8. Binary entropy
# ---------------------------------------------------------------------------
print("\n== 8. Binary entropy and the 'one question' point ==")
print(f"  H2(0.9) = {H2(0.9):.4f} bits; one yes/no question per draw costs 1.0000 bit")

# ---------------------------------------------------------------------------
# 9. Softmax as entropy-regularised argmax
# ---------------------------------------------------------------------------
print("\n== 9. Softmax as entropy-regularised argmax: numeric check ==")
z = np.array([2.0, 1.0, -1.0])
tau = 0.7
q = np.exp(z / tau)
q /= q.sum()


def obj(r):
    """<r, z> + tau * H(r), the objective softmax(z / tau) maximises over the simplex."""
    return float((r * z).sum() + tau * (-(r[r > 0] * np.log(r[r > 0])).sum()))


rng = np.random.default_rng(0)
best = max(obj(r) for r in rng.dirichlet(np.ones(3), 200000))
print(f"  objective at softmax(z/tau) = {obj(q):.6f};  best of 200,000 random simplex points = {best:.6f}")

# ---------------------------------------------------------------------------
# 10. Diffusion-loss floor via I-MMSE
# ---------------------------------------------------------------------------
print("\n== 10. Diffusion-loss floor via I-MMSE: 'total destroyed' in nats ==")
for name, bits in [("linear", 6.6438), ("cosine", 7.2820)]:
    print(f"  {name}: {bits} bits x ln2 = {bits * np.log(2):.3f} nats")
# sanity check of the identity on a unit Gaussian source in d = 1:
# dI/ds = 1/(2(1+s)) = 0.5 * mmse(s), with mmse(s) = 1/(1+s)
for s_ in [0.5, 1.0, 9.0]:
    dI = (0.5 * np.log(1 + s_ + 1e-6) - 0.5 * np.log(1 + s_ - 1e-6)) / 2e-6
    print(f"  s={s_}: dI/ds (numeric) = {dI:.6f}; 0.5*mmse = {0.5 / (1 + s_):.6f}")

# ---------------------------------------------------------------------------
# 11. Held-out floors under label smoothing
# ---------------------------------------------------------------------------
print(
    "\n== 11. Held-out floors at P(Y=1|x)=0.9, K=2, eps=0.1 (convention A): "
    "which loss owns which floor =="
)
p = np.array([0.9, 0.1])
eps = 0.1
r = (1 - eps) * p + eps / 2


def ce(t, q):
    """Cross-entropy of q against target t, in bits."""
    return float(-(t * np.log2(q)).sum())


print(
    f"  smoothed target r_eps = ({r[0]:.2f}, {r[1]:.2f});  smoothed-objective floor H(r_eps) = "
    f"{H_bits(r):.4f} bits (attained at q = r_eps)"
)
print(f"  hard-label floor H(Y|X) = H2(0.9) = {H_bits(p):.4f} bits (attained at q = P(Y|X))")
print(
    f"  smoothed optimum scored with hard labels: CE(p, r_eps) = {ce(p, r):.4f} bits"
    "  -- attains neither floor"
)
print(
    f"  additive (wrong) formula: H(Y|X) + H(t_y) = {H_bits(p) + H_bits([0.95, 0.05]):.4f} bits, "
    "overstating the minimum"
)

# ---------------------------------------------------------------------------
# 12. Diffusion schedules: the endpoints decide the totals
# ---------------------------------------------------------------------------
print(
    "\n== 12. Diffusion schedules of the notebook, d = 1, unit-variance source: "
    "endpoints decide the totals =="
)
T = 1000
ab_lin = np.cumprod(1 - np.linspace(1e-4, 0.02, T))
t = np.arange(1, T + 1) / T
s_ = 0.008
ab_cos = np.cos((t + s_) / (1 + s_) * np.pi / 2) ** 2 / np.cos(s_ / (1 + s_) * np.pi / 2) ** 2
tot = {}
for name, ab in (("linear", ab_lin), ("cosine", ab_cos)):
    snr = ab / (1 - ab)
    I = 0.5 * np.log2(1 + snr)
    tot[name] = I[0] - I[-1]
    print(
        f"  {name:>6}: alpha_bar_1 = {ab[0]:.6f}  SNR_1 = {snr[0]:.1f}  I(t=1) = {I[0]:.4f} bits"
        f"  I(t=T) = {I[-1]:.6f}  total drop = {I[0] - I[-1]:.4f} bits = {(I[0] - I[-1]) * LOG2:.4f} nats"
        f" (the article quotes the 4-dp bits x ln2: {round(I[0] - I[-1], 4) * LOG2:.3f})"
    )
ratio = (ab_cos[0] / (1 - ab_cos[0])) / (ab_lin[0] / (1 - ab_lin[0]))
print(
    f"  SNR_1 ratio cosine/linear = {ratio:.2f};  0.5*log2(ratio) = {0.5 * np.log2(ratio):.3f} bits"
    f" = {0.5 * np.log2(ratio) * LOG2:.3f} nats;  gap of totals = {tot['cosine'] - tot['linear']:.3f} bits"
)
print(
    "  -> the two floors differ because the SNR endpoints differ, "
    "not because the SNR-integral ELBO invariance failed"
)

# ---------------------------------------------------------------------------
# 13. Fano's inequality can be tight
# ---------------------------------------------------------------------------
print("\n== 13. Binary Fano can be tight: uniform bit through BSC(f) ==")


def H2inv(h):
    """Inverse of the binary entropy function on [0, 1/2], by bisection."""
    lo, hi = 1e-12, 0.5
    for _ in range(200):
        mid = (lo + hi) / 2
        if H2(mid) < h:
            lo = mid
        else:
            hi = mid
    return mid


for f_ in (0.01, 0.1, 0.25):
    print(
        f"  f={f_:.2f}: H(X|Y)=H2(f)={H2(f_):.4f} bits;  Fano: P_e >= H2^-1(H(X|Y)) = "
        f"{H2inv(H2(f_)):.4f};  Bayes error = f = {f_:.2f}  (equality)"
    )

# ---------------------------------------------------------------------------
# 14. Self-improvement toy: mean regret from the printed mean/std columns
# ---------------------------------------------------------------------------
print(
    "\n== 14. Self-improvement toy (§12.1): mean regret from the printed mean/std columns, "
    "E[L] = Var + bias^2 =="
)
# verifier noise -> ((fixed mean, fixed std), (annealed mean, annealed std))
tab = {
    0.0: ((0.9991, 0.0233), (1.0000, 0.0001)),
    0.5: ((0.9688, 0.2108), (1.0355, 0.1987)),
    2.0: ((0.9543, 0.3652), (1.1119, 0.4735)),
    8.0: ((0.9347, 0.6228), (0.8788, 0.7998)),
}
print(
    f"{'verifier noise':>15} {'fixed: E[L]':>13} {'= var':>9} {'+ bias^2':>9}  | "
    f"{'annealed: E[L]':>15} {'= var':>9} {'+ bias^2':>9}"
)
for sv, ((mf, sf), (ma, sa)) in tab.items():
    print(
        f"{sv:>15.2f} {sf**2 + (mf - 1) ** 2:>13.5f} {sf**2:>9.5f} {(mf - 1) ** 2:>9.5f}  | "
        f"{sa**2 + (ma - 1) ** 2:>15.5f} {sa**2:>9.5f} {(ma - 1) ** 2:>9.5f}"
    )
print(
    f"  fixed regret ratio sigma_v=8 / sigma_v=0: "
    f"{(0.6228**2 + 0.0653**2) / (0.0233**2 + 0.0009**2):.0f}x"
)

# ---------------------------------------------------------------------------
# 15. Quantisation in d dimensions
# ---------------------------------------------------------------------------
print("\n== 15. Quantisation in d dimensions: H(X^Delta) = h(X) - d log2(Delta) + o(1) ==")
h1 = 0.5 * np.log2(2 * np.pi * np.e)
h2 = np.log2(2 * np.pi * np.e)
print(
    f"  d=1 unit Gaussian: h = {h1:.4f} bits; Delta=0.1: {h1:.3f} + 1*log2(10) = "
    f"{h1 + np.log2(10):.3f} bits  (exact binned 5.3696)"
)
print(
    f"  d=2 independent standard coords: h = log2(2 pi e) = {h2:.4f} bits; Delta=0.1: "
    f"{h2:.3f} + 2*log2(10) = {h2 + 2 * np.log2(10):.3f} bits"
)
print(
    f"  the one-dimensional formula would give {h2 + np.log2(10):.3f} bits, "
    f"wrong by log2(10) = {np.log2(10):.3f}"
)
edges = np.arange(-12, 12.0001, 0.1)
cdf = 0.5 * (1 + np.vectorize(erf)(edges / sqrt(2)))
pm = np.diff(cdf)
print(f"  exact 2-D grid entropy (product of the two 1-D binned marginals) = {2 * H_bits(pm):.4f} bits")

# ---------------------------------------------------------------------------
# 16. A one-point I-MMSE rule mis-scores a perfect denoiser
# ---------------------------------------------------------------------------
print(
    "\n== 16. Diffusion diagnostic: a one-point rule mis-scores a PERFECT denoiser "
    "(unit Gaussian, mmse = 1/(1+s)) =="
)


def mmse(s):
    """Minimum mean-square error of a unit Gaussian observed at SNR s."""
    return 1 / (1 + s)


lo, hi = 1.0, 2.0
ex = 0.5 * np.log((1 + hi) / (1 + lo))
lower = 0.5 * (hi - lo) * mmse(lo)
upper = 0.5 * (hi - lo) * mmse(hi)
trap = 0.25 * (hi - lo) * (mmse(lo) + mmse(hi))
print(
    f"  interval [1,2]: exact dI = 0.5 ln(3/2) = {ex:.4f} nats; lower-end rule = {lower:.4f} "
    f"({lower - ex:+.4f}); upper-end = {upper:.4f} ({upper - ex:+.4f}); trapezoid = {trap:.4f} "
    f"({trap - ex:+.4f})"
)
for name, ab in (("linear", ab_lin), ("cosine", ab_cos)):
    snr = ab / (1 - ab)
    s_hi, s_lo = snr[:-1], snr[1:]
    exs = 0.5 * np.log((1 + s_hi) / (1 + s_lo))
    low = 0.5 * (s_hi - s_lo) * mmse(s_lo)
    up = 0.5 * (s_hi - s_lo) * mmse(s_hi)
    print(
        f"  {name:>6} schedule, oracle denoiser over 999 steps: exact {exs.sum():.4f} nats, "
        f"lower-end rule {low.sum():.4f}, upper-end {up.sum():.4f}, "
        f"max one-step |err| {np.abs(low - exs).max():.4f}"
    )

# ---------------------------------------------------------------------------
# 17. Kraft: real-valued relaxation vs integer codeword lengths
# ---------------------------------------------------------------------------
print("\n== 17. Kraft: the real-valued relaxation vs integer codeword lengths, p = (0.9, 0.1) ==")
p = np.array([0.9, 0.1])
ideal = -np.log2(p)
print(
    f"  ideal real-valued lengths -log2 p = ({ideal[0]:.3f}, {ideal[1]:.3f}) bits; "
    f"H(p) = {H_bits(p):.4f} bits"
)
print(
    f"  optimal binary prefix code (Huffman) = (1, 1): expected length 1.0000 bits; "
    f"Shannon code ceil(-log2 p) = (1, 4): {0.9 * 1 + 0.1 * 4:.3f} bits (< H+1 = {H_bits(p) + 1:.3f})"
)

# ---------------------------------------------------------------------------
# 18. Hypothesis testing: Stein's exponent vs Chernoff information
# ---------------------------------------------------------------------------
print(
    "\n== 18. Hypothesis testing: Stein's exponent D(P||Q) vs Chernoff information C(P,Q), "
    "P = Bern(0.9), Q = Bern(0.5) =="
)
P = np.array([0.9, 0.1])
Q = np.array([0.5, 0.5])


def D(a, b):
    """KL divergence D(a || b) between two probability vectors, in nats."""
    return float((a * np.log(a / b)).sum())


ts = np.linspace(0, 1, 100001)
C = -min(np.log((P**t * Q ** (1 - t)).sum()) for t in ts)
print(
    f"  D(P||Q) = {D(P, Q):.4f} nats = {D(P, Q) / LOG2:.4f} bits"
    "   (Stein: type-II exponent at fixed type-I error, testing Q against P)"
)
print(f"  D(Q||P) = {D(Q, P):.4f} nats = {D(Q, P) / LOG2:.4f} bits   (the other direction)")
print(
    f"  C(P,Q)  = {C:.4f} nats = {C / LOG2:.4f} bits"
    "   (Chernoff: the symmetric Bayesian error exponent; sits between the two)"
)
print(
    "  exponent-only readings at n = 20 -- NOT error probabilities, see section 23 for the exact "
    f"binomial errors: exp(-n D) = {np.exp(-20 * D(P, Q)):.2e}; exp(-n C) = {np.exp(-20 * C):.2e}"
)

# ---------------------------------------------------------------------------
# 19. Barber-Agakov bound, units
# ---------------------------------------------------------------------------
print("\n== 19. Barber-Agakov in nats on the BSC(0.1), X uniform, q = the true conditional ==")
f = 0.1
Hn = np.log(2)
ElnQ = -(H_bits([f, 1 - f]) * LOG2)
print(
    f"  H_nats(X) = ln 2 = {Hn:.4f}; E ln p(x|y) = -H_nats(X|Y) = {ElnQ:.4f}; "
    f"sum = {Hn + ElnQ:.4f} nats = {(Hn + ElnQ) / LOG2:.4f} bits = I(X;Y) exactly"
)
print(
    f"  mixing units (H in bits, E ln q in nats) would give {1 + ElnQ:.4f}, "
    "which is not a bound on anything"
)

# ---------------------------------------------------------------------------
# 20. Self-improvement toy: regret ratios and bias shares
# ---------------------------------------------------------------------------
print("\n== 20. Self-improvement toy: regret ratios and bias shares from the printed columns ==")
# verifier noise -> mean regret E[L] and its bias^2 part, fixed and annealed policies
fixed = {0.0: 0.00054, 0.5: 0.04541, 2.0: 0.13546, 8.0: 0.39214}
fb = {0.0: 0.0, 0.5: 0.00097, 2.0: 0.00209, 8.0: 0.00426}
ann = {0.0: 0.0, 0.5: 0.04074, 2.0: 0.23672, 8.0: 0.65437}
ab = {0.0: 0.0, 0.5: 0.00126, 2.0: 0.01252, 8.0: 0.01469}
print(
    f"  sigma_v=8: fixed/annealed = {fixed[8.0] / ann[8.0]:.3f} -> holding exploration open cuts "
    f"the mean regret by {100 * (1 - fixed[8.0] / ann[8.0]):.0f}% "
    f"(annealing costs {100 * (ann[8.0] / fixed[8.0] - 1):.0f}% more)"
)
for sv in (0.5, 2.0, 8.0):
    print(
        f"  sigma_v={sv}: bias share  fixed {100 * fb[sv] / fixed[sv]:.1f}%"
        f"   annealed {100 * ab[sv] / ann[sv]:.1f}%"
    )
print("  the annealed bias share peaks at sigma_v = 2 and falls again at 8: small and not monotone")

# ---------------------------------------------------------------------------
# 21. AR(1) at phi = 0.9: arithmetic vs geometric mean of the spectrum
# ---------------------------------------------------------------------------
print("\n== 21. AR(1) at phi = 0.9, unit innovation: arithmetic vs geometric mean of S(omega) ==")
phi = 0.9
w = np.linspace(-np.pi, np.pi, 400001)
S = 1 / (1 - 2 * phi * np.cos(w) + phi**2)
print(
    "  S at omega = 0, pi/4, pi/2, 3pi/4, pi: "
    + ", ".join(
        f"{1 / (1 - 2 * phi * np.cos(x) + phi**2):.3f}"
        for x in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi)
    )
)
arith = np.trapezoid(S, w) / (2 * np.pi)
geom = np.exp(np.trapezoid(np.log(S), w) / (2 * np.pi))
print(f"  arithmetic mean (1/2pi) int S = {arith:.4f} = marginal variance 1/(1-phi^2) = {1 / (1 - phi**2):.4f}")
print(
    f"  geometric mean exp((1/2pi) int log S) = {geom:.4f} = innovation variance; "
    f"spectral flatness = {geom / arith:.4f} = 1 - phi^2 = {1 - phi**2:.2f}"
)
print(
    f"  entropy rate 0.5 ln(2 pi e * 1) = {0.5 * np.log(2 * np.pi * np.e):.4f} nats; "
    f"of white noise at the same POWER 5.263: {0.5 * np.log(2 * np.pi * np.e / (1 - phi**2)):.4f} nats; "
    f"gap = 0.5 ln(1-phi^2) = {0.5 * np.log(1 - phi**2):.4f}"
)

# ---------------------------------------------------------------------------
# 22. The beta-VAE curve vs Shannon's R(D)
# ---------------------------------------------------------------------------
print(
    "\n== 22. The beta-VAE curve of the running family vs Shannon's R(D): the gap, "
    "and the one point where it is zero =="
)
# running example: z ~ N(0,1), x|z ~ N(z, 1/10)  =>  x ~ N(0, 1.1);
# encoder q(z|x) = N(a x, s2); decoder mean c*z (c = 1 unrestricted)
sx2 = 1.1
dv = 0.1


def vae_point(a, s2, c=1.0):
    """Rate, distortion and the Shannon comparison for one encoder/decoder in the family."""
    delta = (1 - c * a) ** 2 * sx2 + c**2 * s2  # E (x - c z)^2: squared error of the reproduction
    R = 0.5 * (sx2 * a * a + s2 - 1 - np.log(s2))  # E_x KL(q(z|x) || N(0,1)), nats
    v = sx2 * a * a + s2  # variance of the aggregate posterior q(z)
    Iq = 0.5 * np.log(v / s2)  # I_q(X;Z), nats
    gap_prior = 0.5 * (v - 1 - np.log(v))  # KL(q(z) || p(z)), nats
    Dn = 0.5 * np.log(2 * np.pi * dv) + delta / (2 * dv)  # E -ln N(x; c z, dv), nats
    Rsh = 0.5 * np.log(sx2 / delta)  # Shannon R(delta) for N(0, 1.1) under squared error, nats
    return delta, R, Iq, gap_prior, Dn, Rsh


print(
    "  unrestricted decoder (c = 1); the family's optimum of D + beta R is "
    "a = 10/(beta+10), s2 = beta/(beta+10):"
)
print(
    "   beta      a     s2   delta=E(x-z)^2   R_VAE [bits]   I_q [bits]   KL(q(z)||p) [bits]"
    "   R(D) Shannon [bits]   R_VAE - R(D)"
)
for beta in (0.25, 0.5, 1.0, 2.0, 4.0):
    a = 10 / (beta + 10)
    s2 = beta / (beta + 10)
    d, R, Iq, g, Dn, Rsh = vae_point(a, s2)
    print(
        f"  {beta:5.2f}  {a:.4f} {s2:.4f}        {d:.4f}        {R / LOG2:.4f}      {Iq / LOG2:.4f}"
        f"              {g / LOG2:.4f}              {Rsh / LOG2:.4f}        {(R - Rsh) / LOG2:+.4f}"
    )
print(
    "  at beta = 1 the family contains the exact posterior, q(z) = N(0,1) = p(z), and "
    "R_VAE = I_q = R(D): the curve TOUCHES R(D) there and sits above it elsewhere"
)
print(
    "  restricted decoder, mean c z with c = 1/2 (the decoder cannot use z at full gain); "
    "best encoder in the family at beta = 1 is a = 10c/(1+10c^2), s2 = 1/(1+10c^2):"
)
c = 0.5
a = 10 * c / (1 + 10 * c * c)
s2 = 1 / (1 + 10 * c * c)
d, R, Iq, g, Dn, Rsh = vae_point(a, s2, c)
print(
    f"  a = {a:.4f}, s2 = {s2:.4f}, delta = {d:.4f}: R_VAE = {R / LOG2:.4f} bits, I_q = {Iq / LOG2:.4f}, "
    f"prior gap = {g / LOG2:.4f}, Shannon R(D) at that delta = {Rsh / LOG2:.4f} bits, "
    f"gap = {(R - Rsh) / LOG2:.4f} bits"
)
hx = 0.5 * np.log(2 * np.pi * np.e * sx2)
xent = 0.5 * np.log(2 * np.pi * (c * c + dv)) + sx2 / (2 * (c * c + dv))
print(
    f"  -ELBO = {(R + Dn) / LOG2:.4f} bits against h(x) = {hx / LOG2:.4f} bits; "
    f"the excess {(R + Dn - hx) / LOG2:.4f} bits is the cross-entropy "
    f"H(N(0,1.1), N(0,{c * c + dv:.2f})) - h(x) = {(xent - hx) / LOG2:.4f}"
)

# ---------------------------------------------------------------------------
# 23. Hypothesis testing at n = 20: exponents are not finite-sample errors
# ---------------------------------------------------------------------------
print("\n== 23. Hypothesis testing at n = 20: asymptotic exponents are not finite-sample error probabilities ==")


def binom_pmf(n, k, p):
    """Binomial probability mass P(K = k) for K ~ Bin(n, p)."""
    return comb(n, k) * p**k * (1 - p) ** (n - k)


n = 20
pP, pQ = 0.9, 0.5
pmfP = np.array([binom_pmf(n, k, pP) for k in range(n + 1)])
pmfQ = np.array([binom_pmf(n, k, pQ) for k in range(n + 1)])
# equal-prior Bayes test: say P iff k >= kstar, where the likelihood ratio pmfP/pmfQ >= 1
# (monotone in k)
kstar = int(np.argmax(pmfP >= pmfQ))
bayes = 0.5 * (pmfP[:kstar].sum() + pmfQ[kstar:].sum())
print(
    f"  equal-prior Bayes test says P iff k >= {kstar} ones of 20; exact error = "
    f"0.5*[P_P(k<={kstar - 1}) + P_Q(k>={kstar})] = "
    f"0.5*[{pmfP[:kstar].sum():.5f} + {pmfQ[kstar:].sum():.5f}] = {bayes:.5f}"
)
print(
    f"  exponent-only reading exp(-n C) with C = {C:.4f}: {np.exp(-n * C):.5f}"
    f"   -- off by a factor {np.exp(-n * C) / bayes:.1f}"
)
# Neyman-Pearson: hold type-I error P_P(say Q) <= alpha, minimise type-II P_Q(say P);
# reject P for small k, randomise at the boundary
alpha = 0.05
cum = 0.0
kb = 0
while cum + pmfP[kb] <= alpha:
    cum += pmfP[kb]
    kb += 1
gamma = (alpha - cum) / pmfP[kb]  # randomisation probability at k = kb
typeII = pmfQ[kb + 1 :].sum() + (1 - gamma) * pmfQ[kb]
print(
    f"  Neyman-Pearson at type-I <= {alpha}: say Q when k <= {kb - 1}, and at k = {kb} with "
    f"probability {gamma:.4f}; exact type-II error = {typeII:.5f}"
)
print(
    f"  exponent-only reading exp(-n D(P||Q)) with D = {D(P, Q):.4f}: {np.exp(-n * D(P, Q)):.6f}"
    f"   -- off by a factor {typeII / np.exp(-n * D(P, Q)):.1f} the other way"
)
print(
    "  Stein's lemma is a statement about lim (1/n) ln(type-II error) = -D(P||Q); at n = 20, "
    f"-(1/n) ln(exact type-II) = {-np.log(typeII) / n:.4f} nats against D = {D(P, Q):.4f}"
)
print(f"  Chernoff: -(1/n) ln(exact Bayes error) = {-np.log(bayes) / n:.4f} nats against C = {C:.4f}")
for nn in (20, 50, 100, 200):
    pm = np.array([binom_pmf(nn, k, pP) for k in range(nn + 1)])
    qm = np.array([binom_pmf(nn, k, pQ) for k in range(nn + 1)])
    ks = int(np.argmax(pm >= qm))
    be = 0.5 * (pm[:ks].sum() + qm[ks:].sum())
    print(
        f"    n = {nn:4d}: -(1/n) ln(Bayes error) = {-np.log(be) / nn:.4f}  (C = {C:.4f}); "
        f"Bayes error = {be:.2e} vs exp(-nC) = {np.exp(-nn * C):.2e}"
    )
P6 = np.array([0.6, 0.4])
C6 = -min(np.log((P6**t * Q ** (1 - t)).sum()) for t in ts)
print(
    f"  a second coin, Bern(0.6) vs fair: Stein exponent D(Bern(0.6)||Bern(0.5)) = {D(P6, Q):.4f} nats; "
    f"D(Bern(0.5)||Bern(0.6)) = {D(Q, P6):.4f}; Chernoff information = {C6:.4f} nats, "
    "strictly smaller than both"
)

# ---------------------------------------------------------------------------
# 24. Bode's sensitivity integral needs relative degree >= 2
# ---------------------------------------------------------------------------
print("\n== 24. Bode's sensitivity integral needs relative degree >= 2: L(s) = 1/(s+1) is a stable counter-example ==")
w = np.linspace(0, 2e5, 4000001)
S_mag = np.sqrt((w**2 + 1) / (w**2 + 4))  # S = (s+1)/(s+2)
val = np.trapezoid(np.log(S_mag), w)
print("  L = 1/(s+1), S = 1/(1+L) = (s+1)/(s+2): both stable, no unstable open-loop poles, relative degree 1")
print(
    f"  int_0^inf ln|S(jw)| dw = (1/2) int ln((w^2+1)/(w^2+4)) dw = (pi/2)(1-2) = {-np.pi / 2:.5f};"
    f"  numerically (to 2e5 rad/s) {val:.5f}"
)
print(
    "  the general relative-degree-1 formula: int ln|S| dw = pi * sum Re(p_k) - (pi/2) lim_{s->inf} s L(s)"
    f" = 0 - (pi/2)*1 = {-np.pi / 2:.5f}"
)
print(
    "  with relative degree >= 2 the limit term is 0 and the integral is pi * sum Re(p_k) over "
    "unstable open-loop poles: zero only when there are none"
)

# ---------------------------------------------------------------------------
# 25. A second coin at n = 20: Bern(0.6) against fair
# ---------------------------------------------------------------------------
print(
    "\n== 25. A second coin at n = 20: Bern(0.6) against fair — Stein exponent, "
    "exact Neyman-Pearson error, Chernoff strictly below =="
)
pP6 = 0.6
pmfP6 = np.array([binom_pmf(n, k, pP6) for k in range(n + 1)])
pmfQ20 = np.array([binom_pmf(n, k, pQ) for k in range(n + 1)])
# Stein's regime: protect P = Bern(0.6) at type-I <= alpha, minimise type-II = Pr_Q[say P];
# the likelihood ratio is monotone in k, so "say Q" for small k
alpha = 0.05
cum = 0.0
kb = 0
while cum + pmfP6[kb] <= alpha:
    cum += pmfP6[kb]
    kb += 1
gamma = (alpha - cum) / pmfP6[kb]
typeII6 = pmfQ20[kb + 1 :].sum() + (1 - gamma) * pmfQ20[kb]
print(
    f"  P = Bern(0.6) protected at type-I <= {alpha}: say Q when k <= {kb - 1}, and at k = {kb} with "
    f"probability {gamma:.4f}; exact type-II error at n = 20 = {typeII6:.4f}"
)
print(
    f"  Stein exponent D(P||Q) = {D(P6, Q):.4f} nats -> exponent-only reading exp(-20 D) = "
    f"{np.exp(-n * D(P6, Q)):.4f}; realised -(1/n) ln(type-II) = {-np.log(typeII6) / n:.4f}"
)
ks6 = int(np.argmax(pmfP6 >= pmfQ20))
bayes6 = 0.5 * (pmfP6[:ks6].sum() + pmfQ20[ks6:].sum())
print(
    f"  equal-prior Bayes test says P iff k >= {ks6}: exact Bayes error = {bayes6:.4f}; "
    f"Chernoff information C = {C6:.4f} nats -> exp(-20 C) = {np.exp(-n * C6):.4f}; "
    f"realised -(1/n) ln(Bayes error) = {-np.log(bayes6) / n:.4f}"
)
print(
    f"  Chernoff {C6:.4f} < Stein {D(P6, Q):.4f}: the minimax exponent is strictly below the "
    "one-sided one, and neither is an n = 20 error probability"
)


def binom_pmf_big(nn, p):
    """Binomial pmf in log space, so n = 2000 does not overflow comb()."""
    k = np.arange(nn + 1)
    lg = np.array([lgamma(nn + 1) - lgamma(kk + 1) - lgamma(nn - kk + 1) for kk in k])
    return np.exp(lg + k * np.log(p) + (nn - k) * np.log(1 - p))


for nn in (20, 100, 500, 2000):
    pm = binom_pmf_big(nn, pP6)
    qm = binom_pmf_big(nn, pQ)
    # log pmfP - log pmfQ, exact in log space
    llr = np.arange(nn + 1) * np.log(pP6 / pQ) + (nn - np.arange(nn + 1)) * np.log((1 - pP6) / (1 - pQ))
    ks = int(np.argmax(llr >= 0))
    be = 0.5 * (pm[:ks].sum() + qm[ks:].sum())
    print(f"    n = {nn:5d}: Bayes error = {be:.3e}, -(1/n) ln = {-np.log(be) / nn:.4f}  (C = {C6:.4f})")

# ---------------------------------------------------------------------------
# 26. A variance-coded latent: activity zero, mutual information positive
# ---------------------------------------------------------------------------
print("\n== 26. A variance-coded latent: posterior-mean activity A_j = 0, yet I(X;Z_j) > 0 ==")
# X a fair bit; q(z|X=0) = N(0, 0.25), q(z|X=1) = N(0, 1). Both conditionals have mean 0, so
# Var_x E[z|x] = 0 (Burda activity 0), but they differ, so |z| carries information about X.
# I(X;Z) = H(X) - H(X|Z) = the Jensen-Shannon-type value, by quadrature.
s0, s1 = 0.5, 1.0
z = np.linspace(-12, 12, 2_000_001)
f0 = np.exp(-(z**2) / (2 * s0**2)) / np.sqrt(2 * np.pi * s0**2)
f1 = np.exp(-(z**2) / (2 * s1**2)) / np.sqrt(2 * np.pi * s1**2)
m = 0.5 * (f0 + f1)
with np.errstate(divide="ignore", invalid="ignore"):
    i0 = np.trapezoid(np.where(f0 > 0, f0 * np.log2(f0 / m), 0.0), z)
    i1 = np.trapezoid(np.where(f1 > 0, f1 * np.log2(f1 / m), 0.0), z)
I_var = 0.5 * (i0 + i1)
kl0 = 0.5 * (s0**2 - 1 - np.log(s0**2))
kl1 = 0.5 * (s1**2 - 1 - np.log(s1**2))
print(
    f"  q(z|X=0) = N(0, {s0**2}), q(z|X=1) = N(0, {s1**2}), X fair: E[z|x] = 0 for both inputs "
    "-> activity A_j = Var_x E[z|x] = 0 (below the 0.01 threshold)"
)
print(
    f"  I(X;Z_j) = 0.5*[KL(f0||m) + KL(f1||m)] = {I_var:.4f} bits = {I_var * np.log(2):.4f} nats"
    "  (> 0: |z| is informative about X)"
)
print(
    f"  per-dimension KL to the prior N(0,1): {kl0:.4f} nats for X=0, {kl1:.4f} nats for X=1, "
    f"average {(0.5 * (kl0 + kl1)):.4f} nats"
)
print(
    f"  so the activity column calls this dimension inactive while it carries {I_var:.3f} bits; "
    "the earlier N(0,0.25)-for-every-x example is the converse (KL 0.318 nats, I = 0)"
)

# ---------------------------------------------------------------------------
# 27. Appendix C.2: three variances, three centred proxies; the mixture counter-example
# ---------------------------------------------------------------------------
print("\n== 27. Appendix C.2: three variances, three centred proxies; and the mixture counter-example ==")
thr = 2.5
n, k = 200, 300


def tail(v):
    """Two-sided tail P(|x| > thr) of a centred Gaussian of variance v."""
    return erfc(thr / sqrt(2 * v))


p0 = tail(1.0)
for name, v in (
    ("identity (1-1/n)^k", (1 - 1 / n) ** k),
    ("5,000-chain mean within-chain variance (notebook 01, cell 23)", 0.2030),
    ("200-chain pilot mean within-chain variance (notebook 01, cell 22, 0.4829^2)", 0.4829**2),
):
    print(
        f"  {name:>70}: v = {v:.4f} -> P(|x|>{thr}) = {tail(v):.3e}, "
        f"factor {p0 / tail(v):,.0f} below gen 0 ({p0:.3e})"
    )
print(
    "  (tails here come from the PRINTED 4-decimal variances; notebook 01 uses the unrounded means, "
    f"hence its 2.881e-08.) pilot/5,000-chain proxy ratio = {tail(0.4829**2) / tail(0.2030):.1f}; "
    f"median-chain drop 1.242e-2 / 3.336e-17 = 10^{np.log10(1.242e-2 / 3.336e-17):.2f} "
    "(more than fourteen orders, not fifteen)"
)


def two(m, s):
    """Two-sided tail P(|x| > thr) of N(m, s^2)."""
    return 0.5 * erfc((thr - m) / (s * sqrt(2))) + 0.5 * erfc((thr + m) / (s * sqrt(2)))


mix = 0.5 * two(-1, 0.1) + 0.5 * two(1, 0.1)
print(
    f"  counter-example: 0.5 N(-1,0.01) + 0.5 N(1,0.01) has variance 1.01 and P(|x|>{thr}) = {mix:.2e}; "
    f"N(0,1.01) has {tail(1.01):.3e} — the mixture is LIGHTER"
)
print(
    "  so 'a mixture of narrow Gaussians with scattered means is heavier-tailed than the Gaussian "
    "of the same variance' is not a theorem;"
)
print(
    "  notebook 01's mixture is heavier because its fitted means are not Gaussian-distributed "
    "and its within-chain variances are unequal"
)
