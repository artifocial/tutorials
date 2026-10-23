#!/usr/bin/env python3
"""Worked numbers for "The Probability Domain".

Companion script to the tutorial at
https://artifocial.com/blog/probability-domain-2026-oct-22 and to the three
notebooks in this folder. Every value printed below is one the article quotes
in a code block, and the article cites this file by section number (for
example "worked_numbers.py §21"), so the numbering here is fixed: 53 sections,
each opened by ``sec(N, title)``.

Pure NumPy, float64, CPU-only. Sections 1-31 draw in order from one generator
seeded 20261022; sections 32-53 each own a generator seeded ``2026_1016_NN``
(``§51`` draws nothing), so no section's numbers depend on a later one.
``rng_states.json`` records the generator state at the start of every section
(captured by ``capture_rng_states.py``) so the notebooks can run any section
on its own. Run with

    python worked_numbers.py > worked_numbers_output.txt

(about two and a quarter minutes) and diff against the committed output file.
Closed-form lines are also re-derived by ``test_worked_numbers.py``.
"""

import numpy as np, math, sys

sys.stdout.reconfigure(line_buffering=True)
from math import erf, sqrt, log, exp, pi

rng = np.random.default_rng(20261022)
np.set_printoptions(precision=4, suppress=True)


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def sec(n, title):
    print(f"\n=== §{n} {title} ===")


# Standard-error helpers: computed from samples the sections already draw — no random draws of their own.
def se_var(a):
    """SE of the sample variance of an iid sample: sqrt((m4 - s^4)/n), m4 the sample fourth central moment"""
    a = np.asarray(a, float)
    c = a - a.mean()
    s2 = np.mean(c * c)
    return sqrt(max(np.mean(c**4) - s2 * s2, 0.0) / len(a))


def se_sd(a):
    """SE of the sample SD: SE(var)/(2 sd)"""
    a = np.asarray(a, float)
    s = a.std()
    return se_var(a) / (2 * s) if s > 0 else 0.0


def se_median(a):
    """distribution-free SE of a sample median: half the width of the order-statistic ~68% interval, ranks n/2 -/+ sqrt(n)/2"""
    a = np.sort(np.asarray(a, float))
    n = len(a)
    if n < 4:
        return float('nan')
    lo, hi = max(int(math.floor(n / 2 - sqrt(n) / 2)), 0), min(int(math.ceil(n / 2 + sqrt(n) / 2)), n - 1)
    return (a[hi] - a[lo]) / 2


def jackknife_kurt(y):
    """delete-one jackknife SE of the sample excess kurtosis (sums updated in O(n); no resampling, no draws)"""
    y = np.asarray(y, float)
    n = len(y)
    m = n - 1
    S = [np.sum(y**k) for k in (1, 2, 3, 4)]
    r1 = (S[0] - y) / m
    r2 = (S[1] - y**2) / m
    r3 = (S[2] - y**3) / m
    r4 = (S[3] - y**4) / m
    mu2 = r2 - r1**2
    mu4 = r4 - 4 * r3 * r1 + 6 * r2 * r1**2 - 3 * r1**4
    ki = mu4 / mu2**2 - 3
    return sqrt((n - 1) / n * np.sum((ki - ki.mean()) ** 2))


# ----------------------------------------------------------------------------------------------
sec(1, "Bertrand's chord: three 'uniform' measures, three answers (unit circle, side = sqrt(3))")
N = 1_000_000
# (a) random endpoints
t1, t2 = rng.uniform(0, 2 * pi, N), rng.uniform(0, 2 * pi, N)
L = 2 * np.abs(np.sin((t1 - t2) / 2))
pa = np.mean(L > sqrt(3))
print(f"(a) two uniform endpoints        P(L > sqrt3) = {pa:.4f}   (theory 1/3 = {1/3:.4f})")
# (b) random radius point: distance from centre uniform on [0,1]
d = rng.uniform(0, 1, N)
L = 2 * np.sqrt(1 - d**2)
pb = np.mean(L > sqrt(3))
print(f"(b) uniform distance to centre   P(L > sqrt3) = {pb:.4f}   (theory 1/2 = {0.5:.4f})")
# (c) random midpoint uniform in the disc
r = np.sqrt(rng.uniform(0, 1, N))
L = 2 * np.sqrt(1 - r**2)
pc = np.mean(L > sqrt(3))
print(f"(c) uniform midpoint in the disc P(L > sqrt3) = {pc:.4f}   (theory 1/4 = {0.25:.4f})")
print(
    f"(SE of the three lines above: ± {sqrt(pa*(1-pa)/N):.4f}, ± {sqrt(pb*(1-pb)/N):.4f}, ± {sqrt(pc*(1-pc)/N):.4f} — binomial sqrt(p(1-p)/N) at N = {N:,}; the largest |measured - theory| is {max(abs(pa-1/3)/sqrt(pa*(1-pa)/N), abs(pb-0.5)/sqrt(pb*(1-pb)/N), abs(pc-0.25)/sqrt(pc*(1-pc)/N)):.1f} SE)"
)

# ----------------------------------------------------------------------------------------------
sec(2, "Monty Hall as a conditioning statement")
N = 1_000_000
car = rng.integers(0, 3, N)
pick = rng.integers(0, 3, N)
ps = np.mean(car == pick)
print(f"stay wins   = {ps:.4f}   switch wins = {1-ps:.4f}   (theory 1/3, 2/3)")
print(
    f"(SE of the line above: ± {sqrt(ps*(1-ps)/N):.4f} for either frequency — binomial sqrt(p(1-p)/N) at N = {N:,})"
)

# ----------------------------------------------------------------------------------------------
sec(
    3,
    "Vitali-free but still sharp: a dense set of measure zero (rationals in [0,1]) — the Lebesgue integral of 1_Q",
)
# Riemann: upper sum 1, lower sum 0 on every partition; Lebesgue: measure of Q is 0.
print("Riemann upper sum = 1, lower sum = 0 on every partition -> not Riemann integrable")
print("Lebesgue: lambda(Q cap [0,1]) = 0  ->  integral of 1_Q = 0  (a countable union of points)")
print(
    "Lebesgue builds the integral from the RANGE (level sets), Riemann from the DOMAIN (intervals): that is the whole difference"
)

# ----------------------------------------------------------------------------------------------
sec(
    4,
    "E2 — two-class Gaussian mixture, means ±mu e1, identity covariance, equal priors: Bayes risk = Phi(-mu)",
)
for mu in (0.5, 1.0, 1.5, 2.0):
    print(f"mu = {mu:.1f}:  Bayes risk Phi(-mu) = {Phi(-mu):.4f}")
mu, dsz, n = 1.0, 20, 200_000
y = rng.integers(0, 2, n) * 2 - 1
X = rng.standard_normal((n, dsz))
X[:, 0] += mu * y
print(f"Monte Carlo Bayes classifier error at mu=1, d={dsz}, n={n:,}: {np.mean(np.sign(X[:,0]) != y):.4f}")


# logistic regression by Newton on 2,000 points, 20 dims; then its test error vs Bayes
def fit_logreg(X, y, iters=25, l2=1e-3):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        z = (y + 1) / 2
        g = X.T @ (p - z) / len(y) + l2 * w
        H = (X.T * (p * (1 - p))) @ X / len(y) + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


for ntr in (50, 200, 2000, 20000):
    Xtr, ytr = X[:ntr], y[:ntr]
    w = fit_logreg(Xtr, ytr)
    err = np.mean(np.sign(X[20000:] @ w) != y[20000:])
    print(
        f"logistic regression, n_train={ntr:>6}, d=20: held-out error {err:.4f}  (excess over Bayes {err-Phi(-mu):+.4f})"
    )

# ----------------------------------------------------------------------------------------------
sec(
    5, "Conditional expectation is an L2 projection: E[(Y - E[Y|X]) g(X)] = 0 for every g, and MSE decomposes"
)
n = 1_000_000
Xc = rng.standard_normal(n)
Y = Xc**2 + np.sin(3 * Xc) + 0.5 * rng.standard_normal(n)
m = Xc**2 + np.sin(3 * Xc)  # E[Y|X] exactly
resid = Y - m
for name, g in (("X", Xc), ("X^2", Xc**2), ("sin 3X", np.sin(3 * Xc)), ("1[X>0]", (Xc > 0).astype(float))):
    print(f"E[(Y - E[Y|X]) * {name:7s}] = {np.mean(resid*g):+.4f}")
f_lin = np.polyfit(Xc, Y, 1)
pred = np.polyval(f_lin, Xc)
print(f"MSE of best LINEAR predictor      = {np.mean((Y-pred)**2):.4f}")
print(f"MSE of E[Y|X]                     = {np.mean((Y-m)**2):.4f}   (= noise variance 0.25)")
print(
    f"E[(E[Y|X] - linear)^2]            = {np.mean((m-pred)**2):.4f}   -> sum {np.mean((Y-m)**2)+np.mean((m-pred)**2):.4f} (Pythagoras)"
)

# ----------------------------------------------------------------------------------------------
sec(6, "Borel–Kolmogorov: conditioning a uniform point on the sphere on a great circle, two ways")
n = 4_000_000
v = rng.standard_normal((n, 3))
v /= np.linalg.norm(v, axis=1, keepdims=True)
lat = np.arcsin(v[:, 2])
lon = np.arctan2(v[:, 1], v[:, 0])
eps = 0.02
# (i) the great circle as the equator: band |lat| < eps -> distribution of longitude
sel = np.abs(lat) < eps
h, _ = np.histogram(lon[sel], bins=6, range=(-pi, pi))
print(f"(i) equator via latitude band:  longitude histogram (6 bins, normalised) = {h/h.sum()}")
# (ii) the same great circle as a meridian (rotate axes): band |lon| < eps -> distribution of latitude
sel = np.abs(lon) < eps
h, edges = np.histogram(lat[sel], bins=6, range=(-pi / 2, pi / 2))
centres = (edges[:-1] + edges[1:]) / 2
print(f"(ii) meridian via longitude band: latitude histogram (6 bins, normalised) = {h/h.sum()}")
print(
    f"     cos(latitude) at bin centres, normalised                             = {np.cos(centres)/np.cos(centres).sum()}"
)
print(
    "Same curve on the sphere, two conditional densities: uniform in (i), proportional to cos(lat) in (ii)."
)

# ----------------------------------------------------------------------------------------------
sec(7, "Modes of convergence: the typewriter sequence and escaping mass")
# typewriter: indicator of [k/2^m, (k+1)/2^m) enumerated; X_n(u) -> 0 in probability and in L1, but limsup = 1 for every u.
u = rng.uniform(0, 1, 100_000)
hits = np.zeros_like(u)
idx = 0
pm = []
for m in range(0, 12):
    for k in range(2**m):
        Xn = ((u >= k / 2**m) & (u < (k + 1) / 2**m)).astype(float)
        pm.append((idx, Xn.mean()))
        hits += Xn
        idx += 1
print(
    f"typewriter: P(X_n = 1) at n = 1, 3, 7, 1023, 4095: {[round(float(pm[i][1]),4) for i in (0,2,6,1022,4094)]}  -> 0"
)
print(
    f"typewriter: fraction of sample points u hit at least once per dyadic level (12 levels): {np.mean(hits >= 12):.4f}  -> limsup X_n = 1 a.s."
)
print(
    f"(SE of the P(X_n = 1) line: each entry is a frequency over the {len(u):,} points u, ± sqrt(p(1-p)/{len(u):,}) = "
    + ", ".join(f"{sqrt(pm[i][1]*(1-pm[i][1])/len(u)):.4f}" for i in (0, 2, 6, 1022, 4094))
    + "; the 1.0000 is exact — every u is hit at every level)"
)
# escaping mass: X_n = n * 1[U < 1/n]
esc7 = []
for nn in (10, 100, 1000, 10000):
    Xn = nn * (u < 1 / nn)
    print(
        f"escaping mass n={nn:>5}: P(X_n != 0) = {np.mean(Xn != 0):.4f}, E[X_n] = {Xn.mean():.3f}  (-> 0 in probability, E stays 1: no L1 convergence)"
    )
    esc7.append((nn, np.mean(Xn != 0)))
print(
    "(SE of the four lines above, from the same "
    + f"{len(u):,}"
    + " points: P(X_n != 0) ± "
    + ", ".join(f"{sqrt(q*(1-q)/len(u)):.4f}" for _, q in esc7)
    + ";  E[X_n] = n · P(X_n != 0) here, so ± "
    + ", ".join(f"{nn*sqrt(q*(1-q)/len(u)):.3f}" for nn, q in esc7)
    + " — the E[X_n] column is a mean of 100,000 terms of which only n^-1 of them are nonzero, which is why it wanders while its expectation is exactly 1)"
)

# ----------------------------------------------------------------------------------------------
sec(8, "SLLN vs LIL: how far a running mean of fair coins strays, against sqrt(2 log log n / n) * sigma")
n = 2_000_000
flips = rng.integers(0, 2, n)
S = np.cumsum(flips - 0.5)
ns = np.arange(1, n + 1)
mask = ns >= 100
env = 0.5 * np.sqrt(2 * np.log(np.log(ns[mask])) / ns[mask])  # sigma = 1/2 for a fair coin
dev = np.abs(S[mask]) / ns[mask]
print(f"fraction of n in [100, 2e6] with |S_n/n| > LIL envelope: {np.mean(dev > env):.5f}")
print(f"fraction of n in [100, 2e6] with |S_n/n| > 0.5*envelope : {np.mean(dev > 0.5*env):.4f}")
print(
    f"fraction of n in [100, 2e6] with |S_n/n| > 1/sqrt(n)*0.5 (a CLT-width band): {np.mean(dev > 0.5/np.sqrt(ns[mask])):.4f}"
)
for nn in (100, 1_000, 10_000, 100_000, 1_000_000):
    print(
        f"n={nn:>9}: envelope = {0.5*sqrt(2*log(log(nn))/nn):.5f}   1/sqrt(n) CLT scale = {0.5/sqrt(nn):.5f}   ratio = {sqrt(2*log(log(nn))):.3f}"
    )

# ----------------------------------------------------------------------------------------------
sec(
    9,
    "Berry–Esseen for Bernoulli(p) sums: the bound 0.4748 * rho / (sigma^3 sqrt n) against the measured sup distance",
)
for p, nn in ((0.5, 30), (0.1, 30), (0.1, 300), (0.01, 300)):
    sig = sqrt(p * (1 - p))
    rho = p * (1 - p) * ((1 - p) ** 2 + p**2)  # E|X-p|^3
    bound = 0.4748 * rho / (sig**3 * sqrt(nn))
    # exact binomial CDF of standardised sum vs Phi
    from math import comb

    ks = np.arange(nn + 1)
    pmf = np.array([comb(nn, k) * p**k * (1 - p) ** (nn - k) for k in ks])
    cdf = np.cumsum(pmf)
    z = (ks - nn * p) / (sig * sqrt(nn))
    supd = max(
        max(abs(cdf[i] - Phi(z[i])) for i in range(nn + 1)),
        max(abs(cdf[i - 1] - Phi(z[i])) for i in range(1, nn + 1)),
    )
    print(
        f"p={p:<5} n={nn:>4}: measured sup|F_n - Phi| = {supd:.4f}   Berry–Esseen bound = {bound:.4f}   (rho/sigma^3 = {rho/sig**3:.3f})"
    )

# ----------------------------------------------------------------------------------------------
sec(
    10,
    "The CLT is a Fourier theorem: empirical characteristic function of the standardised sum vs exp(-t^2/2)",
)
nn, reps = 50, 200_000
Xs = rng.exponential(1.0, (reps, nn))
Z = (Xs.sum(1) - nn) / sqrt(nn)
se10 = []
for t in (0.5, 1.0, 2.0):
    ecf = np.mean(np.exp(1j * t * Z))
    print(
        f"t={t}: |empirical cf| = {abs(ecf):.4f}, Re = {ecf.real:+.4f}, Im = {ecf.imag:+.4f};  exp(-t^2/2) = {exp(-t*t/2):.4f}"
    )
    se10.append((t, np.std(np.cos(t * Z)) / sqrt(reps), np.std(np.sin(t * Z)) / sqrt(reps)))
print(
    "(SE of the three lines above, sd/sqrt("
    + f"{reps:,}"
    + ") of cos(tZ) and sin(tZ): "
    + ";  ".join(f"t={t}: Re ± {a:.4f}, Im ± {b:.4f}" for t, a, b in se10)
    + " — the Im parts sit "
    + ", ".join(f"{abs(np.mean(np.sin(t*Z)))/b:.1f}" for t, a, b in se10)
    + " SE from 0: at n = 50 the exponential sum is still skewed, which the next line measures)"
)
print(
    f"skewness of standardised sum (n=50 exponentials): {np.mean(Z**3):.4f}  (theory gamma1/sqrt(n) = {2/sqrt(nn):.4f})"
)
print(f"(SE of the line above: ± {np.std(Z**3)/sqrt(reps):.4f}, sd(Z^3)/sqrt({reps:,}))")

# ----------------------------------------------------------------------------------------------
sec(
    11,
    "Where the CLT fails: Cauchy means never settle; Pareto(alpha=1.5) has a mean but its standardised sum is not Gaussian",
)
c = rng.standard_cauchy(1_000_000)
print(
    "Cauchy running mean at n = 10^2..10^6:",
    [round(float(c[:k].mean()), 3) for k in (100, 1000, 10000, 100000, 1000000)],
)
alpha = 1.5
par = (rng.uniform(0, 1, (20_000, 1000))) ** (
    -1 / alpha
)  # Pareto with x_min=1, mean alpha/(alpha-1)=3, infinite variance
s = par.sum(1) / 1000
print(
    f"Pareto(1.5): sample means of blocks of 1000: mean {s.mean():.3f} (theory 3.000), sd {s.std():.3f}, max {s.max():.1f}"
)
print(
    f"(no standard error is printed for the line above: a block mean of Pareto(1.5) draws has infinite variance, so sd/sqrt({len(s):,}) = {s.std()/sqrt(len(s)):.4f} is a statistic of these {len(s):,} blocks with no population value behind it — §53 shows how far it moves between batches)"
)
zz = (s - s.mean()) / s.std()
print(
    f"excess kurtosis of the block means: {np.mean(zz**4)-3:.2f}  (a Gaussian limit would give ~0; it is diverging, the limit is 1.5-stable)"
)

# ----------------------------------------------------------------------------------------------
sec(12, "E3 — standard Gaussian vectors in d dimensions: norms, cosines, attention logits")
for d in (16, 256, 4096):
    G = rng.standard_normal((2000, d))
    nrm = np.linalg.norm(G, axis=1)
    cos = (G[:1000] * G[1000:]).sum(1) / (nrm[:1000] * nrm[1000:])
    dots = (G[:1000] * G[1000:]).sum(1)
    print(
        f"d={d:>5}: ||g|| mean {nrm.mean():8.3f} sd {nrm.std():.3f} (sqrt d = {sqrt(d):.3f}, sd -> 1/sqrt2 = 0.707);  "
        f"cosine sd {cos.std():.4f} (1/sqrt d = {1/sqrt(d):.4f});  dot-product sd {dots.std():.2f} (sqrt d = {sqrt(d):.2f})"
    )
    print(
        f"    (SE of the line above: ||g|| mean ± {nrm.std(ddof=1)/sqrt(len(nrm)):.3f}, its sd ± {se_sd(nrm):.3f};  cosine sd ± {se_sd(cos):.4f};  dot-product sd ± {se_sd(dots):.2f}   — from the same {len(nrm):,} vectors / {len(cos):,} pairs, SE(sd) = sqrt((m4 - s^4)/n)/(2 s))"
    )

# ----------------------------------------------------------------------------------------------
sec(13, "Johnson–Lindenstrauss: 1,000 points in d=4096 projected to k dims by a Gaussian matrix / sqrt k")


def pdist(A):
    sq = (A * A).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2 * A @ A.T
    return np.sqrt(np.maximum(D2, 0))


P = rng.standard_normal((1000, 4096))
D0 = pdist(P)
iu = np.triu_indices(1000, 1)
for k in (64, 256, 1024):
    R = rng.standard_normal((4096, k)) / sqrt(k)
    Q = P @ R
    D1 = pdist(Q)
    ratio = D1[iu] / D0[iu]
    print(
        f"k={k:>5}: distance ratio mean {ratio.mean():.4f}, sd {ratio.std():.4f}, max distortion {np.abs(ratio-1).max():.4f}   (theory sd ~ 1/sqrt(2k) = {1/sqrt(2*k):.4f})"
    )
    M13 = np.zeros((1000, 1000))
    M13[iu] = ratio
    M13 = M13 + M13.T  # delete-one-POINT jackknife: the 499,500 pairs share points
    S1, S2, T1, T2, Np = M13.sum(1), (M13 * M13).sum(1), ratio.sum(), (ratio * ratio).sum(), len(ratio)
    m_i = (T1 - S1) / (Np - 999)
    sd_i = np.sqrt(np.maximum((T2 - S2) / (Np - 999) - m_i**2, 0))
    jk13 = lambda th: sqrt(999 / 1000 * np.sum((th - th.mean()) ** 2))
    print(
        f"    (SE of the line above: ratio mean ± {jk13(m_i):.4f}, sd ± {jk13(sd_i):.4f}   — delete-one-point jackknife over the 1,000 points, since the 499,500 pairs share points; the max distortion is an extreme value, no SE)"
    )

# ----------------------------------------------------------------------------------------------
sec(
    14,
    "Covariance estimation: ||Sigma_hat - I||_op for n = d/4, d, 4d, 16d, 64d at d=256 (Vershynin Thm 4.7.1 scale sqrt(d/n))",
)
d = 256
for f in (0.25, 1, 4, 16, 64):
    n = int(f * d)
    G = rng.standard_normal((n, d))
    Sh = G.T @ G / n
    op = np.max(np.abs(np.linalg.eigvalsh(Sh) - 1))
    print(
        f"n = {f:>5}d = {n:>6}: ||Sigma_hat - I||_op = {op:.3f}    sqrt(d/n) = {sqrt(d/n):.3f}   (2*sqrt(d/n)+d/n = {2*sqrt(d/n)+d/n:.3f})"
    )

# ----------------------------------------------------------------------------------------------
sec(
    15,
    "Eval-set error bars: Hoeffding n for a half-width eps at confidence 1-delta, and a binomial check at n=1000, acc=0.80",
)
for eps in (0.05, 0.02, 0.01, 0.005):
    for delta in (0.05,):
        print(
            f"eps={eps:<6} delta={delta}: Hoeffding n >= ln(2/delta)/(2 eps^2) = {math.ceil(log(2/delta)/(2*eps**2)):>8,}"
        )
acc, n = 0.80, 1000
se = sqrt(acc * (1 - acc) / n)
print(
    f"acc=0.80, n=1000: binomial sd = {se:.4f}; 95% normal interval half-width = {1.96*se:.4f}; Hoeffding half-width sqrt(ln(2/0.05)/(2n)) = {sqrt(log(40)/(2*n)):.4f}"
)
# two models, same test set, paired: variance depends on disagreement rate
dis = 0.10
p_win = 0.55  # of the disagreements, model A wins 55%
diff_sd = sqrt(
    dis * (1 - (2 * p_win - 1) ** 2 * dis) / n
)  # approx sd of (accA - accB) via paired disagreements
print(
    f"paired comparison, n=1000, disagreement rate 10%, A wins 55% of them: gap = {dis*(2*p_win-1):.4f}, sd ~ {diff_sd:.4f}  -> {dis*(2*p_win-1)/diff_sd:.2f} sd"
)

# ----------------------------------------------------------------------------------------------
sec(
    16,
    "Finite hypothesis class bound: sqrt((ln|H| + ln(2/delta)) / (2n)) and the DKW / Glivenko–Cantelli measurement",
)
for H, n in ((10**6, 10**4), (10**6, 10**5), (2 ** (32 * 1000), 10**5)):
    print(
        f"|H|={H if H < 1e12 else '2^32000':>8}, n={n:>7}: uniform deviation bound {sqrt((log(H) + log(40))/(2*n)):.4f}"
    )
for n in (100, 1000, 10000, 100000):
    xs = np.sort(rng.standard_normal(n))
    Fn = np.arange(1, n + 1) / n
    F = np.array([Phi(x) for x in xs])
    sup = max(np.max(np.abs(Fn - F)), np.max(np.abs(Fn - 1 / n - F)))
    print(
        f"n={n:>7}: sup|F_n - F| = {sup:.4f}    DKW 95% bound sqrt(ln(2/0.05)/(2n)) = {sqrt(log(40)/(2*n)):.4f}"
    )

# ----------------------------------------------------------------------------------------------
sec(17, "Random Fourier features: Hoeffding on Bochner — RBF kernel approximation error vs D")


def rbf(x, y, s=1.0):
    return np.exp(-np.sum((x - y) ** 2) / (2 * s * s))


pts = rng.standard_normal((200, 8))
Ktrue = np.array([[rbf(a, b) for b in pts] for a in pts])
for D in (100, 1000, 10000):
    W = rng.standard_normal((8, D))
    b = rng.uniform(0, 2 * pi, D)
    Z = sqrt(2 / D) * np.cos(pts @ W + b)
    Kapp = Z @ Z.T
    print(
        f"D={D:>6}: max |K_approx - K| = {np.abs(Kapp-Ktrue).max():.4f}, rms = {np.sqrt(np.mean((Kapp-Ktrue)**2)):.4f}   (1/sqrt D = {1/sqrt(D):.4f})"
    )
    G17, gs17 = 10, D // 10  # delete-one-GROUP jackknife over the D iid features
    th17 = np.array(
        [
            (
                (lambda Kg: (np.abs(Kg - Ktrue).max(), np.sqrt(np.mean((Kg - Ktrue) ** 2))))(
                    (Kapp - Z[:, g * gs17 : (g + 1) * gs17] @ Z[:, g * gs17 : (g + 1) * gs17].T)
                    * D
                    / (D - gs17)
                )
            )
            for g in range(G17)
        ]
    )
    jk17 = lambda t: sqrt((G17 - 1) / G17 * np.sum((t - t.mean()) ** 2))
    print(
        f"    (SE of the line above: max ± {jk17(th17[:, 0]):.4f}, rms ± {jk17(th17[:, 1]):.4f}   — delete-one-group jackknife over {G17} groups of {gs17:,} features; the {D:,} features are the iid draws, the 200 points are fixed)"
    )

# ----------------------------------------------------------------------------------------------
sec(
    18,
    "Gradient of an expectation: score-function vs pathwise estimators of d/dmu E[X^2], X ~ N(mu, 1), true value 2 mu",
)
mu, n = 1.5, 1000
reps = 2000
sf, pw = [], []
for _ in range(reps):
    eps = rng.standard_normal(n)
    x = mu + eps
    sf.append(np.mean(x**2 * (x - mu)))  # score function: f(x) * d/dmu log p(x)
    pw.append(np.mean(2 * x))  # pathwise: d/dmu (mu + eps)^2 = 2(mu+eps)
sf, pw = np.array(sf), np.array(pw)
print(
    f"true 2mu = {2*mu}; score-function: mean {sf.mean():.4f}, sd {sf.std():.4f};  pathwise: mean {pw.mean():.4f}, sd {pw.std():.4f};  variance ratio {sf.var()/pw.var():.1f}"
)
d18s, d18p = (sf - sf.mean()) ** 2, (
    pw - pw.mean()
) ** 2  # per-replication contributions to the two variances
R18 = sf.var() / pw.var()
cov18 = np.mean(
    (d18s - d18s.mean()) * (d18p - d18p.mean())
)  # the two estimators share each replication's draws, so their variance estimates covary
se_R18 = R18 * sqrt(
    (d18s.var() / sf.var() ** 2 + d18p.var() / pw.var() ** 2 - 2 * cov18 / (sf.var() * pw.var())) / reps
)  # joint delta method for a ratio of two correlated variance estimates
se_R18_naive = R18 * sqrt(
    (d18s.var() / sf.var() ** 2 + d18p.var() / pw.var() ** 2) / reps
)  # the same without the covariance term
print(
    f"(SE of the line above: the two means ± {sf.std()/sqrt(reps):.4f} and ± {pw.std()/sqrt(reps):.4f} (sd/sqrt({reps:,}));  the sds ± {se_sd(sf):.4f} and ± {se_sd(pw):.4f};  the variance ratio ± {se_R18:.1f} by the joint delta method — the two estimators use the same draws in each replication (correlation {np.corrcoef(sf, pw)[0, 1]:.2f}; of their squared deviations {np.corrcoef(d18s, d18p)[0, 1]:.2f}), so the ratio's bar carries their covariance; ignoring it would print ± {se_R18_naive:.1f})"
)

# ----------------------------------------------------------------------------------------------
sec(19, "Jensen's gap and the ELBO: E[log X] vs log E[X] for X lognormal(0, s^2)")
for s in (0.1, 0.5, 1.0, 2.0):
    print(f"s={s}: log E[X] = s^2/2 = {s*s/2:.4f},  E[log X] = 0  ->  gap {s*s/2:.4f} nats")

# ----------------------------------------------------------------------------------------------
sec(20, "Divergences under disjoint support: U[0,1] vs U[theta, 1+theta]")
for th in (0.0, 0.01, 0.1, 0.5, 1.0, 2.0):
    ov = max(0.0, 1 - th)
    tv = 1 - ov
    js = log(2) if th >= 1 else None
    kl = 0.0 if th == 0 else float('inf')
    print(
        f"theta={th:<4}: KL = {kl},  TV = {tv:.2f},  W1 = {th:.2f}"
        + (f",  JS = ln2 = {js:.4f} (saturated)" if js else "")
    )
print(
    "Pinsker check on Bernoulli(0.5) vs Bernoulli(0.6): TV = 0.1, KL(nats) = %.5f, sqrt(KL/2) = %.4f >= TV"
    % (0.5 * log(0.5 / 0.6) + 0.5 * log(0.5 / 0.4), sqrt((0.5 * log(0.5 / 0.6) + 0.5 * log(0.5 / 0.4)) / 2))
)

# ----------------------------------------------------------------------------------------------
sec(
    21,
    "Robbins–Monro SGD on E2 (logistic loss, d=20, mu=1): step 1/t vs constant 0.1, excess risk over Bayes",
)
d, mu = 20, 1.0


def stream(n):
    yy = rng.integers(0, 2, n) * 2 - 1
    XX = rng.standard_normal((n, d))
    XX[:, 0] += mu * yy
    return XX, yy


Xte, yte = stream(50_000)


def run(step_fn, T=20_000):
    w = np.zeros(d)
    out = []
    XX, yy = stream(T)
    for t in range(T):
        x, yv = XX[t], yy[t]
        g = -yv * x / (1 + exp(yv * (x @ w)))
        w -= step_fn(t + 1) * g
        if (t + 1) in (100, 1000, 10000, 20000):
            out.append(((t + 1), np.mean(np.sign(Xte @ w) != yte)))
    return out


traj21 = (
    {}
)  # the four trajectories are kept so that the notebook's figure plots THESE runs, not a fresh replicate
for name, fn in (
    ("1/t", lambda t: 1.0 / t),
    ("1/sqrt t", lambda t: 1.0 / sqrt(t)),
    ("const 0.1", lambda t: 0.1),
    ("const 1.0", lambda t: 1.0),
):
    res = run(fn)
    traj21[name] = res
    print(
        f"step {name:>9}: "
        + "  ".join(f"t={t:>5} err={e:.4f}" for t, e in res)
        + f"   (Bayes {Phi(-mu):.4f})"
    )

# ----------------------------------------------------------------------------------------------
sec(
    22,
    "Optional stopping: peeking at a z-test every 100 samples up to 10,000 vs a test martingale with Ville's inequality (alpha = 0.05)",
)
reps, N_, look = 4000, 10_000, 100
fp_peek = 0
fp_ville = 0
fp_fixed = 0
lam = 0.5
for _ in range(reps):
    x = rng.integers(0, 2, N_).astype(float)  # null: fair coin, mean 0.5
    S = np.cumsum(x - 0.5)
    ns = np.arange(1, N_ + 1)
    z = S[look - 1 :: look] / (0.5 * np.sqrt(ns[look - 1 :: look]))
    if np.any(np.abs(z) > 1.96):
        fp_peek += 1
    if abs(z[-1]) > 1.96:
        fp_fixed += 1
    # test martingale: product of (1 + lam*(2x-1)) with lam in (-1,1) -> nonneg martingale with E=1 under null; two-sided via mixture of +lam/-lam
    E_plus = np.cumprod(1 + lam * (2 * x - 1))
    E_minus = np.cumprod(1 - lam * (2 * x - 1))
    E = 0.5 * (E_plus + E_minus)
    if np.any(E >= 1 / 0.05):
        fp_ville += 1
print(f"fixed-n z-test at n=10,000:            false positive rate {fp_fixed/reps:.4f}")
print(f"z-test peeked every 100 (100 looks):   false positive rate {fp_peek/reps:.4f}")
print(
    f"test martingale, stop when E_n >= 20:  false positive rate {fp_ville/reps:.4f}   (Ville: <= 0.05 at EVERY stopping rule)"
)

# ----------------------------------------------------------------------------------------------
sec(
    23,
    "E4 — AR(1) with phi = 0.9, unit stationary variance: ESS fraction (1-phi)/(1+phi), measured variance of the time average",
)
phi, n, reps = 0.9, 10_000, 2000
e = rng.standard_normal((n, reps)) * sqrt(1 - phi**2)
x = rng.standard_normal(reps)
acc = x.copy()
for t in range(1, n):
    x = phi * x + e[t]
    acc += x
means = acc / n
print(
    f"var(time average) * n = {means.var()*n:.3f};  theory (1+phi)/(1-phi) = {(1+phi)/(1-phi):.3f};  ESS = n*(1-phi)/(1+phi) = {n*(1-phi)/(1+phi):.0f} of {n}"
)
print(
    f"    (SE of the line above: var(time average) * n ± {se_var(means)*n:.3f}   — sqrt((m4 - s^4)/reps) x n from the same {reps:,} independent replicated time averages)"
)

# ----------------------------------------------------------------------------------------------
sec(
    24,
    "Metropolis–Hastings for N(0,1) with a Gaussian random walk: acceptance and integrated autocorrelation vs step size",
)


def mh(step, n=200_000):
    x = 0.0
    xs = np.empty(n)
    acc = 0
    u = np.log(rng.uniform(0, 1, n))
    prop = rng.standard_normal(n) * step
    for t in range(n):
        y = x + prop[t]
        if u[t] < -0.5 * (y * y - x * x):
            x = y
            acc += 1
        xs[t] = x
    return xs, acc / n


for step in (0.1, 0.5, 2.4, 10.0):
    xs, a = mh(step)
    xc = xs - xs.mean()
    var = xc.var()
    rho = [np.dot(xc[:-k], xc[k:]) / ((len(xc) - k) * var) for k in range(1, 400)]
    tau = 1 + 2 * sum(r for r in rho if r > 0.01)
    print(
        f"step={step:>4}: acceptance {a:.3f}, integrated autocorrelation time ~ {tau:6.1f}, ESS per 10^4 draws ~ {10_000/tau:7.0f}, sample var {var:.3f}"
    )

# ----------------------------------------------------------------------------------------------
sec(
    25,
    "Unadjusted Langevin for N(0,1): x <- x - h x + sqrt(2h) xi has stationary variance 1/(1 - h/2), not 1",
)
for h in (0.01, 0.1, 0.5, 1.0):
    n = 2_000_000 if h < 0.05 else 400_000
    x = 0.0
    xs = np.empty(n)
    xi = rng.standard_normal(n)
    for t in range(n):
        x = x - h * x + sqrt(2 * h) * xi[t]
        xs[t] = x
    print(
        f"h={h:<5}: measured stationary var {xs[int(20/h):].var():.4f}   theory 1/(1-h/2) = {1/(1-h/2):.4f}"
    )
    ys25 = xs[int(20 / h) :]
    nb25 = 100
    yb25 = ys25[: len(ys25) // nb25 * nb25].reshape(nb25, -1)
    print(
        f"    (SE of the line above: measured stationary var ± {yb25.var(axis=1).std(ddof=1)/sqrt(nb25):.4f}   — batch means, {nb25} batches of {len(ys25)//nb25:,} steps of the same chain;  AR(1) asymptotic sqrt((2/n)(1 + rho^2)/(1 - rho^2)) x var with rho = 1 - h: {sqrt(2/len(ys25)*(1 + (1-h)**2)/(1 - (1-h)**2))*ys25.var():.4f})"
    )

# ----------------------------------------------------------------------------------------------
sec(
    26,
    "Diffusion as SDE reversal: OU forward on a two-point mixture, reverse with the EXACT score recovers the modes",
)


# forward: dx = -x dt + sqrt2 dW, T = 3, from x0 in {-2 (prob 0.3), +2 (prob 0.7)}; marginal at t is a mixture of N(m e^{-t}, 1-e^{-2t})
def score(x, t):
    m = 2 * exp(-t)
    v = 1 - exp(-2 * t)
    w1, w2 = 0.3, 0.7
    p1 = w1 * np.exp(-((x + m) ** 2) / (2 * v))
    p2 = w2 * np.exp(-((x - m) ** 2) / (2 * v))
    return (-(x + m) / v * p1 - (x - m) / v * p2) / (p1 + p2)


T, K, n = 3.0, 3000, 100_000
dt_ = T / K
x = (
    rng.standard_normal(n) * sqrt(1 - exp(-2 * T)) + 0
)  # start near the stationary N(0,1): mixture at T has means ±2e^-3 ≈ ±0.1
for k in range(K):
    t = T - k * dt_
    # reverse SDE (Anderson): dx = [-x - 2*score] (-dt) ... in reverse-time form: x <- x + (x + 2 score(x,t)) dt + sqrt(2 dt) xi
    x = x + (x + 2 * score(x, t)) * dt_ + sqrt(2 * dt_) * rng.standard_normal(n)
pneg = np.mean(x < 0)
npos = int(np.sum(x > 0))
print(
    f"reverse-time OU with exact score, T=3, {K} steps: mass at x<0 = {np.mean(x<0):.4f} (target 0.300), mean of x>0 group = {x[x>0].mean():.3f} (target 2.0), sd {x[x>0].std():.3f} (noise of the last Euler step, sqrt(2 dt) = {sqrt(2*dt_):.3f}; exact target 0 at t = 0)"
)
print(
    f"(SE of the line above, from the same {n:,} paths: mass at x<0 ± {sqrt(pneg*(1-pneg)/n):.4f} (binomial);  mean of the x>0 group ± {x[x>0].std()/sqrt(npos):.4f} (sd/sqrt({npos:,}));  the mass is {abs(pneg-0.3)/sqrt(pneg*(1-pneg)/n):.1f} SE from 0.300 and the mean {abs(x[x>0].mean()-2.0)/(x[x>0].std()/sqrt(npos)):.1f} SE from 2.0)"
)

# ----------------------------------------------------------------------------------------------
sec(27, "Minimax: Le Cam two-point for a Gaussian mean, and the nonparametric density rate n^(-2s/(2s+d))")
sigma, n = 1.0, 100
# two-point: theta0 = 0, theta1 = delta; KL(P0^n || P1^n) = n delta^2 / (2 sigma^2); choose delta so KL = 1/2 -> delta = sigma/sqrt(n)
delta = sigma / sqrt(n)
print(
    f"n={n}, sigma=1: Le Cam separation delta = sigma/sqrt n = {delta:.3f}; Le Cam lower bound on minimax MSE >= (delta/2)^2 * (1 - sqrt(KL/2)) / 2 = {(delta/2)**2*(1-sqrt(0.5/2))/2:.6f};  sample-mean risk sigma^2/n = {sigma**2/n:.5f}"
)
print("nonparametric density estimation, smoothness s=2, squared-L2 minimax rate n^(-2s/(2s+d)):")
for d in (1, 2, 10, 100, 1000):
    for n in (10**4, 10**6, 10**9):
        print(f"  d={d:>5} n=10^{int(round(math.log10(n)))}: rate = {n**(-4/(4+d)):.2e}", end="")
    print()
print("  n needed for rate 0.01 at s=2: " + ", ".join(f"d={d}: 10^{(4+d)/4*2:g}" for d in (1, 2, 10, 100)))

# ----------------------------------------------------------------------------------------------
sec(
    28,
    "Exchangeable but not independent: the Pólya urn, and de Finetti's latent (the limiting fraction is Beta(1,1))",
)
reps, n = 20_000, 2000
a = np.ones(reps)
b = np.ones(reps)
for _ in range(n):
    red = rng.uniform(size=reps) < a / (a + b)
    a += red
    b += ~red
fr = (a - 1) / n
print(
    f"P(first ball red) = 0.5; P(second red | first red) = 2/3; P(second red) = {0.5*2/3+0.5*1/3:.4f} = P(first red): exchangeable, not independent"
)
h, _ = np.histogram(fr, bins=5, range=(0, 1))
print(f"limiting red fraction over {reps:,} urns, 5 bins: {h/h.sum()}  (Beta(1,1) = uniform -> 0.2 each)")
print(
    f"(SE of the line above: each bin frequency ± {sqrt(0.2*0.8/reps):.4f}, binomial sqrt(0.2 · 0.8 / {reps:,}); the farthest entry is {np.max(np.abs(h/h.sum()-0.2))/sqrt(0.2*0.8/reps):.1f} SE from 0.2)"
)

# ----------------------------------------------------------------------------------------------
sec(
    29,
    "Infinite width is a CLT: one hidden ReLU layer, output at two inputs, vs the arc-cosine kernel (Cho & Saul 2009)",
)
x1 = np.array([1.0, 0.0])
x2 = np.array([0.6, 0.8])  # unit vectors, angle = acos(0.6)
theta = math.acos(0.6)
k_arccos = (1 / (2 * pi)) * (
    math.sin(theta) + (pi - theta) * math.cos(theta)
)  # kernel for ||x||=1 with var-1 weights, normalised by width
for width in (10, 100, 1000, 10000):
    reps, chunk = 20000, 500
    f1 = np.empty(reps)
    f2 = np.empty(reps)
    for c in range(0, reps, chunk):
        W = rng.standard_normal((chunk, width, 2))
        v = rng.standard_normal((chunk, width))
        h1 = np.maximum(W @ x1, 0)
        h2 = np.maximum(W @ x2, 0)
        f1[c : c + chunk] = (v * h1).sum(1) / sqrt(width)
        f2[c : c + chunk] = (v * h2).sum(1) / sqrt(width)
    print(
        f"width={width:>6}: var f(x1) = {f1.var():.4f} (theory 0.5000 = k(x1,x1)), cov(f(x1), f(x2)) = {np.mean(f1*f2):.4f} (arc-cos kernel {k_arccos:.4f}), excess kurtosis of f(x1) = {np.mean(((f1-f1.mean())/f1.std())**4)-3:+.3f}"
    )
    print(
        f"        (SE of the line above, from the same {reps:,} networks: var ± {se_var(f1):.4f};  cov ± {np.std(f1*f2)/sqrt(reps):.4f} (sd of the products / sqrt n);  the kurtosis SE at this width is §41's, not sqrt(24/n))"
    )

# ----------------------------------------------------------------------------------------------
sec(30, "St Petersburg: a game with infinite expectation, and what the sample mean of n games does")
n = 1_000_000
k = rng.geometric(0.5, n)  # number of tosses until first head
pay = 2.0**k
print(
    "sample mean of 2^K at n = 10^2..10^6:",
    [round(float(pay[:m].mean()), 1) for m in (100, 1000, 10000, 100000, 1000000)],
    "  (grows like log2(n) — Feller: S_n/(n log2 n) -> 1 in probability — and never settles)",
)

# ----------------------------------------------------------------------------------------------
sec(
    31, "Hoeffding vs Bernstein when the variance is small: Bernoulli(0.01), n=10,000, delta=0.05 half-widths"
)
p, n, delta = 0.01, 10_000, 0.05
hoeff = sqrt(log(2 / delta) / (2 * n))
bern = sqrt(2 * p * (1 - p) * log(2 / delta) / n) + 2 * log(2 / delta) / (3 * n)
print(
    f"Hoeffding half-width {hoeff:.4f};  Bernstein half-width {bern:.4f};  the true sd {sqrt(p*(1-p)/n):.4f}"
)

print("\nDONE")

# ==============================================================================================
# §32–§39. Each section below uses its OWN generator (g32 … g39, seeded 2026_1016_NN) and never
# the shared `rng`, so the §1–§31 output above (ending in the first DONE marker) does not depend on them.
# ==============================================================================================

# ----------------------------------------------------------------------------------------------
sec(
    32,
    "Exact finite-width excess kurtosis of the §29 ReLU network: 15/N, with the Monte Carlo standard error of the sample kurtosis",
)
# Z = v * ReLU(g), v and g independent N(0,1); f(x1) = sum_i Z_i / sqrt(N), so excess kurtosis of f = excess kurtosis of Z / N.
# Half-Gaussian moments: E[ReLU(g)^2] = E[g^2]/2 = 1/2, E[ReLU(g)^4] = E[g^4]/2 = 3/2; E[v^2] = 1, E[v^4] = 3.
# Numerical cross-check with a deterministic grid integral of the standard normal density (no random draws).
_gx = np.linspace(-12, 12, 2_000_001)
_gw = np.exp(-(_gx**2) / 2) / sqrt(2 * pi) * (_gx[1] - _gx[0])
E_relu2 = float(np.sum(np.maximum(_gx, 0) ** 2 * _gw))
E_relu4 = float(np.sum(np.maximum(_gx, 0) ** 4 * _gw))
E_v4 = float(np.sum(_gx**4 * _gw))
E_Z2 = 1.0 * E_relu2
E_Z4 = E_v4 * E_relu4
kappa_Z = E_Z4 / E_Z2**2 - 3
print(
    f"E[ReLU(g)^2] = {E_relu2:.4f} (theory 1/2), E[ReLU(g)^4] = {E_relu4:.4f} (theory 3/2), E[v^4] = {E_v4:.4f} (theory 3)"
)
print(
    f"Z = v*ReLU(g): E[Z^2] = {E_Z2:.4f} (theory 1/2), E[Z^4] = {E_Z4:.4f} (theory 9/2), excess kurtosis E[Z^4]/E[Z^2]^2 - 3 = {kappa_Z:.4f} (theory 15)"
)
se_kurt = sqrt(24 / 20000)
print(
    f"MC standard error of the sample excess kurtosis, 20,000 Gaussian replicates: sqrt(24/20000) = {se_kurt:.4f}"
)
for N, meas in ((10, 1.472), (100, 0.156), (1000, 0.016), (10000, 0.029)):
    th = 15 / N
    print(
        f"width={N:>6}: theory 15/N = {th:+.4f}   §29 measured {meas:+.3f}   |measured - theory| / SE = {abs(meas-th)/se_kurt:.2f}"
    )

# ----------------------------------------------------------------------------------------------
sec(
    33,
    "Score-function gradient with a parameter-dependent support: X ~ U(0, theta), f(x) = x — the naive estimator gives -1/2, the boundary term 1 restores the true 1/2",
)
g33 = np.random.default_rng(2026_1016_33)
theta, n = 2.0, 200_000
U = g33.uniform(0, 1, n)
X = theta * U
print(f"true d/dtheta E[X] = d/dtheta (theta/2) = {0.5:.4f}")
naive = X * (-1 / theta)  # f(x) * d/dtheta log p_theta(x) = x * (-1/theta)
print(
    f"naive score-function estimate E[X * d/dtheta log p_theta(X)] = E[-X/theta] = {naive.mean():+.4f}   (SE {naive.std(ddof=1)/sqrt(n):.4f}; theory -1/2)"
)
boundary = theta * (1 / theta)  # f(theta) * p_theta(theta) * d(theta)/d(theta)
print(f"boundary term f(theta) p_theta(theta) = theta * (1/theta) = {boundary:.4f}")
print(f"corrected = naive + boundary = {naive.mean()+boundary:+.4f}   (theory +0.5000)")
h = 0.01
fd = (np.mean((theta + h) * U) - np.mean((theta - h) * U)) / (
    2 * h
)  # common random numbers: same U at theta +/- h
print(f"finite-difference check with common random numbers, theta ± {h}: {fd:+.4f}   (theory +0.5000)")

# ----------------------------------------------------------------------------------------------
sec(
    34,
    "Constant-step SGD on the noisy quadratic f(x) = x^2/2: exact gradient converges, additive noise leaves a floor alpha sigma^2/(2 - alpha), noise that vanishes at the optimum converges a.s. only under the contraction condition E log|1 - alpha(1+zeta)| < 0",
)
g34 = np.random.default_rng(2026_1016_34)
x0, T, chains = 5.0, 20_000, 200
# (a) exact gradient g_t = x_t: x_{t+1} = (1 - alpha) x_t
alpha = 0.1
print(
    f"(a) exact gradient, alpha={alpha}: "
    + "   ".join(
        f"|x_{t}| = {abs(x0*(1-alpha)**t):.4e} (theory (1-alpha)^t * 5 = {x0*(1-alpha)**t:.4e})"
        for t in (10, 100, 1000)
    )
)
# (b) additive noise g_t = x_t + xi_t, xi ~ N(0,1): x_{t+1} = (1-alpha) x_t - alpha xi_t -> stationary var alpha^2/(1-(1-alpha)^2) = alpha/(2-alpha)
for alpha in (0.5, 0.1, 0.01):
    x = np.full(chains, x0)
    acc2 = 0.0
    acc1 = 0.0
    acc1c = np.zeros(chains)
    acc2c = np.zeros(chains)  # per-chain sums for the SE line (same iterates, no new draw)
    xi = g34.standard_normal((T, chains))
    for t in range(T):
        x = x - alpha * (x + xi[t])
        if t >= T // 2:
            acc1 += x.sum()
            acc2 += (x * x).sum()
            acc1c += x
            acc2c += x * x
    m = acc1 / (chains * (T - T // 2))
    v = acc2 / (chains * (T - T // 2)) - m * m
    print(
        f"(b) additive noise, alpha={alpha:<5}: stationary var of x_t over the last 10,000 steps x 200 chains = {v:.4f}   (theory alpha/(2-alpha) = {alpha/(2-alpha):.4f}; mean x_t {m:+.4f})"
    )
    vc = acc2c / (T - T // 2) - (acc1c / (T - T // 2)) ** 2
    print(
        f"    (SE of the line above: stationary var ± {vc.std(ddof=1)/sqrt(chains):.4f}   — SD across the {chains} independent chains of each chain's own last-10,000-step variance, divided by sqrt({chains}))"
    )
# (c) multiplicative noise g_t = x_t (1 + zeta_t), zeta ~ N(0, 0.25) (variance 0.25, sd 0.5): x_{t+1} = x_t (1 - alpha (1 + zeta_t))
alpha, zsd = 0.1, 0.5
zeta = g34.standard_normal((10_000, chains)) * zsd
logabs = np.log(x0) + np.cumsum(
    np.log(np.abs(1 - alpha * (1 + zeta))), axis=0
)  # log|x_t| tracked exactly (no underflow)
for t in (100, 1000, 10000):
    la = logabs[t - 1]
    med, p90 = np.median(la), np.percentile(la, 90)
    print(
        f"(c) multiplicative noise, alpha={alpha}, zeta sd {zsd}: t={t:>5}: median |x_t| = {exp(med):.3e} (log10 {med/log(10):+.2f}), 90th pct |x_t| = {exp(p90):.3e} (log10 {p90/log(10):+.2f})"
    )
contr = np.log(np.abs(1 - alpha * (1 + zeta)))
print(
    f"(c) expected log-contraction E log|1 - alpha(1 + zeta)| = {contr.mean():+.4f} per step (MC over {contr.size:,} draws; log(1-alpha) = {log(1-alpha):+.4f} without noise)  -> |x_t| ~ 5 exp({contr.mean():+.4f} t): noise that vanishes at the optimum converges a.s. only because this contraction condition E log|1 - alpha(1+zeta)| < 0 holds here (§40: zeta = ±20 makes it positive and the iterates diverge a.s.)"
)
_gx34 = np.linspace(-12, 12, 2_000_001)
_gw34 = np.exp(-(_gx34**2) / 2) / sqrt(2 * pi) * (_gx34[1] - _gx34[0])
print(
    f"    (SE of the line above: E log|1 - alpha(1 + zeta)| ± {contr.std(ddof=1)/sqrt(contr.size):.4f}   — sd/sqrt(n) from the same {contr.size:,} draws; exact value by Gaussian quadrature {float(np.sum(np.log(np.abs(1 - alpha*(1 + zsd*_gx34)))*_gw34)):+.4f})"
)

# ----------------------------------------------------------------------------------------------
sec(
    35,
    "McNemar on a paired evaluation: the same one-point gap at n = 1,000 is noise with 100 discordant pairs and decisive with 10",
)
from math import comb


def binom_two_sided(m, k):
    """exact two-sided p = 2 * P(Bin(m, 1/2) >= k) for k >= m/2, capped at 1"""
    return min(1.0, 2 * sum(comb(m, j) for j in range(k, m + 1)) / 2**m)


for m, k, note, n_eval in (
    (100, 55, "(i)", 1000),
    (10, 10, "(ii)", 1000),
    (160, 95, "(iii) exercise 7", 2000),
):  # exercise 7 scores a 2,000-example test set
    gap = (k - (m - k)) / n_eval * 100
    p = binom_two_sided(m, k)
    z = (k - m / 2) / sqrt(m / 4)
    print(
        f"{note:<17} discordant = {m:>3}, B wins {k:>3}: gap = {gap:.1f} points of {n_eval};  exact two-sided p = {p:.6f}"
        + (
            f";  normal-approximation z = {z:.3f}"
            if note != "(ii)"
            else f"  (= 2 * (1/2)^10 = {2*0.5**10:.6f})"
        )
    )

# ----------------------------------------------------------------------------------------------
sec(
    36,
    "Two effective sample sizes that share a name: importance-sampling ESS (sum w)^2 / sum w^2 vs Markov-chain ESS n / (1 + 2 sum rho_k)",
)
g36 = np.random.default_rng(2026_1016_36)
n = 10_000
# (a) IS: q = N(0,1), p = N(mu,1), w = p/q = exp(mu x - mu^2/2); E_q[w] = 1, E_q[w^2] = exp(mu^2) -> ESS/n -> exp(-mu^2)
xq = g36.standard_normal(n)
for mu in (0.5, 1.0, 2.0, 3.0):
    w = np.exp(mu * xq - mu * mu / 2)
    ess = w.sum() ** 2 / (w * w).sum()
    print(
        f"(a) IS, target N({mu},1) from N(0,1), n={n:,}: ESS = {ess:9.1f}, ESS/n = {ess/n:.4f}   (theory limit E[w]^2/E[w^2] = exp(-mu^2) = {exp(-mu*mu):.4f}; max weight share {w.max()/w.sum():.4f})"
    )
    A36, B36 = w.mean(), (w * w).mean()
    C36 = np.cov(w, w * w) / n
    g36v = np.array([2 * A36 / B36, -A36 * A36 / B36**2])
    se36 = sqrt(float(g36v @ C36 @ g36v))
    print(
        f"    (SE of the line above: ESS/n ± {se36:.4f}, ESS ± {n*se36:.1f}   — delta method on (mean w)^2 / mean w^2 from the same n = {n:,} weights; at mu >= 2 the weights are heavy-tailed and this SE is itself unreliable, cf. §48)"
    )
# (b) MCMC: AR(1) x_t = phi x_{t-1} + sqrt(1-phi^2) e_t; rho_k = phi^k -> ESS = n (1-phi)/(1+phi)
for phi in (0.5, 0.9, 0.99):
    e = g36.standard_normal(n) * sqrt(1 - phi**2)
    x = np.empty(n)
    x[0] = g36.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    xc = x - x.mean()
    var = xc.var()
    s = 0.0
    K = 0
    for k in range(1, n // 2):
        r = np.dot(xc[:-k], xc[k:]) / ((n - k) * var)
        if r < 0.01:
            break
        s += r
        K = k
    tau = 1 + 2 * s
    print(
        f"(b) AR(1) phi={phi:<4}, n={n:,}: theory ESS = n(1-phi)/(1+phi) = {n*(1-phi)/(1+phi):7.1f};  empirical (rho_k summed to lag {K}, first rho_k < 0.01) tau = {tau:6.1f}, ESS = n/tau = {n/tau:7.1f}"
    )
    se_tau = tau * sqrt(2 * (2 * K + 1) / n)
    print(
        f"    (SE of the line above: tau ± {se_tau:.1f}, ESS = n/tau ± {n*se_tau/tau**2:.1f}   — Sokal's asymptotic Var(tau_hat) ~ 2(2K + 1) tau^2/n with the window K = {K}, from the same chain)"
    )

# ----------------------------------------------------------------------------------------------
sec(
    37,
    "Monte Carlo standard errors for the §22 false-positive rates, and a mixture test martingale that bets on several lambdas at once",
)
g37 = np.random.default_rng(2026_1016_37)
R22 = 4000
for name, p in (
    ("fixed-n z-test", 0.0460),
    ("z-test peeked every 100", 0.3835),
    ("test martingale E_n >= 20", 0.0437),
):
    print(
        f"(a) §22 {name:<27} rate {p:.4f} at {R22:,} replications: SE = sqrt(p(1-p)/{R22}) = {sqrt(p*(1-p)/R22):.4f}"
    )
lams = (0.1, 0.2, 0.3, 0.5)
for p1 in (0.6, 0.55):
    print(
        f"(b) growth rate per step at P(1)={p1}: "
        + ", ".join(f"lambda={l}: {p1*log(1+l)+(1-p1)*log(1-l):+.4f}" for l in lams)
        + "   (p log(1+lambda) + (1-p) log(1-lambda); the optimum is lambda = 2p-1)"
    )
reps, N_, thresh = 2000, 10_000, 20.0


def run_martingales(p1):
    """returns (crossed, cross_time) for the 8-branch mixture and for the fixed lambda=0.5 two-sided (§22) martingale"""
    Emix = np.ones((8, reps))
    Efix = np.ones((2, reps))
    t_mix = np.full(reps, -1)
    t_fix = np.full(reps, -1)
    for t in range(N_):
        s = 2.0 * (g37.uniform(0, 1, reps) < p1) - 1.0  # ±1 with P(+1) = p1
        for j, l in enumerate(lams):
            Emix[2 * j] *= 1 + l * s
            Emix[2 * j + 1] *= 1 - l * s
        Efix[0] *= 1 + 0.5 * s
        Efix[1] *= 1 - 0.5 * s
        new = (t_mix < 0) & (Emix.mean(0) >= thresh)
        t_mix[new] = t + 1
        new = (t_fix < 0) & (Efix.mean(0) >= thresh)
        t_fix[new] = t + 1
    return t_mix, t_fix


t_mix, t_fix = run_martingales(0.5)
fp = np.mean(t_mix >= 0)
fpf = np.mean(t_fix >= 0)
print(
    f"(b) null P(1)=0.5, {reps:,} replications, n={N_:,}: mixture (8 branches) false-positive rate of 'stop when E_n >= 20' = {fp:.4f} (SE {sqrt(fp*(1-fp)/reps):.4f});  fixed lambda=0.5 (§22) = {fpf:.4f} (SE {sqrt(fpf*(1-fpf)/reps):.4f});  Ville bound 0.0500"
)
for p1 in (0.6, 0.55):
    t_mix, t_fix = run_martingales(p1)
    cm, cf = np.mean(t_mix >= 0), np.mean(t_fix >= 0)
    med_m = np.median(t_mix[t_mix >= 0]) if cm > 0 else float('nan')
    med_f = np.median(t_fix[t_fix >= 0]) if cf > 0 else float('nan')
    print(
        f"(b) alternative P(1)={p1}: mixture crosses 20 by n={N_:,} in {cm:.4f} of runs, median crossing time {med_m:.0f};  fixed lambda=0.5 crosses in {cf:.4f} of runs, median crossing time {med_f:.0f}"
    )
    print(
        f"    (SE of the line above: mixture fraction ± {sqrt(cm*(1-cm)/reps):.4f}, median crossing time ± {se_median(t_mix[t_mix >= 0]):.0f};  fixed lambda=0.5 fraction ± {sqrt(cf*(1-cf)/reps):.4f}, median ± {se_median(t_fix[t_fix >= 0]):.0f}   — binomial SEs from the same {reps:,} runs; median SEs are half the order-statistic 68% interval of the crossing times)"
    )

# ----------------------------------------------------------------------------------------------
sec(
    38,
    "The §4.3 learning curve over 20 independent training sets per n: mean excess risk, its standard error, and the per-decade ratio",
)
g38 = np.random.default_rng(2026_1016_38)
mu, dsz, reps38, n_ho = 1.0, 20, 20, 100_000
bayes = Phi(-mu)
yho = g38.integers(0, 2, n_ho) * 2 - 1
Xho = g38.standard_normal((n_ho, dsz))
Xho[:, 0] += mu * yho
print(
    f"held-out set: {n_ho:,} points drawn once; Bayes classifier error on it {np.mean(np.sign(Xho[:,0]) != yho):.4f} (theory Phi(-1) = {bayes:.4f})"
)
bayes_ho = np.mean(
    np.sign(Xho[:, 0]) != yho
)  # the held-out set's own Bayes-classifier error (its sampling offset from Phi(-1))
exc = {}
exc_ho = {}
with np.errstate(over='ignore'):
    for ntr in (50, 200, 2000, 20000):
        errs = np.empty(reps38)
        for r in range(reps38):
            ytr = g38.integers(0, 2, ntr) * 2 - 1
            Xtr = g38.standard_normal((ntr, dsz))
            Xtr[:, 0] += mu * ytr
            w = fit_logreg(Xtr, ytr, iters=25)
            errs[r] = np.mean(np.sign(Xho @ w) != yho)
        exc[ntr] = errs - bayes
        exc_ho[ntr] = errs - bayes_ho
        print(
            f"n_train={ntr:>6}, d=20, {reps38} reps: mean held-out error {errs.mean():.4f}, mean excess over Bayes Phi(-1) {exc[ntr].mean():+.4f}, SE of the mean excess {exc[ntr].std(ddof=1)/sqrt(reps38):.4f}  (min {errs.min():.4f}, max {errs.max():.4f}; excess over the held-out set's own Bayes error {bayes_ho:.4f}: {exc_ho[ntr].mean():+.4f})"
        )
for a, b in ((50, 200), (200, 2000), (2000, 20000)):
    r_n = b / a
    print(
        f"excess({a})/excess({b}) = {exc[a].mean()/exc[b].mean():.2f} against Phi(-1), {exc_ho[a].mean()/exc_ho[b].mean():.2f} against the held-out set's Bayes error   (n ratio {r_n:g}: a 1/sqrt(n) rate predicts sqrt({r_n:g}) = {sqrt(r_n):.2f}, a d/n rate predicts {r_n:g})"
    )

# ----------------------------------------------------------------------------------------------
sec(
    39,
    "Median-of-means on Pareto(alpha = 1.5): a finite-variance-free estimator against the sample mean, n = 10,000, 1,000 replications",
)
g39 = np.random.default_rng(2026_1016_39)
alpha, n, reps39, blocks = 1.5, 10_000, 1000, 20
true_mean = alpha / (alpha - 1)  # x_min = 1, same construction as §11: U^(-1/alpha)
par = g39.uniform(0, 1, (reps39, n)) ** (-1 / alpha)
sm = par.mean(1)
mom = np.median(par.reshape(reps39, blocks, n // blocks).mean(2), axis=1)
print(
    f"Pareto(1.5), x_min=1: true mean alpha/(alpha-1) = {true_mean:.4f}; n={n:,}, {reps39:,} replications, median-of-means with {blocks} blocks of {n//blocks}"
)
for name, est in (("sample mean    ", sm), ("median-of-means", mom)):
    err = est - true_mean
    ae = np.abs(err)
    print(
        f"{name}: RMSE {np.sqrt(np.mean(err**2)):.4f}, median |error| {np.median(ae):.4f}, 99th pct |error| {np.percentile(ae, 99):.4f}   (bias = mean error {err.mean():+.4f}, max |error| {ae.max():.3f})"
    )

print("\nDONE (§32–§39)")

# ==============================================================================================
# §40–§46 are appended AFTER the "DONE (§32–§39)" marker so that the first 255 output lines stay byte-identical.
# Each section owns its generator gNN = default_rng(2026_1016_NN) and never touches the shared `rng`.

# ----------------------------------------------------------------------------------------------
sec(
    40,
    "Multiplicative-noise SGD on f(x) = x^2/2: a.s. convergence with E[x_t^2] -> infinity, and the two exact contraction conditions",
)
g40 = np.random.default_rng(2026_1016_40)
alpha, x0, paths, T40 = 0.1, 1.0, 2000, 200
# x_{t+1} = x_t (1 - alpha (1 + zeta_t)); zeta = ±5 equiprobable -> multipliers 0.4 / 1.4
m_lo, m_hi = 1 - alpha * (1 + 5), 1 - alpha * (1 - 5)
Elog, Em2 = 0.5 * (log(abs(m_lo)) + log(abs(m_hi))), 0.5 * (m_lo**2 + m_hi**2)
print(
    f"(a) zeta = ±5, alpha={alpha}, x_0={x0}: multipliers {m_lo:.1f} / {m_hi:.1f};  E log|m| = (1/2) ln(0.56) = {Elog:+.4f} < 0 (a.s. contraction)   E[m^2] = (0.16 + 1.96)/2 = {Em2:.2f} > 1 (mean-square blow-up)   [exact]"
)
mult = np.where(g40.integers(0, 2, (T40, paths)) == 0, m_lo, m_hi)
logabs = log(x0) + np.cumsum(np.log(np.abs(mult)), axis=0)  # log|x_t| tracked exactly, as §34(c)
print(
    f"(a) {paths:,} paths: "
    + "   ".join(
        f"t={t}: median |x_t| = {exp(np.median(logabs[t-1])):.2e}, frac |x_t| < 1e-6 = {np.mean(logabs[t-1] < log(1e-6)):.4f}"
        for t in (50, 100, 200)
    )
    + f"   (theory median ~ exp({Elog:+.4f} t))"
)
print(
    f"    (SE of the line above: "
    + "   ".join(
        f"t={t}: median within a factor exp(±{se_median(logabs[t-1]):.3f}), frac ± {sqrt(np.mean(logabs[t-1] < log(1e-6))*(1 - np.mean(logabs[t-1] < log(1e-6)))/paths):.4f}"
        for t in (50, 100, 200)
    )
    + f"   — order-statistic SE of the median on the log scale and binomial SEs, from the same {paths:,} paths)"
)
Em4 = 0.5 * (m_lo**4 + m_hi**4)  # E[m^4], for the exact SD of x_t^2
for t in (10, 20, 40):
    x2 = np.exp(2 * logabs[t - 1])
    print(
        f"(a) t={t:>2}: MC mean of x_t^2 over {paths:,} paths = {x2.mean():.3e}   (exact E[x_t^2] = 1.06^{t} = {Em2**t:.3e};  max path x_t^2 = {x2.max():.3e}, median {np.median(x2):.3e} — the mean is carried by rare paths, so the MC mean undershoots the exact value once those paths are rarer than 1 in {paths:,})"
    )
    print(
        f"    (SE of the line above: MC mean of x_t^2 ± {x2.std(ddof=1)/sqrt(paths):.3e}   — sd/sqrt({paths:,}) from the same paths;  the exact SE, sqrt((E[m^4]^t - 1.06^(2t))/{paths:,}) with E[m^4] = {Em4:.4f}, is {sqrt((Em4**t - Em2**(2*t))/paths):.3e}: the empirical SE understates by the same rare-path mechanism)"
    )
# (b) zeta = ±20 -> multipliers -1.1 / 2.9
m_lo, m_hi = 1 - alpha * (1 + 20), 1 - alpha * (1 - 20)
Elog20 = 0.5 * (log(abs(m_lo)) + log(abs(m_hi)))
mult = np.where(g40.integers(0, 2, (100, paths)) == 0, m_lo, m_hi)
logabs20 = log(x0) + np.cumsum(np.log(np.abs(mult)), axis=0)
print(
    f"(b) zeta = ±20, alpha={alpha}: multipliers {m_lo:.1f} / {m_hi:.1f};  E log|m| = (1/2)(ln 1.1 + ln 2.9) = {Elog20:+.4f} > 0;  frac of {paths:,} paths with |x_100| > 1e6 = {np.mean(logabs20[99] > log(1e6)):.4f}  (median |x_100| = {exp(np.median(logabs20[99])):.2e})"
)
f20 = np.mean(logabs20[99] > log(1e6))
print(
    f"    (SE of the line above: frac ± {sqrt(f20*(1-f20)/paths):.4f} (binomial, from the same {paths:,} paths: every path exceeded 1e6);  median |x_100| within a factor exp(±{se_median(logabs20[99]):.3f}) (order-statistic SE on the log scale; exact median exp(100 E log|m|) = {exp(100*Elog20):.2e}))"
)
# (c) the two exact scalar conditions; Gaussian case by deterministic quadrature on the §32 grid (zeta = 0.5 g, g ~ N(0,1))
_gx = np.linspace(-12, 12, 2_000_001)
_gw = np.exp(-(_gx**2) / 2) / sqrt(2 * pi) * (_gx[1] - _gx[0])
mg = 1 - alpha * (1 + 0.5 * _gx)
rows = (
    ("N(0, 0.5^2) [§34(c)]", float(np.sum(np.log(np.abs(mg)) * _gw)), float(np.sum(mg**2 * _gw))),
    ("±5                 ", Elog, Em2),
    ("±20                ", Elog20, 0.5 * ((1 - alpha * 21) ** 2 + (1 - alpha * (-19)) ** 2)),
)
print(
    f"(c) alpha={alpha}: a.s. contraction iff E log|1 - alpha(1+zeta)| < 0;  mean-square contraction iff E[(1 - alpha(1+zeta))^2] < 1   (no noise: log(1-alpha) = {log(1-alpha):+.4f}, (1-alpha)^2 = {(1-alpha)**2:.4f})"
)
for name, el, em in rows:
    print(
        f"(c)   zeta ~ {name}:  E log|m| = {el:+.4f} ({'a.s. converges' if el < 0 else 'a.s. diverges '})   E[m^2] = {em:.4f} ({'L2 converges' if em < 1 else 'L2 diverges '})"
    )
# (d) additive noise, alpha = 1, xi = ±1: x_{t+1} = x_t - alpha (x_t + xi_t) = -xi_t
xi = np.where(g40.integers(0, 2, 10_000) == 0, -1.0, 1.0)
xd = np.empty(10_000)
x = x0
for t in range(10_000):
    x = x - 1.0 * (x + xi[t])
    xd[t] = x
vals, cnts = np.unique(xd, return_counts=True)
print(
    f"(d) additive noise, alpha=1, xi = ±1, 10,000 steps from x_0={x0}: x_{{t+1}} = -xi_t;  empirical support {{{', '.join(f'{v:+.0f}' for v in vals)}}} with frequencies {np.round(cnts/cnts.sum(), 4).tolist()}  — a two-point law, not Gaussian (excess kurtosis {np.mean(xd**4)/np.mean(xd**2)**2 - 3:+.1f} vs 0 for a Gaussian)"
)
p40 = np.mean(xd == 1)
print(
    f"    (SE of the line above: each frequency ± {sqrt(p40*(1-p40)/len(xd)):.4f} (binomial at p = 1/2, from the same 10,000 steps);  the excess kurtosis -2 is exact for any ±1 sample (x^4 = x^2 = 1), no SE)"
)
print(
    f"(d) stationary variance: empirical {xd.var():.4f}   (theory alpha sigma^2/(2 - alpha) = 1 * 1/(2 - 1) = {1.0*1.0/(2-1.0):.4f} still holds — the formula fixes the variance, not the shape)"
)
print(
    f"    (SE of the line above: empirical variance ± {se_var(xd):.4f}   — sqrt((m4 - s^4)/n) from the same 10,000 steps; for a ±1 sample s^2 = 1 - mean^2, so the SE is 2 |mean| sqrt(p(1-p)/n) up to O(1/n))"
)

# ----------------------------------------------------------------------------------------------
sec(
    41,
    "Monte Carlo standard error of the sample excess kurtosis at finite width: the Gaussian sqrt(24/m) understates it at small N",
)
g41 = np.random.default_rng(2026_1016_41)
# §29 model, copied exactly: f(x1) = N^{-1/2} sum_i v_i ReLU(w_i . x1), w_i ~ N(0, I_2), v_i ~ N(0,1), x1 = e_1, so g_i = w_i . x1 ~ N(0,1).
# Exact sampler of the same law: conditional on g, f ~ N(0, S/N) with S = sum_i ReLU(g_i)^2 = sum of K squared normals, K ~ Bin(N, 1/2):
#   f = sqrt(S/N) u,  K ~ Bin(N, 1/2),  S | K ~ chi^2_K (= Gamma(K/2, scale 2); S = 0 when K = 0),  u ~ N(0,1).   Three draws per f at any width.
m41, R41 = (
    20_000,
    1000,
)  # R = 1,000 replicated estimates per width (the exact sampler costs ~2 s per 400 at any width)


def excess_kurt(f):
    return np.mean(((f - f.mean()) / f.std()) ** 4) - 3  # the §29 estimator


def kurt_reps_exact(N, R):
    K = g41.binomial(N, 0.5, (R, m41))
    S = g41.gamma(K / 2, 2.0)
    u = g41.standard_normal((R, m41))
    f = np.sqrt(S / N) * u
    return np.array([excess_kurt(f[r]) for r in range(R)])


def kurt_reps_direct(N, R, chunk):
    out = np.empty(R)
    for r in range(R):
        f = np.empty(m41)
        for c in range(0, m41, chunk):
            W = g41.standard_normal((chunk, N, 2))
            v = g41.standard_normal((chunk, N))
            f[c : c + chunk] = (v * np.maximum(W @ np.array([1.0, 0.0]), 0)).sum(1) / sqrt(N)
        out[r] = excess_kurt(f)
    return out


se_gauss = sqrt(24 / m41)
print(
    f"(a) m = {m41:,} samples per kurtosis estimate, R = {R41} replicated estimates per width (exact sampler); Gaussian SE sqrt(24/m) = {se_gauss:.4f}; reviewer's asymptotic SE sqrt((24 + 1080/N + 12810/N^2 + 32490/N^3)/m)"
)
direct = {}
direct[10] = kurt_reps_direct(10, 400, 2000)
direct[100] = kurt_reps_direct(100, 100, 500)
for N in (10, 100, 1000, 10000):
    k = kurt_reps_exact(N, R41)
    se_rev = sqrt((24 + 1080 / N + 12810 / N**2 + 32490 / N**3) / m41)
    sd = k.std(ddof=1)
    extra = (
        f";  direct §29 sampler ({len(direct[N])} reps): SD {direct[N].std(ddof=1):.4f}, mean {direct[N].mean():+.4f}"
        if N in direct
        else ""
    )
    print(
        f"(b) width={N:>6}: empirical SD of the estimate = {sd:.4f} (its own SE ~ SD/sqrt(2(R-1)) = {sd/sqrt(2*(R41-1)):.4f})   vs (i) Gaussian {se_gauss:.4f}   (ii) reviewer {se_rev:.4f} [ratio emp/(ii) = {sd/se_rev:.3f}];   mean of the {R41} estimates {k.mean():+.4f} ± {sd/sqrt(R41):.4f} vs exact 15/N = {15/N:+.4f}{extra}"
    )

# ----------------------------------------------------------------------------------------------
sec(
    42,
    "Selecting the best of m = 100 fixed models on n = 1,000 held-out examples: the variance-proxy bound sigma sqrt(2 ln m) is not a Hoeffding guarantee",
)
g42 = np.random.default_rng(2026_1016_42)
p, n42, m42, R42 = 0.8, 1000, 100, 2000
mgf = p * exp(-(1 - p)) + (1 - p) * exp(
    p
)  # E[exp(-(X - p))], X ~ Bernoulli(p), lambda = -1 (the lower tail of the accuracy)
print(
    f"(a) X ~ Bernoulli({p}): E[e^(-(X-{p}))] = {p} e^-{1-p:.1f} + {1-p:.1f} e^{p} = {mgf:.5f}  >  e^(Var(X)/2) = e^{p*(1-p)/2:.2f} = {exp(p*(1-p)/2):.5f}  -> p(1-p) is NOT a valid sub-Gaussian variance proxy;  Hoeffding's proxy 1/4 holds: e^(1/8) = {exp(1/8):.5f} >= {mgf:.5f}"
)
sigma = sqrt(p * (1 - p) / n42)
b_var, b_hoeff = sigma * sqrt(2 * log(m42)), sqrt(log(m42) / (2 * n42))
print(
    f"(b) n={n42:,}, m={m42}, p={p}: sigma = sqrt(p(1-p)/n) = {sigma:.5f};  variance-proxy bound sigma sqrt(2 ln m) = {b_var:.4f} (not a guarantee for Bernoulli)   Hoeffding-valid sqrt(ln m / (2n)) = {b_hoeff:.4f}"
)
acc = (
    g42.binomial(n42, p, (R42, m42)) / n42
)  # m independent models, each scored on its own n Bernoulli(p) correctness indicators
mx = (acc - p).max(1)
print(
    f"(c) {R42:,} replications: mean of max_j (accuracy_j - {p}) = {mx.mean():.4f} (MC SE {mx.std(ddof=1)/sqrt(R42):.4f};  median {np.median(mx):.4f}, 99th pct {np.percentile(mx, 99):.4f})   — under both {b_var:.4f} and {b_hoeff:.4f}; only the second is a proven bound on E[max]"
)
print(
    f"(c) fraction of replications whose max exceeds the variance-proxy bound {b_var:.4f}: {np.mean(mx > b_var):.4f};  exceeds the Hoeffding bound {b_hoeff:.4f}: {np.mean(mx > b_hoeff):.4f}"
)

# ----------------------------------------------------------------------------------------------
sec(
    43,
    "The paired standard error of a checkpoint difference is sqrt(Var(D_i)/n), and sqrt(2) x SE is not an upper bound",
)
g43 = np.random.default_rng(2026_1016_43)
pA = pB = 0.18
n43 = 50_000
se_one = sqrt(pA * (1 - pA) / n43)
print(
    f"two classifiers, error rate {pA} each, n = {n43:,}, D_i = 1{{A wrong}} - 1{{B wrong}};  per-classifier SE sqrt(0.18 * 0.82/n) = {se_one:.5f},  sqrt(2) x that = {sqrt(2)*se_one:.5f}"
)
for name, p_both in (
    ("errors disjoint        ", 0.0),
    ("errors independent     ", pA * pB),
    ("errors 90 % shared     ", 0.162),
):
    varD = (
        pA * (1 - pA) + pB * (1 - pB) - 2 * (p_both - pA * pB)
    )  # Var(1A - 1B) = Var 1A + Var 1B - 2 Cov; P(both wrong) = p_both
    print(
        f"{name}: P(both wrong) = {p_both:.4f}, Var(D) = {varD:.4f}, exact SE(D-bar) = sqrt(Var(D)/n) = {sqrt(varD/n43):.5f}   = {sqrt(varD/n43)/se_one:.3f} x per-classifier SE ({'above' if sqrt(varD/n43) > sqrt(2)*se_one else 'below'} sqrt(2) x SE = {sqrt(2)*se_one:.5f})"
    )
u = g43.uniform(0, 1, n43)
D = np.where(
    u < pA, 1.0, np.where(u < pA + pB, -1.0, 0.0)
)  # disjoint: A-only wrong w.p. 0.18, B-only wrong w.p. 0.18, neither w.p. 0.64
print(
    f"simulated disjoint case once, n = {n43:,}: mean D = {D.mean():+.5f}, empirical SE from the per-example D_i = sqrt(var(D)/n) = {D.std(ddof=1)/sqrt(n43):.5f}   (exact {sqrt(0.36/n43):.5f})"
)

# ----------------------------------------------------------------------------------------------
sec(44, "Where JS saturates and W1 does not, and where even a finite W1 has zero generator gradient")
uq = (
    np.arange(100_000) + 0.5
) / 100_000  # quantile grid for the 1-D formula W1 = int_0^1 |F^-1(u) - G^-1(u)| du


def js_two_points(a, b):
    """JS(delta_a, delta_b) in nats: M = (delta_a + delta_b)/2 has mass 1/2 at each point when a != b, so each KL term is ln 2"""
    return 0.0 if a == b else 0.5 * log(1 / 0.5) + 0.5 * log(1 / 0.5)


print(
    f"(a) P = delta_0, Q_theta = delta_theta:   JS in nats (ln 2 = {log(2):.4f}),  W1 = |theta|,  dW1/dtheta = sign(theta)"
)
for th in (0.1, 0.5, 1.0, 2.0, 5.0):
    w1_num = float(np.mean(np.abs(np.zeros_like(uq) - np.full_like(uq, th))))
    print(
        f"(a)   theta = {th:<4}: JS = {js_two_points(0.0, th):.4f} (= ln 2, flat in theta: zero gradient)   W1 = {abs(th):.4f} (quantile formula {w1_num:.4f})   dW1/dtheta = {math.copysign(1.0, th):+.1f}"
    )
print(
    f"(b) P = delta_1, Q_theta = delta_(theta^2):   W1 = |1 - theta^2|,  dW1/dtheta = -2 theta sign(1 - theta^2)"
)
for th in (0.0, 0.5, 0.9, 1.1):
    w1 = abs(1 - th**2)
    grad = -2 * th * math.copysign(1.0, 1 - th**2) + 0.0  # + 0.0 turns IEEE -0.0 into 0.0 at theta = 0
    w1_num = float(np.mean(np.abs(np.ones_like(uq) - np.full_like(uq, th**2))))
    print(
        f"(b)   theta = {th:<4}: W1 = {w1:.4f} (quantile formula {w1_num:.4f})   dW1/dtheta = {grad:+.4f}"
        + (
            "   <- distance 1, gradient exactly 0: the generator is stuck at a stationary point"
            if th == 0.0
            else ""
        )
    )

# ----------------------------------------------------------------------------------------------
sec(45, "Stationary but not ergodic: a latent fair coin Z, X_t = Z for every t")
g45 = np.random.default_rng(2026_1016_45)
R45, T45, B45, nboot = 1000, 1000, 50, 200
Z = g45.integers(0, 2, R45).astype(float)
X = np.repeat(Z[:, None], T45, axis=1)  # R trajectories of length T, each constant
means = X.mean(1)
print(
    f"{R45:,} trajectories x T = {T45:,} steps: per-trajectory sample mean takes the value 0 in {np.mean(means == 0):.3f} of trajectories, 1 in {np.mean(means == 1):.3f}, exactly 1/2 in {np.mean(means == 0.5):.3f}   (marginal law of X_t is Bernoulli(1/2) at every t: stationary ✓)"
)
print(
    f"    (SE of the line above: each fraction ± {sqrt(0.5*0.5/R45):.3f}   — binomial at p = 1/2 from the same {R45:,} trajectories)"
)
print(
    f"variance of the sample mean across trajectories: empirical {means.var(ddof=1):.4f}   (theory 1/4 = {0.25:.4f}; an ergodic iid Bernoulli(1/2) chain would give 1/(4T) = {0.25/T45:.5f})   -> time average != ensemble average: ergodic ✗"
)
print(
    f"    (SE of the line above: empirical variance ± {se_var(means):.4f}   — sqrt((m4 - s^4)/R) from the same {R45:,} trajectory means)"
)
x1 = X[0]  # ONE trajectory
starts = g45.integers(
    0, T45 - B45 + 1, (nboot, T45 // B45)
)  # moving-block bootstrap, 20 blocks of length 50 per resample
boot_means = np.array([np.concatenate([x1[s : s + B45] for s in st]).mean() for st in starts])
batch_means = x1.reshape(T45 // B45, B45).mean(1)
print(
    f"from ONE trajectory (Z = {x1[0]:.0f}): block-bootstrap variance estimate (block length {B45}, {nboot} resamples) = {boot_means.var(ddof=1):.4f};  batch-means estimate ({T45//B45} batches of {B45}) = {batch_means.var(ddof=1)/(T45//B45):.4f}   (truth 1/4)"
)
print(
    "stationary ✓, ergodic ✗: both one-trajectory estimators report zero — no amount of T or resampling reveals the latent coin from a single path"
)

# ----------------------------------------------------------------------------------------------
sec(
    46,
    "Two uncertainties on the §38 learning curve at n = 2,000: variation across training seeds vs the paired test-set uncertainty of the excess risk",
)
g46 = np.random.default_rng(2026_1016_46)
mu, dsz, seeds46, ntr46, n_ho = 1.0, 20, 20, 2000, 100_000


def draw_e2(g, n):
    y = g.integers(0, 2, n) * 2 - 1
    X = g.standard_normal((n, dsz))
    X[:, 0] += mu * y
    return X, y


Xho1, yho1 = draw_e2(g46, n_ho)  # ONE shared held-out set, as §38
bayes_ho1 = np.mean(np.sign(Xho1[:, 0]) != yho1)
ws = []
with np.errstate(over='ignore'):
    for r in range(seeds46):
        Xtr, ytr = draw_e2(g46, ntr46)
        ws.append(fit_logreg(Xtr, ytr, iters=25))  # §4's fit_logreg, re-used
Xho2, yho2 = draw_e2(g46, n_ho)  # a SECOND independent held-out set, drawn after the training sets
bayes_ho2 = np.mean(np.sign(Xho2[:, 0]) != yho2)
err1 = np.array([np.mean(np.sign(Xho1 @ w) != yho1) for w in ws])
err2 = np.array([np.mean(np.sign(Xho2 @ w) != yho2) for w in ws])
exc1 = err1 - bayes_ho1
exc2 = err2 - bayes_ho2
print(
    f"held-out set 1: {n_ho:,} points, its own Bayes-classifier error {bayes_ho1:.4f} (population floor Phi(-1) = {Phi(-mu):.4f});  {seeds46} seeds, each trained on n = {ntr46:,}, d = {dsz}"
)
print(
    f"(a) across {seeds46} seeds on set 1: mean excess error over the set's Bayes error = {exc1.mean():+.4f} (over Phi(-1): {(err1 - Phi(-mu)).mean():+.4f});  SD across seeds = {exc1.std(ddof=1):.4f};  SE of the mean = SD/sqrt({seeds46}) = {exc1.std(ddof=1)/sqrt(seeds46):.4f}   (min {exc1.min():+.4f}, max {exc1.max():+.4f})"
)
pse = np.empty(seeds46)
for r, w in enumerate(ws):
    Dr = (np.sign(Xho1 @ w) != yho1).astype(float) - (np.sign(Xho1[:, 0]) != yho1).astype(float)
    pse[r] = Dr.std(ddof=1) / sqrt(n_ho)
print(
    f"(b) per seed, paired SE of the excess on set 1 = sqrt(Var(D_i)/{n_ho:,}), D_i = 1{{model wrong}} - 1{{Bayes wrong}}: mean {pse.mean():.5f}, min {pse.min():.5f}, max {pse.max():.5f}   (vs the unpaired sqrt(2 * 0.16 * 0.84/n) = {sqrt(2*0.16*0.84/n_ho):.5f})"
)
print(
    f"(c) the SAME {seeds46} models on an independent held-out set 2 ({n_ho:,} points, its own Bayes error {bayes_ho2:.4f}): mean excess over the set's Bayes error = {exc2.mean():+.4f} (over Phi(-1): {(err2 - Phi(-mu)).mean():+.4f});  SD across seeds {exc2.std(ddof=1):.4f}"
)
print(
    f"(c) shared-test-set offset between the two sets: mean excess set 1 - set 2 = {exc1.mean() - exc2.mean():+.4f} (raw error {err1.mean() - err2.mean():+.4f});  per-seed correlation of excess across the two sets = {np.corrcoef(exc1, exc2)[0, 1]:.3f}   — compare with the across-seed SE {exc1.std(ddof=1)/sqrt(seeds46):.4f} and the paired SE ~{pse.mean():.5f}"
)

# ----------------------------------------------------------------------------------------------
sec(
    47,
    "The covariance bound needs K^2: a bounded, mean-0, variance-1 three-point law (eps = 1e-4) whose sample covariance is 0 in nine datasets of ten at n = 1,000, d = 1",
)
g47 = np.random.default_rng(2026_1016_47)
# §47–§49 sit before the final DONE marker, after every earlier line of output; each owns its
# generator gNN = default_rng(2026_1016_NN) and never touches the shared `rng`, so the §1–§46 output does not depend on them.
eps47, n47, R47, chunk47 = 1e-4, 1000, 20_000, 1000
a47 = sqrt(1 / eps47)  # X = ±eps^(-1/2) w.p. eps/2 each, 0 w.p. 1 - eps
print(
    f"(a) X = ±eps^(-1/2) = ±{a47:.1f} w.p. eps/2 each, 0 w.p. 1 - eps, eps = {eps47:g}:  E X = 0,  E X^2 = eps * eps^(-1) = {eps47*a47**2:.4f}  [exact];  Sigma_hat = (1/n) sum X_i^2 = N/(n eps) with N ~ Bin(n, eps)"
)
p0 = (1 - eps47) ** n47
print(
    f"(b) exact P(all n = {n47:,} samples zero) = (1 - eps)^n = {p0:.4f}  -> Sigma_hat = 0, operator-norm error ||Sigma|| = 1, in that fraction of datasets"
)
allzero = np.empty(R47, dtype=bool)
s2 = np.empty(R47)
for c in range(0, R47, chunk47):
    u = g47.uniform(0, 1, (chunk47, n47))
    X = np.where(u < eps47 / 2, -a47, np.where(u < eps47, a47, 0.0))
    allzero[c : c + chunk47] = np.all(X == 0, axis=1)
    s2[c : c + chunk47] = (X * X).mean(1)
fz = allzero.mean()
print(
    f"(b) simulated, R = {R47:,} datasets of n = {n47:,}: fraction all-zero = {fz:.4f}  (binomial SE sqrt(p(1-p)/R) = {sqrt(p0*(1-p0)/R47):.4f};  |sim - exact| / SE = {abs(fz-p0)/sqrt(p0*(1-p0)/R47):.2f})"
)
err47 = np.abs(s2 - 1)
pmf = [
    comb(n47, k) * eps47**k * (1 - eps47) ** (n47 - k) for k in range(0, 61)
]  # Bin(1000, 1e-4): mass beyond k = 60 is below 1e-100
E_err = sum(pk * abs(k / (n47 * eps47) - 1) for k, pk in enumerate(pmf))
print(
    f"(c) d = 1 operator-norm error |Sigma_hat - 1|: simulated mean over R = {R47:,} = {err47.mean():.4f} (SE {err47.std(ddof=1)/sqrt(R47):.4f});  exact E|N/(n eps) - 1| = sum_k P(N=k)|k/(n eps) - 1| = {E_err:.4f}  (= 2(1 - eps)^n = {2*p0:.4f}, since E[N/(n eps)] = 1)"
)
ref = sqrt(1 / n47) + 1 / n47
K2 = 1 / (eps47 * log((1 + eps47) / eps47))
print(
    f"(c) constant-free Gaussian-type reference sqrt(d/n) + d/n at d = 1, n = {n47:,} = {ref:.4f}  -> the mean error is {E_err/ref:.0f}x it: no universal C can cover this law"
)
print(
    f"(d) ||X||_psi2 = inf{{t: E exp(X^2/t^2) <= 2}};  E exp(X^2/t^2) = 1 - eps + eps exp(1/(eps t^2)) = 2  =>  K^2 = ||X||_psi2^2 = 1/(eps ln((1 + eps)/eps)) = {K2:.1f}  (check: E exp(X^2/K^2) = {1 - eps47 + eps47*exp(1/(eps47*K2)):.6f})"
)
print(
    f"(d) the bounded-variable bound ||X||_psi2 <= ||X||_inf/sqrt(ln 2) gives K^2 <= 1/(eps ln 2) = {1/(eps47*log(2)):.1f};  K^2 (sqrt(d/n) + d/n) = {K2*ref:.1f} (C = 1), which does cover the mean error {E_err:.4f}"
)
print(
    f"(d) for contrast, a standard Gaussian: E exp(g^2/t^2) = (1 - 2/t^2)^(-1/2) = 2  =>  K^2 = 8/3 = {8/3:.4f} at every scale (absorbed into C)"
)
for e in (1e-2, 1e-3, 1e-4):
    k2 = 1 / (e * log((1 + e) / e))
    print(
        f"(e) eps = {e:g}: exact K^2 = 1/(eps ln((1+eps)/eps)) = {k2:9.1f}   1/(eps ln(1/eps)) = {1/(e*log(1/e)):9.1f} (ratio {k2*e*log(1/e):.4f})   bound 1/(eps ln 2) = {1/(e*log(2)):9.1f}   (1 - eps)^n at n = 1,000 = {(1-e)**n47:.4f}"
    )

# ----------------------------------------------------------------------------------------------
sec(
    48,
    "Weight-based IS ESS is not a standard error: the §36 setting at mu = 2 (target N(2,1), proposal N(0,1), f(x) = x, n = 10,000), closed-form IS and SNIS standard errors against sqrt(Var_p f / ESS)",
)
g48 = np.random.default_rng(2026_1016_48)
mu, n48, R48, chunk48 = 2.0, 10_000, 2000, 100
var_is = (
    exp(mu * mu) * (1 + 4 * mu * mu) - mu * mu
)  # E_q[w^2 x^2] = e^(mu^2) E[(X + 2mu)^2], minus (E_q[w x])^2 = mu^2
var_snis = exp(mu * mu) * (1 + mu * mu)  # E_q[w^2 (x - mu)^2] = e^(mu^2) E[(X + mu)^2];  E_q w = 1
se_is, se_snis = sqrt(var_is / n48), sqrt(var_snis / n48)
print(
    f"(a) closed forms, from E[g(X) e^(aX)] = e^(a^2/2) E[g(X + a)] for X ~ N(0,1):  Var_q(w f) = e^(mu^2)(1 + 4 mu^2) - mu^2 = 17 e^4 - 4 = {var_is:.1f};  IS SE = sqrt({var_is:.1f}/n) = {se_is:.4f}"
)
print(
    f"(a) SNIS asymptotic variance E_q[w^2 (f - mu)^2]/(E_q w)^2 = e^(mu^2)(1 + mu^2) = 5 e^4 = {var_snis:.1f};  SNIS SE = sqrt({var_snis:.1f}/n) = {se_snis:.4f}"
)
_gx = np.linspace(-12, 12, 2_000_001)
_gw = (
    np.exp(-(_gx**2) / 2) / sqrt(2 * pi) * (_gx[1] - _gx[0])
)  # the §32 grid: deterministic quadrature under q
_w = np.exp(mu * _gx - mu * mu / 2)
print(
    f"(a) quadrature check on the §32 grid: Var_q(w f) = {float(np.sum((_w*_gx)**2*_gw)) - float(np.sum(_w*_gx*_gw))**2:.1f},  E_q[w^2 (f - mu)^2] = {float(np.sum(_w**2*(_gx - mu)**2*_gw)):.1f},  E_q[w^2] = {float(np.sum(_w**2*_gw)):.2f} (e^(mu^2) = {exp(mu*mu):.2f})"
)
ess_pop = n48 * exp(-mu * mu)
xq36 = np.random.default_rng(2026_1016_36).standard_normal(
    n48
)  # §36's draw, re-created from §36's seed (§36 itself is untouched)
w36 = np.exp(mu * xq36 - mu * mu / 2)
ess36 = w36.sum() ** 2 / (w36 * w36).sum()
bar_s, bar_p = sqrt(1 / ess36), sqrt(1 / ess_pop)
print(
    f"(b) weight-based bar sqrt(Var_p f / ESS), Var_p f = 1:  §36's sample ESS on its own draw = {ess36:.1f} -> sqrt(1/{ess36:.1f}) = {bar_s:.4f};  population ESS n/E_q[w^2] = n e^(-mu^2) = {ess_pop:.1f} -> {bar_p:.4f}"
)
print(
    f"(b) understatement: IS SE / bar = {se_is/bar_s:.2f}x (sample ESS {ess36:.1f}), {se_is/bar_p:.2f}x (population ESS {ess_pop:.1f});  SNIS SE / bar = {se_snis/bar_s:.2f}x, {se_snis/bar_p:.2f}x"
)
wf = w36 * xq36
snis36 = wf.sum() / w36.sum()
wn = w36 / w36.sum()
print(
    f"(b) from §36's single draw: IS estimate {wf.mean():.4f}, draw-based SE sd(w_i f_i)/sqrt(n) = {wf.std(ddof=1)/sqrt(n48):.4f};  SNIS estimate {snis36:.4f}, delta-method SE sqrt(sum wn_i^2 (f_i - SNIS)^2) = {sqrt(np.sum(wn**2*(xq36 - snis36)**2)):.4f}   (truth mu = {mu})"
)
is_est = np.empty(R48)
snis_est = np.empty(R48)
one_est = np.empty(R48)
ess_r = np.empty(R48)
for c in range(0, R48, chunk48):
    x = g48.standard_normal((chunk48, n48))
    w = np.exp(mu * x - mu * mu / 2)
    sw = w.sum(1)
    is_est[c : c + chunk48] = (w * x).mean(1)
    snis_est[c : c + chunk48] = (w * x).sum(1) / sw
    one_est[c : c + chunk48] = (w * 1.0).sum(1) / sw
    ess_r[c : c + chunk48] = sw**2 / (w * w).sum(1)
for name, est, se in (("IS  ", is_est, se_is), ("SNIS", snis_est, se_snis)):
    sd = est.std(ddof=1)
    print(
        f"(c) {name}, R = {R48:,} replications of n = {n48:,}: mean {est.mean():.4f} (truth {mu}, MC SE {sd/sqrt(R48):.4f}), empirical SD {sd:.4f} vs closed-form SE {se:.4f} (ratio {sd/se:.3f})"
    )
mom = [
    exp(j * (j - 1) * mu * mu / 2)
    * sum(
        comb(j, i) * (j * mu) ** (j - i) * (0 if i % 2 else math.prod(range(i - 1, 0, -2)))
        for i in range(j + 1)
    )
    for j in range(1, 5)
]  # E_q[(w x)^j] = e^(j(j-1)mu^2/2) E[(X + j mu)^j]
m4c = mom[3] - 4 * mom[2] * mom[0] + 6 * mom[1] * mom[0] ** 2 - 3 * mom[0] ** 4
kap = (m4c - 3 * var_is**2) / (n48 * var_is**2)  # exact excess kurtosis of one n-draw IS estimate
print(
    f"(c) why the empirical SDs sit below the closed forms: the IS estimate's exact excess kurtosis is kappa_4(w f)/(n Var^2) = {kap:.0f}, so an SD over R = {R48:,} replications has relative SE ~ (1/2) sqrt((kappa + 2)/R) = {0.5*sqrt((kap + 2)/R48):.1f} — the variance is carried by draws (x near 2 mu = 4 and beyond under q) that a few million samples rarely contain"
)
print(
    f"(c) sample ESS across the {R48:,} replications: median {np.median(ess_r):.1f}, 5th-95th pct {np.percentile(ess_r, 5):.1f}-{np.percentile(ess_r, 95):.1f}, fraction above the population {ess_pop:.1f} = {np.mean(ess_r > ess_pop):.4f}  — the sample ESS overstates n e^(-mu^2) because the weights that would pull it down are rarely drawn"
)
print(
    f"(d) limiting case f = 1: the SNIS estimate sum w_i / sum w_i equals 1 in every one of the {R48:,} replications (max |est - 1| = {np.abs(one_est - 1).max():.1e}) whatever the ESS ({ess_r.min():.1f} to {ess_r.max():.1f})"
)
_erfc = np.frompyfunc(math.erfc, 1, 1)


def cauchy_from_normal(x):
    """f(x) = tan(pi (Phi(x) - 1/2)): a standard Cauchy draw via its quantile; written as sign(x) / tan(pi Phi(-|x|)) for tail precision"""
    tail = 0.5 * _erfc(np.abs(x) / sqrt(2)).astype(float)  # Phi(-|x|)
    return np.sign(x) / np.tan(pi * tail)


iqr = {}
_fq = 1 / (2 * pi)  # standard Cauchy density at its quartiles ±1
se_iqr = sqrt(
    (2 * 0.25 * 0.75 - 2 * 0.25 * 0.25) / (R48 * _fq**2)
)  # asymptotic SE of the sample IQR (quartile variances minus their covariance)
for nn in (1000, 10_000):
    means = np.empty(R48)
    for c in range(0, R48, chunk48):
        means[c : c + chunk48] = cauchy_from_normal(g48.standard_normal((chunk48, nn))).mean(1)
    q25, q75 = np.percentile(means, [25, 75])
    iqr[nn] = q75 - q25
    print(
        f"(e) p = q, every w_i = 1, ESS = n = {nn:,} exactly; f(x) = tan(pi(Phi(x) - 1/2)) ~ Cauchy, Var_p f = infinity:  IQR of the {R48:,} replicated sample means = {iqr[nn]:.3f}  (median {np.median(means):+.3f};  exact: the mean of n Cauchy draws is Cauchy, IQR 2 at every n; MC SE of the IQR at R = {R48:,}: {se_iqr:.3f})"
    )
print(
    f"(e) IQR ratio n = 1,000 : 10,000 = {iqr[1000]/iqr[10_000]:.3f}   (a 1/sqrt(n) CLT bar would shrink it by sqrt(10) = {sqrt(10):.3f}): weights blameless, no bar at all"
)

# ----------------------------------------------------------------------------------------------
sec(
    49,
    "Recalibration cannot buy sharpness: the calibration-refinement split of the Brier score for a calibrated two-level report, the constant 0.5, and the canonical recalibration of a miscalibrated report",
)
g49 = np.random.default_rng(2026_1016_49)
N49 = 200_000
print(
    "(a) exact, Q uniform on {0.2, 0.8}, Y ~ Bernoulli(Q): Brier = calibration E[(q - c(q))^2] + refinement E[c(q)(1 - c(q))]"
)
for name, lv, cv in (
    ("Q itself, calibrated       ", (0.2, 0.8), (0.2, 0.8)),
    ("constant 0.5, calibrated   ", (0.5, 0.5), (0.5, 0.5)),
    ("q = 0.3 / 0.9, miscalibrated", (0.3, 0.9), (0.2, 0.8)),
    ("its recalibration c(q)     ", (0.2, 0.8), (0.2, 0.8)),
):
    cal = 0.5 * sum((l - c) ** 2 for l, c in zip(lv, cv))
    ref49 = 0.5 * sum(c * (1 - c) for c in cv)
    print(f"(a)   {name}: calibration {cal:.4f}   refinement {ref49:.4f}   Brier {cal + ref49:.4f}")
Q = np.where(g49.integers(0, 2, N49) == 0, 0.2, 0.8)
Y = (g49.uniform(0, 1, N49) < Q).astype(float)


def brier_split(q, y, cmap=None):
    """calibration and refinement terms with c estimated per level of q (or taken from cmap), the Brier score, and SEs"""
    levels = np.unique(q)
    cal = ref49 = 0.0
    cq = np.empty_like(q)
    floor = 0.0
    for l in levels:
        s = q == l
        c = y[s].mean() if cmap is None else cmap[l]
        cq[s] = c
        pi_l = s.mean()
        cal += pi_l * (l - c) ** 2
        ref49 += pi_l * c * (1 - c)
        floor += c * (1 - c) / len(y)
    b = (q - y) ** 2
    r = (cq - y) ** 2
    return cal, ref49, b.mean(), b.std(ddof=1) / sqrt(len(y)), r.std(ddof=1) / sqrt(len(y)), floor, cq


for l in (0.2, 0.8):
    s = Q == l
    c = Y[s].mean()
    print(
        f"(b) simulated N = {N49:,}: level Q = {l}: {s.sum():,} cases, empirical c = {c:.4f} (SE {sqrt(c*(1-c)/s.sum()):.4f})"
    )
for name, q in (("Q itself    ", Q), ("constant 0.5", np.full(N49, 0.5))):
    cal, ref49, b, seb, ser, floor, _ = brier_split(q, Y)
    print(
        f"(b)   {name}: calibration {cal:.2e} (sampling floor sum_l c_l(1 - c_l)/N = {floor:.1e})   refinement {ref49:.4f} (SE {ser:.4f})   Brier {b:.4f} (SE {seb:.4f})"
    )
qm = np.where(Q == 0.2, 0.3, 0.9)
cal, ref49, b, seb, ser, floor, cq = brier_split(qm, Y)
print(
    f"(c) miscalibrated q = 0.3 / 0.9 on the same N = {N49:,}: calibration {cal:.4f}   refinement {ref49:.4f} (SE {ser:.4f})   Brier {b:.4f} (SE {seb:.4f})   (exact 0.0100 / 0.1600 / 0.1700)"
)
cal2, ref2, b2, seb2, ser2, _, _ = brier_split(cq, Y)
print(
    f"(c) canonical recalibration c(q), c estimated in-sample: levels {np.unique(cq).round(4).tolist()};  calibration {cal2:.1e}   refinement {ref2:.4f} (equal to q's: {ref2 == ref49})   Brier {b2:.4f} (SE {seb2:.4f})"
)
h = N49 // 2
cfit = {l: Y[:h][qm[:h] == l].mean() for l in (0.3, 0.9)}  # fit c on the first half, evaluate on the second
qh, yh = qm[h:], Y[h:]
cal_q, ref_q, b_q, _, _, _, _ = brier_split(qh, yh)
cqh = np.where(qh == 0.3, cfit[0.3], cfit[0.9])
cal_c, ref_c, b_c, seb_c, _, _, _ = brier_split(cqh, yh)
floor_c = sum(c * (1 - c) for c in cfit.values()) * (
    1 / h + 1 / (N49 - h)
)  # sum_l pi_l c_l(1-c_l)(1/n_fit,l + 1/n_hold,l), pi_l = 1/2, n_l ~ half of each half
print(
    f"(c) held out (c fitted on the first {h:,}, scored on the other {N49 - h:,}): c = {{0.3: {cfit[0.3]:.4f}, 0.9: {cfit[0.9]:.4f}}};  q: calibration {cal_q:.4f}, refinement {ref_q:.4f}, Brier {b_q:.4f};  c(q): calibration {cal_c:.1e} (sampling-noise expectation sum_l c_l(1 - c_l)(1/h + 1/(N - h)) = {floor_c:.1e}), refinement {ref_c:.4f}, Brier {b_c:.4f} (SE {seb_c:.4f})"
)
print(
    f"(c) recalibration moved the calibration term {cal_q:.4f} -> {cal_c:.1e} and left the refinement at {ref_q:.4f} = {ref_c:.4f};  coarsening Q to the constant 0.5 instead keeps calibration and raises refinement 0.16 -> 0.25"
)

# ----------------------------------------------------------------------------------------------
sec(
    50,
    "A finite long-run variance is not a CLT: X_t = V eps_t with V in {1, 2} drawn once per run, eps_t iid N(0,1)",
)
g50 = np.random.default_rng(2026_1016_50)
# §50–§53 sit after §49 and before the final DONE marker; each owns its generator
# gNN = default_rng(2026_1016_NN) and never touches the shared `rng`, so no earlier output line depends on them.
n50, R50, chunk50, B50 = 1000, 20_000, 1000, 50
lrv50 = 0.5 * (1 + 4)  # E V^2 = (1^2 + 2^2)/2
z95 = 1.96
c50 = z95 * sqrt(lrv50)  # the nominal interval is |sqrt(n) Xbar| <= 1.96 sqrt(2.5)
miss_exact = 0.5 * math.erfc(c50 / sqrt(2)) + 0.5 * math.erfc(
    c50 / 2 / sqrt(2)
)  # P(|V Z| > c) = 1/2 P(|Z| > c) + 1/2 P(|Z| > c/2)
kurt_exact = (
    0.5 * 3 * 1**4 + 0.5 * 3 * 2**4
) / lrv50**2 - 3  # E[(VZ)^4] = E V^4 E Z^4 = (1 + 16)/2 * 3 = 25.5;  25.5/6.25 - 3
print(
    f"(a) exact: Var X_t = E V^2 = {lrv50:.4f};  Cov(X_s, X_t) = E[V^2] E[eps_s] E[eps_t] = 0 for s != t;  long-run variance sum_k Cov(X_0, X_k) = {lrv50:.4f}  (finite, and the sample autocorrelations are 0 at every lag)"
)
print(
    f"(a) but sqrt(n) Xbar_n = V (1/sqrt(n)) sum eps_t =d V Z exactly at every n: the law is the scale mixture (1/2) N(0,1) + (1/2) N(0,4), not N(0, 2.5);  its excess kurtosis is E[(VZ)^4]/(E[(VZ)^2])^2 - 3 = 25.5/6.25 - 3 = {kurt_exact:.4f}"
)
print(
    f"(a) the nominal-95% interval Xbar +- 1.96 sqrt(2.5)/sqrt(n) misses 0 with exact probability (1/2) P(|Z| > 1.96 sqrt(2.5)) + (1/2) P(|Z| > 1.96 sqrt(2.5)/2) = (1/2)({math.erfc(c50/sqrt(2)):.4f}) + (1/2)({math.erfc(c50/2/sqrt(2)):.4f}) = {miss_exact:.4f}  (nominal 0.0500)"
)
V = np.empty(R50)
sx = np.empty(R50)
ac = np.empty((R50, 3))
bm = np.empty(R50)
for c in range(0, R50, chunk50):
    v = np.where(g50.integers(0, 2, chunk50) == 0, 1.0, 2.0)
    X = v[:, None] * g50.standard_normal((chunk50, n50))
    V[c : c + chunk50] = v
    sx[c : c + chunk50] = sqrt(n50) * X.mean(1)
    Xc = X - X.mean(1, keepdims=True)
    den = (Xc * Xc).sum(1)
    for k in (1, 2, 3):
        ac[c : c + chunk50, k - 1] = (Xc[:, :-k] * Xc[:, k:]).sum(1) / den
    bm[c : c + chunk50] = B50 * X.reshape(chunk50, n50 // B50, B50).mean(2).var(
        1, ddof=1
    )  # batch-means long-run-variance estimate, 20 batches of 50
print(
    f"(b) simulated, R = {R50:,} runs of n = {n50:,} with g50: fraction of runs with V = 1: {np.mean(V == 1):.4f} (binomial SE {sqrt(0.25/R50):.4f}; exact 0.5000)"
)
print(
    f"(b) sample autocorrelation pooled across runs: "
    + ",  ".join(
        f"lag {k}: {ac[:, k-1].mean():+.5f} ± {ac[:, k-1].std(ddof=1)/sqrt(R50):.5f}" for k in (1, 2, 3)
    )
    + f"   (exact 0 at every lag; the finite-n bias of the estimator is -1/n = {-1/n50:+.5f})"
)
print(
    f"(b) variance of sqrt(n) Xbar_n across runs: {sx.var(ddof=1):.4f} (SE {se_var(sx):.4f})   vs the long-run variance {lrv50:.4f}:  the variance is right"
)
print(
    f"(b) excess kurtosis of sqrt(n) Xbar_n across runs: {np.mean((sx - sx.mean())**4)/sx.var()**2 - 3:+.4f} (delete-one jackknife SE {jackknife_kurt(sx):.4f})   vs the exact scale-mixture value {kurt_exact:+.4f} (a Gaussian limit would give 0):  the shape is not"
)
miss = np.mean(np.abs(sx) > c50)
print(
    f"(b) fraction of runs where the nominal-95% interval misses 0: {miss:.4f} (binomial SE {sqrt(miss*(1-miss)/R50):.4f})   vs exact {miss_exact:.4f} from (a), nominal 0.0500"
)
for vv in (1.0, 2.0):
    s = V == vv
    print(
        f"(c) batch-means long-run-variance estimate from within each run (20 batches of 50): among the {s.sum():,} runs with V = {vv:.0f}: mean {bm[s].mean():.4f} (SE {bm[s].std(ddof=1)/sqrt(s.sum()):.4f}), 5th-95th pct {np.percentile(bm[s], 5):.3f}-{np.percentile(bm[s], 95):.3f}   (it estimates V^2 = {vv*vv:.0f}, the run's own conditional variance)"
    )
print(
    f"(c) the mean of the estimate over all runs is {bm.mean():.4f} (SE {bm.std(ddof=1)/sqrt(R50):.4f}) = E V^2 = 2.5, but the fraction of runs whose estimate lies within 10% of 2.5 ([2.25, 2.75]) is {np.mean((bm >= 2.25) & (bm <= 2.75)):.4f}:  no single run sees the long-run variance 2.5 — the within-run estimator cannot average over V, which is drawn once"
)

# ----------------------------------------------------------------------------------------------
sec(
    51,
    "The differential value's infinite sum need not converge: the alternating two-state chain P = [[0,1],[1,0]], r = (1, -1)",
)
# deterministic — no randomness anywhere in this section
P51 = np.array([[0.0, 1.0], [1.0, 0.0]])
r51 = np.array([1.0, -1.0])
w, vecs = np.linalg.eig(P51.T)
pi51 = np.real(vecs[:, np.argmin(np.abs(w - 1))])
pi51 = pi51 / pi51.sum()
J51 = float(pi51 @ r51)
print(
    f"(a) stationary law pi = {pi51.round(4).tolist()} (left eigenvector of P at eigenvalue 1), gain J = pi . r = {J51:+.4f};  period 2: eigenvalues of P are {np.sort(np.linalg.eigvals(P51).real).tolist()}"
)
e0 = np.array([1.0, 0.0])
law = e0.copy()
S = []
acc = 0.0
for t in range(1000):
    acc += float(law @ r51) - J51
    S.append(acc)
    law = law @ P51
S = np.array(S)
print(
    f"(a) centred partial sums S_t = sum_(k=0)^t (E_0 r(X_k) - J) from state 0, t = 0..9: {[int(round(s)) for s in S[:10]]}  — the sequence alternates 1, 0, 1, 0, ... and never converges"
)
print(
    f"(a) Cesaro average (1/T) sum_(t<T) S_t: "
    + ",  ".join(f"T = {T}: {S[:T].mean():.4f}" for T in (10, 100, 1000))
    + "   -> 0.5, the Cesaro limit exists although the sum does not"
)
A51 = np.vstack(
    [(np.eye(2) - P51)[0], [0.0, 1.0]]
)  # Poisson equation (I - P) h = r - J, with the normalisation h(1) = 0 replacing the redundant second row
h51 = np.linalg.solve(A51, np.array([r51[0] - J51, 0.0]))
print(
    f"(a) Poisson equation h = r - J + P h with h(1) = 0: h = {h51.round(4).tolist()}, h(0) = {h51[0]:.4f};  residual ||(I - P) h - (r - J)||_inf = {np.abs((np.eye(2) - P51) @ h51 - (r51 - J51)).max():.1e}  — h exists and is unique up to constants even though the series for it diverges"
)
print(
    f"(a) time-t law from state 0 and its TV distance to pi: "
    + ",  ".join(
        f"t = {t}: {(e0 @ np.linalg.matrix_power(P51, t)).round(1).tolist()} TV {0.5*np.abs(e0 @ np.linalg.matrix_power(P51, t) - pi51).sum():.4f}"
        for t in range(6)
    )
    + "   (TV = 0.5 at every t: the chain never mixes)"
)
occ = lambda T: np.mean([e0 @ np.linalg.matrix_power(P51, t) for t in range(T)], axis=0)
print(
    f"(a) time-average occupation (1/T) sum_(t<T) P^t(0, .) and its TV distance to pi: "
    + ",  ".join(f"T = {T}: TV {0.5*np.abs(occ(T) - pi51).sum():.4f}" for T in (10, 11, 1000, 1001))
    + "   (-> 0: the time average converges although the time-t law does not)"
)
P51b = np.array([[0.9, 0.1], [0.1, 0.9]])
w, vecs = np.linalg.eig(P51b.T)
pib = np.real(vecs[:, np.argmin(np.abs(w - 1))])
pib = pib / pib.sum()
Jb = float(pib @ r51)
law = e0.copy()
acc = 0.0
Sb = []
for t in range(51):
    acc += float(law @ r51) - Jb
    Sb.append(acc)
    law = law @ P51b
hb = np.linalg.solve(np.vstack([(np.eye(2) - P51b)[0], [0.0, 1.0]]), np.array([r51[0] - Jb, 0.0]))
print(
    f"(b) aperiodic contrast P = [[0.9, 0.1], [0.1, 0.9]], same r: pi = {pib.round(4).tolist()}, J = {Jb:+.4f}, second eigenvalue {0.9 - 0.1:.1f};  partial sums from state 0: "
    + ",  ".join(f"t = {t}: {Sb[t]:.4f}" for t in (0, 10, 50))
    + f"   (E_0 r(X_k) - J = 0.8^k, so S_t = (1 - 0.8^(t+1))/0.2 -> {1/0.2:.4f})"
)
print(
    f"(b) Poisson solution with h(1) = 0: h = {hb.round(4).tolist()};  h(0) - h(1) = {hb[0] - hb[1]:.4f}  [algebra: h(0) - h(1) = (r(0) - r(1)) + (0.9 - 0.1)(h(0) - h(1))  =>  0.2 (h(0) - h(1)) = 2  =>  {2/0.2:.4f}];  the convergent sum's limit is h(0) - pi . h = {hb[0] - float(pib @ hb):.4f}, and |S_50 - 5| = {abs(Sb[50] - 5):.1e} = 0.8^51/0.2"
)
from math import comb as _comb

P51c = np.full((2, 2), 0.5)  # mixes in ONE step: P^t = pi for every t >= 1
w, vecs = np.linalg.eig(P51c.T)
pic = np.real(vecs[:, np.argmin(np.abs(w - 1))])
pic = pic / pic.sum()
Jc = float(pic @ r51)
ES = lambda s0, T: sum(
    float((np.eye(2)[s0] @ np.linalg.matrix_power(P51c, t)) @ r51) - Jc for t in range(T)
)  # E_s0 [ sum_(t<T) (r(X_t) - J) ]
hc = np.linalg.solve(np.vstack([(np.eye(2) - P51c)[0], [0.0, 1.0]]), np.array([r51[0] - Jc, 0.0]))
print(
    f"(c) one-step mixing P = [[0.5, 0.5], [0.5, 0.5]], same r: pi = {pic.round(4).tolist()}, J = {Jc:+.4f}, second eigenvalue {np.sort(np.abs(np.linalg.eigvals(P51c)))[0]:.1f};  E_0 [sum_(t<T) (r(X_t) - J)] = "
    + ",  ".join(f"T = {T}: {ES(0, T):.4f}" for T in (1, 10, 100, 1000))
    + "   (the limit of expected finite-horizon sums exists and is 1: only the first term is off the mean)"
)
print(
    f"(c) but the PATH's partial sum never converges: for t >= 1 each r(X_t) is +1 or -1 with probability 1/2, independently of the past, so S_T - 1 is a simple random walk — Var_0 S_T = T - 1 = "
    + ", ".join(f"{T - 1}" for T in (10, 100, 1000))
    + " at T = 10, 100, 1000, and P_0(S_T = 1) = "
    + ", ".join(f"{_comb(T - 1, (T - 1)//2)/2**(T - 1):.4f}" for T in (11, 101, 1001))
    + " at T = 11, 101, 1001 -> 0   (an a.s. limit would need the increments to vanish; they are +-1 forever)"
)
print(
    f"(c) Poisson solution with h(1) = 0: h = {hc.round(4).tolist()};  h(0) - h(1) = {hc[0] - hc[1]:.4f} = lim_T E_0[sum] - lim_T E_1[sum] = {ES(0, 1000):.4f} - ({ES(1, 1000):+.4f})   — the differential value is the limit of expected finite-horizon sums, not the expectation of a convergent series"
)

# ----------------------------------------------------------------------------------------------
sec(
    52,
    "Correlated maxima and a shifted Cauchy: E max of two correlated normals, of 100 equicorrelated normals, and a W1 that exists when no moment does",
)
g52 = np.random.default_rng(2026_1016_52)
rhos52 = (-1.0, -0.5, 0.0, 0.5, 1.0)
R52a, R52c, m52 = 200_000, 20_000, 100
emax2 = lambda rho: sqrt(
    (1 - rho) / pi
)  # max(Z1, Z2) = (Z1 + Z2)/2 + |Z1 - Z2|/2, Z1 - Z2 ~ N(0, 2(1 - rho))
print(
    f"(a) exact E max(Z1, Z2), Z1, Z2 standard normal with correlation rho: sigma sqrt((1 - rho)/pi) = E|Z1 - Z2|/2;  "
    + ",  ".join(f"rho = {rho:+.1f}: {emax2(rho):.4f}" for rho in rhos52)
)
for rho in rhos52:
    z1 = g52.standard_normal(R52a)
    z2 = rho * z1 + sqrt(1 - rho * rho) * g52.standard_normal(R52a)
    mx = np.maximum(z1, z2)
    print(
        f"(b) Monte Carlo, R = {R52a:,} pairs, rho = {rho:+.1f}: mean max {mx.mean():.4f} ± {mx.std(ddof=1)/sqrt(R52a):.4f}   (exact {emax2(rho):.4f})"
    )
_x = np.linspace(-12, 12, 2_400_001)
_dx = _x[1] - _x[0]
_phi = np.exp(-(_x**2) / 2) / sqrt(2 * pi)
_Phi = 0.5 * np.frompyfunc(math.erfc, 1, 1)(-_x / sqrt(2)).astype(float)
emax100 = float(
    np.sum(_x * m52 * _phi * _Phi ** (m52 - 1)) * _dx
)  # E max of m iid normals = int x m phi(x) Phi(x)^(m-1) dx
print(
    f"(c) m = {m52} iid standard normals: E max = int x m phi(x) Phi(x)^(m-1) dx = {emax100:.4f} (trapezoid on [-12, 12], step {_dx:.0e});  sqrt(2 ln m) = {sqrt(2*log(m52)):.4f} for scale"
)
for rho in (0.0, 0.5, 0.9):
    Wc = g52.standard_normal((R52c, 1))
    E = g52.standard_normal((R52c, m52))
    mx = (sqrt(rho) * Wc + sqrt(1 - rho) * E).max(1)
    print(
        f"(c) m = {m52} scores with common correlation rho = {rho:.1f} (Z_j = sqrt(rho) W + sqrt(1 - rho) e_j), R = {R52c:,}: E max = {mx.mean():.4f} ± {mx.std(ddof=1)/sqrt(R52c):.4f}   (exact: sqrt(rho) E W + sqrt(1 - rho) E max_j e_j = sqrt(1 - rho) x {emax100:.4f} = {sqrt(1 - rho)*emax100:.4f};  independent value {emax100:.4f})"
    )
z1 = g52.standard_normal(R52c)
z2 = -0.5 * z1 + sqrt(1 - 0.25) * g52.standard_normal(R52c)
print(
    f"(c) m = 2, rho = -0.5 (the common-correlation construction needs rho >= -1/(m - 1): -1 for m = 2, -1/99 for m = 100), R = {R52c:,}: E max = {np.maximum(z1, z2).mean():.4f} ± {np.maximum(z1, z2).std(ddof=1)/sqrt(R52c):.4f}   (exact {emax2(-0.5):.4f}, as in (a))"
)
# (d) shifted Cauchy: W1(Cauchy(0,1), Cauchy(1,1)) = int |F_P - F_Q| dx; F_P - F_Q = (arctan x - arctan(x - 1))/pi >= 0 integrates to the shift, 1
xin = np.linspace(-50, 50, 1_000_001)
xtail = np.logspace(log(50) / log(10), 7, 200_001)[1:]
xg = np.concatenate([-xtail[::-1], xin, xtail])
dF = (np.arctan(xg) - np.arctan(xg - 1)) / pi
w1_num = float(np.trapezoid(np.abs(dF), xg))
print(
    f"(d) shifted Cauchy, P = Cauchy(0,1), Q = Cauchy(1,1): F_P - F_Q = (arctan x - arctan(x - 1))/pi >= 0 everywhere (min on the grid {dF.min():.1e});  W1 = int |F_P - F_Q| dx on [-1e7, 1e7] = {w1_num:.4f}  (exact 1: a shift by c costs exactly c, whatever the tails; the mass beyond ±1e7 is 2/(pi 1e7) = {2/(pi*1e7):.1e})"
)
cx = g52.standard_cauchy(1_000_000)
run = np.cumsum(np.abs(cx))
print(
    f"(d) sample first absolute moment (1/n) sum |X_i| of one Cauchy sample path from g52, at n = 10^3, 10^4, 10^5, 10^6: "
    + ",  ".join(f"{run[nn-1]/nn:.3f}" for nn in (1000, 10_000, 100_000, 1_000_000))
    + "   — a single path, no SE: E|X| = infinity, so the running mean has no limit; its typical size is (2/pi) ln n = "
    + ", ".join(f"{2/pi*log(nn):.1f}" for nn in (1000, 10_000, 100_000, 1_000_000))
    + " at these n, with fluctuations of the same order"
)
ey = sqrt(2 / pi)
print(
    f"(d) Cauchy vs N(0,1): W1(P, Q) >= |E_P|X| - E_Q|Y|| for any coupling, and E_Q|Y| = sqrt(2/pi) = {ey:.4f};  lower bound on any coupling's cost, from a truncated sample (the same running means minus {ey:.4f}): "
    + ",  ".join(
        f"n = 10^{int(round(log(nn, 10)))}: {run[nn-1]/nn - ey:.3f}"
        for nn in (1000, 10_000, 100_000, 1_000_000)
    )
    + "   — the bound does not settle and is unbounded in n: W1(Cauchy, N(0,1)) = infinity while W1(Cauchy(0,1), Cauchy(1,1)) = 1"
)

# ----------------------------------------------------------------------------------------------
sec(
    53,
    "An empirical RMSE of an infinite-variance mean (20 batches of the §39 experiment), and the rank of a sample covariance at n < d",
)
g53 = np.random.default_rng(2026_1016_53)
alpha53, n53, reps53, blocks53, nb53 = 1.5, 10_000, 1000, 20, 20
true_mean53 = alpha53 / (alpha53 - 1)
print(
    f"(a) Pareto(alpha = {alpha53}), x_min = 1, as §39 (U^(-1/alpha)): true mean {true_mean53:.4f}, Var = +inf (alpha <= 2), so the POPULATION RMSE of the sample mean is +inf at every n; the numbers below are statistics of the simulation ({nb53} independent batches of {reps53:,} replications of n = {n53:,}, g53)"
)
rm_sm = np.empty(nb53)
rm_mom = np.empty(nb53)
p99_sm = np.empty(nb53)
p99_mom = np.empty(nb53)
for b in range(nb53):
    par = g53.uniform(0, 1, (reps53, n53)) ** (-1 / alpha53)
    e_sm = par.mean(1) - true_mean53
    e_mom = np.median(par.reshape(reps53, blocks53, n53 // blocks53).mean(2), axis=1) - true_mean53
    rm_sm[b], rm_mom[b] = sqrt(np.mean(e_sm**2)), sqrt(np.mean(e_mom**2))
    p99_sm[b], p99_mom[b] = np.percentile(np.abs(e_sm), 99), np.percentile(np.abs(e_mom), 99)
    print(
        f"(a) batch {b+1:>2}: empirical RMSE of the sample mean {rm_sm[b]:.4f}, of median-of-means ({blocks53} blocks of {n53//blocks53}) {rm_mom[b]:.4f};  99th pct |error|: {p99_sm[b]:.4f} / {p99_mom[b]:.4f};  max |error| of the sample mean {np.abs(e_sm).max():.3f}"
    )
print(
    f"(a) across the {nb53} batches — sample mean RMSE: min {rm_sm.min():.4f}, median {np.median(rm_sm):.4f}, max {rm_sm.max():.4f} (max/min {rm_sm.max()/rm_sm.min():.2f}x);  median-of-means RMSE: min {rm_mom.min():.4f}, median {np.median(rm_mom):.4f}, max {rm_mom.max():.4f} (max/min {rm_mom.max()/rm_mom.min():.2f}x)"
)
print(
    f"(a) 99th-percentile |error| across batches — sample mean: min {p99_sm.min():.4f}, max {p99_sm.max():.4f};  median-of-means: min {p99_mom.min():.4f}, max {p99_mom.max():.4f}   (the empirical RMSE of the sample mean is driven by the largest single draw in each batch and does not settle; the median-of-means statistics do)"
)
d53 = 256
for n in (64, 256, 1024):
    G = g53.standard_normal((n, d53))
    ev = np.linalg.eigvalsh(G.T @ G / n)
    gam = d53 / n
    lo, hi = (1 - sqrt(gam)) ** 2, (1 + sqrt(gam)) ** 2
    nz = ev[ev > 1e-10]
    op = np.abs(ev - 1).max()
    print(
        f"(b) d = {d53}, n = {n:>4} (d/n = {gam:g}), single draw (no SE): eigenvalues of (1/n) X^T X below 1e-10: {int(np.sum(ev <= 1e-10)):>3} (exactly max(d - n, 0) = {max(d53 - n, 0)});  smallest nonzero {nz.min():.4e}, largest {ev.max():.4f}   vs the Marchenko-Pastur edges (1 -+ sqrt(d/n))^2 = {lo:.4f}, {hi:.4f}"
        + (
            f" (at d = n the lower edge is 0 and the smallest eigenvalue is of order 1/d^2 = {1/d53**2:.1e})"
            if n == d53
            else ""
        )
        + f";  ||Sigma_hat - I||_op = {op:.4f} vs 2 sqrt(d/n) + d/n = {2*sqrt(gam) + gam:.4f}"
    )

print("\nDONE (§40–§53)")
