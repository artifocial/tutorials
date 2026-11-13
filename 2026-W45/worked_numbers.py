#!/usr/bin/env python3
"""Worked numbers for "The Optimization Domain" (W45 trend tutorial).

Companion script to the tutorial and to the two notebooks in this folder. Every
value printed below is one the article quotes in a code block, and the article
cites this file by section number ("worked_numbers.py §N"), so the numbering is
fixed: 53 sections, each opened by ``sec(N, title)`` (§42–§53 are appended after §41, so the stream §1–§41 draw is unchanged; the norm lines at the end of §14 draw nothing).

Pure NumPy, float64, CPU-only. All sections draw in order from one generator
seeded 20261105 (the W45 trend slot, Thu Nov 5 2026); a section that draws
nothing says so. ``rng_states.json`` records the generator state at the start
of every section (captured by ``capture_rng_states.py``) so the notebooks can
run any section on its own. Run with

    python worked_numbers.py > worked_numbers_output.txt

and diff against the committed output file. Closed-form lines are re-derived
by ``test_worked_numbers.py``.
"""

import numpy as np, math, sys, time
sys.stdout.reconfigure(line_buffering=True)
from math import sqrt, log, exp, pi

rng = np.random.default_rng(20261105)
np.set_printoptions(precision=4, suppress=True)
T0 = time.time()


def sec(n, title):
    print(f"\n=== §{n} {title} ===")


def se_mean(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / sqrt(len(a))


def se_prop(k, n):
    p = k / n
    return sqrt(p * (1 - p) / n)


# ----------------------------------------------------------------------------------------------
# PART I — the puzzles (§0 of the article)
# ----------------------------------------------------------------------------------------------

sec(1, "Puzzle 1: one step size, two verdicts — GD on a quadratic at eta = 0.5/L, 1.9/L, 2.1/L")
# f(x) = 1/2 x^T A x, A diagonal with eigenvalues log-spaced in [1, 100]: L = 100, mu = 1, kappa = 100.
lam = np.logspace(0, 2, 20)
L, mu = lam.max(), lam.min()
x0 = np.ones(20)
for c in (0.5, 1.0, 1.9, 1.99, 2.01, 2.1):
    eta = c / L
    x = x0.copy()
    for t in range(200):
        x = x - eta * lam * x
    f = 0.5 * np.sum(lam * x * x)
    rho = max(abs(1 - eta * lam))
    print(f"  eta = {c:4.2f}/L  f(x_200) = {f:12.4e}   contraction factor max|1-eta*lambda| = {rho:.4f}")
print(f"  L = {L:.1f}, mu = {mu:.1f}, kappa = {L/mu:.0f}; the boundary is eta = 2/L = {2/L:.4f}")

sec(2, "Puzzle 2: same loss, zero training error twice, two different models (d > n least squares)")
n, d = 40, 200
X = rng.standard_normal((n, d))
w_star = np.zeros(d); w_star[:10] = rng.standard_normal(10) * 2   # sparse truth
y = X @ w_star
Xte = rng.standard_normal((2000, d)); yte = Xte @ w_star
def run(update, steps, eta):
    w = np.zeros(d)
    for t in range(steps):
        g = X.T @ (X @ w - y) / n
        w = w - eta * update(g, t)
    return w
w_gd = run(lambda g, t: g, 20000, 0.05)
# sign descent (Adam with beta1 = beta2 = 0, eps -> 0, is sign descent)
w_sign = run(lambda g, t: np.sign(g), 20000, 0.0005)
w_mn = np.linalg.lstsq(X, y, rcond=None)[0]            # minimum-norm interpolant
for name, w in (("GD from 0", w_gd), ("sign descent from 0", w_sign), ("min-norm (pinv)", w_mn)):
    tr = np.mean((X @ w - y) ** 2); te = np.mean((Xte @ w - yte) ** 2)
    print(f"  {name:20s} train MSE = {tr:9.2e}  test MSE = {te:8.4f}  ||w||_2 = {np.linalg.norm(w):7.3f}  ||w||_1 = {np.linalg.norm(w,1):8.2f}  ||w||_inf = {np.abs(w).max():.3f}")
print(f"  ||w_GD - w_minnorm||_2 = {np.linalg.norm(w_gd - w_mn):.2e}   (GD from zero converges to the min-norm interpolant)")

sec(3, "Puzzle 3: a non-convex problem that gradient descent solves from any random start (rank-1 matrix sensing)")
# M* = u u^T with u in R^10; m = 200 Gaussian measurements y_i = <A_i, M*>; f(x) = 1/(4m) sum (<A_i, x x^T> - y_i)^2
k, m = 10, 200
u = rng.standard_normal(k); u /= np.linalg.norm(u)
A = rng.standard_normal((m, k, k)); A = (A + A.transpose(0, 2, 1)) / sqrt(2)
yv = np.einsum('mij,i,j->m', A, u, u)
def f_ms(x):
    r = np.einsum('mij,i,j->m', A, x, x) - yv
    return np.sum(r * r) / (4 * m)
def grad_ms(x):
    r = np.einsum('mij,i,j->m', A, x, x) - yv
    return np.einsum('m,mij,j->i', r, A, x) / m
succ = 0; its = []
for trial in range(100):
    x = rng.standard_normal(k) * 0.1
    for t in range(20000):
        x = x - 0.1 * grad_ms(x)
        if f_ms(x) < 1e-12:
            break
    ok = min(np.linalg.norm(x - u), np.linalg.norm(x + u)) < 1e-4
    succ += ok; its.append(t + 1)
print(f"  100 random starts (scale 0.1), step 0.1: reached a global minimiser (x = +/-u to 1e-4) in {succ}/100 runs; steps to f < 1e-12: median {int(np.median(its))}, max {max(its)}")
H0 = -np.einsum('m,mij->ij', yv, A) / m           # Hessian at x = 0 is -(1/m) sum y_i A_i (a strict saddle: negative eigenvalue)
ev = np.linalg.eigvalsh(H0)
print(f"  Hessian at the saddle x = 0: lambda_min = {ev[0]:.4f}, lambda_max = {ev[-1]:.4f}  (lambda_min < 0: x = 0 is a strict saddle, not a minimum)")
x = np.zeros(k)
for t in range(20000):
    x = x - 0.1 * grad_ms(x)
print(f"  started exactly at x = 0: f after 20000 steps = {f_ms(x):.4e}  (gradient descent never leaves a stationary point)")

# ----------------------------------------------------------------------------------------------
# PART II — convexity, duality, rates (article §1–§3)
# ----------------------------------------------------------------------------------------------

sec(4, "Convexity is checked on chords: Jensen on three functions, 10,000 random chords each")
tests = {"x^2": lambda x: x * x, "exp(x)": np.exp, "log-sum-exp(x, 0)": lambda x: np.logaddexp(x, 0.0), "sin(x)": np.sin, "sqrt|x|": lambda x: np.sqrt(np.abs(x))}
for name, g in tests.items():
    a = rng.uniform(-3, 3, 10000); b = rng.uniform(-3, 3, 10000); th = rng.uniform(0, 1, 10000)
    lhs = g(th * a + (1 - th) * b); rhs = th * g(a) + (1 - th) * g(b)
    viol = np.mean(lhs > rhs + 1e-12)
    print(f"  {name:18s} fraction of chords with f(theta a+(1-theta) b) > theta f(a)+(1-theta) f(b): {viol:.4f}")

sec(5, "Weak and strong duality on a bias-free soft-margin SVM: primal and dual objectives meet; KKT names the support vectors")
n, d = 60, 2
Xs = rng.standard_normal((n, d)); ys = np.where(Xs[:, 0] + 0.5 * Xs[:, 1] + 0.3 * rng.standard_normal(n) > 0, 1.0, -1.0)
C = 1.0
Q = (ys[:, None] * Xs) @ (ys[:, None] * Xs).T
# dual: max 1^T a - 1/2 a^T Q a, 0 <= a <= C  (projected gradient ascent)
a = np.zeros(n); step = 1.0 / np.linalg.eigvalsh(Q).max()
def primal(w):
    return 0.5 * w @ w + C * np.sum(np.maximum(0, 1 - ys * (Xs @ w)))
def dual(a):
    return a.sum() - 0.5 * a @ Q @ a
for t in range(1, 20001):
    a = np.clip(a + step * (1 - Q @ a), 0, C)
    if t in (1, 10, 100, 1000, 20000):
        w = (a * ys) @ Xs
        print(f"  iter {t:6d}: dual = {dual(a):.6f}   primal(w(a)) = {primal(w):.6f}   gap = {primal(w)-dual(a):.2e}")
w = (a * ys) @ Xs
marg = ys * (Xs @ w)
print(f"  support vectors (alpha > 1e-6): {np.sum(a > 1e-6)}/{n};  alpha = C (inside margin or misclassified): {np.sum(a > C - 1e-6)};  0 < alpha < C (exactly on the margin, y w.x = 1): {np.sum((a > 1e-6) & (a < C - 1e-6))}")
on = (a > 1e-6) & (a < C - 1e-6)
print(f"  max |y w.x - 1| over the on-margin set = {np.abs(marg[on]-1).max():.2e};  min y w.x over alpha = 0 points = {marg[a<=1e-6].min():.4f} (>= 1: complementary slackness)")

sec(6, "Fenchel conjugate of log-sum-exp is negative entropy; the KL-penalised objective has a Gibbs closed form (W41 bridge)")
th = np.array([1.0, 0.0, -1.0, 2.0])
A_th = np.log(np.sum(np.exp(th)))
p_soft = np.exp(th - A_th)
# maximise <theta, p> + H(p) on the simplex by exponentiated gradient (mirror descent with KL)
p = np.ones(4) / 4
for t in range(200):
    g = th - np.log(p) - 1                       # gradient of <theta,p> + H(p)
    p = p * np.exp(0.5 * g); p /= p.sum()
val = th @ p - np.sum(p * np.log(p))
print(f"  A(theta) = log sum exp(theta) = {A_th:.6f};  max_p <theta,p> + H(p) by exponentiated gradient = {val:.6f};  argmax p = {p};  softmax(theta) = {p_soft}")
# RLHF-style: max_p <r,p> - beta KL(p||pi) = beta log E_pi exp(r/beta), argmax = pi exp(r/beta)/Z
r = np.array([2.0, 0.5, -1.0, 1.0]); pi0 = np.array([0.4, 0.3, 0.2, 0.1]); beta = 0.7
p = pi0.copy()
for t in range(400):
    g = r - beta * (np.log(p / pi0) + 1)
    p = p * np.exp(0.3 * g); p /= p.sum()
p_gibbs = pi0 * np.exp(r / beta); Z = p_gibbs.sum(); p_gibbs /= Z
print(f"  KL-penalised argmax by EG = {p}; closed form pi exp(r/beta)/Z = {p_gibbs}; optimal value {r @ p - beta*np.sum(p*np.log(p/pi0)):.6f} vs beta log Z = {beta*log(Z):.6f}")

sec(7, "Rates on a quadratic: GD, heavy ball and Nesterov at kappa = 10, 100, 1000, 10,000 — iterations to f - f* < 1e-8, against the theorems' upper bounds")
d = 50
print("  kappa     GD iters   GD bound (1-1/kappa)^2t    HB iters   Nesterov iters   NAG bound (1-1/sqrt kappa)^t")
for kappa in (10, 100, 1000, 10000):
    lam = np.logspace(0, log(kappa, 10), d)       # eigenvalues from 1 to kappa; L = kappa, mu = 1
    L, mu = lam.max(), lam.min()
    x0 = np.ones(d) / sqrt(d)
    f0 = 0.5 * np.sum(lam * x0 * x0)
    def iters(method):
        x = x0.copy(); xp = x0.copy()
        for t in range(1, 400001):
            if method == "GD":
                x = x - (1 / L) * lam * x
            elif method == "HB":                     # Polyak's optimal constants for a quadratic
                a = 4 / (sqrt(L) + sqrt(mu)) ** 2; b = ((sqrt(kappa) - 1) / (sqrt(kappa) + 1)) ** 2
                xn = x - a * lam * x + b * (x - xp); xp, x = x, xn
            else:                                    # Nesterov, strongly convex constants
                yv_ = x + ((sqrt(kappa) - 1) / (sqrt(kappa) + 1)) * (x - xp)
                xn = yv_ - (1 / L) * lam * yv_; xp, x = x, xn
            if 0.5 * np.sum(lam * x * x) < 1e-8:
                return t
        return t
    tg, th_, tn = iters("GD"), iters("HB"), iters("NAG")
    thy_gd = log(1e-8 / f0) / (2 * log(1 - 1 / kappa))          # f contracts by (1-1/kappa)^2 per step (distance by 1-1/kappa)
    thy_nag = log(1e-8 / (2 * f0)) / log(1 - 1 / sqrt(kappa))   # f(x_t)-f* <= (1-1/sqrt kappa)^t (f0-f* + mu/2 ||x0-x*||^2) <= 2 f0 (...)^t here
    print(f"  {kappa:6d}  {tg:9d}  {thy_gd:14.0f}               {th_:8d}   {tn:12d}      {thy_nag:14.0f}")
print("  (a bound is an upper bound: the worst direction holds only 1/d of the initial energy here, so GD beats its bound by a factor ~1.5)")
sec(8, "Nesterov's worst-case function f_{2t+1}: no first-order method beats 3 L ||x0 - x*||^2 / (32 (t+1)^2) at step t, and Nesterov's method is within a constant of it")
L = 1.0
def make(d):
    def f_nest(x):
        return L / 4 * (0.5 * (x[0] ** 2 + np.sum((x[:-1] - x[1:]) ** 2) + x[-1] ** 2) - x[0])
    def g_nest(x):
        g = np.zeros_like(x); dif = x[:-1] - x[1:]
        g[0] += x[0]; g[:-1] += dif; g[1:] -= dif; g[-1] += x[-1]; g[0] -= 1
        return L / 4 * g
    xstar = 1 - np.arange(1, d + 1) / (d + 1)
    return f_nest, g_nest, xstar
print("     t    d=2t+1   GD gap f(x_t)-f*   NAG gap     lower bound 3LR^2/(32(t+1)^2)   GD/bound   NAG/bound   NAG upper bound 4LR^2/(t+2)^2")
for t in (1, 2, 5, 10, 20, 50, 100):
    d = 2 * t + 1
    f_nest, g_nest, xstar = make(d); fstar = f_nest(xstar); R2 = np.sum(xstar ** 2)
    x = np.zeros(d)
    for s in range(t):
        x = x - (1 / L) * g_nest(x)
    gd_gap = f_nest(x) - fstar
    x = np.zeros(d); yk = x.copy(); lam_k = 0.0        # Nesterov's method (Beck–Teboulle / FISTA form), constant step 1/L
    for s in range(t):
        xn = yk - (1 / L) * g_nest(yk); lam_n = (1 + sqrt(1 + 4 * lam_k * lam_k)) / 2
        yk = xn + ((lam_k - 1) / lam_n) * (xn - x); x, lam_k = xn, lam_n
    nag_gap = f_nest(x) - fstar
    lb = 3 * L * R2 / (32 * (t + 1) ** 2); ub = 4 * L * R2 / (t + 2) ** 2
    print(f"  {t:4d}   {d:5d}   {gd_gap:14.6f}   {nag_gap:10.6f}        {lb:12.6f}             {gd_gap/lb:6.2f}      {nag_gap/lb:6.2f}      {ub:12.6f}")
print("  (the function depends on t: at step t the method has only touched coordinates 1..t of a 2t+1-dimensional chain, whatever it did)")
sec(9, "Nonsmooth: subgradient O(1/sqrt t) vs proximal gradient O(1/t) vs FISTA O(1/t^2) on a lasso problem")
n, d = 100, 50
Xl = rng.standard_normal((n, d)); wl = np.zeros(d); wl[:5] = (1, -2, 1.5, -1, 0.5)
yl = Xl @ wl + 0.1 * rng.standard_normal(n); lam1 = 0.1
Ll = np.linalg.eigvalsh(Xl.T @ Xl / n).max()
def F(w): return 0.5 * np.mean((Xl @ w - yl) ** 2) + lam1 * np.abs(w).sum()
def gsm(w): return Xl.T @ (Xl @ w - yl) / n
def prox(v, t): return np.sign(v) * np.maximum(np.abs(v) - t, 0)
# reference optimum: 20,000 FISTA iterations
w = np.zeros(d); z = w.copy(); tk = 1.0
for t in range(20000):
    wn = prox(z - gsm(z) / Ll, lam1 / Ll); tn = (1 + sqrt(1 + 4 * tk * tk)) / 2
    z = wn + (tk - 1) / tn * (wn - w); w, tk = wn, tn
Fstar = F(w)
res = {}
w_sg = np.zeros(d); best = np.inf; w_pg = np.zeros(d); w_fi = np.zeros(d); z = w_fi.copy(); tk = 1.0
for t in range(1, 2001):
    w_sg = w_sg - (0.5 / sqrt(t)) * (gsm(w_sg) + lam1 * np.sign(w_sg)); best = min(best, F(w_sg))
    w_pg = prox(w_pg - gsm(w_pg) / Ll, lam1 / Ll)
    wn = prox(z - gsm(z) / Ll, lam1 / Ll); tn = (1 + sqrt(1 + 4 * tk * tk)) / 2
    z = wn + (tk - 1) / tn * (wn - w_fi); w_fi, tk = wn, tn
    if t in (10, 100, 1000, 2000):
        print(f"  t = {t:5d}: subgradient (best) {best-Fstar:10.3e}   proximal {F(w_pg)-Fstar:10.3e}   FISTA {F(w_fi)-Fstar:10.3e}")
print(f"  support of the lasso solution: {np.sum(np.abs(w) > 1e-8)} nonzeros of {d} (truth 5); L = {Ll:.3f}")

# ----------------------------------------------------------------------------------------------
# PART III — beyond Euclid (article §4)
# ----------------------------------------------------------------------------------------------

sec(10, "Mirror descent on the simplex: exponentiated gradient's log d against projected gradient's sqrt d — the step the theory licenses, d = 10 .. 100,000")
def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u); rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    th = (css[rho] - 1) / (rho + 1.0); return np.maximum(v - th, 0)
T = 500
print(f"  T = {T} rounds, losses in [0,1]^d, expert 0 better by 0.1 on average. Regret per round, and the mass on expert 0 after T rounds:")
print("      d     EG eta=sqrt(2 ln d/T): regret  p[0]     PGD eta=sqrt(2/(dT)) (its bound's step): regret  p[0]     PGD eta=sqrt(2/T) (untuned): regret  p[0]     bounds: EG sqrt(2 ln d/T)   PGD sqrt(2d/T)")
for dd in (10, 100, 1000, 10000, 100000):
    p_eg = np.ones(dd) / dd; p_pg = np.ones(dd) / dd; p_pg2 = np.ones(dd) / dd
    eta_eg = sqrt(2 * log(dd) / T); eta_pg = sqrt(2 / (dd * T)); eta_pg2 = sqrt(2 / T)
    l_eg = l_pg = l_pg2 = 0.0; cum = np.zeros(dd)
    for t in range(T):
        lvec = rng.uniform(0, 1, dd); lvec[0] = rng.uniform(0, 0.8)
        l_eg += p_eg @ lvec; l_pg += p_pg @ lvec; l_pg2 += p_pg2 @ lvec; cum += lvec
        p_eg = p_eg * np.exp(-eta_eg * lvec); p_eg /= p_eg.sum()
        p_pg = proj_simplex(p_pg - eta_pg * lvec); p_pg2 = proj_simplex(p_pg2 - eta_pg2 * lvec)
    best = cum.min()
    print(f"  {dd:7d}     {(l_eg-best)/T:7.4f}  {p_eg[0]:6.3f}                  {(l_pg-best)/T:7.4f}  {p_pg[0]:6.3f}                               {(l_pg2-best)/T:7.4f}  {p_pg2[0]:6.3f}              {sqrt(2*log(dd)/T):7.4f}            {sqrt(2*dd/T):8.4f}")
print("  (with the step its own bound licenses, projected gradient barely leaves the uniform point once d is large: its guarantee scales with the l2 size of a loss vector, sqrt d; exponentiated gradient's scales with the l_inf size, 1. The untuned PGD step works here because the losses are random, not adversarial)")
sec(11, "Fisher information is the Hessian of KL (checked by finite differences), and natural gradient is reparametrisation-invariant")
# Gaussian family N(mu, sigma^2) in two parametrisations: (mu, sigma) and (mu, log sigma)
def kl_gauss(mu1, s1, mu2, s2):
    return log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2) - 0.5
mu0, s0 = 0.3, 1.7
h = 1e-3
def hess_kl(param):
    # second derivative of theta -> KL(p_theta0 || p_theta) at theta0 by central differences, in the given parametrisation
    def unpack(th):
        return (th[0], th[1]) if param == "sigma" else (th[0], exp(th[1]))
    th0 = np.array([mu0, s0 if param == "sigma" else log(s0)])
    Hm = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            def Kf(di, dj):
                th = th0.copy(); th[i] += di; th[j] += dj; m, s = unpack(th); return kl_gauss(mu0, s0, m, s)
            Hm[i, j] = (Kf(h, h) - Kf(h, -h) - Kf(-h, h) + Kf(-h, -h)) / (4 * h * h)
    return Hm
F_sigma = np.array([[1 / s0 ** 2, 0], [0, 2 / s0 ** 2]]); F_logs = np.array([[1 / s0 ** 2, 0], [0, 2.0]])
print(f"  (mu, sigma):     Hessian of KL = {hess_kl('sigma').ravel()}   Fisher closed form = {F_sigma.ravel()}")
print(f"  (mu, log sigma): Hessian of KL = {hess_kl('logsigma').ravel()}   Fisher closed form = {F_logs.ravel()}")
# one natural-gradient step on the expected negative log-likelihood of data from N(2, 0.5^2), from (mu0, s0), in both parametrisations
mu_d, s_d = 2.0, 0.5
def nll_grad(mu, s, param):      # gradient of E_data[-log p] = log s + (s_d^2 + (mu_d-mu)^2)/(2 s^2) + const
    gmu = -(mu_d - mu) / s ** 2; gs = 1 / s - (s_d ** 2 + (mu_d - mu) ** 2) / s ** 3
    return np.array([gmu, gs]) if param == "sigma" else np.array([gmu, gs * s])
for param in ("sigma", "logsigma"):
    g = nll_grad(mu0, s0, param); Fm = F_sigma if param == "sigma" else F_logs
    step_gd = -0.1 * g; step_ng = -0.1 * np.linalg.solve(Fm, g)
    to_ms = (lambda st: (mu0 + st[0], s0 + st[1])) if param == "sigma" else (lambda st: (mu0 + st[0], exp(log(s0) + st[1])))
    m1, s1 = to_ms(step_gd); m2, s2 = to_ms(step_ng)
    print(f"  {param:9s} plain GD step -> (mu, sigma) = ({m1:.4f}, {s1:.4f});  natural-gradient step -> ({m2:.4f}, {s2:.4f})")

sec(12, "Adam's counter-example (Reddi, Kale & Kumar 2018): online linear losses where Adam walks to the wrong end of [-1, 1] and AMSGrad does not")
Cc = 3.0
b1 = 0.0; b2 = 1 / (1 + Cc ** 2); eps = 0.0
def run_adam(ams, T=5000, alpha0=0.1):
    x = 0.0; m_ = 0.0; v = 0.0; vhat = 0.0; regret = 0.0
    for t in range(1, T + 1):
        g = Cc if t % 3 == 1 else -1.0           # f_t(x) = C x on t = 1 mod 3, else -x
        regret += g * x - g * (-1.0 if Cc > 2 else 1.0)   # best fixed point in hindsight: x = -1 (since (C-2)/3 > 0 pushes the sum positive)
        m_ = b1 * m_ + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        vhat = max(vhat, v) if ams else v
        x = x - (alpha0 / sqrt(t)) * m_ / (sqrt(vhat) + eps); x = min(1.0, max(-1.0, x))
    return x, regret / T
xa, ra = run_adam(False); xm, rm = run_adam(True)
print(f"  C = {Cc}, beta2 = 1/(1+C^2) = {b2:.3f}: after 5000 rounds Adam sits at x = {xa:+.3f} (average regret {ra:+.4f});  AMSGrad at x = {xm:+.3f} (average regret {rm:+.4f})")
print(f"  the sum of losses over one period of 3 is (C - 2) x = {Cc-2:.1f} x, minimised at x = -1; Adam's step after the C-gradient is small (v just jumped), after each -1 gradient it is large: net drift is toward +1")

sec(13, "Block heterogeneity (Zhang et al. 2024 picture): one learning rate for all blocks vs one per block, on a 4-block quadratic")
blocks = [np.full(25, s) for s in (1.0, 10.0, 100.0, 1000.0)]       # four blocks whose curvatures differ by 1000x
lamb = np.concatenate(blocks); L = lamb.max()
x0 = rng.standard_normal(100)
def f_b(x): return 0.5 * np.sum(lamb * x * x)
x = x0.copy()
for t in range(1, 100001):
    x = x - (1 / L) * lamb * x
    if f_b(x) < 1e-8: break
print(f"  GD with the one safe step 1/L (L = {L:.0f}): {t} iterations to f < 1e-8 (the kappa = 1000 block is the bottleneck: f_block1 after t steps = (1-1/1000)^2t f0)")
x = x0.copy()
for t in range(1, 100001):
    x = x - (1 / lamb) * lamb * x                 # per-block step 1/L_block: Newton on a block-diagonal problem
    if f_b(x) < 1e-8: break
print(f"  per-block step 1/L_block: {t} iteration(s)  (each block is a scalar quadratic; a diagonal preconditioner is exact here)")
x = x0.copy(); eta = 0.5; rec = {}
for t in range(1, 201):
    x = x - eta * np.sign(lamb * x); eta *= 0.98
    if t in (10, 50, 100, 200): rec[t] = f_b(x)
print("  sign descent, eta_0 = 0.5 decayed x0.98 per step: f at t = " + ", ".join(f"{t}: {v:.2e}" for t, v in rec.items()) + "   (scale-free in the curvature: the floor is set by the step, 0.5 * 0.98^t, not by kappa)")
sec(14, "Steepest descent under three norms is three updates; Newton–Schulz orthogonalisation (Muon's coefficients) in 5 steps")
G = rng.standard_normal((6, 4)) @ np.diag([3.0, 1.0, 0.3, 0.01]) @ rng.standard_normal((4, 4))
U, S, Vt = np.linalg.svd(G, full_matrices=False)
print(f"  singular values of G: {S}")
print(f"  ||G||_F = {np.linalg.norm(G):.4f}  ||G||_2 = {S[0]:.4f}  ||G||_nuc = {S.sum():.4f}")
d_l2 = G / np.linalg.norm(G); d_linf = np.sign(G) / np.sqrt(G.size); d_spec = U @ Vt   # sign(G)/sqrt(24): Frobenius norm 1 like the Euclidean row (the spectral row U V^T has Frobenius norm sqrt(min(m, n)) = 2), l_inf norm 1/sqrt(24) = 0.204; the l_inf-UNIT direction is sign(G) itself, with <G, sign G> = ||G||_1
for nm, D in (("Euclidean (G/||G||_F)", d_l2), ("l_inf   (sign G, scaled)", d_linf), ("spectral (U V^T)", d_spec)):
    print(f"  {nm:26s} <G, D> = {np.sum(G*D):.4f}   singular values of D: {np.linalg.svd(D, compute_uv=False)}")
# Newton-Schulz with Muon's quintic coefficients, 5 iterations, on G / ||G||_F
a_, b_, c_ = 3.4445, -4.7750, 2.0315
Xn = G / np.linalg.norm(G)
for it in range(1, 6):
    A_ = Xn @ Xn.T; B_ = b_ * A_ + c_ * A_ @ A_
    Xn = a_ * Xn + B_ @ Xn
    print(f"  NS step {it}: singular values -> {np.linalg.svd(Xn, compute_uv=False)}")
# no draws: the five-step output is an APPROXIMATION of the polar factor U V^T, whose singular values are all one —
# so Muon's update norms are those of this matrix, not the ideal ||UV^T||_2 = 1 and ||UV^T||_F = sqrt(min(m, n)) = 2 for a 6x4 gradient
sv5 = np.linalg.svd(Xn, compute_uv=False)
print(f"  after 5 steps: ||X||_2 = {sv5[0]:.4f} (exact polar factor: 1), ||X||_F = {np.linalg.norm(Xn):.4f} (exact polar factor: sqrt(min(m, n)) = {sqrt(min(G.shape)):.4f}); the five-step update is {100 * (1 - np.linalg.norm(Xn) / sqrt(min(G.shape))):.1f}% short of the ideal Frobenius norm")
print(f"  the smallest direction ({S[-1]:.4f}; {S[-1]/np.linalg.norm(G):.4f} after Frobenius scaling) is not orthonormalised in 5 steps: NS is a polynomial in the singular values, and small ones lag — the point of the 2026 spectral-scaling-law finding")

sec(15, "Orthogonalised gradient vs plain gradient on an ill-conditioned matrix regression — each method at its best step from a grid (illustration, not a theorem)")
mdim, ndim, N = 32, 16, 400
Xr = rng.standard_normal((ndim, N)) * np.logspace(0, -2, ndim)[:, None]     # input covariance condition number 1e4
Wt = rng.standard_normal((mdim, ndim)); Yr = Wt @ Xr
L_spec = np.linalg.eigvalsh(Xr @ Xr.T / N).max()
def loss_r(W): return 0.5 * np.mean(np.sum((W @ Xr - Yr) ** 2, 0))
def run_r(kind, mom, eta, T=2000):
    W = np.zeros((mdim, ndim)); M = np.zeros_like(W); hist = {}
    for t in range(1, T + 1):
        Gm = (W @ Xr - Yr) @ Xr.T / N
        M = mom * M + Gm
        if kind == "gd":
            W = W - eta * M
        else:
            Uo, _, Vo = np.linalg.svd(M, full_matrices=False); W = W - eta * (Uo @ Vo) * (0.998 ** t)
        if t in (100, 1000, 2000): hist[t] = loss_r(W)
    return hist
grids = {"gd": [c / L_spec for c in (0.25, 0.5, 1.0, 1.5, 1.9)], "orth": [0.003, 0.01, 0.03, 0.1, 0.3]}
for kind, mom, nm in (("gd", 0.0, "GD"), ("gd", 0.9, "GD + momentum 0.9"), ("orth", 0.0, "orthogonalised (Muon-like), no momentum"), ("orth", 0.9, "orthogonalised + momentum 0.9")):
    best = None
    for eta in grids[kind]:
        h = run_r(kind, mom, eta)
        if best is None or h[2000] < best[1][2000]: best = (eta, h)
    eta, h = best
    print(f"  {nm:42s} best step {eta:8.4f}: " + "  ".join(f"t={t}: {l:9.3e}" for t, l in h.items()))
print(f"  (input condition number 1e4, L = {L_spec:.3f}; the orthogonalised step decays as 0.998^t so that a fixed-length update can settle)")

# ----------------------------------------------------------------------------------------------
# PART IV — the stochastic regime (article §5)
# ----------------------------------------------------------------------------------------------

sec(16, "The noise floor: SGD on a quadratic with additive noise — stationary variance alpha sigma^2 / (2 - alpha), and what decay buys (W43 §8.4 hand-off)")
sig = 1.0
print("  alpha   stationary var (theory)   measured (last 50k of 100k steps)  ± SE")
for alpha in (0.5, 0.1, 0.01):
    x = 0.0; xs = np.empty(100000)
    xi = rng.standard_normal(100000)
    for t in range(100000):
        x = (1 - alpha) * x - alpha * sig * xi[t]; xs[t] = x
    tail = xs[50000:]
    var = np.mean(tail ** 2)
    # SE of a time-average variance of an AR(1): var * sqrt(2 * (1+rho^2)/(1-rho^2) / n), rho = 1-alpha
    rho = 1 - alpha; se = var * sqrt(2 * (1 + rho * rho) / (1 - rho * rho) / len(tail))
    print(f"  {alpha:5.2f}   {alpha*sig**2/(2-alpha):10.5f}               {var:10.5f}                ± {se:.5f}")
# decay: alpha_t = a/t with a*lambda > 1/2 gives O(1/t); here lambda = 1
x = 0.0; xi = rng.standard_normal(100000); rec = {}
for t in range(1, 100001):
    a_t = 1.0 / t
    x = (1 - a_t) * x - a_t * xi[t - 1]
    if t in (100, 1000, 10000, 100000): rec[t] = x * x
print("  alpha_t = 1/t (a lambda = 1 > 1/2): x_t^2 at t = 1e2..1e5 for ONE run: " + ", ".join(f"{t}: {v:.2e}" for t, v in rec.items()) + "   (E x_t^2 = 1/t exactly: the running mean of the noise)")

sec(17, "Gradient noise scale and the batch-size curve: steps to a target loss vs batch size on a logistic regression, each batch at its best step, with the fitted S = S_min (1 + B_noise/B)")
n, d = 20000, 30
Xg = rng.standard_normal((n, d)); wg = rng.standard_normal(d) / sqrt(d) * 3
pg = 1 / (1 + np.exp(-(Xg @ wg))); yg = (rng.uniform(size=n) < pg).astype(float)
Xh, yh = Xg[:4000], yg[:4000]                       # the loss is checked on a 4,000-example subset every step
def loss_g(w, X=Xg, y=yg):
    z = X @ w; return np.mean(np.logaddexp(0, z) - y * z)
w_opt = np.zeros(d)
for t in range(3000):
    z = Xg @ w_opt; w_opt -= 0.5 * (Xg.T @ (1 / (1 + np.exp(-z)) - yg) / n)
Lstar = loss_g(w_opt, Xh, yh); L0 = loss_g(np.zeros(d), Xh, yh); target = Lstar + 0.1 * (L0 - Lstar)
z = np.zeros(n); per = Xg * (1 / (1 + np.exp(-z)) - yg)[:, None]
gbar = per.mean(0); Sigma_tr = np.sum(per.var(0)); B_noise0 = Sigma_tr / (gbar @ gbar)
z = Xg @ w_opt; per = Xg * (1 / (1 + np.exp(-z)) - yg)[:, None]
gbar = per.mean(0); B_noise1 = np.sum(per.var(0)) / (gbar @ gbar)
print(f"  gradient noise scale B_noise = tr(Sigma)/||g||^2: {B_noise0:.1f} at w = 0, {B_noise1:.1e} at the optimum (||g|| -> 0: the scale grows along the path)")
print(f"  target = L* + 10% of the initial gap on the 4,000-example check set, checked every step (L* = {Lstar:.4f}, L0 = {L0:.4f})")
print("      B    best eta    steps to target (median of 3)   examples processed")
steps = {}
for B in (1, 4, 16, 64, 256, 1024, 4096):
    best = None
    for eta in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
        runs = []
        for rep in range(3):
            w = np.zeros(d); done = 20000
            for t in range(1, 20001):
                idx = rng.integers(0, n, B); zz = Xg[idx] @ w
                w -= eta * (Xg[idx].T @ (1 / (1 + np.exp(-zz)) - yg[idx]) / B)
                if loss_g(w, Xh, yh) < target: done = t; break
            runs.append(done)
        med = np.median(runs)
        if best is None or med < best[1]: best = (eta, med)
    steps[B] = best[1]
    print(f"  {B:5d}    {best[0]:6.2f}      {int(best[1]):8d}                          {int(best[1]*B):10d}")
Bs = np.array(list(steps), float); Ss = np.array([steps[b] for b in steps], float)
Aa = np.vstack([np.ones_like(Bs), 1 / Bs]).T; coef = np.linalg.lstsq(Aa, Ss, rcond=None)[0]
print(f"  fit S = S_min (1 + B_noise/B): S_min = {coef[0]:.1f} steps, B_noise(fit) = {coef[1]/coef[0]:.1f}")
sec(18, "Variance reduction: SVRG's linear rate against SGD's noise floor on a finite-sum strongly convex logistic regression")
n, d = 1000, 20
Xv = rng.standard_normal((n, d)); wv = rng.standard_normal(d); yv2 = np.where(Xv @ wv + 0.5 * rng.standard_normal(n) > 0, 1.0, 0.0)
lam2 = 1e-2
def Fv(w): z = Xv @ w; return np.mean(np.logaddexp(0, z) - yv2 * z) + lam2 / 2 * w @ w
def gv(w, idx): z = Xv[idx] @ w; return Xv[idx].T @ (1 / (1 + np.exp(-z)) - yv2[idx]) / len(idx) + lam2 * w
w = np.zeros(d)
for t in range(5000): w -= 0.5 * gv(w, np.arange(n))
Fst = Fv(w)
w_sgd = np.zeros(d); w_svrg = np.zeros(d); eta = 0.1
print("  epochs   SGD (eta=0.1/sqrt(epoch))   SGD (eta=0.1)    SVRG (eta=0.1)")
w_sgdd = np.zeros(d)
for ep in range(1, 31):
    for i in range(n):
        idx = rng.integers(0, n, 1)
        w_sgd -= eta * gv(w_sgd, idx)
        w_sgdd -= (eta / sqrt(ep)) * gv(w_sgdd, idx)
    # SVRG epoch: full gradient at snapshot, then n corrected steps
    snap = w_svrg.copy(); gfull = gv(snap, np.arange(n))
    for i in range(n):
        idx = rng.integers(0, n, 1)
        w_svrg -= eta * (gv(w_svrg, idx) - gv(snap, idx) + gfull)
    if ep in (1, 2, 5, 10, 20, 30):
        print(f"  {ep:5d}     {Fv(w_sgdd)-Fst:12.3e}             {Fv(w_sgd)-Fst:12.3e}     {Fv(w_svrg)-Fst:12.3e}")

sec(19, "Schedules on the noisy quadratic model (exact expectations, no sampling): constant, cosine, WSD, 1/sqrt t — and tail averaging, simulated")
# NQM: f(x) = 1/2 sum h_i x_i^2, gradient noise variance h_i * c / B per coordinate (Zhang et al. 2019). Per-coordinate recursions for E x_i^2 are exact.
dq = 1000; hq = 1.0 / np.arange(1, dq + 1); cq = 1.0; Bq = 64
Tq = 2000; eta_max = 1.0 / hq.max()
def nqm_loss(sched):
    m2 = np.ones(dq)            # E x_i^2 at t = 0 (x_0 = 1 in every coordinate)
    loss = []
    for t in range(Tq):
        eta = sched(t)
        m2 = (1 - eta * hq) ** 2 * m2 + eta ** 2 * hq * cq / Bq
        loss.append(0.5 * np.sum(hq * m2))
    return np.array(loss)
scheds = {
    "constant eta_max": lambda t: eta_max,
    "cosine to 0": lambda t: eta_max * 0.5 * (1 + math.cos(pi * t / Tq)),
    "WSD (80% const, 20% linear cooldown)": lambda t: eta_max if t < 0.8 * Tq else eta_max * (Tq - t) / (0.2 * Tq),
    "1/sqrt(t)": lambda t: eta_max / sqrt(1 + t / 50),
    "constant eta_max/10": lambda t: eta_max / 10,
}
print(f"  d = {dq}, h_i = 1/i, noise c/B = {cq/Bq}, T = {Tq}; expected loss at t = 400, 1600, 2000:")
for nm, s in scheds.items():
    lo = nqm_loss(s)
    print(f"  {nm:38s} {lo[399]:9.5f}   {lo[1599]:9.5f}   {lo[-1]:9.5f}")
last, tail = [], []
for rep in range(20):
    xs = np.ones(dq); avg = np.zeros(dq); eta = eta_max; cnt = 0
    noise = rng.standard_normal((Tq, dq))
    for t in range(Tq):
        xs = xs - eta * (hq * xs + np.sqrt(hq * cq / Bq) * noise[t])
        if t >= 0.75 * Tq: cnt += 1; avg += (xs - avg) / cnt
    last.append(0.5 * np.sum(hq * xs * xs)); tail.append(0.5 * np.sum(hq * avg * avg))
print(f"  constant eta_max, 20 simulated runs: last-iterate loss {np.mean(last):.5f} ± {se_mean(last):.5f}, loss of the average over the last 25% {np.mean(tail):.5f} ± {se_mean(tail):.5f}  (averaging buys most of what the cooldown buys)")
sec(20, "Adam's first step is a sign step: ||x_1 - x_0||_2 = eta sqrt(d) regardless of the gradient — why warmup exists")
for dd in (10, 10 ** 4, 10 ** 6):
    g = rng.standard_normal(dd) * 1e-3           # tiny gradients
    m_ = 0.1 * g; v = 0.001 * g * g               # beta1 = 0.9, beta2 = 0.999, t = 1
    mhat = m_ / 0.1; vhat = v / 0.001
    upd = mhat / (np.sqrt(vhat) + 1e-8)
    print(f"  d = {dd:8d}: ||update||_2 / eta = {np.linalg.norm(upd):10.2f}   sqrt(d) = {sqrt(dd):10.2f}   (||g||_2 was {np.linalg.norm(g):.4f})")
print("  draws: three Gaussian vectors; the identity is exact up to eps")

sec(21, "Weight decay sets an equilibrium norm, not a penalty: ||w|| under AdamW-style updates converges to sqrt(eta / (2 lambda)) per coordinate, and the angular step to sqrt(2 eta lambda)")
dd = 1000; eta = 1e-2; lam_wd = 0.1
w = rng.standard_normal(dd) * 3
print("  (random unit-RMS update directions, uncorrelated with w: the W43 AR(1) with a different name; timescale 1/(eta lambda) = 1000 steps)")
for t in range(1, 20001):
    u = rng.standard_normal(dd)                   # Adam-like update with RMS 1 per coordinate
    w_new = w - eta * (u + lam_wd * w)
    if t in (1, 100, 1000, 3000, 10000, 20000):
        ang = math.acos(min(1.0, (w @ w_new) / (np.linalg.norm(w) * np.linalg.norm(w_new))))
        print(f"  t = {t:6d}: ||w||_2 / sqrt(d) = {np.linalg.norm(w_new)/sqrt(dd):.4f}   equilibrium sqrt(eta/(2 lambda)) = {sqrt(eta/(2*lam_wd)):.4f}   angular step this update = {ang:.4f} rad   equilibrium sqrt(2 eta lambda) = {sqrt(2*eta*lam_wd):.4f}")
    w = w_new
print(f"\nDONE (§1–§21) in {time.time()-T0:.1f} s")

# ----------------------------------------------------------------------------------------------
# PART V — non-convex, local (article §6)
# ----------------------------------------------------------------------------------------------

sec(22, "What a stationary-point guarantee says: GD on an L-smooth function reaches min_t ||grad||^2 <= 2L (f0 - f*) / T; SGD's O(1/sqrt T) tail")
# deterministic case on Nesterov's chain function with d = 2001 (L-smooth, L = 1): GD is slow here, so the bound is informative
d = 2001; L = 1.0
def f_ch(x): return L / 4 * (0.5 * (x[0] ** 2 + np.sum((x[:-1] - x[1:]) ** 2) + x[-1] ** 2) - x[0])
def g_ch(x):
    g = np.zeros_like(x); dif = x[:-1] - x[1:]
    g[0] += x[0]; g[:-1] += dif; g[1:] -= dif; g[-1] += x[-1]; g[0] -= 1
    return L / 4 * g
xstar = 1 - np.arange(1, d + 1) / (d + 1); fstar = f_ch(xstar)
x = np.zeros(d); f0 = f_ch(x); gmin2 = np.inf
print("      T    min_t ||grad f(x_t)||^2 (GD, eta=1/L)   bound 2L(f0 - f*)/T   ratio   (chain function, d = 2001)")
for t in range(1, 10001):
    g = g_ch(x); gmin2 = min(gmin2, g @ g); x = x - g / L
    if t in (10, 100, 1000, 10000):
        bnd = 2 * L * (f0 - fstar) / t
        print(f"  {t:6d}    {gmin2:14.3e}                      {bnd:12.3e}        {gmin2/bnd:6.3f}")
# stochastic case on a non-convex function: f(x) = sum x_i^2/2 + 2 cos(3 x_i), L = 19, Gaussian gradient noise
dn = 50; Lnc = 19.0; sig = 2.0
def g_nc(x): return x - 6.0 * np.sin(3 * x)
x = rng.uniform(-3, 3, dn); gmin2 = np.inf; ref = None
print("  SGD on a non-convex f (d = 50, L = 19, noise sigma = 2 per coordinate), eta_t = 1/(L sqrt t): min_t ||grad||^2 and a T^-1/2 reference line through T = 10")
for t in range(1, 10001):
    g = g_nc(x); gmin2 = min(gmin2, g @ g)
    x = x - (1 / (Lnc * sqrt(t))) * (g + sig * rng.standard_normal(dn))
    if t in (10, 100, 1000, 10000):
        if ref is None: ref = gmin2
        print(f"  T = {t:6d}: min_t ||grad||^2 = {gmin2:10.4f}    reference {ref:.4f} * sqrt(10/T) = {ref*sqrt(10/t):10.4f}")
print("  (a stationary point is what is promised, and a stationary point of a non-convex function may be a saddle; nothing above says 'minimum')")
sec(23, "Saddles: GD started r away from a strict saddle escapes in ~ log(1/r) / (eta |lambda_min|) steps; started on the saddle it stays")
# f(x, y) = (x^2 - y^2)/2 + y^4/4: saddle at the origin with Hessian diag(1, -1); minima at y = +/-1, x = 0
eta = 0.1
def gsad(p): return np.array([p[0], -p[1] + p[1] ** 3])
print("      r (initial |y|)   steps to |y| > 0.5 (measured)   log(0.5/r)/(eta*1) predicted")
for r in (1e-1, 1e-2, 1e-4, 1e-8, 1e-12):
    p = np.array([1.0, r])
    for t in range(1, 100000):
        p = p - eta * gsad(p)
        if abs(p[1]) > 0.5: break
    print(f"  {r:12.0e}          {t:8d}                             {log(0.5/r)/(eta*1.0):8.1f}")
p = np.array([1.0, 0.0])
for t in range(10000): p = p - eta * gsad(p)
print(f"  r = 0 (on the stable manifold): after 10,000 steps p = {p}  — gradient descent converges to the saddle; the set of such starts has measure zero (Lee et al. 2016)")
# escape with isotropic noise of scale 1e-3 added once at step 0 from the exact saddle, 1000 trials
ts = []
for trial in range(1000):
    p = np.array([1.0, 0.0]) + 1e-3 * rng.standard_normal(2)
    for t in range(1, 100000):
        p = p - eta * gsad(p)
        if abs(p[1]) > 0.5: break
    ts.append(t)
print(f"  one perturbation of scale 1e-3 at the saddle, 1,000 trials: escape in {np.mean(ts):.1f} ± {se_mean(ts):.1f} steps (median {int(np.median(ts))}, max {max(ts)})")

sec(24, "The Polyak–Łojasiewicz inequality: a non-convex function GD solves at a linear rate, with its PL constant measured and used")
# f(x) = x^2 + 3 sin^2(x) (Karimi, Nutini & Schmidt 2016): non-convex, f* = 0 at x = 0, satisfies PL with mu = 1/32 (their bound); L = 8
def f_pl(x): return x * x + 3 * np.sin(x) ** 2
def g_pl(x): return 2 * x + 3 * np.sin(2 * x)
xs = np.linspace(-20, 20, 400001)
ratio = (0.5 * g_pl(xs) ** 2) / np.where(f_pl(xs) > 1e-12, f_pl(xs), np.nan)
mu_pl = np.nanmin(ratio)
print(f"  f(x) = x^2 + 3 sin^2 x: measured PL constant mu = min (1/2)|f'|^2 / (f - f*) over [-20, 20] = {mu_pl:.4f} (paper's bound 1/32 = {1/32:.4f});  L = 8;  f'' < 0 on {np.mean(2 + 6*np.cos(2*xs) < 0)*100:.1f}% of the line")
print("      t    f(x_t) from x0 = 10 (GD, eta = 1/L)   PL bound (1 - mu/L)^t f(x0)")
x = 10.0; f0 = f_pl(x)
for t in range(1, 16):
    x = x - g_pl(x) / 8
    if t in (1, 2, 3, 5, 8, 10, 12, 15):
        print(f"  {t:5d}    {f_pl(x):12.4e}                        {(1 - mu_pl/8)**t * f0:12.4e}")
print("  (the bound is a bound: near 0 the step 1/L is Newton's step for this f, so the real rate is superlinear; the guarantee is linear and holds on the whole line, including the 38% of it where f is concave)")
sec(25, "Nonsmooth: what autodiff returns at a kink depends on a convention, and the conservative-gradient theorem is why that is allowed")
# h(x) = relu(x) - relu(-x) = x for every real x, so h'(0) = 1. Autodiff with relu'(0) := 0 returns 0; with relu'(0) := 1 returns 2.
def relu(x): return np.maximum(x, 0.0)
def relu_prime(x, at0): return np.where(x > 0, 1.0, np.where(x < 0, 0.0, at0))
for at0 in (0.0, 0.5, 1.0):
    hp = relu_prime(0.0, at0) - relu_prime(-0.0, at0) * (-1)
    print(f"  relu'(0) := {at0}:  autodiff h'(0) = {hp:+.1f}   (true derivative of h(x) = x is +1)")
# on a random line through 0 the kink has measure zero: the fraction of sampled points where the convention matters
xs = rng.standard_normal(10 ** 6)
print(f"  of 10^6 Gaussian inputs, exactly {np.sum(xs == 0.0)} hit the kink: the convention is invisible almost everywhere, which is the theorem (Bolte & Pauwels 2021)")
# a 2-layer ReLU net on a kink: the loss landscape is piecewise smooth; count distinct activation patterns along a segment
W1 = rng.standard_normal((16, 2)); b1 = rng.standard_normal(16)
pts = np.linspace(-1, 1, 10001)[:, None] * np.array([[1.0, 0.7]])
pat = (pts @ W1.T + b1 > 0)
changes = np.sum(np.any(pat[1:] != pat[:-1], axis=1))
print(f"  a 16-unit ReLU layer along one input segment: {changes} activation-pattern changes in 10,001 points — {changes} kinks of the loss on that line, each of measure zero")

# ----------------------------------------------------------------------------------------------
# PART VI — landscapes (article §7)
# ----------------------------------------------------------------------------------------------

sec(26, "Benign landscapes: rank-r matrix sensing (Burer–Monteiro) from 100 random starts, exact rank and over-parameterised")
k, r_true, m = 12, 2, 300
Ut = rng.standard_normal((k, r_true)); Mstar = Ut @ Ut.T
A2 = rng.standard_normal((m, k, k)); A2 = (A2 + A2.transpose(0, 2, 1)) / sqrt(2)
y2 = np.einsum('mij,ij->m', A2, Mstar)
def f_bm(U): r = np.einsum('mij,ij->m', A2, U @ U.T) - y2; return np.sum(r * r) / (4 * m)
def g_bm(U): r = np.einsum('mij,ij->m', A2, U @ U.T) - y2; return np.einsum('m,mij->ij', r, A2) @ U / m
for rr in (2, 4):
    ok4 = ok2 = 0; its = []; errs = []
    for trial in range(100):
        U = rng.standard_normal((k, rr)) * 0.1
        for t in range(20000):
            U = U - 0.02 * g_bm(U)
            if f_bm(U) < 1e-10: break
        e = np.linalg.norm(U @ U.T - Mstar); errs.append(e); ok4 += e < 1e-4; ok2 += e < 1e-2; its.append(t + 1)
    print(f"  factor rank {rr} (truth {r_true}): ||U U^T - M*||_F < 1e-4 in {ok4}/100 runs, < 1e-2 in {ok2}/100; steps to f < 1e-10: median {int(np.median(its))}, max {max(its)}; median final error {np.median(errs):.2e}")
print("  (every run heads to M*; the exact-rank runs get there at a linear rate, the over-parameterised runs at a sublinear one — the extra directions are flat, which is the price of r > r*)")
print("  (the theorem: every second-order stationary point is global when m is large enough relative to k r; the saddles are strict — Ge, Jin & Zheng 2017)")
sec(27, "Lazy versus feature learning: a two-layer ReLU net at widths 64 .. 4096 under NTK scaling (1/sqrt m) and mean-field scaling (1/m)")
# 1-D regression y = sin(2 pi x) on 40 points; manual gradients; the step is 0.5 under NTK scaling and 0.05 m under mean-field scaling (so that each neuron can move O(1))
xtr = np.linspace(-1, 1, 40); ytr = np.sin(2 * pi * xtr)
def train(mw, scaling, steps=3000):
    w1 = rng.standard_normal(mw); b1_ = rng.standard_normal(mw); w2 = rng.standard_normal(mw)
    w10, b10, w20 = w1.copy(), b1_.copy(), w2.copy()
    scale = 1 / sqrt(mw) if scaling == "ntk" else 1 / mw
    eta = 0.5 if scaling == "ntk" else 0.05 * mw
    for t in range(steps):
        pre = np.outer(xtr, w1) + b1_; act = np.maximum(pre, 0); out = scale * act @ w2
        err = out - ytr
        gw2 = scale * act.T @ err / len(xtr)
        gact = scale * np.outer(err, w2) * (pre > 0)
        gw1 = (gact * xtr[:, None]).sum(0) / len(xtr); gb1 = gact.sum(0) / len(xtr)
        w1 -= eta * gw1; b1_ -= eta * gb1; w2 -= eta * gw2
    pre = np.outer(xtr, w1) + b1_; out = scale * np.maximum(pre, 0) @ w2
    rel = np.linalg.norm(w1 - w10) / np.linalg.norm(w10)
    return 0.5 * np.mean((out - ytr) ** 2), rel, np.mean((pre > 0) != (np.outer(xtr, w10) + b10 > 0))
print("   width      NTK: loss     ||w1 - w1_0||/||w1_0||   activations flipped      mean-field: loss     rel. movement   flipped")
for mw in (64, 256, 1024, 4096):
    ln, rn, fn = train(mw, "ntk"); lm, rm, fm = train(mw, "mf")
    print(f"  {mw:6d}      {ln:9.5f}        {rn:9.5f}               {fn:7.4f}             {lm:9.5f}           {rm:9.5f}      {fm:7.4f}")
print("  (NTK scaling: the relative movement shrinks with width — the kernel regime; mean-field: it does not — features move)")
sec(28, "Edge of stability: full-batch GD on a small tanh network drives the top Hessian eigenvalue to 2/eta and holds it there")
# 1-D regression, 1 hidden layer of 32 tanh units, 50 points; Hessian top eigenvalue by power iteration on finite-difference Hessian-vector products
xe = np.linspace(-2, 2, 50); ye = np.sin(3 * xe) + 0.3 * xe ** 2
H = 32
def unpack(th): return th[:H], th[H:2 * H], th[2 * H:3 * H], th[3 * H]
def loss_e(th):
    w1, b1_, w2, b2 = unpack(th); return 0.5 * np.mean((np.tanh(np.outer(xe, w1) + b1_) @ w2 + b2 - ye) ** 2)
def grad_e(th):
    w1, b1_, w2, b2 = unpack(th); pre = np.outer(xe, w1) + b1_; act = np.tanh(pre); out = act @ w2 + b2; err = (out - ye) / len(xe)
    gw2 = act.T @ err; gb2 = err.sum(); gact = np.outer(err, w2) * (1 - act ** 2)
    return np.concatenate([(gact * xe[:, None]).sum(0), gact.sum(0), gw2, [gb2]])
def top_eig(th, iters=60):
    v = rng.standard_normal(len(th)); v /= np.linalg.norm(v); lam_ = 0.0; h = 1e-4
    for _ in range(iters):
        hv = (grad_e(th + h * v) - grad_e(th - h * v)) / (2 * h); lam_ = v @ hv; v = hv / (np.linalg.norm(hv) + 1e-30)
    return lam_
th0 = np.concatenate([rng.standard_normal(H), rng.standard_normal(H) * 0.5, rng.standard_normal(H) / sqrt(H) * 0.1, [0.0]])
print(f"  initial top Hessian eigenvalue (sharpness): {top_eig(th0):.3f}, initial loss {loss_e(th0):.4f}")
for eta in (0.05, 0.1, 0.2):
    th = th0.copy(); rec = []
    for t in range(1, 10001):
        th = th - eta * grad_e(th)
        if t in (200, 1000, 3000, 6000, 10000): rec.append((t, loss_e(th), top_eig(th)))
    print(f"  eta = {eta:4.2f}, 2/eta = {2/eta:5.1f}:  " + "  ".join(f"t={t}: loss {l:.4f} sharp {s:5.2f}" for t, l, s in rec))
print("  (sharpness rises until it meets 2/eta and then hovers there while the loss keeps falling non-monotonically — Cohen et al. 2021; at eta = 0.05 the run never reaches 2/eta = 40 and sits where progressive sharpening leaves it)")
sec(29, "Implicit bias on separable data: GD on the logistic loss turns toward the l2 max-margin direction slowly, normalised GD fast, and sign descent toward a different margin")
n = 60
Xi = 0.5 * rng.standard_normal((n, 2)) + np.array([3.0, 1.0]); yi = np.ones(n)
Xi2 = 0.5 * rng.standard_normal((n, 2)) - np.array([3.0, 1.0]); Xi = np.vstack([Xi, Xi2]); yi = np.concatenate([yi, -np.ones(n)])
# class means at (3, 1): the l2 max-margin direction leans along the means, the l_inf one (an LP over the square) toward the diagonal
Q = (yi[:, None] * Xi) @ (yi[:, None] * Xi).T; a = np.zeros(2 * n); st = 1 / np.linalg.eigvalsh(Q).max()
for t in range(50000): a = np.maximum(a + st * (1 - Q @ a), 0)     # hard-margin SVM dual (bias-free): l2 max-margin direction
w_svm = (a * yi) @ Xi; w_svm /= np.linalg.norm(w_svm)
def margin(w, norm): return np.min(yi * (Xi @ w)) / norm(w)
angs = np.linspace(0, 2 * pi, 36001)
best_inf = max(angs, key=lambda th_: margin(np.array([math.cos(th_), math.sin(th_)]), lambda w: np.abs(w).max()))
w_inf = np.array([math.cos(best_inf), math.sin(best_inf)])
print(f"  data separable: min_i y_i <w_svm, x_i> = {np.min(yi * (Xi @ w_svm)):.4f} > 0")
print(f"  l2 max-margin direction {w_svm} (l2 margin {margin(w_svm, np.linalg.norm):.4f}); l_inf max-margin direction {w_inf} (l_inf margin {margin(w_inf, lambda w: np.abs(w).max()):.4f}); angle between them {math.degrees(math.acos(abs(w_svm @ w_inf))):.2f} deg")
def grad_log(w): z = yi * (Xi @ w); return -(Xi * (yi / (1 + np.exp(z)))[:, None]).mean(0)
def grad_dir(w):                                                 # the gradient up to a positive factor, computed in the log domain so it never underflows
    z = yi * (Xi @ w); lw = -np.logaddexp(0, z); wt = np.exp(lw - lw.max()); return -(Xi * (yi * wt)[:, None]).mean(0)
def ang(u, v): return math.degrees(math.acos(min(1.0, abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))))
w = np.zeros(2); wn = np.zeros(2); ws = np.zeros(2)
print("        t    GD: angle to l2-mm   ||w||    normalised GD: angle to l2-mm    sign descent: angle to l_inf-mm   angle to l2-mm   l_inf margin of ws / best")
for t in range(1, 10 ** 5 + 1):
    w = w - 0.5 * grad_log(w)
    gd_ = grad_dir(wn); wn = wn - 0.5 * gd_ / np.linalg.norm(gd_)
    ws = ws - 0.01 * np.sign(grad_dir(ws))
    if t in (10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5):
        print(f"  {t:8d}    {ang(w, w_svm):10.3f}        {np.linalg.norm(w):7.3f}          {ang(wn, w_svm):10.3f}                    {ang(ws, w_inf):10.3f}                 {ang(ws, w_svm):10.3f}        {margin(ws, lambda v: np.abs(v).max())/margin(w_inf, lambda v: np.abs(v).max()):.4f}")
print("  (GD's direction converges at a 1/log t rate — Soudry et al. 2018 — which is why its angle barely moves after t = 10^4 while ||w|| grows like log t; normalised GD converges in direction at a polynomial rate — Nacson et al. 2019; sign descent is steepest descent in the l_inf norm and heads for the l_inf margin — Gunasekar et al. 2018)")
sec(30, "Sharpness is not reparametrisation-invariant (Dinh et al. 2017): the same function, Hessian top eigenvalue multiplied by alpha^2")
# f(a, b) = 1/2 E (a b x - y)^2 with x ~ N(0,1), y = c x: loss = 1/2 (ab - c)^2 (+ const); the function x -> abx is unchanged by (a, b) -> (alpha a, b/alpha)
c = 1.0
def hess_ab(a_, b_):
    return np.array([[b_ ** 2, 2 * a_ * b_ - c], [2 * a_ * b_ - c, a_ ** 2]])
for alpha in (1.0, 10.0, 100.0):
    a_, b_ = alpha * 1.0, 1.0 / alpha                       # a b = c = 1: a global minimum in every case
    ev = np.linalg.eigvalsh(hess_ab(a_, b_))
    print(f"  (a, b) = ({a_:6.2f}, {b_:6.4f}): a b = {a_*b_:.1f}, loss 0, Hessian eigenvalues {ev}, top = {ev[-1]:.4f}, trace = {np.trace(hess_ab(a_, b_)):.4f}")
print("  (every row is the same predictor with the same loss on every input; 'flat' is a statement about coordinates, not about the function)")

sec(31, "Linear mode connectivity: the loss barrier between two SGD runs from the same initialisation versus from different ones")
# 2-D two-moons-like data, 1 hidden layer of 64 ReLU units, SGD with different data orders
nn_ = 400
tt = rng.uniform(0, pi, nn_ // 2)
Xm = np.vstack([np.c_[np.cos(tt), np.sin(tt)], np.c_[1 - np.cos(tt), 0.5 - np.sin(tt)]]) + 0.1 * rng.standard_normal((nn_, 2))
ym = np.concatenate([np.ones(nn_ // 2), -np.ones(nn_ // 2)])
Hm_ = 64
def init(): return [rng.standard_normal((2, Hm_)) / sqrt(2), np.zeros(Hm_), rng.standard_normal(Hm_) / sqrt(Hm_), 0.0]
def fwd(p, X): return np.maximum(X @ p[0] + p[1], 0) @ p[2] + p[3]
def loss_m(p, X=Xm, y=ym): return np.mean(np.logaddexp(0, -y * fwd(p, X)))
def sgd(p, epochs=60, B=16, eta=0.1):
    p = [q.copy() if hasattr(q, 'copy') else q for q in p]
    for ep in range(epochs):
        perm = rng.permutation(nn_)
        for i in range(0, nn_, B):
            idx = perm[i:i + B]; X = Xm[idx]; y = ym[idx]
            pre = X @ p[0] + p[1]; act = np.maximum(pre, 0); out = act @ p[2] + p[3]
            err = -y / (1 + np.exp(y * out)) / len(idx)
            gw2 = act.T @ err; gb2 = err.sum(); gact = np.outer(err, p[2]) * (pre > 0)
            p[0] -= eta * X.T @ gact; p[1] -= eta * gact.sum(0); p[2] -= eta * gw2; p[3] -= eta * gb2
    return p
def barrier(pa, pb):
    ls = [loss_m([(1 - s) * a + s * b for a, b in zip(pa, pb)]) for s in np.linspace(0, 1, 21)]
    return max(ls) - max(ls[0], ls[-1]), ls[10]
p0 = init(); pA = sgd(p0); pB = sgd(p0); pC = sgd(init())
bAB, midAB = barrier(pA, pB); bAC, midAC = barrier(pA, pC)
print(f"  end losses: A {loss_m(pA):.4f}, B {loss_m(pB):.4f} (same init, different order), C {loss_m(pC):.4f} (different init)")
print(f"  barrier along the segment A–B: {bAB:.4f} (midpoint loss {midAB:.4f});  A–C: {bAC:.4f} (midpoint loss {midAC:.4f})")
print("  (same-init runs are linearly connected, different-init runs are not — Frankle et al. 2020; permuting C's hidden units is what Git Re-Basin does)")

# ----------------------------------------------------------------------------------------------
# PART VII — global (article §8)
# ----------------------------------------------------------------------------------------------

sec(32, "Certified global optimisation by branch-and-bound with a Lipschitz lower bound on Rastrigin: boxes explored grow exponentially with dimension")
import heapq
def rastrigin(x): return 10 * len(x) + np.sum(x * x - 10 * np.cos(2 * pi * x))
def bnb(dim, eps, max_boxes=200000):
    Lip = sqrt(dim) * (2 * 5.12 + 20 * pi)                      # per-coordinate |f'| <= 2*5.12 + 20 pi on [-5.12, 5.12]
    lo0, hi0 = -5.12 * np.ones(dim), 5.12 * np.ones(dim)
    c = (lo0 + hi0) / 2; best = rastrigin(c); best_x = c
    heap = [(best - Lip * np.linalg.norm(hi0 - lo0) / 2, 0, lo0, hi0)]; cnt = 1; k_ = 1
    while heap:
        lb, _, lo, hi = heapq.heappop(heap)
        if lb > best - eps: break                               # the global minimum is certified to within eps
        j = int(np.argmax(hi - lo)); mid = (lo[j] + hi[j]) / 2
        for a_, b_ in (((lo, np.where(np.arange(dim) == j, mid, hi))), (np.where(np.arange(dim) == j, mid, lo), hi)):
            c = (a_ + b_) / 2; fc = rastrigin(c)
            if fc < best: best, best_x = fc, c
            heapq.heappush(heap, (fc - Lip * np.linalg.norm(b_ - a_) / 2, k_, a_, b_)); k_ += 1; cnt += 1
            if cnt >= max_boxes: return best, best_x, cnt, False
    return best, best_x, cnt, True
print("   dim   boxes to certify f* within eps = 1.0   certified f*   (true f* = 0 at the origin)")
for dim in (1, 2, 3, 4):
    best, bx, cnt, done = bnb(dim, 1.0)
    print(f"  {dim:4d}   {cnt:10d}{'' if done else ' (cap hit, not certified)'}                          {best:8.4f}")
print("  (each box is split in half along its longest side; the lower bound is f(centre) - Lip * radius; a tighter bound — interval or alpha-BB — cuts the constant, not the exponent)")

sec(33, "Local search, annealed Langevin, and random restarts on 2-D Rastrigin: fraction of runs that end in the global basin")
def g_ras(x): return 2 * x + 20 * pi * np.sin(2 * pi * x)
def gd_run(x, steps=2000, eta=0.002):
    for _ in range(steps): x = x - eta * g_ras(x)
    return x
def sgld_run(x, steps=20000, eta=0.005, T0_=10.0):
    for t in range(steps):
        Tt = T0_ * (1 - t / steps) ** 2 + 1e-3                   # annealed temperature
        x = x - eta * g_ras(x) + sqrt(2 * eta * Tt) * rng.standard_normal(2)
        x = np.clip(x, -5.12, 5.12)
    return gd_run(x, 500)
ngd = nsg = 0; N_ = 200
for trial in range(N_):
    x0 = rng.uniform(-5.12, 5.12, 2)
    ngd += rastrigin(gd_run(x0)) < 0.5; nsg += rastrigin(sgld_run(x0)) < 0.5
print(f"  GD from a uniform random start: global basin reached in {ngd/N_:.3f} ± {se_prop(ngd, N_):.3f} of {N_} runs  (basin of the origin has width ~1 in each coordinate on a 10.24-wide square: (1/10.24)^2 = {(1/10.24)**2:.4f} by area)")
print(f"  annealed Langevin (T from 10 to ~0 over 20,000 steps, eta = 0.005) then GD: {nsg/N_:.3f} ± {se_prop(nsg, N_):.3f}   (the best of a small grid over T0 and eta; a factor of ~10 over local search, and nowhere near reliable)")
k_restart = int(math.ceil(log(0.01) / log(1 - max(ngd, 1) / N_)))
print(f"  random restarts of plain GD needed for 99% success at the measured rate: {k_restart}  (restarts are the cheapest global method when the basin is not tiny; none of this certifies anything)")
sec(34, "Bayesian optimisation (GP + expected improvement) versus random search on 2-D Rastrigin with 60 evaluations (W28 bridge)")
def Phi(x): return 0.5 * (1 + math.erf(x / sqrt(2)))
def phi(x): return exp(-x * x / 2) / sqrt(2 * pi)
def kern(Xa, Xb, ell=0.6): d2 = np.sum((Xa[:, None, :] - Xb[None, :, :]) ** 2, -1); return np.exp(-d2 / (2 * ell * ell))
bo_best, rs_best = [], []
cand = rng.uniform(-5.12, 5.12, (4000, 2))
for trial in range(10):
    Xo = rng.uniform(-5.12, 5.12, (5, 2)); yo = np.array([rastrigin(x) for x in Xo])
    for it in range(55):
        mu_y, sd_y = yo.mean(), yo.std() + 1e-9; yn = (yo - mu_y) / sd_y
        K = kern(Xo, Xo) + 1e-6 * np.eye(len(Xo)); Linv = np.linalg.inv(np.linalg.cholesky(K))
        alpha_ = Linv.T @ (Linv @ yn)
        Ks = kern(cand, Xo); mu_c = Ks @ alpha_; v = Linv @ Ks.T; var_c = np.maximum(1 - np.sum(v * v, 0), 1e-12); sd_c = np.sqrt(var_c)
        zbest = yn.min(); zc = (zbest - mu_c) / sd_c
        ei = sd_c * (zc * np.vectorize(Phi)(zc) + np.vectorize(phi)(zc))
        xnew = cand[np.argmax(ei)]; Xo = np.vstack([Xo, xnew]); yo = np.append(yo, rastrigin(xnew))
    bo_best.append(yo.min())
    rs = np.array([rastrigin(x) for x in rng.uniform(-5.12, 5.12, (60, 2))]); rs_best.append(rs.min())
print(f"  best value after 60 evaluations, 10 trials: BO {np.mean(bo_best):.3f} ± {se_mean(bo_best):.3f}   random search {np.mean(rs_best):.3f} ± {se_mean(rs_best):.3f}   (f* = 0)")
diff_ = np.array(rs_best) - np.array(bo_best)
print(f"  difference random search - BO over the 10 trials: {diff_.mean():.3f} ± {se_mean(diff_):.3f} (SE of the mean difference) = {diff_.mean()/se_mean(diff_):.2f} standard errors; BO's best was lower in {int(np.sum(diff_ > 0))}/10 trials")
print("  (BO is a global method under a prior; its guarantee is a regret bound for the GP it assumed, not a certificate)")

sec(35, "Verification as the home of certified global optimisation: interval bounds, then branch-and-bound on the input box, on a 2-16-1 ReLU network at five radii")
W1v = rng.standard_normal((16, 2)); b1v = rng.standard_normal(16) * 0.5; W2v = rng.standard_normal(16) / 4; b2v = 0.0
x0v = np.array([0.8, -0.3])
def net(x): return np.maximum(W1v @ x + b1v, 0) @ W2v + b2v
f0v = net(x0v); sgn = 1.0 if f0v > 0 else -1.0
def ibp(lo, hi):                                                 # interval bound propagation: a sound lower/upper bound on net over the box
    c = (lo + hi) / 2; r = (hi - lo) / 2
    cz = W1v @ c + b1v; rz = np.abs(W1v) @ r
    zl, zu = cz - rz, cz + rz
    al, au = np.maximum(zl, 0), np.maximum(zu, 0)
    wpos = np.maximum(W2v, 0); wneg = np.minimum(W2v, 0)
    return zl, zu, wpos @ al + wneg @ au + b2v, wpos @ au + wneg @ al + b2v
def bab(eps, cap=200000):                                         # split the input box; certify a leaf when its IBP bound is on the right side; falsify on a counter-example
    leaves = [(x0v - eps, x0v + eps)]; decided = 0; undecided = 0; nodes = 1
    while leaves:
        lo, hi = leaves.pop()
        zl, zu, olo, ohi = ibp(lo, hi)
        if sgn * (olo if sgn > 0 else ohi) > 0: decided += 1; continue
        c = (lo + hi) / 2
        corners = [np.array([a_, b_]) for a_ in (lo[0], hi[0]) for b_ in (lo[1], hi[1])]
        if any(sgn * net(p_) <= 0 for p_ in [c] + corners): return "FALSIFIED (counter-example found)", nodes, decided
        if nodes >= cap: undecided += 1; continue
        j = int(np.argmax(hi - lo)); mid = (lo[j] + hi[j]) / 2
        leaves.append((lo, np.where(np.arange(2) == j, mid, hi))); leaves.append((np.where(np.arange(2) == j, mid, lo), hi)); nodes += 2
    return ("CERTIFIED" if undecided == 0 else "undecided (node cap)"), nodes, decided
print(f"  property: sign(net(x)) = sign(net(x0)) = {'+' if sgn > 0 else '-'} for every ||x - x0||_inf <= eps;  net(x0) = {f0v:+.4f}")
print("    eps    IBP bound on sgn*net (lower)   unstable ReLUs   grid min of sgn*net (401^2)   branch-and-bound verdict                 boxes   certified leaves")
grid = np.linspace(-1, 1, 401)
for eps_v in (0.1, 0.25, 0.4, 0.5, 0.75):
    zl, zu, olo, ohi = ibp(x0v - eps_v, x0v + eps_v)
    lb = sgn * (olo if sgn > 0 else ohi); unstable = int(np.sum((zl < 0) & (zu > 0)))
    GX, GY = np.meshgrid(x0v[0] + eps_v * grid, x0v[1] + eps_v * grid)
    gmin = np.min(sgn * (np.maximum(np.stack([GX.ravel(), GY.ravel()], 1) @ W1v.T + b1v, 0) @ W2v + b2v))   # min of the SIGNED output; sgn * min(v) is wrong when sgn < 0
    verdict, nodes, dec = bab(eps_v)
    print(f"  {eps_v:5.2f}    {lb:+10.4f}                       {unstable:3d}/16          {gmin:+10.4f}                   {verdict:38s} {nodes:6d}   {dec:6d}")
print("  (interval arithmetic is sound and loose; once it says 'undecided', splitting the box tightens it, and a counter-example ends the search — the shape of alpha-beta-CROWN in miniature; VNN-COMP 2025 ran 8 tools on 25 benchmarks this way)")
sec(36, "Quantisation is a constrained second-order problem: round-to-nearest vs OBS/GPTQ-style sequential rounding with Hessian compensation on a 64-input linear layer")
din, nsamp = 64, 2000
Cx = np.linalg.cholesky(np.array([[0.8 ** abs(i - j) for j in range(din)] for i in range(din)]))
Xq = rng.standard_normal((nsamp, din)) @ Cx.T                      # correlated inputs: Hessian H = X^T X / n is far from diagonal
wq = rng.standard_normal(din)
levels = 4                                                         # 2-bit grid, symmetric
scale = np.abs(wq).max() / (levels / 2 - 0.5)
def quant(v): return np.clip(np.round(v / scale), -levels / 2, levels / 2 - 1) * scale
Hq = Xq.T @ Xq / nsamp + 1e-3 * np.eye(din)
w_rtn = quant(wq)
# GPTQ (Frantar et al. 2023) in its simplest form: quantise coordinates in order, push the error onto the remaining weights via the Hessian inverse
w_g = wq.copy(); Hinv = np.linalg.inv(Hq)
for i in range(din):
    qi = quant(np.array([w_g[i]]))[0]; err = (w_g[i] - qi) / Hinv[i, i]
    w_g[i:] -= err * Hinv[i, i:]; w_g[i] = qi
    # keep the already-quantised prefix fixed: Hinv restricted to the remaining block (Cholesky form in the paper)
    Hinv_sub = Hinv[i + 1:, i + 1:] - np.outer(Hinv[i + 1:, i], Hinv[i, i + 1:]) / Hinv[i, i]
    Hinv = np.block([[Hinv[:i + 1, :i + 1], Hinv[:i + 1, i + 1:]], [Hinv[i + 1:, :i + 1], Hinv_sub]]) if i + 1 < din else Hinv
w_g = quant(w_g)
def out_err(w): return np.mean((Xq @ (w - wq)) ** 2)
print(f"  2-bit grid ({levels} levels): output MSE round-to-nearest {out_err(w_rtn):.4f}   GPTQ-style {out_err(w_g):.4f}   (ratio {out_err(w_rtn)/out_err(w_g):.2f}x);  weight MSE RTN {np.mean((w_rtn-wq)**2):.4f} vs GPTQ {np.mean((w_g-wq)**2):.4f}")
print("  (the objective is the quadratic (w - w_q)^T H (w - w_q) over a discrete grid: minimising it exactly is an integer program; OBS/GPTQ is a greedy second-order heuristic with a closed-form compensation step)")

# ----------------------------------------------------------------------------------------------
# PART VIII — minimax and relaxation (article §9–§10)
# ----------------------------------------------------------------------------------------------

sec(37, "Minimax is not descent: simultaneous gradient descent-ascent spirals out of the bilinear saddle; extragradient and optimistic steps spiral in")
eta = 0.1
def run_game(kind, T=200):
    z = np.array([1.0, 1.0]); zp = z.copy()        # min_x max_y x y: the saddle is (0, 0)
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])          # v(z) = (dL/dx, -dL/dy) = (y, -x)
    for t in range(T):
        if kind == "GDA": z = z - eta * J @ z
        elif kind == "alternating": x = z[0] - eta * z[1]; y = z[1] + eta * x; z = np.array([x, y])
        elif kind == "extragradient": zh = z - eta * J @ z; z = z - eta * J @ zh
        elif kind == "optimistic": zn = z - 2 * eta * J @ z + eta * J @ zp; zp, z = z, zn
    return np.linalg.norm(z)
for kind in ("GDA", "alternating", "extragradient", "optimistic"):
    print(f"  {kind:14s} ||z_200|| = {run_game(kind):12.4e}")
print(f"  simultaneous GDA multiplies ||z|| by sqrt(1 + eta^2) = {sqrt(1+eta**2):.5f} per step: {sqrt(1+eta**2)**200:.3e} after 200 (a theorem about a 2x2 matrix); extragradient contracts by sqrt(1 - eta^2 + eta^4) = {sqrt(1-eta**2+eta**4):.5f}")

sec(38, "Compressed sensing: l1 recovery (lasso with lambda = 1e-4, support threshold, least-squares debias — a practical proxy for basis pursuit) of a 10-sparse signal in R^200 from m Gaussian measurements; least squares never does")
nsig, ksp = 200, 10
print("     m    lasso+threshold+debias exact recovery rate (20 trials)   min-norm l2 recovery rate   l1 median rel. error")
for m_ in (20, 40, 60, 80, 120):
    ok1 = ok2 = 0; errs = []
    for trial in range(20):
        Am = rng.standard_normal((m_, nsig)) / sqrt(m_); s = np.zeros(nsig); idx = rng.choice(nsig, ksp, replace=False); s[idx] = rng.standard_normal(ksp)
        bm = Am @ s
        # NOT the equality-constrained basis pursuit LP min ||x||_1 s.t. Ax = b that the theorem is about: FISTA on the lasso 1/2||Ax-b||^2 + lam ||x||_1
        # with a small lam, the support thresholded at 1e-3, then least squares on that support (debiasing). The machine-precision errors below are the
        # debiased ones; the lasso iterate itself is off by O(lam). Exact recovery of the support is what the two procedures share
        Lm = np.linalg.eigvalsh(Am.T @ Am).max(); lam_ = 1e-4; x = np.zeros(nsig); z = x.copy(); tk = 1.0
        for t in range(3000):
            xn = prox(z - Am.T @ (Am @ z - bm) / Lm, lam_ / Lm); tn = (1 + sqrt(1 + 4 * tk * tk)) / 2
            z = xn + (tk - 1) / tn * (xn - x); x, tk = xn, tn
        supp = np.abs(x) > 1e-3
        if supp.sum() <= m_: xd = np.zeros(nsig); xd[supp] = np.linalg.lstsq(Am[:, supp], bm, rcond=None)[0]
        else: xd = x
        rel = np.linalg.norm(xd - s) / np.linalg.norm(s); errs.append(rel); ok1 += rel < 1e-4
        x2 = np.linalg.lstsq(Am, bm, rcond=None)[0]; ok2 += np.linalg.norm(x2 - s) / np.linalg.norm(s) < 1e-4
    print(f"  {m_:4d}       {ok1/20:5.2f} ± {se_prop(ok1,20):.2f}                                {ok2/20:5.2f}                      {np.median(errs):10.2e}")
print("  (the convex relaxation of an NP-hard l0 problem is exact with high probability once m >~ C k log(n/k), and k log(n/k) ~ 30 here; the theorem is about the basis-pursuit LP and leaves C unstated — the transition above, between m = 40 and 60, is this lasso+debias proxy's and only estimates C ~ 2)")

sec(39, "Over-parameterised interpolation: GD on a d > n linear model reaches zero loss at a linear rate on the data subspace — the PL* picture — and the rate is the smallest nonzero singular value")
n, d = 50, 400
Xo = rng.standard_normal((n, d)); yo = rng.standard_normal(n)
sv = np.linalg.svd(Xo, compute_uv=False); Lo = sv[0] ** 2 / n; mu_o = sv[-1] ** 2 / n
w = np.zeros(d)
print(f"  tangent kernel K = X X^T / n: largest eigenvalue {Lo:.3f}, smallest (of n = {n}) {mu_o:.3f}; PL* constant mu = {mu_o:.3f}; predicted rate (1 - mu/L)^2t")
for t in range(1, 2001):
    w = w - (1 / Lo) * Xo.T @ (Xo @ w - yo) / n
    if t in (10, 100, 500, 1000, 2000):
        print(f"  t = {t:5d}: loss {0.5*np.mean((Xo @ w - yo)**2):10.3e}   predicted {0.5*np.mean(yo**2)*(1 - mu_o/Lo)**(2*t):10.3e}")
print("  (the Hessian in weight space has d - n = 350 zero eigenvalues and the loss is not strongly convex; the condition that does the work is on the n x n kernel, not on the Hessian — Liu, Zhu & Belkin 2022)")

sec(40, "Three optimisation errors in one excess risk (Bottou & Bousquet 2008): approximation, estimation and optimisation, measured on E2 of the Probability Domain")
# E2: two-class Gaussian mixture in d = 20, means +/- mu e1 with mu = 1, identity covariance; Bayes risk Phi(-1) = 0.1587
d = 20; mu_e = 1.0
def sample_e2(n):
    y = np.where(rng.uniform(size=n) < 0.5, 1.0, -1.0); X = rng.standard_normal((n, d)); X[:, 0] += mu_e * y; return X, y
Xte, yte = sample_e2(200000)
def err(w): return np.mean(np.sign(Xte @ w) != yte)
bayes = Phi(-mu_e)
w_bayes = np.zeros(d); w_bayes[0] = 1.0
print(f"  Bayes risk Phi(-1) = {bayes:.4f}; the Bayes classifier is in the linear class: approximation error = 0 here by construction")
print("      n     exact empirical minimiser (full GD): excess risk     one-pass SGD (eta=0.05): excess risk — a TOTAL, not an added term")
for n in (100, 1000, 10000):
    Xn, yn = sample_e2(n)
    w = np.zeros(d)
    for t in range(2000): z = yn * (Xn @ w); w += 0.5 * (Xn * (yn / (1 + np.exp(z)))[:, None]).mean(0)
    e_full = err(w)
    ws = np.zeros(d)
    for i in range(n): z = yn[i] * (Xn[i] @ ws); ws += 0.05 * Xn[i] * yn[i] / (1 + exp(z))
    e_sgd = err(ws)
    print(f"  {n:6d}     {e_full-bayes:10.4f}                                       {e_sgd-bayes:10.4f}")
print("  (the left column is the estimation term; the right column is the one-pass model's whole excess risk, which can be BELOW the left one — under-optimising is a regulariser at small n; past a point, more passes buy less than more data — the argument for SGD at scale)")

sec(41, "Double descent: test risk of the minimum-norm interpolant against p/n, with the exact bias and variance, and the ridge at the Gaussian-prior Bayes lambda beside it")
# y = X beta + noise, X Gaussian n x p (isotropic), ||beta|| = r, noise sd sigma. Test risk of w is E_x (x.(w - beta))^2 = ||w - beta||^2 exactly (isotropic x).
# Min-norm interpolant w = X^+ y: bias = ||(I - X^+ X) beta||^2, variance = ||X^+ eps||^2. Gaussian design gives both in closed form in expectation:
#   p < n - 1:  E bias = 0,            E var = sigma^2 p / (n - p - 1)       (inverse Wishart, W ~ W_p(n))
#   p > n + 1:  E bias = r^2 (1 - n/p),  E var = sigma^2 n / (p - n - 1)       (inverse Wishart, W ~ W_n(p))
# and the expectation is infinite at |p - n| <= 1. Ridge at lambda = p sigma^2 / r^2, the Bayes lambda for beta ~ N(0, r^2/p I), is the explicit regulariser. The simulation
# draws beta on the SPHERE of radius r (same second moment, not Gaussian), so this ridge is the best linear estimator for that prior scale, not the Bayes rule for the fixed-norm prior.
n_dd, r_dd, sig_dd, trials_dd = 100, 1.0, 0.5, 200
print(f"  n = {n_dd}, ||beta|| = {r_dd:.1f}, noise sd {sig_dd:.1f} (noise variance {sig_dd**2:.2f}); {trials_dd} trials per p; risk = ||w - beta||^2 (isotropic x, so exact)")
print("       p    p/n     min-norm risk (mean ± SE)     median      E bias    E var    E risk        ridge at Bayes lambda (mean ± SE)")
for p_ in (10, 25, 50, 75, 90, 95, 100, 105, 110, 125, 150, 200, 400, 1000):
    risks = np.empty(trials_dd); rr = np.empty(trials_dd); lam_dd = p_ * sig_dd ** 2 / r_dd ** 2
    for k_ in range(trials_dd):
        beta = rng.standard_normal(p_); beta *= r_dd / np.linalg.norm(beta)
        Xd = rng.standard_normal((n_dd, p_)); yd = Xd @ beta + sig_dd * rng.standard_normal(n_dd)
        w_mn_ = np.linalg.lstsq(Xd, yd, rcond=None)[0]
        w_rd_ = Xd.T @ np.linalg.solve(Xd @ Xd.T + lam_dd * np.eye(n_dd), yd)      # ridge via the n x n system (valid for every p)
        risks[k_] = np.sum((w_mn_ - beta) ** 2); rr[k_] = np.sum((w_rd_ - beta) ** 2)
    if p_ < n_dd - 1: eb, ev = 0.0, sig_dd ** 2 * p_ / (n_dd - p_ - 1)
    elif p_ > n_dd + 1: eb, ev = r_dd ** 2 * (1 - n_dd / p_), sig_dd ** 2 * n_dd / (p_ - n_dd - 1)
    else: eb, ev = (0.0 if p_ <= n_dd else r_dd ** 2 * (1 - n_dd / p_)), float("inf")
    etot = f"{eb+ev:8.4f}" if np.isfinite(ev) else "     inf"
    evs = f"{ev:8.4f}" if np.isfinite(ev) else "     inf"
    print(f"  {p_:6d}  {p_/n_dd:5.2f}     {risks.mean():9.4f} ± {se_mean(risks):8.4f}      {np.median(risks):8.4f}    {eb:7.4f}  {evs}  {etot}        {rr.mean():8.4f} ± {se_mean(rr):6.4f}")
print("  (the classical U-curve is the p < n half; the peak at p = n is the variance of inverting a square random matrix, infinite in expectation; the second descent is the min-norm interpolant that GD from zero reaches (§2), whose bias RISES like r^2 (1 - n/p) while its variance falls like n/(p-n-1); ridge at the Bayes lambda has no peak at all)")
print("  --- interpolating all D = 1000 features at n = 100: the covariance tail decides whether it is harmless (signal in the top 5 directions; risk = (w-beta)^T Sigma (w-beta)) ---")
print("   tail variance per direction    min-norm interpolant (p = 1000)     oracle least squares on the 5 signal features     ridge at lambda = sigma^2 k / r^2")
D_dd, k_dd = 1000, 5
for eps_ in (1.0, 0.1, 0.01, 0.001):
    lam_cov = np.r_[np.ones(k_dd), eps_ * np.ones(D_dd - k_dd)]; sq = np.sqrt(lam_cov)
    rm, ro, rr2 = np.empty(trials_dd), np.empty(trials_dd), np.empty(trials_dd)
    for k_ in range(trials_dd):
        beta = np.zeros(D_dd); beta[:k_dd] = rng.standard_normal(k_dd); beta[:k_dd] *= r_dd / np.linalg.norm(beta[:k_dd])
        Xd = rng.standard_normal((n_dd, D_dd)) * sq; yd = Xd @ beta + sig_dd * rng.standard_normal(n_dd)
        w_mn_ = np.linalg.lstsq(Xd, yd, rcond=None)[0]
        w_or_ = np.zeros(D_dd); w_or_[:k_dd] = np.linalg.lstsq(Xd[:, :k_dd], yd, rcond=None)[0]
        lam_dd = sig_dd ** 2 / (r_dd ** 2 / k_dd)                                  # Bayes lambda for the signal's prior variance per direction
        w_rd_ = Xd.T @ np.linalg.solve(Xd @ Xd.T + lam_dd * np.eye(n_dd), yd)
        rm[k_] = np.sum(lam_cov * (w_mn_ - beta) ** 2); ro[k_] = np.sum(lam_cov * (w_or_ - beta) ** 2); rr2[k_] = np.sum(lam_cov * (w_rd_ - beta) ** 2)
    print(f"      {eps_:6.3f} (isotropic)" if eps_ == 1.0 else f"      {eps_:6.3f}            ", end="")
    print(f"                {rm.mean():8.4f} ± {se_mean(rm):6.4f}                       {ro.mean():8.4f} ± {se_mean(ro):6.4f}                     {rr2.mean():8.4f} ± {se_mean(rr2):6.4f}")
print("  (with an isotropic tail the interpolant loses 90% of the signal; as the tail's variance shrinks the 995 weak directions absorb the noise and the min-norm interpolant comes within a factor of three of the oracle that knew the five features — Bartlett, Long, Lugosi & Tsigler's benign overfitting, with the optimiser's implicit bias doing the absorbing)")

# ----------------------------------------------------------------------------------------------
# PART IX — the counter-examples, computed. Appended after §41 so §1–§41 draw the same stream.
# ----------------------------------------------------------------------------------------------

sec(42, "Heavy ball's rate has a prefactor that grows with T: at the extreme eigenvalues the companion matrix has a double root, so x_T = (1 + c T) q^T x_0, not C q^T x_0")
# Polyak's constants on a quadratic with eigenvalues in [mu, L]: alpha = 4/(sqrt L + sqrt mu)^2, beta = ((sqrt k - 1)/(sqrt k + 1))^2 = q^2.
# Along an eigen-direction of curvature lam the iteration is x_{t+1} = (1 + beta - alpha lam) x_t - beta x_{t-1}; its two characteristic roots have
# modulus sqrt(beta) = q for every lam in [mu, L] (that is the sqrt(kappa) rate), and at lam = mu and lam = L they COINCIDE (double root +q, resp. -q),
# so the solution there is (A + B t) q^t: a prefactor linear in t. With x_{-1} = x_0 (zero initial momentum): x_t = (1 + (1 - q) t) q^t x_0 at lam = mu
# and x_t = (1 + (1 + q) t) (-q)^t x_0 at lam = L. No T-independent C satisfies ||x_T|| <= C q^T ||x_0||; the correct statements carry a polynomial
# prefactor, or read C_delta (q + delta)^T for any delta > 0 — and the O(sqrt(kappa) log(1/eps)) iteration count survives both.
kappa_hb = 100.0; L_hb, mu_hb = kappa_hb, 1.0
alpha_hb = 4 / (sqrt(L_hb) + sqrt(mu_hb)) ** 2; q_hb = (sqrt(kappa_hb) - 1) / (sqrt(kappa_hb) + 1); beta_hb = q_hb ** 2
def hb_ratio(lam_, T_):
    x_prev, x_cur = 1.0, 1.0
    for _ in range(T_):
        x_prev, x_cur = x_cur, (1 + beta_hb - alpha_hb * lam_) * x_cur - beta_hb * x_prev
    return abs(x_cur)
print(f"  kappa = {kappa_hb:.0f}: alpha = {alpha_hb:.5f}, beta = q^2 = {beta_hb:.5f}, q = {q_hb:.5f}; double root +q at lam = mu, -q at lam = L; x_{{-1}} = x_0")
print("      T    |x_T|/|x_0| at lam = mu    (1 + (1-q) T) q^T    at lam = L     (1 + (1+q) T) q^T    max over 2001 lam in [mu, L]     q^T alone")
lams_hb = np.linspace(mu_hb, L_hb, 2001)
for T_ in (10, 100, 1000):
    r_mu = hb_ratio(mu_hb, T_); r_L = hb_ratio(L_hb, T_); r_max = max(hb_ratio(l_, T_) for l_ in lams_hb)
    print(f"  {T_:5d}      {r_mu:11.4e}          {(1 + (1 - q_hb) * T_) * q_hb ** T_:11.4e}      {r_L:11.4e}     {(1 + (1 + q_hb) * T_) * q_hb ** T_:11.4e}         {r_max:11.4e}          {q_hb ** T_:11.4e}")
print(f"  (the prefactor at T = 100 is {1 + (1 - q_hb) * 100:.1f} at the soft end and {1 + (1 + q_hb) * 100:.1f} at the stiff end, and both grow without bound: the rate q^T is right, the 'constant' in front of it is not a constant)")

sec(43, "The KL-penalised objective is concave in the policy's probabilities and NOT in its logits: a two-action counter-example")
# max_p <r, p> - beta KL(p || pi_ref) is concave on the simplex (Identity 2, §6). Parametrise a two-action policy by one logit z, p = sigmoid(z); take
# r = 0, pi_ref uniform, beta = 1: J(z) = -KL(Ber(p(z)) || Ber(1/2)) = H(p(z)) - log 2.  dJ/dz = -z p (1 - p);  d2J/dz2 = -p (1 - p) [1 + z (1 - 2p)].
# Concavity in z needs d2J/dz2 <= 0 everywhere; at z = 2 it is positive.
def J_logit(z): p = 1 / (1 + exp(-z)); return -(p * log(2 * p) + (1 - p) * log(2 * (1 - p)))
def J2_closed(z): p = 1 / (1 + exp(-z)); return -p * (1 - p) * (1 + z * (1 - 2 * p))
print("      z     J(z) = -KL(Ber(sigmoid z) || Ber(1/2))     J''(z) closed form     J''(z) by central differences (h = 1e-4)")
for z_ in (0.0, 1.0, 1.5, 2.0, 3.0):
    h_ = 1e-4; fd_ = (J_logit(z_ + h_) - 2 * J_logit(z_) + J_logit(z_ - h_)) / h_ ** 2
    print(f"  {z_:5.1f}     {J_logit(z_):12.6f}                                {J2_closed(z_):+10.5f}             {fd_:+10.5f}")
print("  (J'' > 0 at z = 2 and z = 3: the objective is not concave in the logit although it is strictly concave in p; the sign flips once z (2p - 1) > 1, i.e. once the policy is confident — the sigmoid's own curvature overwhelms the KL's)")

sec(44, "Interval bound propagation is not exact even with every ReLU stable: two copies of one pre-activation, subtracted")
# x in [0, 1]; h1 = relu(x + 1), h2 = relu(x + 1); out = h1 - h2 = 0 for every x. Both pre-activations lie in [1, 2]: both ReLUs are stable (always on).
# Interval arithmetic carries h1 and h2 as independent intervals [1, 2] and returns out in [1 - 2, 2 - 1] = [-1, 1]: the dependency problem, not the ReLUs.
# A symbolic (linear) bound keeps out = (x + 1) - (x + 1) = 0 and is exact — which is why the verifiers of §35 propagate linear bounds (CROWN, beta-CROWN).
W1_d = np.array([[1.0], [1.0]]); b1_d = np.array([1.0, 1.0]); W2_d = np.array([1.0, -1.0]); b2_d = 0.0
lo_x, hi_x = 0.0, 1.0
lo1 = W1_d.clip(min=0) @ [lo_x] + W1_d.clip(max=0) @ [hi_x] + b1_d; hi1 = W1_d.clip(min=0) @ [hi_x] + W1_d.clip(max=0) @ [lo_x] + b1_d    # one affine layer: exact per coordinate
lo1r, hi1r = np.maximum(lo1, 0), np.maximum(hi1, 0)
lo_out = W2_d.clip(min=0) @ lo1r + W2_d.clip(max=0) @ hi1r + b2_d; hi_out = W2_d.clip(min=0) @ hi1r + W2_d.clip(max=0) @ lo1r + b2_d
xs_d = np.linspace(lo_x, hi_x, 10001); true_out = np.maximum(xs_d + 1, 0) - np.maximum(xs_d + 1, 0)
print(f"  pre-activation intervals: h1 in [{lo1[0]:.0f}, {hi1[0]:.0f}], h2 in [{lo1[1]:.0f}, {hi1[1]:.0f}]  ->  unstable ReLUs: 0/2 (both always on)")
print(f"  IBP bound on out = h1 - h2: [{lo_out:+.0f}, {hi_out:+.0f}];  true range over x in [0, 1]: [{true_out.min():+.0f}, {true_out.max():+.0f}]   (loose by 1 on each side with no unstable ReLU)")
A_sym = W2_d @ W1_d; c_sym = W2_d @ b1_d + b2_d                                            # carry each neuron as an affine function of x (exact through an always-on ReLU)
print(f"  linear (symbolic) bound: out = {A_sym[0]:+.0f} x {c_sym:+.0f}  ->  [{min(A_sym[0]*lo_x, A_sym[0]*hi_x)+c_sym:+.0f}, {max(A_sym[0]*lo_x, A_sym[0]*hi_x)+c_sym:+.0f}]: exact, because it remembers that h1 and h2 are the same function of x")
print("  (stable ReLUs remove the activation's contribution to looseness; they do not remove the dependency loss of composing affine maps through intervals — that is what linear bounds fix, and unstable ReLUs are what branching pays for)")

sec(45, "GRPO's group-mean baseline includes the sample's own reward: the estimator's expectation is (1 - 1/G) times the policy gradient, and leave-one-out restores it")
# Tabular softmax policy over K actions with logits theta; fixed rewards r; J(theta) = sum_a pi_a r_a; exact grad J = sum_a pi_a r_a (e_a - pi).
# Group estimator with G on-policy samples: g_hat = (1/G) sum_i (r_i - rbar) grad log pi(a_i), rbar = (1/G) sum_j r_j — which includes r_i.
# E[g_hat] = (1 - 1/G) grad J: the j = i term of the baseline is not independent of the score. Leave-one-out baseline rbar_{-i}: unbiased.
# GRPO also divides by the group's reward std: a ratio of random variables, no longer a scaled gradient in expectation; its cosine with grad J is measured.
K_g = 4; theta_g = np.array([0.5, -0.2, 0.1, -0.4]); r_g = np.array([1.0, 0.0, 0.5, 2.0])
pi_g = np.exp(theta_g) / np.exp(theta_g).sum(); grad_J = pi_g * r_g - pi_g * (pi_g @ r_g)
print(f"  pi = {pi_g}, r = {r_g}, J = {pi_g @ r_g:.4f}, ||grad J|| = {np.linalg.norm(grad_J):.4f}; 100,000 groups per G")
print("      G    <E g_hat, grad J> / ||grad J||^2:  group mean incl. self    expected 1 - 1/G    leave-one-out mean       GRPO (mean, then / std): scale and cosine with grad J")
N_g = 100000
for G_ in (2, 4, 8, 16):
    a_ = rng.choice(K_g, size=(N_g, G_), p=pi_g); r_s = r_g[a_]
    score = np.eye(K_g)[a_] - pi_g[None, None, :]                                           # grad log pi(a) = e_a - pi, shape (N_g, G, K)
    adv_mean = r_s - r_s.mean(1, keepdims=True)
    adv_loo = r_s - (r_s.sum(1, keepdims=True) - r_s) / (G_ - 1)
    sd_ = r_s.std(1, keepdims=True); adv_grpo = adv_mean / np.where(sd_ > 0, sd_, 1.0)
    g_mean = np.einsum('ng,ngk->nk', adv_mean, score) / G_; g_loo = np.einsum('ng,ngk->nk', adv_loo, score) / G_; g_grpo = np.einsum('ng,ngk->nk', adv_grpo, score) / G_
    proj = lambda g: (g.mean(0) @ grad_J) / (grad_J @ grad_J)
    proj_se = lambda g: se_mean(g @ grad_J / (grad_J @ grad_J))
    cos_ = g_grpo.mean(0) @ grad_J / (np.linalg.norm(g_grpo.mean(0)) * np.linalg.norm(grad_J))
    print(f"  {G_:5d}                                        {proj(g_mean):7.4f} ± {proj_se(g_mean):6.4f}        {1 - 1/G_:7.4f}         {proj(g_loo):7.4f} ± {proj_se(g_loo):6.4f}          scale {proj(g_grpo):7.4f}, cosine {cos_:7.4f}")
print("  (the group mean including the sample's own reward is biased by exactly the factor 1 - 1/G — a scaled gradient, so the direction survives and the learning rate absorbs the scale; the std-normalised GRPO advantage is not a scaled gradient, and the last column is how far its direction is from grad J on this problem)")

sec(46, "Adam's per-coordinate step is not capped at eta: after a long quiet stretch one gradient moves a coordinate by up to eta (1 - beta1)/sqrt(1 - beta2) = 3.16 eta")
# m_t = b1 m + (1-b1) g, v_t = b2 v + (1-b2) g^2, both bias-corrected; step = eta m_hat / sqrt(v_hat). The 'trust region of radius eta' reading holds while
# g is CONSTANT (then m_hat = g, sqrt(v_hat) = |g|, step = eta); a Jensen bound |m_hat| <= sqrt(v_hat) would need both averages to carry the same
# weights, and with b1 != b2 they do not (§48 is the supremum that gap allows). It fails outright when a large gradient arrives after many near-zero ones:
# m is then (1-b1) g and v is (1-b2) g^2, so the ratio is (1-b1)/sqrt(1-b2) once the bias corrections have decayed (Kingma & Ba 2015, §2.1).
b1_a, b2_a = 0.9, 0.999
print("      t0 (quiet steps, then one unit gradient)     |step| / eta at t0     bound (1-b1)/sqrt(1-b2)")
for t0_ in (1, 10, 100, 1000, 10000):
    m_a = v_a = 0.0
    for t_ in range(1, t0_ + 1):
        g_ = 1.0 if t_ == t0_ else 0.0
        m_a = b1_a * m_a + (1 - b1_a) * g_; v_a = b2_a * v_a + (1 - b2_a) * g_ * g_
    mhat = m_a / (1 - b1_a ** t0_); vhat = v_a / (1 - b2_a ** t0_)
    print(f"  {t0_:10d}                                         {abs(mhat)/sqrt(vhat):8.4f}                {(1-b1_a)/sqrt(1-b2_a):8.4f}")
print("  (at t0 = 1 the step is the sign step of §20; it falls below eta while v's bias correction is still large and climbs to the 3.16 eta cap as the quiet stretch outlives 1/(1 - beta2): eta bounds the step only up to this factor, and only through v's memory)")

sec(47, "Checkpoint 2: 1.99/L and 1/L have the SAME contraction factor 0.99 on E1 and losses 63x apart — the oscillating direction carries curvature L, the slow one carries curvature mu")
lam47 = np.logspace(0, 2, 20); L47 = lam47.max()
for c in (1.0, 1.99):
    eta = c / L47; fac = 1 - eta * lam47; x = fac ** 200                                      # x_200 per eigen-direction from x_0 = 1
    f_dir = 0.5 * lam47 * x * x; worst = int(np.argmax(np.abs(fac)))
    print(f"  eta = {c:4.2f}/L: max|1 - eta lambda| = {np.abs(fac).max():.4f} at lambda = {lam47[worst]:6.2f}; that direction's share of f(x_200) = {f_dir[worst]:.4e} = (1/2) lambda 0.99^400; total f(x_200) = {f_dir.sum():.4e}")
ratio47 = (0.5 * lam47 * (1 - 1.99 / L47 * lam47) ** 400).sum() / (0.5 * lam47 * (1 - 1 / L47 * lam47) ** 400).sum()
print(f"  ratio f_200(1.99/L) / f_200(1/L) = {ratio47:.1f}: the same amplitude decay 0.99^200 in the worst direction, weighted by a curvature 100x larger (less what the other directions still hold at 1/L)")

# ----------------------------------------------------------------------------------------------
# PART X — five more counter-examples, computed. Appended after §47 so §1–§47 draw the same stream.
# ----------------------------------------------------------------------------------------------

sec(48, "Adam's per-coordinate step over ALL gradient histories: Cauchy-Schwarz bounds |m_hat|/sqrt(v_hat) by (1-b1)/sqrt(1-b2) * sqrt((1-(b1^2/b2)^t)/(1-b1^2/b2)) * sqrt(1-b2^t)/(1-b1^t) -> 7.27 at the defaults, and the history g_j = (b1/b2)^j attains it")
# m_t = (1-b1) sum_j b1^j g_{t-j}, v_t = (1-b2) sum_j b2^j g_{t-j}^2 (j = age). Write b1^j g = (b1^j / b2^{j/2}) (b2^{j/2} g) and apply Cauchy-Schwarz:
# |m_t| <= (1-b1) sqrt(sum_j (b1^2/b2)^j) sqrt(sum_j b2^j g^2) = (1-b1) sqrt((1-rho^t)/(1-rho)) sqrt(v_t/(1-b2)), rho = b1^2/b2 < 1. Equality when
# b2^{j/2} g_{t-j} is proportional to b1^j / b2^{j/2}, i.e. g_{t-j} proportional to (b1/b2)^j: a gradient that DECAYS geometrically into the past, the
# history that §46's single spike is only the first term of. Bias corrections multiply by sqrt(1-b2^t)/(1-b1^t). The supremum at the defaults is 7.27,
# not 3.16: §46 is one history, this is the maximum over all of them (epsilon = 0 throughout; a non-negligible epsilon lowers both).
b1_s, b2_s = 0.9, 0.999; rho_s = b1_s ** 2 / b2_s
def adam_sup(t_): return (1 - b1_s) / sqrt(1 - b2_s) * sqrt((1 - rho_s ** t_) / (1 - rho_s)) * sqrt(1 - b2_s ** t_) / (1 - b1_s ** t_)
def adam_ratio(gs):                                                # run Adam's two averages on a gradient sequence and return |m_hat| / sqrt(v_hat) at its end
    m_ = v_ = 0.0
    for t_, g_ in enumerate(gs, 1):
        m_ = b1_s * m_ + (1 - b1_s) * g_; v_ = b2_s * v_ + (1 - b2_s) * g_ * g_
    return (m_ / (1 - b1_s ** len(gs))) / sqrt(v_ / (1 - b2_s ** len(gs)))
print(f"  b1 = {b1_s}, b2 = {b2_s}: rho = b1^2/b2 = {rho_s:.6f}; single-spike figure (1-b1)/sqrt(1-b2) = {(1-b1_s)/sqrt(1-b2_s):.4f}; supremum over histories (t -> inf) = {(1-b1_s)/sqrt(1-b2_s)/sqrt(1-rho_s):.4f}")
print("      t      single spike at t (§46)     Cauchy-Schwarz bound at t     geometric history g_{t-j} = (b1/b2)^j, run through Adam")
for t_ in (1, 10, 100, 1000, 10000):
    spike = adam_ratio([0.0] * (t_ - 1) + [1.0]); geo = adam_ratio([(b1_s / b2_s) ** (t_ - s_) for s_ in range(1, t_ + 1)])
    print(f"  {t_:6d}          {spike:8.4f}                    {adam_sup(t_):8.4f}                        {geo:8.4f}")
print("  (the single spike reaches 3.16 eta; a gradient that decays into the past at the rate b1/b2 reaches 7.27 eta, and nothing reaches more: eta is a per-coordinate trust radius only up to that factor, which the §4.4 clipping-and-epsilon advice is priced for)")

sec(49, "Goemans-Williamson on the triangle: the maximum cut is 2, the SDP relaxation's value is 2.25 (three unit vectors at 120 degrees), so the first level is NOT exact - 0.878 is an approximation ratio, not a certificate")
# max-cut on K3: maximise (1/2) sum_{edges} (1 - x_i x_j) over x in {-1, +1}^3. The SDP replaces x_i x_j by v_i . v_j with unit vectors v_i (any dimension).
# Upper bound on the SDP: ||v1 + v2 + v3||^2 >= 0 gives sum_{i<j} v_i . v_j >= -3/2, so (1/2)(3 - sum) <= (1/2)(3 + 3/2) = 2.25; three coplanar unit vectors at 120 degrees attain it.
cuts = [0.5 * sum(1 - x[i] * x[j] for i, j in ((0, 1), (1, 2), (0, 2))) for x in [(a_, b_, c_) for a_ in (-1, 1) for b_ in (-1, 1) for c_ in (-1, 1)]]
V_gw = np.array([[math.cos(2 * pi * k_ / 3), math.sin(2 * pi * k_ / 3)] for k_ in range(3)])
sdp_val = 0.5 * sum(1 - V_gw[i] @ V_gw[j] for i, j in ((0, 1), (1, 2), (0, 2)))
print(f"  integer optimum over the 8 cuts of the triangle: {max(cuts):.2f}  (any one vertex against the other two cuts 2 of the 3 edges; no cut takes all three)")
print(f"  SDP relaxation: three unit vectors at 120 degrees give (1/2) sum (1 - v_i . v_j) = (1/2) * 3 * (1 - ({V_gw[0] @ V_gw[1]:+.2f})) = {sdp_val:.2f};  Gram inequality ||v1+v2+v3||^2 >= 0 caps the relaxation at {0.5 * (3 + 1.5):.2f}: the SDP optimum is 2.25")
print(f"  integrality gap on this instance: {sdp_val / max(cuts):.4f} (the relaxation overstates the cut by {100 * (sdp_val / max(cuts) - 1):.1f}%); the SDP value is an UPPER bound on the cut, not its value")
thetas = np.linspace(1e-6, pi, 200001); gw_curve = (thetas / pi) / ((1 - np.cos(thetas)) / 2)
print(f"  hyperplane rounding on the triangle: each edge at angle 2pi/3 is cut with probability (2pi/3)/pi = {(2 * pi / 3) / pi:.4f}, expected cut = {3 * (2 * pi / 3) / pi:.4f} = the optimum, {3 * (2 * pi / 3) / pi / sdp_val:.4f} of the relaxed value")
print(f"  Goemans-Williamson's constant: min over theta of (theta/pi) / ((1 - cos theta)/2) = {gw_curve.min():.5f} at theta = {thetas[gw_curve.argmin()]:.4f} rad = {math.degrees(thetas[gw_curve.argmin()]):.1f} degrees — a guarantee about the ROUNDING, relative to the relaxed value; the instance is certified only when the rounded cut meets the SDP value")

sec(50, "AdamW's decoupled decay w <- (1 - eta lambda) w is the FIRST-ORDER approximation of the exact proximal step w / (1 + eta lambda) for (lambda/2)||w||^2: they differ at order (eta lambda)^2")
# prox_{eta h}(v) = argmin_u (lambda/2)||u||^2 + ||u - v||^2 / (2 eta) = v / (1 + eta lambda). AdamW (Loshchilov & Hutter 2019, Alg. 2) multiplies by 1 - eta lambda instead:
# 1/(1 + x) = 1 - x + x^2 - ..., so the two agree to first order in x = eta lambda and differ by x^2 - x^3 + ... Zhuang et al. 2022 (arXiv:2202.00089) is the paper on the exact-prox variant.
print("      eta*lambda     AdamW factor 1 - x     exact prox factor 1/(1+x)     difference     x^2 (leading term)")
for x_ in (0.001, 0.01, 0.1, 0.5):
    print(f"  {x_:10.3f}         {1 - x_:10.5f}             {1 / (1 + x_):10.5f}            {1 / (1 + x_) - (1 - x_):9.5f}       {x_ * x_:9.5f}")
print("  (at the eta*lambda of a training run, 1e-4 to 1e-2, the two are indistinguishable; the statement 'AdamW is a proximal step' is first-order true and exactly false, and §5.6's equilibrium-norm argument needs only the first order)")

sec(51, "The Gibbs law exp(-f/T) is the stationary law of the CONTINUOUS Langevin diffusion; its Euler discretisation x <- (1 - eta) x + sqrt(2 eta T) xi on f = x^2/2 has stationary variance T / (1 - eta/2), measured")
# Continuous: dx = -x dt + sqrt(2T) dW has stationary variance T (Ornstein-Uhlenbeck). Euler at step eta: x_{t+1} = (1 - eta) x_t + sqrt(2 eta T) xi_t is an AR(1) with
# stationary variance 2 eta T / (1 - (1 - eta)^2) = T / (1 - eta/2): the discretisation samples a WIDER law than exp(-f/T), by a factor that is O(eta), and that is what a
# Metropolis correction (MALA) removes. Measured below with T = 1 on an ensemble of chains after a burn-in.
T_lan = 1.0; n_ch, n_st, burn = 2000, 4000, 1000
print("      eta     closed form T/(1 - eta/2)     measured stationary variance (2000 chains x 3000 post-burn-in steps)     Gibbs target variance T")
for eta_ in (0.5, 0.1, 0.01):
    x_ = np.zeros(n_ch); acc_ = np.zeros(n_ch); cnt_ = 0
    for t_ in range(n_st):
        x_ = (1 - eta_) * x_ + sqrt(2 * eta_ * T_lan) * rng.standard_normal(n_ch)
        if t_ >= burn: acc_ += x_ * x_; cnt_ += 1
    per_chain = acc_ / cnt_                                       # each chain's time-average of x^2; chains are independent, so the SE is over chains
    print(f"  {eta_:6.2f}          {T_lan / (1 - eta_ / 2):8.4f}                        {per_chain.mean():8.4f} ± {se_mean(per_chain):6.4f}                                         {T_lan:8.4f}")
print("  (the chain at eta = 0.5 sits a third too wide, at 0.1 five per cent, at 0.01 half a per cent: the Gibbs statement is exact for the diffusion and O(eta) wrong for the update a loop actually runs; Hajek's log-cooling theorem is about a third object again, a finite-state annealing chain)")

sec(52, "Optimistic gradient and extragradient on the bilinear saddle contract at COMPARABLE, not identical, rates: spectral radius 0.99494 vs 0.99504 at eta = 0.1")
# Extragradient on v(z) = J z: z <- (I - eta J + eta^2 J^2) z, a normal matrix, so ||z|| contracts by exactly its spectral radius sqrt(1 - eta^2 + eta^4) per step.
# Optimistic gradient (§37's update): z_{t+1} = z_t - 2 eta J z_t + eta J z_{t-1}, a two-step recursion whose 4x4 companion matrix [[I - 2 eta J, eta J], [I, 0]] is not normal:
# its spectral radius sets the asymptotic rate and its transient explains why ||z_200|| in §37 is not the 200th power of it times sqrt(2).
eta_52 = 0.1; J_52 = np.array([[0.0, 1.0], [-1.0, 0.0]]); I2 = np.eye(2)
M_eg = I2 - eta_52 * J_52 + eta_52 ** 2 * J_52 @ J_52
M_og = np.block([[I2 - 2 * eta_52 * J_52, eta_52 * J_52], [I2, np.zeros((2, 2))]])
rho_eg = np.abs(np.linalg.eigvals(M_eg)).max(); rho_og = np.abs(np.linalg.eigvals(M_og)).max()
print(f"  extragradient:  spectral radius {rho_eg:.6f} = sqrt(1 - eta^2 + eta^4) = {sqrt(1 - eta_52 ** 2 + eta_52 ** 4):.6f};  rho^200 * sqrt(2) = {rho_eg ** 200 * sqrt(2):.4f}  (§37 printed 5.2290e-01: exact, the matrix is normal)")
print(f"  optimistic:     spectral radius {rho_og:.6f};                                      rho^200 * sqrt(2) = {rho_og ** 200 * sqrt(2):.4f}  (§37 printed 5.1762e-01: the companion matrix is not normal, so the 200-step norm carries a transient)")
print(f"  ratio of the two per-step factors: {rho_og / rho_eg:.6f}; over 200 steps the optimistic method is ahead by a factor {(rho_eg / rho_og) ** 200:.4f} — comparable geometric rates, the same O(1/T) duality-gap theorem (Nemirovski 2004; Mokhtari, Ozdaglar & Pattathil 2020 for the optimistic method), not the same number")

sec(53, "The §22 table read correctly: on the convex chain min_t ||grad||^2 falls like T^-3/2, so the ratio to the O(1/T) bound falls like T^-1/2 (not T^-2 and T^-1)")
# no draws: recompute §22's deterministic GD run on Nesterov's chain and fit log-log slopes to the four printed rows.
d53 = 2001; L53 = 1.0
def f_53(x): return L53 / 4 * (0.5 * (x[0] ** 2 + np.sum((x[:-1] - x[1:]) ** 2) + x[-1] ** 2) - x[0])
def g_53(x):
    g = np.zeros_like(x); dif = x[:-1] - x[1:]
    g[0] += x[0]; g[:-1] += dif; g[1:] -= dif; g[-1] += x[-1]; g[0] -= 1
    return L53 / 4 * g
xs53 = 1 - np.arange(1, d53 + 1) / (d53 + 1); fs53 = f_53(xs53)
x53 = np.zeros(d53); f053 = f_53(x53); gm53 = np.inf; rows53 = []
for t in range(1, 10001):
    g = g_53(x53); gm53 = min(gm53, g @ g); x53 = x53 - g / L53
    if t in (10, 100, 1000, 10000): rows53.append((t, gm53, 2 * L53 * (f053 - fs53) / t))
Ts = np.array([r[0] for r in rows53], float); G2 = np.array([r[1] for r in rows53]); Bd = np.array([r[2] for r in rows53])
slope_g2 = np.polyfit(np.log(Ts), np.log(G2), 1)[0]; slope_ratio = np.polyfit(np.log(Ts), np.log(G2 / Bd), 1)[0]
print("      T    min_t ||grad||^2   per-decade factor   (same run as §22; the printed column is the SQUARED norm)")
for i, (t, g2, b) in enumerate(rows53):
    print(f"  {t:6d}    {g2:14.3e}   {'' if i == 0 else f'{rows53[i-1][1] / g2:8.2f}':>8s}")
print(f"  log-log slope of min_t ||grad||^2 against T: {slope_g2:.3f}  (T^-3/2 would be -1.500; a factor {10 ** 1.5:.1f} per decade, not 100)")
print(f"  log-log slope of the ratio to the bound 2L(f0 - f*)/T: {slope_ratio:.3f}  (the bound is exactly T^-1, so ratio ~ T^{slope_g2 + 1:+.3f})")
print(f"  unsquared ||grad|| therefore falls like T^{slope_g2 / 2:.3f}; the earlier prose's 'gradient norm decays like 1/T^2' and 'ratio falls like 1/T' were misreadings of this table, corrected in round 3")

print(f"\nDONE (§1–§53) in {time.time()-T0:.1f} s")
