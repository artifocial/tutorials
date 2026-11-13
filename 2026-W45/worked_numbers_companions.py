#!/usr/bin/env python3
"""worked_numbers_companions.py — every number printed in the two W46 companion basics of "The Optimization Domain":
   A§  -> weekly/2026-W46/basics/gradient-descent-explained.md
   B§  -> weekly/2026-W46/basics/adam-muon-and-friends-explained.md
NumPy only, float64, one fixed seed (20261112), sections draw from the generator in order.
Run:  python3 worked_numbers_companions.py > worked_numbers_companions_output.txt
(The W39 notebook venv has NumPy: repo/notebooks/2026-W39/.venv/bin/python.)  About half a minute.
Where a section re-computes a quantity of the flagship's worked_numbers.py (§7, §12, §13, §14, §19, §20, §21, §46, §48) it does so
under its own seed and its own instance, so a reader running this script reproduces the companions' numbers exactly and sees the
flagship's agree where the quantity is deterministic and to sampling error where it is not.
"""

import sys, time, math
import numpy as np
from math import sqrt, log, exp, pi

sys.stdout.reconfigure(line_buffering=True)
T0 = time.time()
rng = np.random.default_rng(20261112)
np.set_printoptions(precision=4, suppress=True)


def sec(tag, title):
    print(f"\n=== {tag} {title} ===")


# =============================================================================================
# PART A — gradient-descent-explained.md
# =============================================================================================
sec("A1", "One coordinate, f(x) = h x^2 / 2: each gradient step multiplies x by (1 - eta h); the step is stable iff eta h < 2")
h, x0 = 1.0, 1.0
print("  eta*h    factor (1 - eta h)    x after 20 steps (x0 = 1)")
for eh in [0.5, 1.0, 1.5, 1.9, 2.0, 2.1]:
    fac = 1 - eh
    print(f"  {eh:4.1f}      {fac:+8.3f}             {fac**20:12.4e}")
print("  eta h = 1 is the Newton step (one step, exact); 1 < eta h < 2 oscillates and still converges; eta h = 2 bounces forever; eta h > 2 diverges")

sec("A2", "Two coordinates with curvatures 1 and kappa: the safe step 1/L is set by the stiff one, the iteration count by the soft one — linear in kappa")
print("  kappa    iters to f < 1e-8 at eta = 1/L    ln(f0/eps)/(-2 ln(1 - 1/kappa))    iters at eta = 2/(L + mu)    ((kappa-1)/(kappa+1))^2t bound")
for kappa in [10, 100, 1000, 10000]:
    H = np.array([1.0, kappa])
    L, mu = kappa, 1.0
    eps = 1e-8
    out = []
    for eta in [1 / L, 2 / (L + mu)]:
        x = np.array([1.0, 1.0])
        f0 = 0.5 * np.sum(H * x**2)
        t = 0
        while 0.5 * np.sum(H * x**2) >= eps:
            x = x - eta * H * x
            t += 1
        out.append((t, f0))
    b1 = log(out[0][1] / eps) / (-2 * log(1 - 1 / kappa))
    q = (kappa - 1) / (kappa + 1)
    b2 = log(out[1][1] / eps) / (-2 * log(q))
    print(f"  {kappa:5d}         {out[0][0]:7d}                          {b1:9.1f}                      {out[1][0]:7d}                   {b2:9.1f}")
print("  the bound charges all of f0 to the soft coordinate; at eta = 1/L the stiff one, which holds kappa/(kappa+1) of f0, dies in one step, so the measured count sits under the bound by ln(1+kappa)/(-2 ln(1-1/kappa)) steps: " + ", ".join(f"{log(1+k)/(-2*log(1-1/k)):.0f}" for k in [10, 100, 1000, 10000]))

sec("A3", "The descent lemma on a logistic regression: one step of 1/L must cut the loss by at least ||g||^2 / (2L) — measured against the guarantee")
n, d = 200, 5
X = rng.normal(size=(n, d))
w_true = rng.normal(size=d)
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ w_true))).astype(float) * 2 - 1  # labels in {-1, +1}
lam = 1e-3


def loss(w):
    z = y * (X @ w)
    return np.mean(np.logaddexp(0, -z)) + 0.5 * lam * np.sum(w**2)


def grad(w):
    z = y * (X @ w)
    s = -y / (1 + np.exp(z))  # d/dz log(1 + e^{-z}) * y
    return X.T @ s / n + lam * w


def hess(w):
    z = X @ w
    p = 1 / (1 + np.exp(-z))
    return X.T @ (X * (p * (1 - p))[:, None]) / n + lam * np.eye(d)


Lsm = np.linalg.eigvalsh(X.T @ X).max() / (4 * n) + lam  # smoothness bound: sigma'(z) <= 1/4
mu_sc = lam  # strong convexity from the ridge (the data term is only convex)
w0 = rng.normal(size=d) * 2
g0 = grad(w0)
f0 = loss(w0)
print(f"  n = {n}, d = {d}, ridge {lam:g}: L <= lambda_max(X^T X)/(4n) + lambda = {Lsm:.4f}, mu = {mu_sc:g}, kappa <= {Lsm/mu_sc:.0f}")
G1 = np.sum(g0**2) / (2 * Lsm)
print(f"  f(w0) = {f0:.6f}, ||grad f(w0)||^2 / (2L) = {G1:.6f} (the guaranteed decrease at eta = 1/L)")
print("  eta * L    f(w0 - eta g)      actual decrease    lemma's guarantee at THIS eta: eta (1 - eta L/2) ||g||^2    decrease / own guarantee    decrease / the 1/L guarantee")
for k in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
    f1 = loss(w0 - k / Lsm * g0)
    dec = f0 - f1
    gk = G1 * k * (2 - k)  # eta (1 - eta L/2) ||g||^2 at eta = k/L; zero at 2/L, none beyond
    gk_s = f"{gk:.6f}" if gk > 0 else "none (eta >= 2/L)"
    own = f"{dec/gk:+7.3f}" if gk > 0 else "      -"
    print(f"  {k:4.1f}       {f1:.6f}         {dec:+.6f}          {gk_s:>26}                        {own}                     {dec/G1:+7.3f}")
print("  the lemma guarantees eta (1 - eta L/2) ||g||^2 at every eta < 2/L: largest at eta = 1/L, zero at 2/L, nothing beyond; each row beats its own guarantee, and 3/L and 4/L, which have none, still helped here because L bounds the worst curvature and this start is not in it")

sec("A4", "Fixed step 1/L against backtracking line search (Armijo, start 10/L, halve) on the same logistic regression: iterations and loss evaluations to ||grad|| < 1e-6")
# reference minimiser by Newton
w = w0.copy()
for _ in range(200):  # damped Newton: a full step from a far start (where the Hessian is ~ lambda I) would fly off
    g = grad(w)
    if np.linalg.norm(g) < 1e-13:
        break
    dn = np.linalg.solve(hess(w), g)
    ts = 1.0
    while loss(w - ts * dn) > loss(w) - 1e-4 * ts * (g @ dn):
        ts *= 0.5
    w = w - ts * dn
w_star, f_star = w.copy(), loss(w)
print(f"  reference minimiser by damped Newton's method: f* = {f_star:.10f}, ||grad f(w*)|| = {np.linalg.norm(grad(w_star)):.1e}, ||w*|| = {np.linalg.norm(w_star):.4f}")
w = w0.copy()
t, fe = 0, 0
while np.linalg.norm(grad(w)) >= 1e-6 and t < 200000:
    w = w - grad(w) / Lsm
    t += 1
print(f"  fixed eta = 1/L:          {t:6d} iterations, {t:6d} gradients, f - f* = {loss(w)-f_star:.2e}")
w = w0.copy()
t, fe = 0, 0
while np.linalg.norm(grad(w)) >= 1e-6 and t < 200000:
    g = grad(w)
    f = loss(w)
    eta = 10 / Lsm
    fe += 1
    while loss(w - eta * g) > f - 0.5 * eta * np.sum(g**2):
        eta *= 0.5
        fe += 1
    w = w - eta * g
    t += 1
print(f"  backtracking (Armijo):    {t:6d} iterations, {t:6d} gradients + {fe:6d} loss evaluations, f - f* = {loss(w)-f_star:.2e}, last accepted eta * L = {eta*Lsm:.2f}")

sec("A5", "The two convex rates on the logistic regression: f(w_T) - f* against L||w0 - w*||^2 / (2T) and against (1 - mu/L)^T (L/2)||w0 - w*||^2")
D2 = np.sum((w0 - w_star) ** 2)
print(f"  ||w0 - w*||^2 = {D2:.4f}")
print("      T      f(w_T) - f*      smooth-convex bound L D^2/(2T)      strongly-convex bound (1 - mu/L)^T L D^2/2")
w = w0.copy()
T = 0
for Tmark in [1, 10, 100, 1000, 10000]:
    while T < Tmark:
        w = w - grad(w) / Lsm
        T += 1
    print(f"  {Tmark:6d}     {loss(w)-f_star:12.4e}         {Lsm*D2/(2*Tmark):12.4e}                   {(1-mu_sc/Lsm)**Tmark*Lsm*D2/2:12.4e}")
print("  both bounds hold at every T and both are loose: the data term is itself strongly convex near w*, with a curvature the ridge-only mu = 0.001 does not see, so the measured gap falls far faster than either; 5.55e-17 is float64's floor")

sec("A6", "Momentum on a 50-dimensional quadratic (curvatures log-spaced from 1 to kappa): GD, heavy ball, Nesterov — iterations to f < 1e-8")
dq = 50
print("  kappa     GD iters    GD theory kappa ln(f0/eps)/2    heavy ball iters    Nesterov iters    sqrt(kappa) ln(.)")
for kappa in [100, 1000, 10000]:
    H = np.logspace(0, log(kappa, 10), dq)
    L, mu = H.max(), H.min()
    x0 = rng.normal(size=dq)
    x0 = x0 / np.sqrt(np.mean(x0**2))  # unit RMS start
    f0 = 0.5 * np.sum(H * x0**2)
    eps = 1e-8
    # GD
    x, t = x0.copy(), 0
    while 0.5 * np.sum(H * x**2) >= eps:
        x -= H * x / L
        t += 1
    tg = t
    # heavy ball at Polyak's constants
    q = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
    alpha, beta = 4 / (sqrt(L) + sqrt(mu)) ** 2, q**2
    x, xp, t = x0.copy(), x0.copy(), 0
    while 0.5 * np.sum(H * x**2) >= eps:
        xn = x - alpha * H * x + beta * (x - xp)
        xp, x = x, xn
        t += 1
    th = t
    # Nesterov, strongly convex constants
    beta_n = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
    x, xp, t = x0.copy(), x0.copy(), 0
    while 0.5 * np.sum(H * x**2) >= eps:
        yv = x + beta_n * (x - xp)
        xn = yv - H * yv / L
        xp, x = x, xn
        t += 1
    tn = t
    print(f"  {kappa:5d}      {tg:7d}        {kappa*log(f0/eps)/2:9.0f}                   {th:7d}             {tn:7d}           {sqrt(kappa)*log(f0/eps):8.0f}")
# effective learning rate of heavy ball at a constant gradient
alpha, beta, g = 0.1, 0.9, 1.0
v, steps = 0.0, []
for t in range(60):
    v = beta * v + alpha * g
    steps.append(v)
print(f"  heavy ball at a constant unit gradient, alpha = {alpha}, beta = {beta}: step / alpha after t = 1, 5, 10, 20, 60 steps = {steps[0]/alpha:.3f}, {steps[4]/alpha:.3f}, {steps[9]/alpha:.3f}, {steps[19]/alpha:.3f}, {steps[59]/alpha:.3f}  (limit alpha/(1-beta) / alpha = {1/(1-beta):.1f})")

sec("A7", "SGD on one noisy coordinate: the loss floor is proportional to the step — measured over 4000 chains against eta sigma^2 / (2 (2 - eta h))")
h, sigma, chains, steps = 1.0, 1.0, 4000, 3000
print("    eta     measured E f (last 1500 steps)    theory eta sigma^2 / (2(2 - eta h))    ratio")
for eta in [0.5, 0.1, 0.02, 0.004]:
    x = np.full(chains, 3.0)
    acc = 0.0
    for t in range(steps):
        x = x - eta * (h * x + sigma * rng.normal(size=chains))
        if t >= steps // 2:
            acc += np.mean(0.5 * h * x**2)
    meas = acc / (steps - steps // 2)
    th = eta * sigma**2 / (2 * (2 - eta * h))
    print(f"  {eta:6.3f}        {meas:10.5f}                        {th:10.5f}                   {meas/th:6.3f}")
print(f"  (eta = 0.004 has a time constant 1/(eta h) = {1/(0.004*h):.0f} steps from a start of 3, so its last-half average still carries a little of the transient)")

sec("A8", "Schedules on a noisy quadratic, curvatures 1/i, exact expectations: constant, cosine, warmup-stable-decay, 1/sqrt t — in a transient-limited and a floor-limited run; tail averaging, simulated")
Tn, cB, eta_max = 1000, 1 / 64, 1.0  # largest curvature is 1, so eta = 1 is the Newton step for it and stable for all


def exact_loss(Hn, etas):
    s = np.ones(len(Hn))  # E x_i^2, start at 1 in every coordinate
    out = {}
    for t, eta in enumerate(etas, 1):
        s = (1 - eta * Hn) ** 2 * s + eta**2 * Hn * cB
        if t in (200, 800, 1000):
            out[t] = 0.5 * np.sum(Hn * s)
    return out


sched = {
    "constant eta_max": np.full(Tn, eta_max),
    "cosine to 0": eta_max * 0.5 * (1 + np.cos(pi * np.arange(Tn) / Tn)),
    "WSD (80% const, 20% linear)": np.concatenate([np.full(int(0.8 * Tn), eta_max), eta_max * np.linspace(1, 0, Tn - int(0.8 * Tn), endpoint=False)]),
    "1/sqrt(t)": eta_max / np.sqrt(np.arange(1, Tn + 1)),
    "constant eta_max/10": np.full(Tn, eta_max / 10),
}
for dn, label in [(1000, "transient-limited: d = 1000, slowest time constant 1/(eta h_min) = 1000 = T"), (200, "floor-limited: d = 200, slowest time constant 200 = T/5")]:
    Hn = 1 / np.arange(1, dn + 1)
    print(f"  {label}; h_i = 1/i, noise c/B = {cB:g}, T = {Tn}; expected loss at t = 200, 800, 1000:")
    for name, etas in sched.items():
        o = exact_loss(Hn, etas)
        print(f"    {name:32s} {o[200]:9.5f}   {o[800]:9.5f}   {o[1000]:9.5f}")
# tail averaging, simulated, on the transient-limited case
dn = 1000
Hn = 1 / np.arange(1, dn + 1)
runs = 20
last, avg = [], []
for r in range(runs):
    x = np.ones(dn)
    tail = []
    for t in range(Tn):
        x = x - eta_max * (Hn * x + np.sqrt(Hn * cB) * rng.normal(size=dn))
        if t >= int(0.75 * Tn):
            tail.append(x.copy())
    last.append(0.5 * np.sum(Hn * x**2))
    xa = np.mean(tail, axis=0)
    avg.append(0.5 * np.sum(Hn * xa**2))
print(f"  d = 1000, constant eta_max, {runs} simulated runs: last-iterate loss {np.mean(last):.5f} +- {np.std(last)/sqrt(runs):.5f}; loss of the average over the last 25% {np.mean(avg):.5f} +- {np.std(avg)/sqrt(runs):.5f}")

sec("A9", "Non-convex, two ways: f(x) = x^2 + 3 sin^2 x (PL: every stationary point is the minimum) against f(x) = x^4/4 - x^2/2 (two minima, one saddle)")
print("  PL example, eta = 1/L = 1/8, 200 steps:        start     final x      final f      |f'(x)|")
for s in [-3.0, 2.0, 5.0, 10.0]:
    x = s
    for _ in range(200):
        x -= (2 * x + 3 * math.sin(2 * x)) / 8
    print(f"                                                {s:5.1f}    {x:+9.5f}    {x**2+3*math.sin(x)**2:9.2e}    {abs(2*x+3*math.sin(2*x)):8.1e}")
print("  double well, eta = 0.1, 200 steps:                start     final x      final f")
for s in [-2.0, -0.5, 0.0, 0.5, 2.0, 3.0, 4.0, 5.0]:
    x = s
    first = []
    for _ in range(200):
        if abs(x) > 1e30:
            x = float("inf")
            break
        x -= 0.1 * (x**3 - x)
        if len(first) < 3:
            first.append(x)
    tail = f"    (first steps {', '.join(f'{v:+.4g}' for v in first)})" if abs(s) >= 3 else ""
    if np.isfinite(x):
        print(f"                                                {s:5.1f}    {x:+9.5f}    {x**4/4-x**2/2:+9.5f}" + tail)
    else:
        print(f"                                                {s:5.1f}     overflow     overflow" + tail)
print("  f(0) = 0 is a stationary point of the double well and a start of exactly 0 stays there forever; the tested starts with |x0| <= 2 fall to the NEARER minimum f = -0.25; x0 = 4 overshoots into the FAR well and x0 = 5 diverges — f''(x) = 3 x^2 - 1 passes 2/eta = 20 at |x| = 2.65, so eta = 0.1 is a step that is safe only near the wells (the local-smoothness trap of A11)")

sec("A10", "Checking a gradient by central finite differences: the relative error is U-shaped in the step h (truncation on the right, rounding on the left)")
w_chk = rng.normal(size=d)
g_an = grad(w_chk)
x_c, g_true, g_wrong, eta_c = 1.0, 1.0, 0.8, 1.0  # f(x) = x^2/2: f'(1) = 1; a wrong gradient of 0.8 at step 1
dec_c = x_c**2 / 2 - (x_c - eta_c * g_wrong) ** 2 / 2
print(f"  first, why 'the loss went down' is not the test: f(x) = x^2/2 at x = 1 has gradient 1; a WRONG gradient of 0.8 at step 1 lands at x = {x_c - eta_c*g_wrong:.1f} and cuts the loss by {dec_c:.3f}, more than the Armijo threshold (1/2) eta g_hat^2 = {0.5*eta_c*g_wrong**2:.3f} computed with the wrong gradient — the sufficient-decrease test accepts a wrong gradient; the finite-difference check below is the test")
print("        h       max relative error over 5 coordinates")
for hh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10]:
    g_fd = np.array([(loss(w_chk + hh * np.eye(d)[i]) - loss(w_chk - hh * np.eye(d)[i])) / (2 * hh) for i in range(d)])
    print(f"  {hh:8.0e}       {np.max(np.abs(g_fd - g_an) / np.abs(g_an)):10.2e}")

sec("A11", "Smoothness is local: gradient descent on f(x) = x^4 from x0 = 1 — eta = 0.1 converges, eta = 0.6 diverges, and clipping the gradient to norm 1 rescues it")
for eta, clip, x0_ in [(0.1, None, 1.0), (0.1, None, 2.0), (0.1, None, 3.0), (0.05, None, 2.0), (0.6, None, 1.0), (0.6, 1.0, 1.0)]:
    x = x0_
    traj = [x]
    for t in range(10):
        if abs(x) > 1e30:
            traj.append(float("inf"))
            x = float("inf")
            continue
        g = 4 * x**3
        if clip is not None and abs(g) > clip:
            g = clip * np.sign(g)
        x = x - eta * g
        traj.append(x)
    lab = (f"eta = {eta}, x0 = {x0_:.0f}" + (f", clip {clip}" if clip else "")).ljust(26)
    fmt = lambda v: "overflow" if not np.isfinite(v) else f"{v:+.4g}"
    print(f"  {lab}: x at t = 0..5 = {', '.join(fmt(v) for v in traj[:6])};  x at t = 10 = {fmt(traj[10])}")
print(f"  f''(x) = 12 x^2, so the local step ceiling 2/f''(x) is {2/12:.3f} at x = 1, {2/48:.3f} at x = 2, {2/108:.3f} at x = 3: eta = 0.1 is under the ceiling at x0 = 1 and converges, over it at x0 = 2 and still recovers (the overshoot lands at -1.2, inside the safe region), over it at x0 = 3 and diverges; eta = 0.6 is already 3.6x over the ceiling at x0 = 1, and every overshoot lands where the curvature is larger still; no single eta is safe from every start, and clipping bounds the step's length instead")
print(f"  2/f''(x) is the contraction threshold of the LOCALLY LINEARISED map, not an overshoot threshold: the exact x^4 update is x (1 - 4 eta x^2), whose sign flips only at eta = 1/(4 x^2) = {1/16:.4f} at x = 2, above the local ceiling 1/24 = {1/24:.4f}; eta = 0.05 from x0 = 2 is over the ceiling and lands at {2*(1-4*0.05*4):+.4g} with no overshoot, eta = 0.1 crosses to {2*(1-4*0.1*4):+.4g} and still shrinks |x|, so the run recovers")

# =============================================================================================
# PART B — adam-muon-and-friends-explained.md
# =============================================================================================
b1, b2, eps_adam = 0.9, 0.999, 1e-8


def adam_run(grads, eta, b1=b1, b2=b2, eps=eps_adam, x0=None, amsgrad=False, decay=1.0, wd=0.0, coupled=False):
    """Adam on a given sequence of gradients (callable g(x, t) or array); returns iterates and steps."""
    dim = np.shape(x0) if callable(grads) else np.shape(grads[0])
    x = np.zeros(dim) if x0 is None else np.array(x0, dtype=float)
    m = np.zeros(dim)
    v = np.zeros(dim)
    vmax = np.zeros(dim)
    xs, steps = [x.copy()], []
    nT = len(grads) if not callable(grads) else grads.n
    for t in range(1, nT + 1):
        g = grads(x, t) if callable(grads) else grads[t - 1]
        if coupled:
            g = g + wd * x
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        if amsgrad:
            vmax = np.maximum(vmax, vh)
            vh = vmax
        step = eta * decay ** (t - 1) * mh / (np.sqrt(vh) + eps)
        if wd and not coupled:
            x = x - eta * decay ** (t - 1) * wd * x
        x = x - step
        xs.append(x.copy())
        steps.append(step)
    return np.array(xs), np.array(steps)


sec("B1", "Adam's first step is a sign step: ||x1 - x0||_2 is approximately eta sqrt(d) whatever the gradient's size — exactly at eps = 0, and wherever every |g_i| >> eps at the default")
for dd in [10, 1000, 1000000]:
    g = rng.normal(size=dd) * 1e-3
    _, st = adam_run([g], eta=1.0)
    print(f"  d = {dd:8d}: ||update||_2 / eta = {np.linalg.norm(st[0]):10.3f}   sqrt(d) = {sqrt(dd):10.3f}   (||g||_2 was {np.linalg.norm(g):.4f})")
print("  with eps = 1e-8 the first step is -eta g / (|g| + eps), a sign step only where |g| >> eps: |step| / eta at |g| = 1, 1e-6, 1e-8, 1e-10, 0 = " + ", ".join(f"{a_g/(a_g+1e-8):.5f}" for a_g in [1.0, 1e-6, 1e-8, 1e-10, 0.0]) + f"; the {np.linalg.norm(st[0]):.3f} against {sqrt(dd):.0f} above is that shortfall: each coordinate steps |g|/(|g| + eps) of a full sign step, short by eps/(|g| + eps) (about eps/|g| where |g| >> eps)")
sh = 1e-8 / (np.abs(g) + 1e-8)
print(f"  the per-coordinate shortfall summed over the million coordinates is {sh.sum():.1f}, and sqrt(d) - ||update||/eta = {sqrt(dd) - np.linalg.norm(st[0]):.3f} is that sum / sqrt(d) = {sh.sum()/sqrt(dd):.3f} to first order; the median shortfall is {np.median(sh):.1e} (|g| ~ 1e-3), so the sum is carried by the coordinates whose |g| is far below the RMS — the {np.sum(np.abs(g) < 1e-6)} coordinates with |g| < 1e-6 alone contribute {sh[np.abs(g) < 1e-6].sum():.1f} of it")

sec("B2", "One coordinate under Adam: a constant gradient steps exactly eta; a gradient of constant size and alternating sign steps at most eta; a random sign steps far less")
_, st = adam_run([np.array([0.3])] * 6, eta=1.0)
print(f"  constant g = 0.3:          |step| / eta at t = 1..6 = {', '.join(f'{abs(s[0]):.5f}' for s in st)}")
_, st = adam_run([np.array([(-1.0) ** t]) for t in range(6)], eta=1.0)
print(f"  g = +1, -1, +1, -1, ...:   |step| / eta at t = 1..6 = {', '.join(f'{abs(s[0]):.5f}' for s in st)}   (t = 2 is 1/19 = {1/19:.5f}: m_hat = (0.09 - 0.1)/0.19, v_hat = 1)")
signs = rng.choice([-1.0, 1.0], size=20000)
_, st = adam_run([np.array([s]) for s in signs], eta=1.0)
print(f"  g = random +-1 (stationary): mean |step| / eta over steps 1000..20000 = {np.mean(np.abs(st[999:, 0])):.4f}, max = {np.max(np.abs(st[999:, 0])):.4f}   (the EMA of a random sign is near zero; its RMS is 1)")

sec("B3", "Adam's per-coordinate step is not capped at eta: a quiet stretch then one unit gradient gives up to (1 - b1)/sqrt(1 - b2) = 3.16 eta; the worst history gives 7.27 eta")
rho = b1**2 / b2
print(f"  b1 = {b1}, b2 = {b2}: single-spike bound (1-b1)/sqrt(1-b2) = {(1-b1)/sqrt(1-b2):.4f}; supremum over all histories (Cauchy-Schwarz, t -> inf) = {(1-b1)/sqrt(1-b2)/sqrt(1-rho):.4f}")
print(f"  the supremum factorises: m = (1-b1) sum_j b1^j g_(t-j), v = (1-b2) sum_j b2^j g_(t-j)^2; write b1^j = (b1^j / b2^(j/2)) b2^(j/2) and Cauchy-Schwarz gives |m|/sqrt(v) <= (1-b1)/sqrt(1-b2) x 1/sqrt(1 - b1^2/b2) = {(1-b1)/sqrt(1-b2):.4f} x {1/sqrt(1-rho):.4f} = {(1-b1)/sqrt(1-b2)/sqrt(1-rho):.4f}, with equality iff g_(t-j) is proportional to (b1/b2)^j")
print("      t0 quiet steps, then one unit gradient    |step| / eta at t0    geometric history g_{t-j} = (b1/b2)^j, run through Adam, |step| / eta at t = t0")
for t0 in [1, 10, 100, 1000, 10000]:
    gs = [np.array([0.0])] * (t0 - 1) + [np.array([1.0])]
    _, st = adam_run(gs, eta=1.0)
    j = np.arange(t0)
    hist = (b1 / b2) ** j[::-1]  # g_1 ... g_{t0} with g_{t0 - j} = (b1/b2)^j
    _, st2 = adam_run([np.array([v]) for v in hist], eta=1.0)
    print(f"  {t0:8d}                                        {abs(st[-1][0]):8.4f}                  {abs(st2[-1][0]):8.4f}")

sec("B4", "Block heterogeneity: a 4-block quadratic (25 coordinates each, curvatures 1, 10, 100, 1000) — one learning rate for all against one per block, and Adam at a constant step")
Hb = np.repeat([1.0, 10.0, 100.0, 1000.0], 25)
xb0 = rng.normal(size=100)
fb = lambda x: 0.5 * np.sum(Hb * x**2)
x, t = xb0.copy(), 0
while fb(x) >= 1e-8:
    x -= Hb * x / Hb.max()
    t += 1
print(f"  f(x0) = {fb(xb0):.2f}; GD with the one safe step 1/L (L = 1000): {t} iterations to f < 1e-8 (the curvature-1 block contracts by (1 - 1/1000)^2 per step)")
x = xb0 - xb0  # per-block step 1/h_block is exact Newton on a diagonal quadratic
print(f"  per-block step 1/L_block: f after 1 step = {fb(xb0 - (Hb*xb0)/Hb):.1e}  (each block is a scalar quadratic; a diagonal preconditioner is exact)")


class QuadGrad:
    def __init__(self, n):
        self.n = n

    def __call__(self, x, t):
        return Hb * x


for eta_a, dec in [(0.1, 1.0), (0.01, 1.0), (0.1, 0.995)]:
    xs, _ = adam_run(QuadGrad(3000), eta=eta_a, x0=xb0, decay=dec)
    fs = [fb(xs[k]) for k in (100, 500, 1000, 2000, 3000)]
    late = max(fb(xs[k]) for k in range(1000, 3001))
    lab = f"Adam eta = {eta_a:<5g}" + (f" decayed x{dec}/step" if dec != 1 else " constant        ")
    print(f"  {lab}: f at t = 100, 500, 1000, 2000, 3000 = {fs[0]:.3e}, {fs[1]:.3e}, {fs[2]:.3e}, {fs[3]:.3e}, {fs[4]:.3e};  max f over t in [1000, 3000] = {late:.3e}")
print("  at a constant step Adam is not monotone even on a noiseless quadratic: once the gradient has been tiny for ~1/(1-b2) steps v forgets the old scale, m/sqrt(v) grows back toward one and a step of ~eta per coordinate kicks the iterate out of the minimum it had reached; a decaying step removes the re-excitation")

sec("B5", "Reddi, Kale & Kumar (2018): online linear losses on [-1, 1], gradient C every third round and -1 otherwise — Adam drifts to the wrong end, AMSGrad does not")
C = 3.0
b2_r = 1 / (1 + C**2)
Tr = 5000


def reddi_run(ams, beta1, bias_correct, T=Tr, alpha0=0.1, beta2=None):
    beta2 = b2_r if beta2 is None else beta2
    x = m = v = vmax = 0.0
    regret = 0.0
    for t in range(1, T + 1):
        g = C if t % 3 == 1 else -1.0  # f_t(x) = C x on t = 1 mod 3, else -x
        regret += g * x - g * (-1.0)  # loss of x minus loss of the best fixed point x = -1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mh, vh = (m / (1 - beta1**t), v / (1 - beta2**t)) if bias_correct else (m, v)
        vmax = max(vmax, vh)
        x = x - (alpha0 / sqrt(t)) * mh / (sqrt(vmax if ams else vh) + 1e-8)
        x = min(1.0, max(-1.0, x))
    return x, regret / T


xa, ra = reddi_run(False, 0.0, False)
xm, rm = reddi_run(True, 0.0, False)
print(f"  C = {C}, beta2 = 1/(1+C^2) = {b2_r:.3f}, beta1 = 0, eta_t = 0.1/sqrt(t) (the setting of their Theorem 1): after {Tr} rounds Adam sits at x = {xa:+.3f} (average regret {ra:+.4f});  AMSGrad at x = {xm:+.3f} (average regret {rm:+.4f})")
xa9, ra9 = reddi_run(False, 0.9, True)
print(f"  same sequence with the default first-moment coefficient beta1 = 0.9 and bias correction, beta2 kept at {b2_r:.1f} (not the full defaults): Adam sits at x = {xa9:+.3f} (average regret {ra9:+.4f}) — the momentum averages the three gradients to (C - 2)/3 > 0 before the normaliser sees them, and the drift reverses; Theorem 2 of the paper builds a different sequence for beta1 > 0")
xa99, ra99 = reddi_run(False, 0.9, True, beta2=0.999)
print(f"  same sequence at the full defaults (beta1, beta2) = (0.9, 0.999) with bias correction: Adam sits at x = {xa99:+.3f} (average regret {ra99:+.4f}); the displayed three-period example uses beta2 = 0.1 — the theorem is not restricted to that value, but the sequence that defeats a long beta2 is a different and longer one")
print(f"  the sum of losses over one period of 3 is (C - 2) x = {C-2:.1f} x, minimised at x = -1; with beta1 = 0 the step after the C-gradient is small (v just jumped) and after each -1 gradient it is large: net drift is toward +1")

sec("B6", "Weight decay under Adam-style updates is an equilibrium: decoupled (AdamW) settles ||w||/sqrt(d) at sqrt(eta/(2 lambda)) and the angular step at sqrt(2 eta lambda); coupled L2 does not")
dw, eta_w, lam_w = 1000, 0.01, 0.1
w = np.full(dw, 3.0) * rng.choice([-1, 1], size=dw)
marks = {1, 100, 1000, 3000, 10000}
print(f"  d = {dw}, eta = {eta_w}, lambda = {lam_w}: equilibrium ||w||/sqrt(d) = sqrt(eta/(2 lambda)) = {sqrt(eta_w/(2*lam_w)):.4f}, angular step sqrt(2 eta lambda) = {sqrt(2*eta_w*lam_w):.4f} rad, timescale 1/(eta lambda) = {1/(eta_w*lam_w):.0f} steps")
print("  decoupled:  t      ||w||/sqrt(d)    angular step this update (rad)")
for t in range(1, 10001):
    u = rng.normal(size=dw)  # unit-RMS update direction, uncorrelated with w
    w_new = w - eta_w * (u + lam_w * w)
    if t in marks:
        cosang = np.dot(w, w_new) / (np.linalg.norm(w) * np.linalg.norm(w_new))
        print(f"            {t:6d}     {np.linalg.norm(w_new)/sqrt(dw):8.4f}         {math.acos(min(1, cosang)):8.4f}")
    w = w_new
# two gradient scales (two "layers"): first half noise RMS 1, second half noise RMS 0.01; Adam's normaliser on the gradient, decay decoupled vs coupled
scale = np.concatenate([np.ones(dw // 2), np.full(dw // 2, 0.01)])
print("  two layers under Adam's normaliser: gradient noise RMS 1 (first half) and 0.01 (second half); ||w||/sqrt(d/2) per half at t = 1000, 3000, 10000")
for coupled in (False, True):
    w = np.full(dw, 3.0) * rng.choice([-1, 1], size=dw)
    m = v = np.zeros(dw)
    rows = []
    for t in range(1, 10001):
        g = rng.normal(size=dw) * scale
        if coupled:
            g = g + lam_w * w
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        u = (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps_adam)
        if coupled:
            w = w - eta_w * u
        else:
            w = w - eta_w * (u + lam_w * w)
        if t in (1000, 3000, 10000):
            rows.append((t, np.linalg.norm(w[: dw // 2]) / sqrt(dw / 2), np.linalg.norm(w[dw // 2 :]) / sqrt(dw / 2)))
    lab = "coupled L2 (lambda w inside the gradient)" if coupled else "decoupled (AdamW)                        "
    print(f"    {lab}: " + "   ".join(f"t={t}: {a:.4f} / {b:.4f}" for t, a, b in rows))
# momentum-filtered noise is temporally correlated: the AR(1) argument needs the autocorrelation, not the one-step RMS
a_w = 1 - eta_w * lam_w
var_m = (1 - b1) / (1 + b1)
var_w_exact = eta_w**2 / (1 - a_w**2) * var_m * (1 + a_w * b1) / (1 - a_w * b1)
rng_proxy = np.random.default_rng(20261112 + 6)  # a private stream, so the main stream's later sections are untouched
wp = np.zeros(dw); mp = np.zeros(dw)
for t in range(1, 20001):
    mp = b1 * mp + (1 - b1) * rng_proxy.normal(size=dw)
    wp = a_w * wp - eta_w * mp
rms_u = np.sqrt(np.mean(u[: dw // 2]**2))
print(f"  why the two-layer rows sit near {sqrt(var_w_exact):.4f} and NOT at {rms_u:.3f} x {sqrt(eta_w/(2*lam_w)):.4f} = {rms_u*sqrt(eta_w/(2*lam_w)):.4f}: Adam's update is a momentum average of the normalised noise, so its one-step RMS is only about sqrt((1-b1)/(1+b1)) = {sqrt(var_m):.4f} of the noise's ({rms_u:.3f} measured), but successive updates are correlated and the AR(1) recursion sums that correlation")
print(f"  scalar proxy m_t = b1 m_(t-1) + (1-b1) g_t, w_(t+1) = (1 - eta lambda) w_t - eta m_t, unit Gaussian g: exact stationary RMS of w = eta sqrt( (1-b1)/(1+b1) * (1 + a b1) / ((1 - a^2)(1 - a b1)) ) = {sqrt(var_w_exact):.4f} with a = 1 - eta lambda = {a_w:.3f}; simulated over {dw} coordinates at t = 20000: {np.linalg.norm(wp)/sqrt(dw):.4f}; the uncorrelated-noise formula sqrt(eta/(2 lambda)) = {sqrt(eta_w/(2*lam_w)):.4f} lands within 0.5% of it because the correlation factor (1 + a b1)/(1 - a b1) = {(1+a_w*b1)/(1-a_w*b1):.2f} almost exactly cancels the variance factor (1-b1)/(1+b1) = {var_m:.4f}")
print("  (so the same-norm-in-both-layers result stands — the normaliser hands both layers unit-RMS noise before the momentum — but it is a statement about a correlated AR(1), not 'the update RMS times the formula'; the coupled form's second layer, where lambda w dominates the gradient, is driven to a different norm by a decay the normaliser has rescaled)")

sec("B7", "Newton-Schulz orthogonalisation (Muon's coefficients 3.4445, -4.7750, 2.0315), five steps on an 8 x 4 Gaussian gradient with one singular value forced small")
A = rng.normal(size=(8, 4))
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S = np.array([S[0], S[1], S[2], 1e-3 * S[0]])  # force the smallest singular value small
A = (U * S) @ Vt
X = A / np.linalg.norm(A)  # Frobenius scaling, as in Muon
print(f"  singular values after Frobenius scaling: {np.linalg.svd(X, compute_uv=False)}")
a_, b_, c_ = 3.4445, -4.7750, 2.0315
for k in range(1, 6):
    XXt = X @ X.T
    X = a_ * X + b_ * XXt @ X + c_ * XXt @ XXt @ X
    print(f"  NS step {k}: singular values -> {np.linalg.svd(X, compute_uv=False)}")
s2 = np.linalg.norm(X, 2)
sF = np.linalg.norm(X)
print(f"  after 5 steps: ||X||_2 = {s2:.4f} (exact polar factor: 1), ||X||_F = {sF:.4f} (exact polar factor: sqrt(min(m, n)) = {sqrt(4):.4f}); the five-step update is {100*(1-sF/2):.1f}% short of the ideal Frobenius norm")
print(f"  near zero p(sigma) = 3.4445 sigma - 4.775 sigma^3 + 2.0315 sigma^5 ~ 3.44 sigma: a value of {S[3]/np.linalg.norm(A):.2e} needs about log(1/sigma)/log(3.44) = {log(np.linalg.norm(A)/S[3])/log(3.4445):.1f} steps to reach the band where the healthy values live")
pq = lambda s_: 3.4445 * s_ - 4.7750 * s_**3 + 2.0315 * s_**5
dpq = lambda s_: 3.4445 - 3 * 4.7750 * s_**2 + 5 * 2.0315 * s_**4
fps = sorted(r.real for r in np.roots([2.0315, 0, -4.7750, 0, 3.4445 - 1, 0]) if abs(r.imag) < 1e-12 and r.real > 1e-9)
print(f"  p(1) = {pq(1.0):.4f}: 1 is NOT a fixed point of these coefficients (the classical Newton-Schulz cubic 1.5 s - 0.5 s^3 has p(1) = 1, p'(1) = 0); the positive fixed points are s = {fps[0]:.4f} and {fps[1]:.4f} with |p'(s)| = {abs(dpq(fps[0])):.2f} and {abs(dpq(fps[1])):.2f}, both > 1 and so both unstable — the iteration never converges to 1, it oscillates in a band (rows above), a trade the coefficients make for the x3.44 lift near zero")
P = U @ Vt
print(f"  per-entry RMS of the exact polar factor on m x n = 8 x 4: sqrt(min/(m n)) = 1/sqrt(max(m, n)) = {1/sqrt(8):.4f} (measured {np.sqrt(np.mean(P**2)):.4f}); Moonshot's 0.2 sqrt(max(m, n)) = {0.2*sqrt(8):.4f} rescales it to RMS {0.2*sqrt(8)/sqrt(8):.3f}, AdamW's typical per-coordinate update RMS")

sec("B8", "Four methods on an ill-conditioned matrix regression (16 x 32 weights, input covariance condition number 1e4), each at the best step from a grid, step decayed x0.998 per iteration, input rotated so the stiffness is not axis-aligned — an illustration, not a theorem")
mo, ni, ns = 16, 32, 256
sv = np.logspace(0, -2, ni)  # input standard deviations: condition number of the covariance = 1e4
Qrot, _ = np.linalg.qr(rng.normal(size=(ni, ni)))  # a random rotation: the ill-conditioning is not aligned with W's coordinates
Xin = Qrot @ (rng.normal(size=(ni, ns)) * sv[:, None])
Wstar = rng.normal(size=(mo, ni))
Yout = Wstar @ Xin
mloss = lambda W: 0.5 * np.mean(np.sum((W @ Xin - Yout) ** 2, axis=0))
mgrad = lambda W: (W @ Xin - Yout) @ Xin.T / ns


def ns5(M):
    X = M / (np.linalg.norm(M) + 1e-12)
    for _ in range(5):
        XXt = X @ X.T
        X = a_ * X + b_ * XXt @ X + c_ * XXt @ XXt @ X
    return X


def run_method(name, eta, Tm=2000):
    W = np.zeros((mo, ni))
    M = np.zeros_like(W)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    out = {}
    for t in range(1, Tm + 1):
        G = mgrad(W)
        lr = eta * 0.998 ** (t - 1)
        if name == "GD":
            W = W - lr * G
        elif name == "GD + momentum 0.9":
            M = 0.9 * M + G
            W = W - lr * M
        elif name == "Adam":
            m = b1 * m + (1 - b1) * G
            v = b2 * v + (1 - b2) * G**2
            W = W - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps_adam)
        elif name == "Muon-like (orth of momentum 0.9)":
            M = 0.9 * M + G
            W = W - lr * ns5(M)
        if t in (100, 1000, 2000):
            out[t] = mloss(W)
    return out


print(f"  loss at W = 0: {mloss(np.zeros((mo, ni))):.4e}; largest eigenvalue of the input covariance {np.linalg.eigvalsh(Xin @ Xin.T / ns).max():.4f}, smallest {np.linalg.eigvalsh(Xin @ Xin.T / ns).min():.2e}")
grids = {
    "GD": [0.5, 1.0, 1.5, 1.8, 1.9, 1.95],
    "GD + momentum 0.9": [0.5, 1.0, 1.5, 2.0, 3.0, 3.5],  # heavy ball is stable for alpha < 2(1 + beta)/L = 3.8 here
    "Adam": [0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
    "Muon-like (orth of momentum 0.9)": [0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
}
for name, grid in grids.items():
    best = None
    for eta in grid:
        o = run_method(name, eta)
        if np.isfinite(o[2000]) and (best is None or o[2000] < best[1][2000]):
            best = (eta, o)
    eta, o = best
    print(f"  {name:36s} best step {eta:7.4f}: t=100: {o[100]:.3e}  t=1000: {o[1000]:.3e}  t=2000: {o[2000]:.3e}")
print("  the problem was built to be ill-conditioned in exactly the way a per-direction step fixes; a fixed-length update in every singular direction overshoots the easy ones early and wins late")

# The same problem BEFORE the rotation (Qrot^T undoes it; no new draw): the stiffness is axis-aligned, which is the case a per-coordinate scale is shaped for
Xin_rot, Yout_rot = Xin, Yout
Xin = Qrot.T @ Xin_rot
Yout = Wstar @ Xin
print("  the same instance unrotated (Qrot^T applied, no new draw; the stiffness is axis-aligned) — the pilot the prose compares against:")
unrot = {}
for name, grid in grids.items():
    best = None
    for eta in grid:
        o = run_method(name, eta)
        if np.isfinite(o[2000]) and (best is None or o[2000] < best[1][2000]):
            best = (eta, o)
    eta, o = best
    unrot[name] = o[2000]
    print(f"    {name:36s} best step {eta:7.4f}: t=100: {o[100]:.3e}  t=1000: {o[1000]:.3e}  t=2000: {o[2000]:.3e}")
others = min(v for k, v in unrot.items() if k != "Adam")
print(f"    unrotated, Adam's t=2000 loss is {np.log10(others / unrot['Adam']):.1f} orders of magnitude below the best of the other three; rotated, it is an ordinary method")
Xin, Yout = Xin_rot, Yout_rot
Hd = np.diag([1.0, 100.0])
x0d = np.array([2.0, 1.0])
gd = Hd @ x0d
newton = -np.linalg.solve(Hd, gd)
adam1 = -np.sign(gd)
print(f"  but axis-aligned is not the same as exactly preconditioned: H = diag(1, 100), x0 = (2, 1), gradient ({gd[0]:.0f}, {gd[1]:.0f}); the exact Newton step is ({newton[0]:.0f}, {newton[1]:.0f}) and Adam's first eps-free step is eta x ({adam1[0]:.0f}, {adam1[1]:.0f}) — component ratios {abs(newton[0]/newton[1]):.0f}:1 against 1:1, so no scalar eta makes them equal; 1/sqrt(v_hat) is a gradient scale, not the inverse Hessian, even on a diagonal quadratic")

sec("B9", "The two memories: how much of a new gradient scale v has absorbed after t steps, 1 - b2^t, and the effective learning rate the momentum gives a constant gradient, 1/(1 - b1)")
print("        t      1 - 0.999^t (share of a new scale in v)     1 - 0.9^t (share in m)")
for t in [1, 10, 100, 1000, 3000, 10000]:
    print(f"  {t:7d}            {1-b2**t:8.4f}                                {1-b1**t:8.4f}")
print(f"  time constant of v: 1/(1 - b2) = {1/(1-b2):.0f} steps; of m: 1/(1 - b1) = {1/(1-b1):.0f} steps; a constant gradient's effective step under plain momentum (no normaliser) is alpha/(1 - beta) = {1/(1-b1):.0f} alpha at beta = 0.9")

sec("B10", "Warmup on the 4-block quadratic from a far start (RMS 10): Adam at eta = 0.1 with no warmup against a 100-step linear warmup")
xw0 = xb0 * 10


class QuadGradW:
    def __init__(self, n):
        self.n = n

    def __call__(self, x, t):
        return Hb * x


for warm in [0, 100]:
    x = xw0.copy()
    m = v = np.zeros(100)
    fs = {}
    for t in range(1, 501):
        g = Hb * x
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        lr = 0.1 * min(1.0, t / warm) if warm else 0.1
        x = x - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps_adam)
        if t in (10, 50, 100, 200, 500):
            fs[t] = fb(x)
    lab = "no warmup           " if warm == 0 else f"linear warmup {warm} st."
    print(f"  {lab}: f at t = 10, 50, 100, 200, 500 = {fs[10]:.3e}, {fs[50]:.3e}, {fs[100]:.3e}, {fs[200]:.3e}, {fs[500]:.3e}")
print(f"  f(x0) = {fb(xw0):.3e}, ||x0||_2 = {np.linalg.norm(xw0):.1f}; the first Adam step moves every coordinate by eta, ||x1 - x0||_2 = eta sqrt(d) = {0.1*sqrt(100):.1f}, whatever the gradient — harmless here, because this quadratic has no noise and no sharpening for warmup to protect against; the gain warmup shows in a transformer is a measurement about the transformer, not a theorem about descent")

print(f"\nDONE (A1-A11, B1-B10) in {time.time()-T0:.1f} s")
