#!/usr/bin/env python3
"""Deterministic re-derivation of the closed-form values quoted beside worked_numbers.py's sections.

Every assertion here is a formula the article states, evaluated independently of the script and
compared to the digits the script prints (and the article quotes). Monte Carlo lines are NOT tested
here — they carry standard errors in the output and are checked by diffing worked_numbers_output.txt.

Run:  python test_worked_numbers.py        (under a second; exit 0 means every closed form reproduces)
"""
import math, re, sys
import numpy as np
from math import sqrt, log, exp, pi

OUT = open("worked_numbers_output.txt").read()
checks = 0


def ok(cond, msg):
    global checks
    checks += 1
    if not cond:
        print("FAIL:", msg); sys.exit(1)


def block(n):
    m = re.search(rf"=== §{n} .*?(?=\n=== §|\nDONE|\Z)", OUT, re.S)
    assert m, f"section {n} not found"
    return m.group(0)


# §1 — contraction factors max|1 - eta lambda| on log-spaced eigenvalues in [1, 100]; boundary 2/L
lam = np.logspace(0, 2, 20); L = 100.0
for c, rho in ((0.5, 0.9950), (1.0, 0.9900), (1.9, 0.9810), (1.99, 0.9901), (2.01, 1.0100), (2.1, 1.1000)):
    ok(abs(max(abs(1 - c / L * lam)) - rho) < 2e-4, f"§1 contraction at {c}/L")
ok(f"{2/L:.4f}" == "0.0200", "§1 boundary 2/L")
ok("contraction factor max|1-eta*lambda| = 0.9950" in block(1), "§1 printed contraction")

# §7 — GD bound t = log(1e-8/f0) / (2 log(1 - 1/kappa)), NAG bound with the 2 f0 factor
for kappa, gd_b, nag_b in ((10, 91, 52), (100, 1036, 204), (1000, 11370, 730), (10000, 123943, 2536)):
    lam = np.logspace(0, log(kappa, 10), 50); x0 = np.ones(50) / sqrt(50); f0 = 0.5 * np.sum(lam * x0 * x0)
    ok(round(log(1e-8 / f0) / (2 * log(1 - 1 / kappa))) == gd_b, f"§7 GD bound kappa={kappa}")
    ok(round(log(1e-8 / (2 * f0)) / log(1 - 1 / sqrt(kappa))) == nag_b, f"§7 NAG bound kappa={kappa}")
    ok(f"{kappa:6d}" in block(7) and f"{gd_b:14.0f}" in block(7), f"§7 printed bound kappa={kappa}")

# §8 — Nesterov's chain: f*, R^2, lower bound 3 L R^2 / (32 (t+1)^2), upper 4 L R^2 / (t+2)^2
for t, lb, ub in ((1, 0.020508, 0.388889), (10, 0.005300, 0.190025), (100, 0.000614, 0.025696)):
    d = 2 * t + 1; xstar = 1 - np.arange(1, d + 1) / (d + 1); R2 = np.sum(xstar ** 2)
    ok(abs(3 * R2 / (32 * (t + 1) ** 2) - lb) < 1e-6, f"§8 lower bound t={t}")
    ok(abs(4 * R2 / (t + 2) ** 2 - ub) < 1e-6, f"§8 upper bound t={t}")
    ok(f"{lb:12.6f}" in block(8), f"§8 printed lower bound t={t}")
# GD on the chain at t = 1 from 0: x_1 = (L/4) e_1, f gap closed form
d = 3; xs = 1 - np.arange(1, 4) / 4
def f_ch(x): return 0.25 * (0.5 * (x[0] ** 2 + np.sum((x[:-1] - x[1:]) ** 2) + x[-1] ** 2) - x[0])
ok(abs((f_ch(np.array([0.25, 0, 0])) - f_ch(xs)) - 0.046875) < 1e-9, "§8 GD gap at t=1")

# §6 — log-sum-exp and the Gibbs closed form
th = np.array([1.0, 0.0, -1.0, 2.0]); A = log(np.sum(np.exp(th)))
ok(f"{A:.6f}" == "2.440190", "§6 A(theta)")
r = np.array([2.0, 0.5, -1.0, 1.0]); pi0 = np.array([0.4, 0.3, 0.2, 0.1]); beta = 0.7
Z = np.sum(pi0 * np.exp(r / beta)); ok(f"{beta*log(Z):.6f}" == "1.459336", "§6 beta log Z")
p = pi0 * np.exp(r / beta) / Z; ok(np.allclose(np.round(p, 4), [0.866, 0.0762, 0.006, 0.0519]), "§6 Gibbs argmax")

# §11 — Fisher of N(mu, sigma^2) at sigma = 1.7 in two parametrisations
s0 = 1.7; ok(abs(1 / s0 ** 2 - 0.346) < 5e-4 and abs(2 / s0 ** 2 - 0.692) < 5e-4, "§11 Fisher (mu, sigma)")
ok("Fisher closed form = [0.346 0.    0.    2.   ]" in block(11), "§11 printed Fisher (mu, log sigma)")

# §12 — Reddi et al. counter-example structure: beta2 = 1/(1+C^2), period sum (C-2)x
C = 3.0; ok(f"{1/(1+C*C):.3f}" == "0.100", "§12 beta2"); ok("x = +1.000" in block(12) and "AMSGrad at x = -1.000" in block(12), "§12 verdicts")

# §13 — one-step bottleneck: (1 - 1/1000)^2t to 1e-8 from f0 ~ 25 * 1000/2 ... just check the printed count is within the analytic band
m = re.search(r"GD with the one safe step 1/L \(L = 1000\): (\d+) iterations", block(13)); n_gd = int(m.group(1))
ok(9000 < n_gd < 12000, "§13 GD iteration count in the (1-1/1000)^2t band")

# §14 — nuclear norm = sum of singular values; spectral direction has unit singular values
m = re.search(r"singular values of G: \[([^\]]+)\]", block(14)); S = np.array(m.group(1).split(), float)
m2 = re.search(r"\|\|G\|\|_nuc = ([0-9.]+)", block(14)); ok(abs(S.sum() - float(m2.group(1))) < 2e-3, "§14 nuclear norm")
ok("singular values of D: [1. 1. 1. 1.]" in block(14), "§14 spectral direction")

# §16 — stationary variance alpha sigma^2 / (2 - alpha)
for a, v in ((0.5, "0.33333"), (0.1, "0.05263"), (0.01, "0.00503")):
    ok(f"{a/(2-a):.5f}" == v, f"§16 floor alpha={a}")

# §19 — NQM closed-form expectations (exact recursion), constant schedule at t = 2000
dq = 1000; hq = 1.0 / np.arange(1, dq + 1); eta = 1.0; m2 = np.ones(dq)
for t in range(2000): m2 = (1 - eta * hq) ** 2 * m2 + eta ** 2 * hq / 64
ok(f"{0.5*np.sum(hq*m2):.5f}" == "0.03653", "§19 constant schedule expected loss")
m2 = np.ones(dq)
for t in range(2000):
    e = eta if t < 1600 else eta * (2000 - t) / 400
    m2 = (1 - e * hq) ** 2 * m2 + e ** 2 * hq / 64
ok(f"{0.5*np.sum(hq*m2):.5f}" == "0.01245", "§19 WSD expected loss")

# §20 — ||update||/eta = sqrt(d) at t = 1
for d_ in (10, 10 ** 4, 10 ** 6): ok(f"sqrt(d) = {sqrt(d_):10.2f}" in block(20), f"§20 sqrt d={d_}")

# §21 — equilibrium norm and angular step
ok(f"{sqrt(1e-2/(2*0.1)):.4f}" == "0.2236" and f"{sqrt(2*1e-2*0.1):.4f}" == "0.0447", "§21 equilibria")

# §22 — bound 2L(f0 - f*)/T on the chain, d = 2001
d = 2001; xstar = 1 - np.arange(1, d + 1) / (d + 1)
def f_ch2(x): return 0.25 * (0.5 * (x[0] ** 2 + np.sum((x[:-1] - x[1:]) ** 2) + x[-1] ** 2) - x[0])
gap0 = f_ch2(np.zeros(d)) - f_ch2(xstar)
for T, b in ((10, "2.499e-02"), (10000, "2.499e-05")): ok(f"{2*gap0/T:.3e}" == b, f"§22 bound T={T}")

# §23 — predicted escape steps log(0.5/r)/eta
for r, pred in ((1e-1, 16.1), (1e-4, 85.2), (1e-12, 269.4)): ok(abs(log(0.5 / r) / 0.1 - pred) < 0.05, f"§23 prediction r={r}")

# §24 — PL constant by the same grid, and the concave fraction
xs = np.linspace(-20, 20, 400001); fpl = xs * xs + 3 * np.sin(xs) ** 2; gpl = 2 * xs + 3 * np.sin(2 * xs)
mu = np.nanmin((0.5 * gpl ** 2) / np.where(fpl > 1e-12, fpl, np.nan)); ok(f"{mu:.4f}" == "0.1755", "§24 PL constant")
ok(f"{np.mean(2 + 6*np.cos(2*xs) < 0)*100:.1f}" == "37.9", "§24 concave fraction")

# §30 — Hessian of (ab - 1)^2/2 at ab = 1: eigenvalues {0, a^2 + b^2}
for alpha in (1.0, 10.0, 100.0):
    a_, b_ = alpha, 1 / alpha; top = a_ ** 2 + b_ ** 2
    ok(f"top = {top:.4f}" in block(30), f"§30 top eigenvalue alpha={alpha}")

# §33 — basin area (1/10.24)^2
ok(f"{(1/10.24)**2:.4f}" == "0.0095", "§33 basin area")

# §37 — GDA growth sqrt(1+eta^2), extragradient contraction sqrt(1 - eta^2 + eta^4)
eta = 0.1; ok(f"{sqrt(1+eta**2):.5f}" == "1.00499" and f"{sqrt(1+eta**2)**200:.3e}" == "2.705e+00", "§37 GDA growth")
ok(f"{sqrt(1-eta**2+eta**4):.5f}" == "0.99504", "§37 EG contraction")
J = np.array([[0.0, 1.0], [-1.0, 0.0]]); z = np.array([1.0, 1.0])
for _ in range(200): z = z - eta * J @ z
ok(f"{np.linalg.norm(z):.4e}" in block(37), "§37 printed GDA norm")

# §40 — Bayes risk Phi(-1)
ok(f"{0.5*(1+math.erf(-1/sqrt(2))):.4f}" == "0.1587", "§40 Bayes risk")

# §41 — double descent: exact expected bias and variance of the min-norm interpolant for Gaussian design
# (inverse-Wishart moments: p < n-1 -> var = s^2 p/(n-p-1), bias 0; p > n+1 -> var = s^2 n/(p-n-1), bias = r^2 (1 - n/p))
n41, r41, s41 = 100, 1.0, 0.5
for p41 in (10, 25, 50, 75, 90, 95, 105, 110, 125, 150, 200, 400, 1000):
    if p41 < n41 - 1: eb41, ev41 = 0.0, s41 ** 2 * p41 / (n41 - p41 - 1)
    else: eb41, ev41 = r41 ** 2 * (1 - n41 / p41), s41 ** 2 * n41 / (p41 - n41 - 1)
    ok(f"{eb41:7.4f}  {ev41:8.4f}  {eb41+ev41:8.4f}" in block(41), f"§41 closed form p={p41}")
ok("     inf       inf" in block(41), "§41 infinite expectation at p = n")

# §42 — heavy-ball prefactor at the extreme eigenvalues: (1 + (1-q)T) q^T (lam = mu) and (1 + (1+q)T) q^T (lam = L), q = (sqrt k - 1)/(sqrt k + 1)
q42 = (sqrt(100) - 1) / (sqrt(100) + 1)
ok(f"{q42:.5f}" == "0.81818" and f"{4/(sqrt(100)+1)**2:.5f}" == "0.03306", "§42 constants")
for T42 in (10, 100, 1000):
    ok(f"{(1 + (1 - q42) * T42) * q42 ** T42:11.4e}" in block(42) and f"{(1 + (1 + q42) * T42) * q42 ** T42:11.4e}" in block(42), f"§42 prefactor T={T42}")
ok(f"{1 + (1 - q42) * 100:.1f}" == "19.2" and f"{1 + (1 + q42) * 100:.1f}" == "182.8" and "19.2 at the soft end and 182.8" in block(42), "§42 annotation prefactors")

# §43 — J''(z) = -p(1-p)[1 + z(1-2p)] for J(z) = -KL(Ber(sigmoid z) || Ber(1/2)); positive at z = 2
p43 = 1 / (1 + exp(-2.0)); ok(f"{-p43*(1-p43)*(1+2.0*(1-2*p43)):+.5f}" == "+0.05493" and "+0.05493" in block(43), "§43 J'' at z=2")
ok(f"{-0.25:+.5f}" == "-0.25000" and "-0.25000" in block(43), "§43 J'' at z=0")

# §44 — IBP on the two-copy network returns [-1, +1]; the symbolic bound returns [0, 0]
ok("[-1, +1]" in block(44) and "[+0, +0]: exact" in block(44) and "unstable ReLUs: 0/2" in block(44), "§44 IBP vs symbolic")

# §45 — the expected factor 1 - 1/G, printed beside the Monte Carlo column
for G45 in (2, 4, 8, 16): ok(f"{1-1/G45:7.4f}" in block(45), f"§45 factor G={G45}")

# §46 — Adam's cap (1-b1)/sqrt(1-b2) = 3.1623
ok(f"{0.1/sqrt(0.001):.4f}" == "3.1623" and "3.1623" in block(46), "§46 cap")

# §47 — same contraction factor 0.99 at 1/L and 1.99/L; the stiff direction's share (1/2) L 0.99^400; the ratio 63.3
lam47 = np.logspace(0, 2, 20); L47 = 100.0
r47 = (0.5*lam47*(1-1.99/L47*lam47)**400).sum() / (0.5*lam47*(1-1/L47*lam47)**400).sum()
ok(f"{r47:.1f}" == "63.3" and "= 63.3:" in block(47), "§47 ratio")
ok(f"{0.5*100*0.99**400:.4e}" == "8.9753e-01" and "8.9753e-01" in block(47) and "8.9753e-03" in block(47), "§47 shares")
ok(block(47).count("max|1 - eta lambda| = 0.9900") == 2, "§47 equal contraction factors")

# §48 — Adam's supremum over histories: (1-b1)/sqrt(1-b2) / sqrt(1 - b1^2/b2) = 7.2703; the Cauchy-Schwarz bound at finite t, and the geometric history attains it
b1_48, b2_48 = 0.9, 0.999; rho_48 = b1_48 ** 2 / b2_48
ok(f"{rho_48:.6f}" == "0.810811" and f"{(1-b1_48)/sqrt(1-b2_48)/sqrt(1-rho_48):.4f}" == "7.2703" and "= 7.2703" in block(48), "§48 supremum")
for t48 in (1, 10, 100, 1000, 10000):
    b48 = (1 - b1_48) / sqrt(1 - b2_48) * sqrt((1 - rho_48 ** t48) / (1 - rho_48)) * sqrt(1 - b2_48 ** t48) / (1 - b1_48 ** t48)
    ok(f"{b48:8.4f}                        {b48:8.4f}" in block(48), f"§48 bound attained at t={t48}")
ok("3.1622                      7.2701" in block(48), "§48 single spike vs supremum at t = 10000")

# §49 — triangle max-cut: integer 2, SDP 2.25, rounding probability 2/3, GW constant 0.87857
th49 = np.linspace(1e-6, pi, 200001); gw49 = (th49 / pi) / ((1 - np.cos(th49)) / 2)
ok(f"{gw49.min():.5f}" == "0.87857" and "0.87857" in block(49), "§49 GW constant")
ok("the 8 cuts of the triangle: 2.00" in block(49) and "= 2.25;" in block(49) and f"{2.25/2:.4f}" == "1.1250" and "1.1250" in block(49), "§49 2 vs 2.25")
ok(f"{(2*pi/3)/pi:.4f}" == "0.6667" and "expected cut = 2.0000" in block(49) and f"{2/2.25:.4f}" == "0.8889", "§49 rounding")

# §50 — exact prox 1/(1+x) vs AdamW 1-x
for x50 in (0.001, 0.01, 0.1, 0.5):
    ok(f"{1 - x50:10.5f}             {1 / (1 + x50):10.5f}" in block(50), f"§50 factors x={x50}")
ok(f"{1/1.1:.5f}" == "0.90909" and "0.90909" in block(50), "§50 0.90909")

# §51 — Euler-Langevin stationary variance T/(1 - eta/2)
for e51 in (0.5, 0.1, 0.01):
    ok(f"{e51:6.2f}          {1 / (1 - e51 / 2):8.4f}" in block(51), f"§51 closed form eta={e51}")
ok(f"{1/(1-0.25):.4f}" == "1.3333", "§51 4/3")

# §52 — spectral radii: extragradient sqrt(1 - eta^2 + eta^4); optimistic from the 4x4 companion matrix
J52 = np.array([[0.0, 1.0], [-1.0, 0.0]]); I52 = np.eye(2); e52 = 0.1
r_eg = np.abs(np.linalg.eigvals(I52 - e52 * J52 + e52 ** 2 * J52 @ J52)).max()
r_og = np.abs(np.linalg.eigvals(np.block([[I52 - 2 * e52 * J52, e52 * J52], [I52, np.zeros((2, 2))]]))).max()
ok(f"{r_eg:.6f}" == "0.995038" and f"{sqrt(1 - e52**2 + e52**4):.6f}" == "0.995038" and "0.995038" in block(52), "§52 extragradient radius")
ok(f"{r_og:.6f}" == "0.994936" and "0.994936" in block(52), "§52 optimistic radius")
ok(f"{r_eg**200*sqrt(2):.4f}" == "0.5229" and "0.5229" in block(52) and f"{(r_eg/r_og)**200:.4f}" == "1.0206" and "1.0206" in block(52), "§52 200-step factors")

# §14 norm lines — the five-step Newton–Schulz output's norms from its own printed singular values (exact polar factor: 1 and sqrt(min(m, n)) = 2)
sv14 = np.array([1.0305, 0.7581, 0.7145, 0.12])
ok("||X||_2 = 1.0305" in block(14) and f"{sv14.max():.4f}" == "1.0305", "§14 five-step spectral norm 1.0305 (exact: 1)")
ok("||X||_F = 1.4702" in block(14) and f"{np.sqrt((sv14 ** 2).sum()):.4f}" == "1.4702" and "sqrt(min(m, n)) = 2.0000" in block(14), "§14 five-step Frobenius norm 1.4702 (exact: 2)")
ok("26.5% short" in block(14) and f"{100 * (1 - np.sqrt((sv14 ** 2).sum()) / 2):.1f}" == "26.5", "§14 five-step update 26.5% short of the ideal Frobenius norm")
# §53 — log-log slopes fitted to the four §22 rows (the printed column is the squared norm): T^-3/2 and T^-1/2, not T^-2 and T^-1
T53 = np.log(np.array([10., 100., 1000., 10000.])); g53 = np.log(np.array([1.607e-03, 4.996e-05, 1.577e-06, 4.987e-08]))
s_g = np.polyfit(T53, g53, 1)[0]; s_r = np.polyfit(T53, g53 + T53, 1)[0]
ok(f"{s_g:.2f}" == "-1.50" and "against T: -1.503" in block(53) and "-1.503" not in block(22), "§53 squared-gradient slope -1.50 (T^-3/2) from the §22 rows")
ok(f"{s_r:.2f}" == "-0.50" and "bound 2L(f0 - f*)/T: -0.503" in block(53), "§53 ratio-to-bound slope -0.50 (T^-1/2)")
ok("1.607e-03" in block(22) and "4.987e-08" in block(22) and "1.607e-03" in block(53) and "4.987e-08" in block(53), "§53 reproduces §22's deterministic rows")

# Adam at the defaults with g = (1, -1): constant |g| is NOT a constant gradient; bias-corrected step is eta/19, not eta
b1, b2 = 0.9, 0.999; m = 0.0; v = 0.0; steps = []
for k, g in enumerate((1.0, -1.0), start=1):
    m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
    steps.append((m / (1 - b1 ** k)) / sqrt(v / (1 - b2 ** k)))
ok(abs(steps[0] - 1.0) < 1e-12 and abs(abs(steps[1]) - 1 / 19) < 1e-12 and f"{abs(steps[1]):.5f}" == "0.05263", "§4.4 Adam g=(1,-1): step 1 is eta, step 2 is eta/19")
# The §17 fit S = 6.4 (1 + 103.2/B) over-predicts the large-batch floor (2 steps) and matches B = 16, 64
ok("S_min = 6.4 steps, B_noise(fit) = 103.2" in block(17), "§17 printed fit parameters")
pred = {B: 6.4 * (1 + 103.2 / B) for B in (16, 64, 256, 1024, 4096)}
ok([f"{pred[B]:.1f}" for B in (16, 64, 256, 1024, 4096)] == ["47.7", "16.7", "9.0", "7.0", "6.6"], "§5.2 fitted steps 47.7/16.7/9.0/7.0/6.6 quoted in the disclosure sentence")
ok(all(re.search(rf"^\s*{B}\s+\S+\s+{obs}\s", block(17), re.M) for B, obs in ((16, 46), (64, 15), (256, 4), (1024, 2), (4096, 2))), "§17 measured steps 46/15/4/2/2 behind the disclosure sentence")

print(f"OK: {checks} closed-form checks reproduce the printed digits")
