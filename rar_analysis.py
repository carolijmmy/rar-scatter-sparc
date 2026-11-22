"""
Does the scatter of the radial acceleration relation (RAR) depend on galaxy properties?
PHYS 474 final project. Data: SPARC (Lelli et al. 2016).

Run from the folder that contains SPARC_Lelli2016c.mrt.txt and Rotmod_LTG/:

    python rar_analysis.py

Figures and tables are written to ./output/.

Steps
  1. Compute g_obs and g_bar for every radial point and fit the RAR.
  2. Fit the intrinsic scatter with a point-level maximum-likelihood model.
  3. Test each galaxy's mean residual against galaxy properties
     (Spearman + weighted slope, galaxy-level bootstrap, Benjamini-Hochberg correction).
  4. Injection-recovery: what is the smallest trend the sample could detect?
  5. Exploratory checks: low/high acceleration split, surface-brightness split, radial trend.
  6. Robustness: different mass-to-light ratios and a Q = 1 subsample.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import least_squares, minimize_scalar

# ============================== SETTINGS ==============================
TABLE_PATH = "SPARC_Lelli2016c.mrt.txt"   # galaxy table
ROTMOD_DIR = "Rotmod_LTG"                 # folder of *_rotmod.dat files
OUT_DIR = "output"                        # figures and csv files go here
SHOW_PLOTS = False                        # True: also open each figure on screen

UPSILON_DISK = 0.5        # fiducial 3.6 micron M/L, disk
UPSILON_BULGE = 0.7       # fiducial 3.6 micron M/L, bulge
MIN_INCLINATION = 29.9    # keep Inc > 29.9, i.e. Inc >= 30 deg (inclinations are whole degrees)
MAX_QUALITY = 2           # keep Q = 1 or 2
MAX_VEL_FRAC_ERR = 0.1    # drop points with errV / Vobs >= 10%
MIN_POINTS = 5            # minimum radial points per galaxy for galaxy-level tests

SIGMA_ML = 0.1            # assumed M/L uncertainty [dex] in the error model
SIGMA_INT_DESMOND = 0.034 # intrinsic scatter from Desmond (2023), drawn on the power figure

N_BOOT = 10000            # galaxy bootstrap resamples (main tests)
N_BOOT_SPLIT = 5000       # ... for the low/high acceleration split
N_BOOT_ROBUST = 2000      # ... for each robustness variant
N_TRIALS = 4000           # injection-recovery trials per trend amplitude
SEED = 474

KMS2_PER_KPC_TO_SI = 1.0e6 / 3.0856775814913673e19   # (km/s)^2/kpc -> m/s^2
LN10 = np.log(10.0)

PRIMARY = ["log_SBeff", "f_gas", "log_Reff", "log_Mbar", "T"]   # the 5 pre-declared properties
LABELS = {"log_SBeff": r"$\log_{10}$ SB$_{\rm eff}$", "f_gas": "gas fraction",
          "log_Reff": r"$\log_{10}$ R$_{\rm eff}$", "log_Mbar": r"$\log_{10}$ M$_{\rm bar}$",
          "T": "morph. type T"}

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)   # one generator shared by all resampling, for reproducibility


# ============================== DATA ==============================
def read_sparc_table(path):
    """Read the SPARC galaxy table (MRT text file) into a DataFrame.

    A line is a data row if it has >= 18 fields and fields 1..17 are numbers.
    The column order below must match the file header.
    """
    cols = ["Galaxy", "T", "D", "e_D", "f_D", "Inc", "e_Inc", "L36", "e_L36",
            "Reff", "SBeff", "Rdisk", "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q"]
    rows = []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if len(tok) < 18:
                continue
            try:
                nums = [float(x) for x in tok[1:18]]
            except ValueError:
                continue
            rows.append([tok[0]] + nums)
    table = pd.DataFrame(rows, columns=cols)
    if len(table) == 0:
        raise ValueError("No data rows parsed; check the file layout.")
    if not table["Q"].isin([1, 2, 3]).all() or not table["Inc"].between(0, 90).all():
        raise ValueError("Q or Inc look wrong: the column order does not match your file.")
    table["Q"] = table["Q"].astype(int)
    return table


def build_points(table, ud=UPSILON_DISK, ub=UPSILON_BULGE,
                 max_q=MAX_QUALITY, vel_cut=MAX_VEL_FRAC_ERR):
    """Return one row per radial point with g_obs, g_bar (m/s^2) and a few error-model inputs.

    SPARC gives Vdisk and Vbul for M/L = 1, so they are scaled by ud and ub.
    V*|V| keeps the sign of the gas term, which can be negative near the centre.
    """
    selected = table[(table["Inc"] > MIN_INCLINATION) & (table["Q"] <= max_q)]
    rows = []
    for name in selected["Galaxy"]:
        path = os.path.join(ROTMOD_DIR, f"{name}_rotmod.dat")
        if not os.path.exists(path):
            print("missing file for", name)
            continue
        d = np.atleast_2d(np.loadtxt(path, comments="#"))
        R, Vobs, errV, Vgas, Vdisk, Vbul = d[:, :6].T      # kpc, km/s (x5)

        gas = Vgas * np.abs(Vgas)
        star = ud * Vdisk * np.abs(Vdisk) + ub * Vbul * np.abs(Vbul)
        g_obs = Vobs**2 / R * KMS2_PER_KPC_TO_SI
        g_bar = (gas + star) / R * KMS2_PER_KPC_TO_SI
        frac_V = errV / np.where(Vobs > 0, Vobs, np.nan)   # fractional velocity error

        ok = (R > 0) & (g_obs > 0) & (g_bar > 0)           # drop points with no baryonic pull
        if vel_cut is not None:
            ok &= frac_V < vel_cut
        for j in np.where(ok)[0]:
            rows.append((name, R[j], g_obs[j], g_bar[j], frac_V[j], star[j] / (gas[j] + star[j])))

    if not rows:
        raise ValueError("No radial points kept: check ROTMOD_DIR and the cuts.")
    pts = pd.DataFrame(rows, columns=["Galaxy", "R_kpc", "g_obs", "g_bar", "frac_V", "f_star"])
    pts["log_gobs"] = np.log10(pts["g_obs"])
    pts["log_gbar"] = np.log10(pts["g_bar"])
    return pts


# ============================== RAR FIT ==============================
def rar_function(g_bar, g_dagger):
    """McGaugh et al. (2016): g_obs = g_bar / (1 - exp(-sqrt(g_bar / g_dagger)))."""
    return g_bar / (1.0 - np.exp(-np.sqrt(g_bar / g_dagger)))


def fit_rar(pts):
    """Least-squares fit of g_dagger in log space. Returns g_dagger and the residual of each point."""
    def residuals(p):
        return pts["log_gobs"].values - np.log10(rar_function(pts["g_bar"].values, 10 ** p[0]))
    fit = least_squares(residuals, x0=[np.log10(1.2e-10)])
    return 10 ** fit.x[0], residuals(fit.x)


def galaxy_summary(pts, table):
    """Mean residual, number of points and median log g_bar per galaxy, joined to the SPARC table."""
    summary = (pts.groupby("Galaxy")
                  .agg(mean_resid=("resid", "mean"), n_points=("resid", "size"),
                       med_log_gbar=("log_gbar", "median"))
                  .reset_index())
    return summary.merge(table, on="Galaxy")


def add_properties(gal, ups, min_points=MIN_POINTS):
    """Keep galaxies with enough points and compute the tested properties.

    Units: L36 and MHI are in 1e9 Lsun and 1e9 Msun. The bulge is ignored in M_star.
    """
    g = gal[gal["n_points"] >= min_points].copy()
    g["M_star"] = ups * g["L36"]
    g["M_gas"] = 1.33 * g["MHI"]                       # 1.33 accounts for helium
    g["f_gas"] = g["M_gas"] / (g["M_gas"] + g["M_star"])
    g["log_Mbar"] = np.log10(g["M_gas"] + g["M_star"])
    g["log_SBeff"] = np.log10(g["SBeff"])
    g["log_Reff"] = np.log10(g["Reff"])
    g["log_lambda"] = g["log_Mbar"] - g["log_Reff"]    # compactness, only used in the regime split
    return g


# ============================== INTRINSIC SCATTER ==============================
# Model: residual ~ Normal(0, var) with
#   var = sig_y^2 + (slope * sig_x)^2 + sig_int^2
# sig_y: errors on log g_obs (velocity, inclination, distance), sig_x: M/L error on log g_bar,
# slope: local slope of the RAR. Errors are treated as independent between points.
def add_error_model(pts, table):
    """Add sig_y and sig_x (dex) to each point, using the galaxy's distance and inclination errors."""
    p = pts.merge(table[["Galaxy", "Inc", "e_Inc", "D", "e_D"]], on="Galaxy", how="left")
    sig_V = 2.0 * p["frac_V"] / LN10                                              # g_obs ~ V^2
    sig_inc = 2.0 * np.radians(p["e_Inc"]) / np.tan(np.radians(p["Inc"])) / LN10  # V ~ 1/sin(i)
    sig_D = (p["e_D"] / p["D"]) / LN10                                            # g_obs ~ 1/D
    p["sig_y"] = np.sqrt(sig_V**2 + sig_inc**2 + sig_D**2)
    p["sig_x"] = np.clip(p["f_star"], 0, 1) * SIGMA_ML       # only the stellar part of g_bar is uncertain
    return p


def log_rar(log_g, log_gdag):
    return np.log10(rar_function(10 ** log_g, 10 ** log_gdag))


def rar_slope(log_g, log_gdag, eps=1e-3):
    """Local slope d log f / d log g_bar (central difference)."""
    return (log_rar(log_g + eps, log_gdag) - log_rar(log_g - eps, log_gdag)) / (2 * eps)


def neg_log_like(log_gdag, sig_int, lgo, lgb, sy, sx):
    r = lgo - log_rar(lgb, log_gdag)
    var = sy**2 + (rar_slope(lgb, log_gdag) * sx) ** 2 + sig_int**2
    return 0.5 * np.sum(r**2 / var + np.log(var))


def profile_nll(sig_int, *data):
    """Minimise over g_dagger at fixed sig_int. Returns (min NLL, best log g_dagger)."""
    res = minimize_scalar(lambda p: neg_log_like(p, sig_int, *data),
                          bounds=(-10.5, -9.5), method="bounded")
    return res.fun, res.x


def fit_sigma_int(lgo, lgb, sy, sx, grid=np.linspace(0.0, 0.2, 81)):
    """Profile-likelihood fit of sigma_int on a grid (step 0.0025 dex).

    The 1-sigma interval is where 2*(NLL - NLL_min) < 1, so it cannot be narrower than the grid step.
    """
    prof = np.array([profile_nll(s, lgo, lgb, sy, sx)[0] for s in grid])
    best = prof.argmin()
    inside = grid[2 * (prof - prof[best]) < 1.0]
    log_gdag = profile_nll(grid[best], lgo, lgb, sy, sx)[1]
    return grid[best], inside.min(), inside.max(), log_gdag


# ============================== GALAXY-LEVEL TESTS ==============================
def benjamini_hochberg(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        running = min(running, p[idx] * m / rank)
        adjusted[idx] = running
    return adjusted


def weighted_slope(x, y, w):
    """Weighted least-squares slope of y on the standardised x (dex per 1 sigma of x)."""
    z = (x - x.mean()) / x.std(ddof=1)
    W = w / w.sum()
    zbar, ybar = (W * z).sum(), (W * y).sum()
    return (W * (z - zbar) * (y - ybar)).sum() / (W * (z - zbar) ** 2).sum()


def analyse(g, cols, nboot=N_BOOT):
    """Spearman rho and weighted slope of mean_resid against each column, with 95% galaxy-bootstrap
    intervals and BH-adjusted p-values. Whole galaxies are resampled because points in a galaxy
    share distance, inclination and M/L errors."""
    y = g["mean_resid"].values
    w = g["n_points"].values.astype(float)
    n = len(g)
    boot_idx = rng.integers(0, n, (nboot, n))      # same resamples for every column
    rows = []
    for c in cols:
        x = g[c].values
        rho, p_raw = stats.spearmanr(x, y)
        rho_b = np.array([stats.spearmanr(x[i], y[i])[0] for i in boot_idx])
        slope_b = np.array([weighted_slope(x[i], y[i], w[i]) for i in boot_idx])
        rows.append(dict(prop=c, rho=rho,
                         rho_lo=np.nanpercentile(rho_b, 2.5), rho_hi=np.nanpercentile(rho_b, 97.5),
                         p_raw=p_raw, slope=weighted_slope(x, y, w),
                         slope_lo=np.nanpercentile(slope_b, 2.5), slope_hi=np.nanpercentile(slope_b, 97.5)))
    out = pd.DataFrame(rows)
    out["p_BH"] = benjamini_hochberg(out["p_raw"].values)
    return out


# ============================== INJECTION-RECOVERY ==============================
def rho_critical(n, alpha):
    """Smallest |Spearman rho| that is significant at level alpha for n galaxies (t approximation)."""
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return t / np.sqrt(n - 2 + t**2)


def power_curve(g, col, betas, alpha, trials=N_TRIALS):
    """Fraction of trials in which a Spearman test detects an injected trend beta*z.

    Each trial shuffles the observed galaxy residuals (destroying any real trend but keeping the
    noise), adds beta times the standardised property z, and tests the rank correlation.
    """
    x = g[col].values
    z = (x - x.mean()) / x.std(ddof=1)
    y = g["mean_resid"].values
    n = len(y)
    rx = stats.rankdata(x)
    rx = (rx - rx.mean()) / np.sqrt(((rx - rx.mean()) ** 2).sum())     # centred, unit-norm ranks of x
    rho_crit = rho_critical(n, alpha)

    power = []
    for beta in betas:
        Y = np.array([rng.permutation(y) for _ in range(trials)]) + beta * z
        ranks = np.argsort(np.argsort(Y, axis=1), axis=1).astype(float)
        ranks -= ranks.mean(axis=1, keepdims=True)
        rho = (ranks * rx).sum(axis=1) / np.sqrt((ranks**2).sum(axis=1))
        power.append((np.abs(rho) > rho_crit).mean())
    return np.array(power)


def beta_at_80_percent(betas, power):
    """Smallest beta detected in 80% of trials (linear interpolation)."""
    if power.max() < 0.8:
        return np.nan
    i = np.argmax(power >= 0.8)
    if i == 0:
        return betas[0]
    return np.interp(0.8, [power[i - 1], power[i]], [betas[i - 1], betas[i]])


# ============================== PLOTTING ==============================
def finish(fig, *names):
    """Save a figure under every given file name in OUT_DIR, then show or close it."""
    for name in names:
        fig.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_rar(pts, g_dagger, scatter):
    gb = np.logspace(pts["log_gbar"].min() - 0.2, pts["log_gbar"].max() + 0.2, 300)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.2, 7.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax.scatter(pts["log_gbar"], pts["log_gobs"], s=5, alpha=0.35, color="tab:blue",
               edgecolors="none", label=f"SPARC points (N = {len(pts)})")
    ax.plot(np.log10(gb), np.log10(gb), "k--", lw=1, label="1:1 (no dark matter)")
    ax.plot(np.log10(gb), np.log10(rar_function(gb, g_dagger)), "r-", lw=1.8,
            label=rf"RAR fit, $g_\dagger$ = {g_dagger / 1e-10:.2f}$\times10^{{-10}}$ m s$^{{-2}}$")
    ax.set_ylabel(r"$\log_{10}\,g_{\rm obs}$  [m s$^{-2}$]")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.text(0.97, 0.05, f"observed scatter = {scatter:.2f} dex", transform=ax.transAxes,
            ha="right", fontsize=9)
    axr.scatter(pts["log_gbar"], pts["resid"], s=4, alpha=0.3, color="tab:blue", edgecolors="none")
    axr.axhline(0, color="r", lw=1.2)
    axr.set_ylim(-0.6, 0.6)
    axr.set_xlabel(r"$\log_{10}\,g_{\rm bar}$  [m s$^{-2}$]")
    axr.set_ylabel("residual [dex]")
    finish(fig, "fig_rar.pdf", "rar_plot.png")


def plot_histogram(resid):
    med = resid.median()
    mad = 1.4826 * np.median(np.abs(resid - med))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=30, color="steelblue", alpha=0.8)
    ax.axvline(resid.mean(), color="red", lw=2, label=f"mean = {resid.mean():+.3f}")
    ax.axvline(med, color="k", lw=2, ls="--", label=f"median = {med:+.3f}")
    ax.set_xlabel("RAR residual [dex]")
    ax.set_ylabel("radial points")
    ax.legend()
    ax.set_title(f"std = {resid.std(ddof=1):.3f} dex, MAD = {mad:.3f} dex, skew = {stats.skew(resid):+.2f}, "
                 f"excess kurtosis = {stats.kurtosis(resid):.2f}", fontsize=9)
    finish(fig, "fig_hist.pdf")


def plot_residuals_vs_properties(g, res):
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.9), sharey=True)
    for ax, c in zip(axes, PRIMARY):
        r = res.loc[res["prop"] == c].iloc[0]
        ax.scatter(g[c], g["mean_resid"], s=5 + 1.5 * g["n_points"], alpha=0.55,
                   color="tab:blue", edgecolors="none")
        ax.axhline(0, color="r", lw=1)
        ax.set_title(rf"$\rho$ = {r.rho:+.2f} [{r.rho_lo:+.2f}, {r.rho_hi:+.2f}]", fontsize=8)
        ax.set_xlabel(LABELS[c], fontsize=9)
    axes[0].set_ylabel("mean residual [dex]")
    finish(fig, "fig_resid.pdf")


def plot_power(betas, pw05, pw01):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.fill_between(betas, np.min([pw05[c] for c in PRIMARY], axis=0),
                    np.max([pw05[c] for c in PRIMARY], axis=0), color="tab:blue", alpha=0.35,
                    label=r"$\alpha$ = 0.05 (range over 5 properties)")
    ax.plot(betas, np.mean([pw01[c] for c in PRIMARY], axis=0), "k--", lw=1.2,
            label=r"$\alpha$ = 0.01 (mean)")
    ax.axhline(0.8, color="gray", lw=0.8, ls=":")
    ax.axvline(SIGMA_INT_DESMOND, color="r", lw=1, label=rf"$\sigma_{{\rm int}}$ = {SIGMA_INT_DESMOND} dex")
    ax.set_xlabel(r"injected trend $\beta$ [dex per 1$\sigma$ of property]")
    ax.set_ylabel("detection probability")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    finish(fig, "fig_power.pdf")


def plot_ml_sensitivity(sens):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    axs[0].plot(sens["ML_disk"], sens["g_dagger_e10"], "o-")
    axs[0].set_xlabel(r"$\Upsilon_d$")
    axs[0].set_ylabel(r"$g_\dagger$ ($10^{-10}$ m s$^{-2}$)")
    axs[1].plot(sens["ML_disk"], sens["scatter"], "o-")
    axs[1].set_xlabel(r"$\Upsilon_d$")
    axs[1].set_ylabel("RAR scatter [dex]")
    fig.tight_layout()
    finish(fig, "fig_ML_sensitivity.pdf")


def plot_radial(centres, mean, sem):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.errorbar(centres, mean, yerr=sem, fmt="o-", capsize=3)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"normalised radius $R/R_{\rm max}$")
    ax.set_ylabel("mean residual [dex]")
    fig.tight_layout()
    finish(fig, "fig_radial.pdf")


# ============================== MAIN ANALYSIS ==============================
def main():
    # ---- 1. Data and RAR fit ----
    table = read_sparc_table(TABLE_PATH)
    print(f"Galaxies in table: {len(table)} (expect 175)")
    n_sel = ((table["Inc"] > MIN_INCLINATION) & (table["Q"] <= MAX_QUALITY)).sum()
    print(f"After Inc >= 30 and Q <= {MAX_QUALITY}: {n_sel} galaxies")

    pts = build_points(table)
    print(f"Radial points kept: {len(pts)} from {pts['Galaxy'].nunique()} galaxies (McGaugh+2016: 2693 points)")

    g_dagger, pts["resid"] = fit_rar(pts)
    scatter = pts["resid"].std(ddof=1)
    mad = 1.4826 * np.median(np.abs(pts["resid"] - pts["resid"].median()))
    print(f"g_dagger = {g_dagger:.3e} m/s^2 (published ~1.2e-10)")
    print(f"scatter std = {scatter:.3f} dex (published ~0.13) | robust MAD = {mad:.3f} dex | "
          f"mean resid = {pts['resid'].mean():+.4f}")

    pts.to_csv(os.path.join(OUT_DIR, "rar_points.csv"), index=False)
    gal = galaxy_summary(pts, table)
    gal.to_csv(os.path.join(OUT_DIR, "rar_galaxies.csv"), index=False)
    plot_rar(pts, g_dagger, scatter)
    plot_histogram(pts["resid"])

    # ---- 2. Intrinsic scatter ----
    ep = add_error_model(pts, table)
    lgo, lgb, sy, sx = (ep[c].values for c in ["log_gobs", "log_gbar", "sig_y", "sig_x"])
    sig_best, sig_lo, sig_hi, log_gdag_best = fit_sigma_int(lgo, lgb, sy, sx)
    print(f"\nmedian sig_y = {np.median(sy):.3f} dex, sig_x = {np.median(sx):.3f} dex")
    print(f"g_dagger = {10 ** log_gdag_best:.3e} m/s^2 | sigma_int = {sig_best:.3f} dex "
          f"(1-sigma {sig_lo:.3f} to {sig_hi:.3f})")
    print(f"Desmond (2023): {SIGMA_INT_DESMOND} dex. If sigma_int = 0 the assumed errors explain "
          "all the scatter: report an upper limit.")

    # ---- 3. Residuals against galaxy properties ----
    g = add_properties(gal, ups=UPSILON_DISK)
    print(f"\n{len(g)} galaxies with >= {MIN_POINTS} points; "
          f"std of per-galaxy mean residual = {g['mean_resid'].std(ddof=1):.3f} dex")
    res = analyse(g, PRIMARY)
    print(res.round(3).to_string(index=False))
    diag = analyse(g, ["D", "Inc"])           # distance and inclination as nuisance diagnostics
    print("\nDiagnostics (distance, inclination):\n", diag.round(3).to_string(index=False))
    plot_residuals_vs_properties(g, res)

    # ---- 4. Injection-recovery ----
    betas = np.round(np.arange(0, 0.1301, 0.01), 3)
    pw05 = {c: power_curve(g, c, betas, 0.05) for c in PRIMARY}
    pw01 = {c: power_curve(g, c, betas, 0.01) for c in PRIMARY}
    print("\nfalse-positive rate at beta=0:", np.mean([pw05[c][0] for c in PRIMARY]).round(3), "(alpha=0.05)")
    print("min detectable trend, 80% power (dex per 1 sigma):")
    for c in PRIMARY:
        print(f"  {c:10s} alpha=0.05: {beta_at_80_percent(betas, pw05[c]):.3f}   "
              f"alpha=0.01: {beta_at_80_percent(betas, pw01[c]):.3f}")
    plot_power(betas, pw05, pw01)

    # ---- 5a. Exploratory: low vs high acceleration galaxies ----
    cut = g["med_log_gbar"].median()
    halves = [("LOW acceleration", g[g["med_log_gbar"] <= cut]),
              ("HIGH acceleration", g[g["med_log_gbar"] > cut])]
    print(f"\nsplit at median log g_bar = {cut:.2f}: N_low = {len(halves[0][1])}, N_high = {len(halves[1][1])}")
    for name, sub in halves:
        out = analyse(sub, PRIMARY + ["log_lambda"], N_BOOT_SPLIT)
        print(f"\n{name}")
        print(out[["prop", "rho", "rho_lo", "rho_hi", "p_raw", "p_BH"]].round(3).to_string(index=False))

    # ---- 5b. Exploratory: high vs low surface brightness (KS test) ----
    med_sb = g["log_SBeff"].median()
    hi_sb, lo_sb = g[g["log_SBeff"] > med_sb], g[g["log_SBeff"] <= med_sb]
    D_ks, p_ks = stats.ks_2samp(hi_sb["mean_resid"], lo_sb["mean_resid"])
    print(f"\nKS: D = {D_ks:.3f}, p = {p_ks:.3f} (N = {len(hi_sb)} vs {len(lo_sb)}); "
          f"mean resid {hi_sb['mean_resid'].mean():+.3f} vs {lo_sb['mean_resid'].mean():+.3f}")

    # ---- 5c. Exploratory: residual vs normalised radius ----
    rp = pts.copy()
    rp["R_norm"] = rp["R_kpc"] / rp.groupby("Galaxy")["R_kpc"].transform("max")   # R / R_max per galaxy
    rp["bin"] = pd.cut(rp["R_norm"], np.linspace(0, 1, 6), include_lowest=True)   # 5 bins of width 0.2
    in_sample = rp[rp["Galaxy"].isin(g["Galaxy"])]
    per_gal_bin = in_sample.groupby(["Galaxy", "bin"], observed=True)["resid"].mean().unstack()
    bin_mean = per_gal_bin.mean()
    bin_sem = per_gal_bin.std() / np.sqrt(per_gal_bin.count())   # standard error across galaxies
    plot_radial(np.linspace(0.1, 0.9, 5), bin_mean.values, bin_sem.values)

    inner = rp[rp["R_norm"] <= 0.5].groupby("Galaxy")["resid"].mean()
    outer = rp[rp["R_norm"] > 0.5].groupby("Galaxy")["resid"].mean()
    d_io = (outer - inner).reindex(g["Galaxy"]).dropna()          # galaxies with both inner and outer points
    print(f"outer minus inner mean residual: {d_io.mean():+.3f} +/- {d_io.std() / np.sqrt(len(d_io)):.3f} dex "
          f"(N = {len(d_io)}), Wilcoxon p = {stats.wilcoxon(d_io).pvalue:.3f}")

    # ---- 6. Robustness: other M/L ratios and Q = 1 only ----
    variants = [("fiducial (0.5, 0.7)", 0.5, 0.7, 2), ("low M/L (0.4, 0.6)", 0.4, 0.6, 2),
                ("high M/L (0.6, 0.8)", 0.6, 0.8, 2), ("Q = 1 only", 0.5, 0.7, 1)]
    rows = []
    for name, ud, ub, max_q in variants:
        v_pts = build_points(table, ud, ub, max_q)
        v_gdag, v_pts["resid"] = fit_rar(v_pts)
        v_g = add_properties(galaxy_summary(v_pts, table), ups=ud)
        v_res = analyse(v_g, PRIMARY, N_BOOT_ROBUST)
        for _, r in v_res.iterrows():
            rows.append(dict(variant=name, N_gal=len(v_g), N_pts=len(v_pts), g_dagger_e10=v_gdag / 1e-10,
                             scatter=v_pts["resid"].std(ddof=1), prop=r["prop"], rho=r["rho"], p_BH=r["p_BH"]))
    rob = pd.DataFrame(rows)
    print("\nSpearman rho by variant:")
    print(rob.pivot_table(index="variant", columns="prop", values="rho").round(2))
    summary = rob.groupby("variant")[["N_gal", "N_pts", "g_dagger_e10", "scatter"]].first().round(3)
    summary["p_BH"] = rob.groupby("variant")["p_BH"].min().round(3)     # smallest adjusted p in each variant
    print("\nRobustness summary:\n", summary)

    # Disk M/L sweep with the bulge ratio fixed at 0.7
    ml_rows = []
    for ud in [0.3, 0.4, 0.5, 0.6, 0.7]:
        s_pts = build_points(table, ud, 0.7, 2)
        s_gdag, s_resid = fit_rar(s_pts)
        ml_rows.append((ud, s_gdag / 1e-10, s_resid.std(ddof=1), len(s_pts)))
    sens = pd.DataFrame(ml_rows, columns=["ML_disk", "g_dagger_e10", "scatter", "N_pts"])
    print("\nDisk M/L sweep:\n", sens.round(3))
    plot_ml_sensitivity(sens)

    # ---- Summary of the numbers quoted in the paper ----
    print("\n=== NUMBERS FOR THE PAPER ===")
    print(f"RAR: {pts['Galaxy'].nunique()} galaxies, {len(pts)} points, g_dagger = {g_dagger / 1e-10:.2f}e-10, "
          f"std = {scatter:.3f}, MAD = {mad:.3f}")
    print(f"point-level sigma_int = {sig_best:.3f} dex (1-sigma {sig_lo:.3f} to {sig_hi:.3f}); "
          f"Desmond {SIGMA_INT_DESMOND}")
    print(f"galaxy-level: {len(g)} galaxies, std of mean residuals = {g['mean_resid'].std(ddof=1):.3f}")
    print(res[["prop", "rho", "rho_lo", "rho_hi", "p_raw", "p_BH", "slope", "slope_lo", "slope_hi"]]
          .round(3).to_string(index=False))
    print("min detectable trend, 80% power, alpha=0.05:",
          {c: round(float(beta_at_80_percent(betas, pw05[c])), 3) for c in PRIMARY})
    print(summary)


if __name__ == "__main__":
    main()
