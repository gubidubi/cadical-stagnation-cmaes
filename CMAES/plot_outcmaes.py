#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_outcmaes.py -- turn a pycma (cma-es) log folder into ONE readable figure.

Reads the plain text files a CMA-ES run writes (fit.dat, xmean.dat,
stddev.dat, axlen.dat) and draws the same style of figure as the demo
"clean_cma_plot.png", with panels:

    (a) objective value:       best-ever f, median of the population,
                               and the 25%-75% quartile band (log y)
    (b) global step size sigma                       (log y)
    (c) axis ratio of the covariance matrix          (log y)
    (d) per-coordinate spread: std of the sampling distribution in each
        coordinate -- or |mean_i - xopt_i| if the optimum is known (--xopt)

Only numpy + matplotlib are needed; pycma itself is NOT imported, and the
column layout is taken from the '% # columns="..."' header lines of the
data files, so logs written by different pycma versions should work too.
Negative or zero objective values are handled (a symmetric log scale is
used instead of a plain log scale in that case).

Data files are produced by pycma's logger:
  * cma.fmin(...) and es.optimize(...) log automatically,
  * in a manual ask()/tell() loop you must call es.logger.add() each
    generation (pycma >= 4 no longer does it for you), and
  * keep the 'verbose' option > -9, otherwise logging is disabled.

Examples
--------
    python3 plot_outcmaes.py outcmaes/                  # -> outcmaes_clean.png
    python3 plot_outcmaes.py runs/trial01 -o trial01.png
    python3 plot_outcmaes.py outcmaes/ --xopt 1         # known optimum x*=1
    python3 plot_outcmaes.py outcmaes/ --xopt 0,0,0     # vector optimum
    python3 plot_outcmaes.py outcmaes/ --xaxis iters --show
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

NAVY, TEAL, ORANGE, GREY = "#0b3d91", "#0e7c7b", "#e07b00", "#6b6b6b"
NMAX_POINTS = 5000          # decimate long runs to this many points per panel
TINY = 1e-300               # clamp for log axes


# ----------------------------------------------------------------------
# reading the data files
# ----------------------------------------------------------------------

def _read_matrix(prefix: str, fname: str):
    """Return (matrix, header_lines) for one logger file, (None, '') if absent.

    Accepts folder prefixes with or without trailing separator, and the
    no-separator 'name_prefix' style ('myrun' -> 'myrunfit.dat')."""
    for path in (os.path.join(prefix, fname),
                 prefix + fname if not prefix.endswith(("/", os.sep)) else None):
        if path is None or not os.path.isfile(path):
            continue
        with open(path, "r") as fh:
            header = "".join(line for line in fh
                             if line.startswith(("%", "#")))
        data = np.loadtxt(path, comments=["%", "#"])
        data = np.atleast_2d(data)
        return (data, header) if data.size else (None, "")
    return None, ""


def _column_names(header: str):
    """Parse the column names out of  % # columns="a, b, c"  lines."""
    m = re.search(r'columns="([^"]*)"', header)
    return [tok.strip() for tok in m.group(1).split(",")] if m else []


def _norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9%]", "", tok.lower())


def _find_column(names, canon: str):
    """Index of the column whose name matches `canon`, else None.
    Rules are chosen so that e.g. 'best' does not match 'bestever'."""
    for i, tok in enumerate(names):
        n = _norm(tok)
        hit = {
            "iteration":  n in ("iteration", "iterations", "iter"),
            "evaluation": n.startswith("evaluation") or n in ("evals",),
            "sigma":      n == "sigma",
            "axis_ratio": n.startswith("axisratio"),
            "bestever":   n.startswith("bestever") or n.startswith("best-ever"),
            "best":       n == "best",
            "median":     n.startswith("median"),
            "worst":      n.startswith("worst"),
            "iqr":        n.startswith("interquart") or n.startswith("iqr"),
            "p25":        n.startswith("25"),
            "maxlen":     n.startswith("maxaxis") or n.startswith("maxprincipal"),
            "minlen":     n.startswith("minaxis") or n.startswith("minprincipal"),
            "stds":       n.startswith("std"),
            "xmean":      n == "xmean" or n.startswith("xmean"),
        }.get(canon, False)
        if hit:
            return i
    return None


def _idx_of(mat, names, canon, default):
    """Column index of `canon` in `mat`, else `default` (both bounds-checked
    against the matrix width; None if out of range)."""
    if mat is None:
        return None
    i = _find_column(names, canon)
    if i is None:
        i = default
    return i if i is not None and i < mat.shape[1] else None


def _dedupe(a):
    """Logger files may hold two rows for the same iteration (fit.dat only
    advances on new iterations, the other files do not): keep the last row
    of each iteration, preserving order."""
    if a is None or len(a) < 2:
        return a
    keep = np.r_[True, np.diff(a[:, 0]) > 0]
    return a[keep]


def _decimate(a, target=NMAX_POINTS):
    """Subsample long runs; always keep the final state."""
    if a is None or len(a) <= target:
        return a
    step = int(np.ceil(len(a) / target))
    idx = np.arange(0, len(a), step)
    if idx[-1] != len(a) - 1:
        idx = np.append(idx, len(a) - 1)
    return a[idx]


# ----------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------

def _set_loglike_y(ax, y):
    """Plain log scale when all values are strictly positive, otherwise a
    symmetric log scale (objective values often cross zero / are negative)."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size and y.min() > 0:
        ax.set_yscale("log")
    elif y.size:
        a = np.abs(y[y != 0])
        if a.size:
            lt = 10.0 ** np.floor(np.log10(a.min()))
            ax.set_yscale("symlog", linthresh=max(lt, np.finfo(float).tiny))
    return ax


def _fan(ax, x, Y, label="coordinate", nmax=30):
    """Color-mapped fan of up to `nmax` curves (one per coordinate),
    with a min-max envelope when there are more coordinates than lines."""
    n = Y.shape[1]
    cmap = plt.get_cmap("viridis")
    if n == 1:
        ax.plot(x, np.maximum(Y[:, 0], TINY), color=cmap(0.5), lw=1.6)
        return
    sel = np.unique(np.round(np.linspace(0, n - 1, min(n, nmax))).astype(int))
    if n > nmax:                       # envelope over ALL coordinates
        lo = np.maximum(Y.min(axis=1), TINY)
        hi = np.maximum(Y.max(axis=1), TINY)
        ax.fill_between(x, lo, hi, color="0.72", alpha=0.35, lw=0, zorder=1)
    for k in sel:
        ax.plot(x, np.maximum(Y[:, k], TINY), color=cmap(k / (n - 1)),
                lw=1.0, alpha=0.9, zorder=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    ticks = np.unique(np.round(np.linspace(0, n - 1, min(n, 6))).astype(int))
    cbar.set_ticks(ticks)
    cbar.set_label(label, fontsize=9)


# ----------------------------------------------------------------------
# main plotting routine (also importable: plot_clean("some/outcmaes"))
# ----------------------------------------------------------------------

def plot_clean(prefix="outcmaes/", out=None, xopt=None, xaxis="evals",
               title=None, show=False, nmax=30):
    """Draw the readable diagnostics figure from a pycma log folder.

    prefix : folder / filename prefix written by the pycma logger
    xopt   : known optimum (scalar or length-N sequence): panel (d) then
             shows |mean_i - xopt_i| instead of per-coordinate stds
    xaxis  : 'evals' (default) or 'iters' as abscissa
    out    : image path to save; default "<prefix>_clean.png"
    show   : pop up an interactive window instead of saving
    Returns a small summary dict (iterations, evals, dim, best_ever, path).
    """
    # ---- read + parse the files -------------------------------------
    fit, hdr_fit = _read_matrix(prefix, "fit.dat")
    if fit is None:
        raise SystemExit(
            f"no fit.dat found in '{prefix}'. Does that folder hold pycma log "
            "files? Manual ask()/tell() loops only write them if you call "
            "es.logger.add() every generation (and 'verbose' > -9); "
            "es.optimize() and cma.fmin() log automatically.")
    fit = _decimate(_dedupe(fit))
    cf = _column_names(hdr_fit)
    it_f = _find_column(cf, "iteration")
    if it_f is None:
        it_f = 0
    ev_f = _find_column(cf, "evaluation")
    i_sig = _idx_of(fit, cf, "sigma", 2)
    i_bever = _idx_of(fit, cf, "bestever", None) or _idx_of(fit, cf, "best", None)
    i_med = _idx_of(fit, cf, "median", None)
    i_iqr = _idx_of(fit, cf, "iqr", None)
    i_p25 = _idx_of(fit, cf, "p25", None)
    i_axr = _idx_of(fit, cf, "axis_ratio", None)

    xm, hdr_xm = _read_matrix(prefix, "xmean.dat")
    sd, hdr_sd = _read_matrix(prefix, "stddev.dat")
    ax, hdr_ax = _read_matrix(prefix, "axlen.dat")
    for m in (xm, sd, ax):
        if m is not None:
            m = _decimate(_dedupe(m))
    xm = _decimate(_dedupe(xm))
    sd = _decimate(_dedupe(sd))
    ax = _decimate(_dedupe(ax))
    cxm = _column_names(hdr_xm)
    csd = _column_names(hdr_sd)
    cax = _column_names(hdr_ax)
    i_xmean = _idx_of(xm, cxm, "xmean", 5)
    i_stds = _idx_of(sd, csd, "stds", 5)

    def dim_of(mat, start):
        d = None if mat is None or start is None else mat.shape[1] - start
        return d if d is not None and d >= 1 else None
    dim = dim_of(xm, i_xmean) or dim_of(sd, i_stds)

    def abscissa(mat, it_idx, ev_idx):
        if xaxis == "evals" and ev_idx is not None:
            return mat[:, ev_idx], "function evaluations"
        return mat[:, it_idx], "iterations"
    x_f, xlab = abscissa(fit, it_f, ev_f)
    niter = int(np.round(fit[-1, it_f]))
    nev = int(np.round(fit[-1, ev_f])) if ev_f is not None else None

    # ---- panel (d) data: optimum distances or per-coordinate stds ----
    Yd, x_d = None, None
    if xopt is not None:
        if xm is None or i_xmean is None:
            raise SystemExit("--xopt needs xmean.dat, which this folder lacks.")
        n_opt = np.size(xopt)
        if n_opt == 1:
            xopt_arr = np.full(dim, float(xopt))
        elif n_opt == dim:
            xopt_arr = np.asarray(xopt, dtype=float)
        else:
            raise SystemExit(f"--xopt has {n_opt} values but this is a {dim}-D run")
        Yd = np.abs(xm[:, i_xmean:] - xopt_arr)
        x_d, _ = abscissa(xm, 0, _idx_of(xm, cxm, "evaluation", 1))
        d_title = "$|\\bar x_i - x_i^*|$ per coordinate"
        d_label = "coordinate"
        d_ylabel = "distance of the distribution mean to the optimum"
    elif sd is not None and i_stds is not None:
        Yd = sd[:, i_stds:]
        x_d, _ = abscissa(sd, 0, _idx_of(sd, csd, "evaluation", 1))
        d_title = "Std of the sampling distribution, per coordinate"
        d_label = "coordinate"
        d_ylabel = "per-coordinate std ($\\sigma \\cdot \\sqrt{\\mathrm{diag}(C)}$)"

    # ---- axis ratio: fit.dat column, else derived from axlen.dat ----
    if i_axr is not None:
        axr, x_axr, xlab_axr = fit[:, i_axr], x_f, xlab
    elif ax is not None:
        i_max = _idx_of(ax, cax, "maxlen", 3)
        i_min = _idx_of(ax, cax, "minlen", 4)
        with np.errstate(divide="ignore", invalid="ignore"):
            axr = ax[:, i_max] / ax[:, i_min]
        x_axr, xlab_axr = abscissa(ax, 0, _idx_of(ax, cax, "evaluation", 1))
    else:
        axr = None

    # ---- assemble the panels we actually have data for --------------
    panels = [("a", (i_bever is not None) or (i_med is not None)),
              ("b", i_sig is not None),
              ("c", axr is not None),
              ("d", Yd is not None)]
    order = [k for k, has in panels if has]
    if not order:
        raise SystemExit("nothing plottable found in this log folder.")
    npanel = len(order)
    if npanel >= 4:            # 2x2
        ncol, nrow = 2, 2
    else:                      # single row of npanel (1x1 / 1x2 / 1x3)
        ncol, nrow = npanel, 1
    height = {1: 3.4, 2: 3.8, 3: 4.2}.get(npanel, 8.6)
    fig, grid = plt.subplots(nrow, ncol, figsize=(13.5, height), squeeze=False)
    flat = [grid[r][c] for r in range(nrow) for c in range(ncol)]
    pos = {k: flat[i] for i, k in enumerate(order)}

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.linewidth": 0.8})
    bottom_row = (nrow == 1) or None

    # ---- (a) objective value ----------------------------------------
    ax = pos["a"]
    ax.set_title("(a) Objective value", fontweight="bold", loc="left")
    ax.set_ylabel("objective $f$")
    if i_bever is not None:
        lname = ("best-ever $f$" if _find_column(cf, "bestever") is not None
                 else "best $f$")
        ax.plot(x_f, fit[:, i_bever], color=NAVY, lw=2.2, label=lname)
    if i_med is not None:
        ax.plot(x_f, fit[:, i_med], color=GREY, lw=1.1, ls=":",
                label="median of population")
    if i_p25 is not None and i_iqr is not None:
        ax.fill_between(x_f, fit[:, i_p25],
                        np.maximum(fit[:, i_p25] + fit[:, i_iqr],
                                   fit[:, i_p25]),
                        color=TEAL, alpha=0.18, lw=0, label="25%-75% band")
    ser = [fit[:, i] for i in (i_bever, i_med) if i is not None]
    if ser:
        _set_loglike_y(ax, np.concatenate(ser))
    if i_bever is not None and np.isfinite(fit[-1, i_bever]):
        unit, nlast = (("evals", nev) if xaxis == "evals"
                       else ("iterations", niter))
        ax.annotate(f"best = {fit[-1, i_bever]:.2e}\nafter {nlast} {unit}",
                    xy=(0.03, 0.05), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="0.8", lw=0.8))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ---- (b) sigma ----------------------------------------------------
    ax = pos["b"]
    ax.set_title("(b) Global step size $\\sigma$", fontweight="bold", loc="left")
    ax.set_ylabel("$\\sigma$")
    ax.plot(x_f, fit[:, i_sig], color=ORANGE, lw=2)
    _set_loglike_y(ax, fit[:, i_sig])

    # ---- (c) axis ratio ------------------------------------------------
    if "c" in pos:
        ax = pos["c"]
        ax.set_title("(c) Axis ratio of the covariance matrix",
                     fontweight="bold", loc="left")
        ax.set_ylabel("$\\lambda_\\mathrm{max}/\\lambda_\\mathrm{min}$")
        ax.plot(x_axr, axr, color=TEAL, lw=2)
        _set_loglike_y(ax, axr)
        ax.set_xlim(x_axr[0], x_axr[-1])

    # ---- (d) per-coordinate spread -------------------------------------
    if "d" in pos:
        ax = pos["d"]
        ax.set_title("(d) " + d_title, fontweight="bold", loc="left")
        ax.set_ylabel(d_ylabel)
        _fan(ax, x_d, np.asarray(Yd, dtype=float), label=d_label, nmax=nmax)
        _set_loglike_y(ax, Yd)

    # ---- abscissa labels + limits --------------------------------------
    labels = {"a": xlab, "b": xlab}
    if "c" in pos:
        labels["c"] = xlab_axr
    if "d" in pos:
        labels["d"] = xlab
    if nrow == 1:                     # single row: every panel gets an x label
        for k, ax in pos.items():
            ax.set_xlabel(labels[k])
    else:                             # 2 rows: only the bottom row
        for k, ax in pos.items():
            if order.index(k) >= len(order) - ncol:
                ax.set_xlabel(labels[k])
    for k in ("a", "b"):
        pos[k].set_xlim(x_f[0], x_f[-1])
    if "c" in pos and i_axr is None:   # c plotted from axlen.dat
        pos["c"].set_xlim(x_axr[0], x_axr[-1])

    # ---- title -----------------------------------------------------------
    pieces = [os.path.basename(os.path.normpath(prefix)) or "cma-run",
              f"{niter} iterations"]
    if nev:
        pieces[1] += f" / {nev} evals"
    if dim:
        pieces.append(f"{dim} dims")
    if i_bever is not None and np.isfinite(fit[-1, i_bever]):
        pieces.append(f"best $f$ = {fit[-1, i_bever]:.1e}")
    top = 0.95 if nrow == 2 else 0.82
    fig.suptitle(title or "   —   ".join(pieces), fontsize=13,
                 fontweight="bold", y=0.99 if nrow == 2 else 0.91)
    fig.tight_layout(rect=(0, 0, 1, top))

    if out is None:
        base = os.path.normpath(prefix)
        out = (base.rstrip(os.sep) if base != "." else "cma") + "_clean.png"
    if show:
        fig.show()
        return dict(iterations=niter, evals=nev, dim=dim,
                    best_ever=None if i_bever is None else float(fit[-1, i_bever]),
                    path=None)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")
    return dict(iterations=niter, evals=nev, dim=dim,
                best_ever=None if i_bever is None else float(fit[-1, i_bever]),
                path=out)


# ----------------------------------------------------------------------

def _parse_xopt(s):
    if s is None:
        return None
    try:
        vals = [float(t) for t in str(s).split(",") if t.strip() != ""]
    except ValueError:
        raise SystemExit(f"cannot parse --xopt value '{s}'")
    return vals if len(vals) > 1 else vals[0]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("-----\n")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n")[-1])
    ap.add_argument("logdir", nargs="?", default="outcmaes/",
                    help="pycma log folder or filename prefix "
                         "(default: %(default)s)")
    ap.add_argument("-o", "--out", default=None,
                    help="output image path (default: '<logdir>_clean.png')")
    ap.add_argument("--xopt", default=None, metavar="V",
                    help="known optimum: scalar ('1.0') or comma list "
                         "('0,0,0'); panel (d) then shows |mean - xopt|")
    ap.add_argument("--xaxis", choices=("evals", "iters"), default="iters",
                    help="abscissa: function evaluations or iterations")
    ap.add_argument("--title", default=None, help="override the figure title")
    ap.add_argument("--max-lines", type=int, default=30, metavar="K",
                    help="max coordinate curves in panel (d); remaining "
                         "coordinates are summarized as a band (default: 30)")
    ap.add_argument("--show", action="store_true",
                    help="open an interactive window instead of saving")
    args = ap.parse_args(argv)

    plot_clean(prefix=args.logdir, out=args.out, xopt=_parse_xopt(args.xopt),
               xaxis=args.xaxis, title=args.title, show=args.show,
               nmax=args.max_lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
