"""Numerical building blocks for the incomplete-markets household (plain numpy).

Grids and shapes used throughout
--------------------------------
* nE : int, number of income levels. nA : int, number of asset levels.
* e_grid : float64 (nE,)   the income levels households can have (labour productivity), mean 1.
* a_grid : float64 (nA,)   the asset levels households can hold, increasing, a_grid[0] is the borrowing limit.
* Pi     : float64 (nE, nE) income transition matrix; Pi[i, j] = probability of income j next period given income i.
* Any 'household' array of shape (nE, nA) is indexed [income level, asset level].
* D      : float64 (nE, nA) the distribution of households: share of the population at each (income, asset) point;
           non-negative and sums to 1.
* Va     : float64 (nE, nA) marginal value of an extra unit of assets (how much better off a household would be with
           one more unit of savings); this is the 'backward variable' of the endogenous gridpoint method.
"""

from __future__ import annotations

from math import comb

import numpy as np


def rouwenhorst(rho: float, sd: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretise a persistent log-income process into n income levels (Rouwenhorst method).

    Parameters
    ----------
    rho : float  persistence of income (autocorrelation), between 0 and 1; closer to 1 = income shocks last longer.
    sd : float   standard deviation of log income across households (inequality of income).
    n : int      number of income levels, at least 2.

    Returns
    -------
    e_grid : float64 (n,)     income levels, rescaled so the population average is exactly 1.
    pi : float64 (n,)         long-run share of households at each income level; sums to 1.
    Pi : float64 (n, n)       transition matrix between income levels; each row sums to 1.
    """
    p = (1.0 + rho) / 2.0
    Pi = np.array([[p, 1.0 - p], [1.0 - p, p]])
    for m in range(3, n + 1):
        P = np.zeros((m, m))
        P[:-1, :-1] += p * Pi
        P[:-1, 1:] += (1.0 - p) * Pi
        P[1:, :-1] += (1.0 - p) * Pi
        P[1:, 1:] += p * Pi
        P[1:-1, :] /= 2.0
        Pi = P
    pi = np.array([comb(n - 1, k) for k in range(n)], dtype=float) / 2.0 ** (n - 1)
    psi = sd * np.sqrt(n - 1)
    e = np.exp(np.linspace(-psi, psi, n))
    e = e / (pi @ e)
    return e, pi, Pi


def make_agrid(amin: float, amax: float, n: int, curv: float) -> np.ndarray:
    """Build the asset grid, with more points near the borrowing limit where behaviour changes fastest.

    Parameters
    ----------
    amin : float  lowest asset level = borrowing limit (0 means households cannot borrow).
    amax : float  highest asset level on the grid.
    n : int       number of grid points.
    curv : float  curvature; larger = points packed more tightly near amin.

    Returns
    -------
    np.ndarray, float64, shape (n,): strictly increasing, first element amin, last element amax.
    """
    u = np.linspace(0.0, 1.0, n)
    return amin + (amax - amin) * ((1.0 + curv) ** u - 1.0) / curv


def backward_step(Va_p, a_grid, e_grid, r, w, beta, eis, d):
    """One period of household decision-making, solved backwards in time (endogenous gridpoint method).

    Given how much households value savings NEXT period, find how much each household saves and consumes THIS period.
    Preferences this period: u(c) = c^(1-1/eis)/(1-1/eis) - d*c, so the first-order condition is
    c^(-1/eis) - d = beta * Va_p: the pleasure of the last unit consumed today must equal the discounted value of
    the last unit saved. Households at the borrowing limit simply save the minimum.

    Parameters
    ----------
    Va_p : float64 (nE, nA)  expected marginal value of assets next period, indexed by (income today, assets chosen today).
    a_grid : float64 (nA,)   asset grid.
    e_grid : float64 (nE,)   income levels.
    r : float                net return on assets this period (rate per period).
    w : float                wage per unit of income level this period.
    beta : float             discount factor between this period and the next.
    eis : float              elasticity of intertemporal substitution (1/curvature of utility).
    d : float                taste shifter this period; positive = less desire to consume now.

    Returns
    -------
    Va : float64 (nE, nA)  marginal value of assets today, indexed by (income today, assets held at the start of today).
    a : float64 (nE, nA)   assets chosen for tomorrow by a household with that income and starting assets.
    c : float64 (nE, nA)   consumption chosen today by that household; always positive.
    """
    c_next = (d + beta * Va_p) ** (-eis)
    coh = (1.0 + r) * a_grid[None, :] + w * e_grid[:, None]
    a = np.empty_like(coh)
    for ie in range(coh.shape[0]):
        a[ie] = np.interp(coh[ie], c_next[ie] + a_grid, a_grid)
    a = np.maximum(a, a_grid[0])
    c = coh - a
    Va = (1.0 + r) * (c ** (-1.0 / eis) - d)
    return Va, a, c


def lottery(a, a_grid):
    """Turn each household's chosen asset level into weights on the two nearest grid points.

    The chosen level usually lies between two grid points; the household is split across them so that average
    assets are preserved exactly.

    Parameters
    ----------
    a : float64 (nE, nA)   chosen assets, from `backward_step`.
    a_grid : float64 (nA,) asset grid.

    Returns
    -------
    idx : int64 (nE, nA)    index of the lower neighbouring grid point (between 0 and nA-2).
    w_hi : float64 (nE, nA) weight, between 0 and 1, on the UPPER neighbour (idx + 1); the lower gets 1 - w_hi.
    """
    idx = np.clip(np.searchsorted(a_grid, a, side="right") - 1, 0, len(a_grid) - 2)
    w_hi = np.clip((a - a_grid[idx]) / (a_grid[idx + 1] - a_grid[idx]), 0.0, 1.0)
    return idx, w_hi


def forward_step(D, idx, w_hi, Pi):
    """Move the population one period forward: households save, then draw next period's income.

    Parameters
    ----------
    D : float64 (nE, nA)     distribution today, indexed by (income today, assets held at the start of today).
    idx : int64 (nE, nA)     lower grid neighbour of each household's chosen assets, from `lottery`.
    w_hi : float64 (nE, nA)  weight on the upper neighbour, from `lottery`.
    Pi : float64 (nE, nE)    income transition matrix.

    Returns
    -------
    float64 (nE, nA): distribution tomorrow, indexed by (income tomorrow, assets held at the start of tomorrow);
    non-negative and with the same total as D.
    """
    nE, nA = D.shape
    Dp = np.zeros_like(D)
    for ie in range(nE):
        Dp[ie] = np.bincount(idx[ie], weights=D[ie] * (1.0 - w_hi[ie]), minlength=nA) + np.bincount(
            idx[ie] + 1, weights=D[ie] * w_hi[ie], minlength=nA
        )
    return Pi.T @ Dp


def solve_household_ss(a_grid, e_grid, Pi, pi_e, r, w, beta, eis, tol, Va0=None, D0=None,
                       maxit_va=100_000, maxit_d=200_000):
    """Find the steady state of the household problem for a fixed interest rate, wage and discount factor.

    Iterates `backward_step` until the value function stops changing, then iterates `forward_step` until the
    distribution of households stops changing.

    Parameters
    ----------
    a_grid, e_grid, Pi : as in the module description.
    pi_e : float64 (nE,)    long-run share of households at each income level (used to start the distribution).
    r, w, beta, eis : float net return, wage, discount factor, elasticity of intertemporal substitution.
    tol : float             stop when the largest change between iterations is below this.
    Va0 : float64 (nE, nA) | None   starting guess for the value function (speeds up repeated calls).
    D0 : float64 (nE, nA) | None    starting guess for the distribution.
    maxit_va, maxit_d : int  iteration limits; RuntimeError is raised if they are exceeded.

    Returns
    -------
    Va : float64 (nE, nA)  steady-state marginal value of assets.
    a : float64 (nE, nA)   steady-state savings rule.
    c : float64 (nE, nA)   steady-state consumption rule.
    D : float64 (nE, nA)   steady-state distribution of households (sums to 1).
    """
    if Va0 is None:
        coh = (1.0 + r) * a_grid[None, :] + w * e_grid[:, None]
        Va = (1.0 + r) * (0.1 * coh) ** (-1.0 / eis)
    else:
        Va = Va0
    for _ in range(maxit_va):
        Va_new, a, c = backward_step(Pi @ Va, a_grid, e_grid, r, w, beta, eis, 0.0)
        err = np.max(np.abs(Va_new - Va))
        Va = Va_new
        if err < tol:
            break
    else:
        raise RuntimeError("household value function did not converge")

    idx, w_hi = lottery(a, a_grid)
    if D0 is None:
        D = np.zeros((len(e_grid), len(a_grid)))
        D[:, 0] = pi_e
    else:
        D = D0
    for _ in range(maxit_d):
        D_new = forward_step(D, idx, w_hi, Pi)
        err = np.max(np.abs(D_new - D))
        D = D_new
        if err < tol:
            break
    else:
        raise RuntimeError("stationary distribution did not converge")
    return Va, a, c, D
