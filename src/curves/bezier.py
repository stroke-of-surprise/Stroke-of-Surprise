#!/usr/bin/env python3
"""
)   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/


Bezier curve utils
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from numpy.linalg import norm
from scipy.interpolate import interp1d
import math


def num_bezier(n_ctrl, degree=3):
    if type(n_ctrl) == np.ndarray:
        n_ctrl = len(n_ctrl)
    return int((n_ctrl - 1) / degree)


def bernstein(n, i):
    bi = math.comb(n, i)  # binom(n, i)
    return lambda t, bi=bi, n=n, i=i: bi * t**i * (1 - t) ** (n - i)


def bezier_mat(p, t, deriv=0):
    n = p + 1
    if deriv > 0:
        return np.diff(np.eye(n), 1) @ bezier_mat(p - 1, t, deriv - 1)
    B = np.vstack([bernstein(p, i)(t) for i in range(n)])
    return B


def bezier(P, t, d=0):
    """Bezier curve of degree len(P)-1. d is the derivative order (0 gives positions)"""
    n = P.shape[0] - 1
    if d > 0:
        Q = np.diff(P, axis=0) * n
        return bezier(Q, t, d - 1)
    B = np.vstack([bernstein(n, i)(t) for i, p in enumerate(P)])
    return (P.T @ B).T


def bezier_piecewise(Cp, subd=100, degree=3, d=0):
    """sample a piecewise Bezier curve given a sequence of control points"""
    num = num_bezier(Cp.shape[0], degree)
    X = []
    for i in range(num):
        P = Cp[i * degree : i * degree + degree + 1, :]
        t = np.linspace(0, 1.0, subd)[:-1]
        Y = bezier(P, t, d)
        X += [Y]
    X.append(Cp[-1])
    X = np.vstack(X)
    return X


def plot_control_polygon(
    Cp, degree=3, lw=0.5, linecolor=np.ones(3) * 0.1, color=[0, 0.5, 1.0]
):
    n_bezier = num_bezier(len(Cp), degree)
    for i in range(n_bezier):
        cp = Cp[i * degree : i * degree + degree + 1, :]
        if degree == 3:
            plt.plot(cp[0:2, 0], cp[0:2, 1], ":", color=linecolor, linewidth=lw)
            plt.plot(cp[2:, 0], cp[2:, 1], ":", color=linecolor, linewidth=lw)
            plt.plot(cp[:, 0], cp[:, 1], "o", color=color, markersize=4)
        else:
            plt.plot(cp[:, 0], cp[:, 1], ":", color=linecolor, linewidth=lw)
            plt.plot(cp[:, 0], cp[:, 1], "o", color=color)


def chain_to_beziers(chain, degree=3):
    """Convert Bezier chain to list of curve segments (4 control points each)"""
    num = num_bezier(chain.shape[0], degree)
    beziers = []
    for i in range(num):
        beziers.append(chain[i * degree : i * degree + degree + 1, :])
    return beziers


def beziers_to_chain(beziers):
    """Convert list of Bezier curve segments to a piecewise bezier chain (shares vertices)"""
    n = len(beziers)
    chain = []
    for i in range(n):
        chain.append(list(beziers[i][:-1]))
    chain.append([beziers[-1][-1]])
    return np.array(sum(chain, []))


def split_cubic(bez, t):
    p1, p2, p3, p4 = bez

    p12 = (p2 - p1) * t + p1
    p23 = (p3 - p2) * t + p2
    p34 = (p4 - p3) * t + p3

    p123 = (p23 - p12) * t + p12
    p234 = (p34 - p23) * t + p23

    p1234 = (p234 - p123) * t + p123

    return np.array([p1, p12, p123, p1234]), np.array([p1234, p234, p34, p4])


def cubic_bspline_to_bezier_chain(P, periodic=False):
    """Converts a bspline to a Bezier chain
    Naive implementation of Bohm's algorithm for knot insertion
    This is a bit confusing, but rest=True assumes that the input (spline)
    control points already have repeated knots at the start and end
    In practice, rest=False needs to be fixed since we do not take knot multiplicity into account here"""

    def lerp(a, b, t):
        return a + t * (b - a)

    if periodic:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([P[0], P[0], P, P[-1], P[-1]])

    n = P.shape[0] - 1
    Cp = []
    for i in range(n - 2):
        p = P[i : i + 4]
        b1 = lerp(p[1], p[2], 1.0 / 3)
        b2 = lerp(p[2], p[1], 1.0 / 3)
        l = lerp(p[1], p[0], 1.0 / 3)
        r = lerp(p[2], p[3], 1.0 / 3)

        if not Cp:
            b0 = lerp(l, b1, 0.5)
            b3 = lerp(b2, r, 0.5)
            Cp += [b0, b1, b2, b3]
        else:
            b3 = lerp(b2, r, 0.5)
            Cp += [b1, b2, b3]
    return np.array(Cp)


def point_segment_distance(p, a, b):
    d = b - a
    # relative projection length
    u = np.dot(p - a, d) / np.dot(d, d)
    u = np.clip(u, 0, 1)

    proj = a + u * d
    return np.linalg.norm(proj - p)


def decasteljau(pts, bez, tol, level=0):
    if level > 12:
        return
    p1, p2, p3, p4 = bez
    p12 = (p1 + p2) * 0.5
    p23 = (p2 + p3) * 0.5
    p34 = (p3 + p4) * 0.5
    p123 = (p12 + p23) * 0.5
    p234 = (p23 + p34) * 0.5
    p1234 = (p123 + p234) * 0.5

    d = point_segment_distance(p1234[:2], p1[:2], p4[:2])
    if d > tol * tol:
        decasteljau(pts, [p1, p12, p123, p1234], tol, level + 1)
        decasteljau(pts, [p1234, p234, p34, p4], tol, level + 1)
    else:
        pts.append(p4)


def cubic_bezier_adaptive(Cp, tol):
    Cp = np.array(Cp)
    pts = [Cp[0]]
    for i in range(0, len(Cp) - 1, 3):
        decasteljau(pts, Cp[i : i + 4], tol)
    return np.array(pts)


# def approx_arc_length_cubic(bez):
#     c0, c1, c2, c3 = bez
#     v0 = torch.norm(c1-c0)*0.15
#     v1 = torch.norm(-0.558983582205757*c0 + 0.325650248872424*c1 + 0.208983582205757*c2 + 0.024349751127576*c3)
#     v2 = torch.norm(c3-c0+c2-c1)*0.26666666666666666
#     v3 = torch.norm(-0.024349751127576*c0 - 0.208983582205757*c1 - 0.325650248872424*c2 + 0.558983582205757*c3)
#     v4 = torch.norm(c3-c2)*.15
#     return v0 + v1 + v2 + v3 + v4
