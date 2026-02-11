#!/usr/bin/env python3
'''
 )   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/

B-Spline construction+smoothing utilities

© Daniel Berio (@colormotor) 2025
'''

import numpy as np
import scipy
from scipy.interpolate import BSpline
from scipy.linalg import block_diag
from scipy.special import binom
from scipy.integrate import quad, fixed_quad
import numbers
import pdb

clamp_considering_mult = True

def is_number(x):
    return isinstance(x, numbers.Number)

def Nk(k, der=0):
    '''Uniform B-spline basis'''
    basis_knots = np.linspace(0, k, k+1)
    bsp = BSpline.basis_element(basis_knots).derivative(der)
    return lambda u: bsp(np.clip(u, 0, k))

def add_multiplicity(Q, multiplicity, noise=0.0):
    '''Repeat key-points of input, resulting in sharper turns and additional
       degrees of freedom for smoothing'''
    Q = np.kron(Q, np.ones((multiplicity, 1)))
    return Q + np.random.uniform(-noise, noise, Q.shape)

def keypoint_times(P, t, k, mult=1, closed=False):
    '''Approximate occurrence of curvature extrema along spline,
    given key-point multiplicity, will work with odd multiplicity only'''
    p = k-1

    if closed:
        o = k/2 - np.floor(k/2)
        kt = np.linspace(t[p]-o+1, t[-k]-o, len(P))
    else:
        o1 = o2 = k/2
        kt = np.linspace(t[p]+o1, t[-k]-o2, len(P)-2*mult)
        kt = np.concatenate([[t[p]]*mult, kt, [t[-k]]*mult])
        # kt = np.concatenate([np.linspace(t[p], kt[0], mult+1)[:-1],
        #                      kt,
        #                      np.linspace(kt[-1], t[-k], mult+1)[1:]])

    return kt

def tc(P,
       k,
       mult=1,
       closed=False,
       clamped=True):
    '''Compute knots and control point given key-points, order of spline and
    key-point multiplicity'''
    p = k-1
    M = len(P)

    if closed:
        # Distribute p repeated control points to ensure periodicity
        half_offset = p//2
        offset_remainder = p - half_offset
        P = np.vstack([P[-(offset_remainder):], P, P[:half_offset]])
    elif clamped:
        # Force rest
        if clamp_considering_mult:
            P = np.vstack([[P[0]]*(p), P[mult:-mult], [P[-1]]*(p)])
        else:
            P = np.vstack([[P[0]]*(p), P[1:-1], [P[-1]]*(p)])
    else:
        pass

    n = len(P)
    m = n+k
    t = np.arange(m-(p)*2)

    t = np.concatenate([-np.arange(p)[::-1]-1, t, np.arange(p)+t[-1]+1]) # [t[-1]]*p])
    return t, P


def tcu(P,
        k,
        mult=1,
        closed=False,
        clamped=True):
    '''Compute knots, control points and key-point passage times for given
    key-points, spline order and key-point multiplicity'''
    t, c = tc(P, k, mult, closed, clamped=clamped)
    u = keypoint_times(P, t, k, mult, closed)
    return t, c, u


def retrieve_control_points(C, k, mult, closed):
    ''' Given control points with given multiplicity (clamping, periodic)
    Retrieve the base control points
    '''
    p = k - 1
    if closed:
        half_offset = p//2
        offset_remainder = p - half_offset
        P = C[offset_remainder:-half_offset]
        return P
    else:
        assert(clamp_considering_mult) # TODO
        o1 = p-mult
        o2 = p-mult
        return C[o1:-o2]


def bspline_samples(P, num_or_u, k, deriv=0, mult=1, closed=False,
                    get_tc=False, output={}):
    ''' Sample a bspline defined by key-points P,
    assuming multiplicity has been set by user on P'''
    if not is_number(deriv):
        return [bspline_samples(P, num_or_u, k, d, mult, closed, False, output=output) for d in deriv]
    t, C = tc(P, k, mult, closed)
    X = eval_bspline(t, C, k, num_or_u, deriv, output=output)
    if get_tc:
        return X, t, C
    return X


# Compute once as this will be slow for realtime
gramian_cache = {}
bgramian_cache = {}


def uniform_gramian(nbasis, k, der, get_Gm=False):
    ''' Compute the gramian for uniform b-spline bases of a given order
        and derivative, as described in:
        - Kano, Nakata & Martin (2005) Optimal Curve Fitting and Smoothing Using Normalized Uniform B-splines: A Tool for Studying Complex Systems, Applied Mathematics and Computation.
        and
        - Vermeulen, Bartels & Heppler (1992) Integrating Products of B-splines, SIAM journal on scientific and statistical computing.
    '''

    def inner(f, g, a, b):
        res, _ = quad(lambda x: f(x)*g(x), a, b)
        return res

    G = np.zeros((nbasis, nbasis))
    N = Nk(k, der=der) #bspline_basis(k, der=der)
    # cache = {}
    for i in range(nbasis):
        for j in range(i, nbasis):
            offset = j - i # j > i
            key = (k, der, offset)
            if abs(offset) < k:
                if key in gramian_cache:
                    G[i, j] = G[j, i] = gramian_cache[key]
                else:
                    f = lambda x: N([x])
                    g = lambda x: N([x-(j-i)])
                    G[i, j] = G[j,i] = inner(f, g, 0, k)
                    gramian_cache[key] = G[i, j]
    # Subtract bounds to get integral over internal knot interval
    # See
    # Fujioka & Kano (2007) Constructing Character Font Models from Measured Human Handwriting Motion, IEEE.
    p = k-1
    Gm = np.zeros((p, p))
    for i in range(0, p):
        for j in range(i, p):
            key = (k, der, i, j)
            if key in bgramian_cache:
                Gm[i, j] = Gm[j, i] = bgramian_cache[key]
            else:
                f = lambda x: N([x])
                g = lambda x: N([x-(j-i)])
                Gm[i,j] = Gm[j, i] = inner(f, g, 0, p-i)
                bgramian_cache[key] = Gm[i, j]
    S = np.eye(p)[:, ::-1]
    # if get_Gm:
    #    return G, Gm # S@Gm@S
    G[:p,:p] -= Gm
    G[-p:,-p:] -= S@Gm@S
    return G


def rot_2d(theta):
    m = np.eye(2)
    ct = np.cos(theta)
    st = np.sin(theta)
    m[0,0] = ct; m[0,1] = -st
    m[1,0] = st; m[1,1] = ct
    return m


def make_sigma(theta, scale):
    ''' Builds a 2d covariance with a rotation and scale'''
    Phi = rot_2d(theta)
    S = np.diag(np.array(scale)**2)
    return Phi @ S @ Phi.T


def make_lambda(theta, scale):
    return np.linalg.inv(make_sigma(theta, scale))


def compute_smoothing_bspline(Q, mult=1, der=3, k=6, r=1000, sigma=1.0,
                            end_weight=100,
                            pspline=False,
                            use_keypoints=False,
                            closed=False,
                            constrain=True,
                            get_keypoints=True):
    ''' Smoothing b-spline that tracks key-points in Q'''
    dim = len(Q[0])
    p = k - 1
    # Add multiplicity
    Qm = Q
    # Compute spline knots and control points
    t, C, uk = tcu(Qm, k, mult=mult, closed=closed)
    # Assume clamping
    skip = None
    if not closed:
        if constrain:
            skip = None
        else:
            skip = p

    n = len(C)

    p = k-1

    def construct_B(u, der=0):
        N = Nk(k, der)
        B = np.zeros((len(u), n))
        for i in range(n):
            B[:, i] = N(u-t[i])
        return B

    if use_keypoints:
        s = keypoint_times(Qm, t, k, mult, closed)
    else:
        s = np.linspace(t[k-1], t[-k], len(Qm))

    B = construct_B(s)
    if skip is not None:
        B = B[:,skip:-skip]
    Bb = np.kron(B, np.eye(dim))
    Qb = np.hstack(Qm)
    Cb = np.hstack(C)
    if pspline:
        D = np.diff(np.eye(n), der)
        G = D @ D.T
    else:
        G = uniform_gramian(n, k, der)
    if skip is not None:
        G = G[skip:-skip, skip:-skip]

    Gb = np.kron(G, np.eye(dim))

    if is_number(sigma):
        Wi = np.eye(dim)*(1.0/sigma**2)
    else: # Assume user provided covariance
        Wi = np.eye(dim)
        Wi[:sigma.shape[0],
           :sigma.shape[1]] = np.linalg.inv(sigma)
    W = block_diag(*([Wi]*len(Qm)))

    # Optimize
    if constrain:
        K = np.zeros((n*dim, n*dim))
        if closed:
            pp = p

            K = np.zeros((pp*dim, n*dim))
            K[:(pp)*dim,:(pp)*dim] = np.eye((pp)*dim)
            K[-(pp)*dim:,-(pp)*dim:] = -np.eye((pp)*dim)
            ksize = pp*dim
            target = np.zeros(pp*dim)
        else:
            p1 = (p)*dim
            p2 = (p)*dim

            K = np.zeros((p1+p2, n*dim))
            K[:p1, :p1] = np.eye(p1)
            K[-p2:, -p2:] = np.eye(p2)
            ksize = p1 + p2
            target = np.concatenate([Cb[:p1], Cb[-p2:]])

        L = np.block([[r*Gb + Bb.T@W@Bb, K.T],
                      [K,                np.zeros((ksize, ksize))]])
        Chat = np.linalg.inv(L)@np.concatenate([Bb.T@W@Qb,
                                                target])
        Chat = Chat[:n*dim]
    else:
        Chat = np.linalg.pinv(r*Gb + Bb.T@W@Bb)@(Bb.T@W@Qb)
    if skip is not None:
        Chat = Chat.reshape(n-skip*2, dim)
        C[skip:-skip] = Chat
    else:
        C = Chat.reshape(n, dim)
    if get_keypoints:
        return t, C, uk
    return t, C


def smoothing_bspline(Q, num_or_u, k=6, deriv=0, **kwargs):
    '''Smoothing spline samples for key-points `Q`'''
    t, C = compute_smoothing_bspline(Q, get_keypoints=False, **kwargs)
    return eval_bspline(t, C, k, num_or_u, deriv)


def eval_bspline(t, C, k, num_or_u, deriv=0, output={}):
    ''' Evaluate a B-pline given knots `t` control points `C` and order `k` (degree+1)'''
    if not is_number(deriv):
        return [eval_bspline(t, C, k, num_or_u, d) for d in deriv]

    p = k - 1
    dim = C.shape[1]
    n = len(C)
    basis_knots = np.linspace(0, k, k+1)
    bsp = BSpline.basis_element(basis_knots).derivative(deriv)
    Bk = lambda u: bsp(np.clip(u, 0, k))
    # Design matrix
    if isinstance(num_or_u, np.ndarray):
        u = num_or_u
    else:
        u = np.linspace(t[p], t[-k], num_or_u)
    output['dt'] = u[1] - u[0]
    output['u'] = u
    Bu = np.zeros((n, len(u)))
    for i in range(n):
        Bu[i, :] = Bk(u-t[i])
    # Samples
    Bu = np.kron(Bu.T, np.eye(dim))
    Chat = np.hstack(C)
    X = (Bu@Chat).reshape(len(u), dim)
    return X


def bspline_to_bezier_mat(n):
    ''' Returns a n X n matrix that converts a b-spline of degree n
    to a Bezier curve of degree n. Adapted from matlab code in:
    - Romani & Sabin (2004) The Conversion Matrix between Uniform B-spline and Bézier Representations, Computer Aided Geometric Design.
    '''
    from math import factorial
    S = np.eye(n + 1)

    for k in range(n - 1, 0, -1):
        nc = round((n - k) / 2 + 1e-9)
        # 1. Shift-and-Subtraction
        fc = n - nc
        for j in range(fc, n):
            for i in range(k, n + 1):
                S[i, j] = S[i, j] - S[i, j + 1]

        # 2. Integration
        for j in range(n - 1, fc - 1, -1):
            S[k - 1, j] = S[n, j + 1]
            for i in range(k, n + 1):
                S[i, j] = S[i - 1, j] + S[i, j]

        if (n + 1 - k) % 2 == 1:
            end = k - 2
            if end < 0:
                end = None
            S[k - 1:n + 1, n - nc - 1] = S[n:end:-1, fc]

    # # 3. Replication of columns
    for j in range(round(n / 2)):
        S[:, j] = S[::-1, n + 1 - j - 1]

    return S/factorial(n)


def bspline_to_bezier_chain_mat(p, num_cps):
    k = p + 1
    S = bspline_to_bezier_mat(p)
    num_pieces = num_cps - p
    num_bezier_cps = num_pieces*p + 1
    Ss = np.zeros((num_bezier_cps,
                   num_cps))
    for i in range(num_pieces):
        Ss[i*p:i*p+k,i:i+k]  = S
    return Ss


def reduction_matrix(n, m, r=1, s=1):
    ''' Bezier curve degree reduction based on
    Sunwoo (2005) Matrix Representation for Multi-Degree Reduction of Bézier Curves, Computer Aided Geometric Design.
    and based on code here:
    https://github.com/salvipeter/bezier-reduction
    '''
    N, M = n - (r + s + 2), m - (r + s + 2)
    L = compute_L(M, 2*r + 2, 2*s + 2)
    E = compute_E(N, 2*r + 2, 2*s + 2)
    a = compute_a(n, m, max(r, s))
    D = np.diag([binom(n, r + 1 + i)/binom(N, i) for i in range(N + 1)]) # Lemma 4
    # Lemma 3 (Eq. 41)
    C = np.zeros((N + 1, n + 1))
    for j in range(r + 1, n + r - m + 1):
        for k in range(r + 1):
            d = -binom(n, k) / binom(n, j)
            C[j-r-1, k] = d * np.sum([binom(n - m, j - i) * a[i-k]
                                  for i in range(max(k, j - (n - m)), r + 1)])

    for j in range(r + 1, n - s):
        C[j-r-1, j] = 1
    for j in range(m - s, n - s):
        for k in range(n - s, n + 1):
            d = -binom(n, k)/binom(n, j)
            C[j-r-1, k] = d * np.sum([binom(n - m, j - m + i) * a[i+k-n]
                                  for i in range(max(n - k, m - j), s + 1)])
    # Lemma 7 (Eq. 45)
    D1 = np.zeros((m + 1, M + 1))
    for i in range(r + 1, m - s):
        D1[i, i-r-1] = binom(M, i - r - 1) / binom(m, i)
    # Theorem 2 (Eq. 36)
    Q1 = np.zeros((m + 1, n + 1))
    for j in range(r + 1):
        for k in range(j + 1):
            Q1[j, k] = (binom(n, k) / binom(m, j)) * a[j-k]
    for j in range(m - s, m + 1):
        for k in range(j + (n - m), n + 1):
            Q1[j, k] = (binom(n, k) / binom(m, j)) * a[k-j-(n-m)]
    # Theorem 3 (Eq. 46)
    Q2 = D1 @ L.T @ np.eye(M + 1, N + 1) @ E.T @ D @ C
    # Theorem 4 (Eq. 47)
    return Q1 + Q2


def bezier_chain_reduction_mat(n, m, num_cps, r=1, s=1):
    num_pieces = int((num_cps - 1)/n)
    Q = reduction_matrix(n,m,r,s)
    Qs = np.zeros((num_pieces*m+1, num_pieces*n+1))
    for i in range(num_pieces):
        Qs[i*m:i*m+m+1,i*n:i*n+n+1]  = Q
    return Qs


# Helpers for Bezier degree reduction
def b_ij(m, n, i, j): # Eq 10.
    return (binom(m, i)*binom(n - m, j - i))/binom(n, j)

def compute_L(n, r, s): # Lemma 1
    L = np.zeros((n+1, n+1))
    for k in range(n+1):
        for j in range(n+1):
            for i in range(max(0, j+k-n), min(j,k)+1):
                L[k, j] += ((-1)**(k + i) *
                            b_ij(k, n, i, j) *
                            ((binom(k+r, i)*binom(k+s, k-i))/binom(k, i)))
    return L

def compute_E(n, r, s): # Lemma 2
    E = np.zeros((n+1, n+1))
    for k in range(n+1):
        for j in range(n+1):
            for i in range(j+1):
                E[k, j] += ((-1)**(j + i) *
                            ((binom(j+r, i)*binom(j+s, j-i))/
                              binom(n+r+s+j, k+s+i)))
            E[k, j] *= (((2*j + r + s + 1)/
                         (n + r + s + j + 1))*
                         (binom(j + r + s, r)*binom(n, k))/
                          binom(j + r, r))
    return E


def compute_a(n, m, l): # Eq. 32
    a = np.zeros(l + 1)
    a[0] = 1
    for k in range(1, l + 1):
        a[k] = -np.sum([binom(n - m, k - i)*a[i] for i in range(k)])
    return a


def resample_bspline(P, dt, degree, max_speed, accel_time, subd=50, get_t=False, deriv=0, output={}):
    n = len(P)*subd
    out = {}
    dX, ddX = bspline_samples(P, n, degree+1, deriv=[1,2], output=out)
    dt_orig = out['dt']
    dX = (dX*dt_orig)/dt
    ddX = np.diff(dX, axis=0)/dt
    #ddX = (dX*dt_orig)/dt
    smax = np.max(np.linalg.norm(dX, axis=1))
    amax = np.max(np.linalg.norm(ddX, axis=1))

    div_s = max_speed/smax

    # Compute time step that satisfies max speed and acceleration bounds
    if accel_time is not None:
        max_accel = max_speed/accel_time
        div_a = np.sqrt(max_accel/amax)

        div = min(div_s, div_a)
    else:
        div = div_s
    n_subd = int(n/div)
    X = bspline_samples(P, n_subd, degree+1, deriv=deriv, output=output)
    return X
