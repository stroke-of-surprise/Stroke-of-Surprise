#!/usr/bin/env python3
"""

)   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/

Losses applied to spline geometry

© Daniel Berio (@colormotor) 2025
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from . import config, diffvg_utils
import torchvision.transforms.functional as Fv

from torch.nn import functional as F
from math import prod
import pdb

device = config.device


def make_deriv_loss(deriv, ref_size=1.0, dimensionless=False, log=False):
    """Smoothing with derivative square magnitude"""
    if dimensionless:
        ref_size = 1.0

    def deriv_loss(paths):
        jloss = 0.0

        c = 0
        for stroke in paths:
            if isinstance(stroke, diffvg_utils.SmoothingBSpline):
                d = stroke.inner(deriv, normalize_size=ref_size)[0, 0]
                a, b = stroke.domain()
                T = b - a
                if dimensionless:
                    k = stroke.spline_degree + 1
                    vmean = stroke.inner(1, normalize_size=ref_size)[0, 0] / T
                    d *= T ** (2 * deriv - 3) / vmean**2
                    if log:
                        d = torch.log(d)
                else:
                    d /= T
                jloss += d
                c += 1
        return jloss / c

    return deriv_loss


def make_bbox_loss(box, pad=5):
    """Keep points inside bounding box"""
    (x_min, y_min), (x_max, y_max) = box
    x_min = x_min + pad
    x_max = x_max - pad
    y_min = y_min + pad
    y_max = y_max - pad

    func = torch.nn.functional.softplus

    def bbox_loss(paths):
        loss = 0.0
        n = 0

        for path in paths:
            if type(path) == torch.Tensor:
                points = path
            else:
                points = path.get_points()
            x = points[:, 0]
            y = points[:, 1]
            x_lo = func(x_min - x)
            x_hi = func(x - x_max)
            y_lo = func(y_min - y)
            y_hi = func(y - y_max)
            loss += torch.sum((x_lo + x_hi + y_lo + y_hi))
            n += len(x)

        return loss / n

    return bbox_loss


def bending_loss(
    nodes,
    cyclic,
):
    if cyclic:
        x = torch.cat((nodes, nodes[:2]), dim=0)
    else:
        x = nodes

    e = x[1:] - x[:-1]  # edge vectors
    l = torch.linalg.norm(e, dim=1)  # edge lengths
    cross = e[:-1, 0] * e[1:, 1] - e[:-1, 1] * e[1:, 0]

    # curvature
    dot = torch.sum(e[:-1] * e[1:], dim=1)
    denom = l[:-1] * l[1:] + dot + 1e-15
    kappa = 2.0 * cross / denom

    # Average segment length
    l_bar = 0.5 * (l[:-1] + l[1:])

    return torch.mean((kappa**2) / (l_bar + 1e-15))


def make_bending_loss(subd):
    def f(shapes):
        l = 0.0
        for path in shapes:
            Q = path.param("points")

            if type(path) == diffvg_utils.SmoothingBSpline:
                u = np.linspace(*path.domain(), len(Q) * subd)[2:-2]
            else:
                u = np.linspace(0, 1, subd)
            x = path.samples(u)
            l += bending_loss(x, path.closed) / len(x)
        return l / len(shapes)

    return f


### Overlap loss


def compute_pixel_color(luminosity, alpha, num_layers, background_lumi):
    """
    Premultiplied alpha pixel luminosity (Porter and Duff)
    for layers of constant luminosity and alpha and a given background luminosity
    """
    # Initialize resulting color and alpha with the background
    result_lumi = background_lumi
    result_alpha = 1.0

    for _ in range(num_layers):
        # Current layer's color and alpha
        layer_lumi = luminosity
        layer_alpha = alpha
        new_alpha = result_alpha + layer_alpha * (1 - result_alpha)
        result_lumi = (
            layer_lumi * layer_alpha + result_lumi * result_alpha * (1 - layer_alpha)
        ) / new_alpha
        result_alpha = new_alpha

    return result_lumi


def make_overlap_loss(alpha, lumi=0.0, bg_lumi=1.0, blur=2, subtract_widths=False):
    func = torch.relu
    # func = lambda x:  torch.nn.functional.softplus(x, 5)
    class Loss:
        def __init__(self):
            pass

        def __call__(self, im, paths=[], lumi=0.0, bg_lumi=1.0):
            if blur > 0:
                sigma = blur
                k = int(np.ceil(4 * sigma) + 1)
                im = Fv.gaussian_blur(im.unsqueeze(0).unsqueeze(0), k, sigma)[0, 0]
            # Subtract 'joint' overlaps (assumung circles)
            stroke_lumi = compute_pixel_color(lumi, 1 - alpha, 1, bg_lumi)
            overlap_lumi = compute_pixel_color(lumi, alpha, 2, bg_lumi)
            lumi_diff = abs(stroke_lumi - overlap_lumi)
            area = 0.0
            if subtract_widths and paths:
                for path in paths:
                    points = path.get_points()
                    r = path.get_stroke_width()
                    n = path.num_segments(points)
                    for i in range(1, n):
                        p = points[i * 3]
                        if (
                            p[0] > 0
                            and p[0] < im.shape[1]
                            and p[1] > 0
                            and p[1] < im.shape[0]
                        ):
                            area += np.pi * r[i * 3] ** 2
            self.im = im.detach().cpu().numpy()
            overlap = func((1.0 - im) - stroke_lumi)
            overlap_loss = (torch.sum(overlap) + lumi_diff * area) / (
                im.shape[0] * im.shape[1]
            )
            self.overlap = overlap.detach().cpu().numpy()
            return overlap_loss

    return Loss()


### Repulsion loss converted to PyTorch from https://github.com/kenji-tojo/fab3dwire
def repulsion_kernel(p, Tp, q, Tq, d0=1, eps=1e-1):
    q0 = q
    q1 = q + Tq
    r0 = q0 - p
    r1 = q1 - p

    s0 = np.sum(r0 * Tp, axis=1)
    s1 = np.sum(r1 * Tp, axis=1)

    mask = ~(s0 * s1 >= 0.0)

    s0 = torch.abs(s0[mask])
    s1 = torch.abs(s1[mask])

    s = s0 / (s0 + s1)
    q = (1.0 - s) * q0 + s * q1

    # Kernel
    r = q - p
    norm = torch.norm(r, dim=1) / d0

    mask = norm < 1
    norm = norm[mask]
    q = q[mask]
    p = p[mask]

    w = 1.0 / (1.0 + eps * eps) / (1.0 + eps * eps)
    norm_sq = norm * norm
    norm_eps_sq = norm_sq + eps * eps

    out = 1.0 / norm_eps_sq + w * norm_sq - w * (2.0 + eps * eps)
    return torch.sum(out)


def repulsion_kernel_vectorized(p, Tp, d0=10, eps=2e-1, signed=True):
    n_points = p.shape[0]

    # Cyclically shift p and Tp to generate all combinations
    p_cyclic = p.unsqueeze(1).expand(n_points, n_points, -1)
    Tp_cyclic = Tp.unsqueeze(1).expand(n_points, n_points, -1)

    # Create q and Tq for all pairs
    q = p_cyclic.transpose(0, 1)
    Tq = Tp_cyclic.transpose(0, 1)

    # Remove diagonal (self-interaction terms)
    mask_diag = torch.eye(n_points, dtype=torch.bool, device=p.device)
    q = q[~mask_diag].view(n_points, n_points - 1, -1)
    Tq = Tq[~mask_diag].view(n_points, n_points - 1, -1)

    # Select q with projection on tangent
    q0 = q
    q1 = q + Tq
    r0 = q0 - p.unsqueeze(1)
    r1 = q1 - p.unsqueeze(1)

    # Compute scalar projections
    Tp_broadcast = Tp.unsqueeze(1)
    s0 = torch.sum(r0 * Tp_broadcast, dim=2)
    s1 = torch.sum(r1 * Tp_broadcast, dim=2)

    # Mask where s0 * s1 < 0
    if signed:
        mask = s0 * s1 < 0
    else:
        mask = torch.abs(s0 * s1) > 0  # < torch.inf
    # Filter valid pairs
    s0 = torch.abs(s0[mask])
    s1 = torch.abs(s1[mask])
    if s0.numel() == 0:  # Return zero loss if no valid pairs
        return torch.tensor(0.0, requires_grad=True, device=p.device)

    # Compute interpolation factor and q
    s = s0 / (s0 + s1)
    q0_filtered = q0[mask]
    q1_filtered = q1[mask]
    s = s.unsqueeze(1)
    q = (1.0 - s) * q0_filtered + s * q1_filtered

    # Compute relative vectors and norms
    p_filtered = p.unsqueeze(1).expand(n_points, n_points - 1, -1)[mask]
    r = q - p_filtered  # Shape: (ValidPairs, D)
    norm = torch.norm(r, dim=1) / d0

    # Apply mask for points within range (norm < 1)
    mask_range = norm < 1
    norm = norm[mask_range]
    q = q[mask_range]

    # Kernel computation
    w = 1.0 / ((1.0 + eps * eps) ** 2)
    norm_sq = norm * norm
    norm_eps_sq = norm_sq + eps * eps

    out = 1.0 / norm_eps_sq + w * norm_sq - w * (2.0 + eps * eps)
    out = torch.sum(out)
    return out


def randspace(a, b, n, minstep=0.1, maxstep=0.6):
    """Generate a sequence from a to b with random steps
    minstep and maxstep define the step magnitude
    """
    v = minstep + np.random.uniform(size=(n - 1)) * (maxstep - minstep)
    v = np.hstack([[0.0], v])
    v = v / np.sum(v)
    v = np.cumsum(v)
    return a + v * (b - a)


def repulsion_kernel_semi_vectorized(
    p, Tp, d0=10, eps=2e-1, signed=True, batch_size=512
):
    """Helps with memory issues"""
    n_points = p.shape[0]
    total_loss = 0.0

    for i_start in range(0, n_points, batch_size):
        i_end = min(i_start + batch_size, n_points)

        p_batch = p[i_start:i_end]
        Tp_batch = Tp[i_start:i_end]

        # Expand p_batch against all other points
        pi = p_batch.unsqueeze(1)
        Tpi = Tp_batch.unsqueeze(1)

        pj = p.unsqueeze(0)
        Tpj = Tp.unsqueeze(0)

        # Skip self-interaction
        mask_self = torch.arange(i_start, i_end, device=p.device).unsqueeze(
            1
        ) != torch.arange(n_points, device=p.device).unsqueeze(
            0
        )  # (B, N)

        # Compute r0, r1
        r0 = pj - pi
        r1 = (pj + Tpj) - pi

        s0 = torch.sum(r0 * Tpi, dim=2)
        s1 = torch.sum(r1 * Tpi, dim=2)

        if signed:
            mask = (s0 * s1 < 0) & mask_self
        else:
            mask = (torch.abs(s0 * s1) > 0) & mask_self

        if not mask.any():
            continue

        # Only compute valid pairs
        s0 = torch.abs(s0[mask])
        s1 = torch.abs(s1[mask])

        s = s0 / (s0 + s1)

        q0 = pj.expand(i_end - i_start, n_points, -1)[mask]
        q1 = (pj + Tpj).expand(i_end - i_start, n_points, -1)[mask]

        q = (1.0 - s.unsqueeze(1)) * q0 + s.unsqueeze(1) * q1
        p_valid = pi.expand(i_end - i_start, n_points, -1)[mask]

        r = q - p_valid
        norm = torch.norm(r, dim=1) / d0

        mask_norm = norm < 1
        norm = norm[mask_norm]

        if norm.numel() == 0:
            continue

        w = 1.0 / ((1.0 + eps * eps) ** 2)
        norm_sq = norm * norm
        norm_eps_sq = norm_sq + eps * eps

        out = 1.0 / norm_eps_sq + w * norm_sq - w * (2.0 + eps * eps)
        total_loss += torch.sum(out)

    return total_loss


def path_repulsion_loss(paths, subd=25, random=False, dist=10, signed=True):
    if type(paths) != list:
        paths = [paths]
    X = []
    dX = []
    count = 0
    for P in paths:
        n = P.num_points()
        if type(P) != diffvg_utils.SmoothingBSpline:
            n = subd
        else:
            n = n * subd
        if random:
            u = randspace(*P.domain(), n)
        else:
            u = np.linspace(*P.domain(), n)
        X.append(P.samples(u)[:, :2])
        dX.append(P.samples(u, der=1)[:, :2])
        count += n

    X = torch.vstack(X)
    dX = torch.vstack(dX)
    dX = dX / (dX.norm(dim=1, keepdim=True) + 1e-10)
    # return repulsion_kernel_loop(X, dX, d0=dist, signed=signed)/(n**2)
    return repulsion_kernel_semi_vectorized(X, dX, d0=dist, signed=signed) / (
        count**2
    )


def make_repulsion_loss(subd=10, random=False, single=True, signed=True, dist=10):
    def repulsion_loss(paths):
        if single:
            return sum(
                [
                    path_repulsion_loss(path, subd, random, dist=dist, signed=signed)
                    for path in paths
                ]
            ) / len(paths)
        else:
            return path_repulsion_loss(
                paths, subd, random, dist=dist, signed=signed
            )  # /len(paths)

    return repulsion_loss
