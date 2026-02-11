#!/usr/bin/env python3
"""
Utilities wrapping diffvg for facilitating differentiable rendering of
complex primitives. Different shapes derive from the Shape class and can be
added to a Scene object.

Adapted from calligraph library:
https://github.com/colormotor/calligraph/blob/main/calligraph/diffvg_utils.py

Original © Daniel Berio (@colormotor) 2025
"""

import torch
import pydiffvg
from collections import defaultdict
import numpy as np
from . import config, bspline, bezier, geom
from easydict import EasyDict as edict

device = config.diffvg_device
pydiffvg.set_use_gpu(config.has_gpu)

# Configuration
cfg = lambda: None
cfg.one_channel_is_alpha = True


def skip_kwargs(skip, kwargs):
    """Filter out specified keys from kwargs."""
    return {k: v for k, v in kwargs.items() if k not in skip}


class Scene:
    """Abstraction of a DiffVG scene, handles colors, params and rendering."""

    scale = 1.0

    def __init__(self):
        self.groups = []
        self.primitives = []
        self.shapes = []
        self.params = defaultdict(list)
        self.shape_groups = []
        self.transforms = []

    def add_shapes(self, shapes, split_primitives=True, transform=None, **kwargs):
        """Add shapes to the scene."""
        params = args_to_params(**kwargs)
        _params = {
            "stroke_color": ([0.0, 0.0, 0.0, 1.0], False),
            "fill_color": (None, False),
            "opacity": (1.0, False),
        }
        _params.update(params)
        _params = {key: convert_param(p) for key, p in _params.items()}

        # Add stroke and fill color
        for key, p in _params.items():
            if p is not None:
                self.params[key].append(p)
                
        # Add parameters for each shape
        for shape in shapes:
            for key, p in shape.params.items():
                if p is not None:
                    self.params[key].append(p)

        primitives = sum([s.primitives for s in shapes], [])
        ind = len(self.primitives)

        # Handle transform
        shape_to_canvas = None
        shape_to_canvas_proxy = None
        for shape in shapes:
            shape_to_canvas = shape.shape_to_canvas()
            if shape_to_canvas is not None:
                shape_to_canvas_proxy = shape
                break

        if transform is not None:
            for shape in shapes:
                shape.transform = transform
            self.transforms.append(transform)

        diffvg_groups = []
        
        if split_primitives:
            if _params["fill_color"] is not None and _params["stroke_color"] is not None:
                group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor(list(range(ind, ind + len(primitives)))),
                    fill_color=get_color_param(_params["fill_color"], _params["opacity"]),
                    use_even_odd_rule=False,
                    stroke_color=None,
                )
                group._fill_opt = _params["fill_color"]
                group._stroke_opt = _params["stroke_color"]
                group._opacity_opt = _params["opacity"]
                group._shape_to_canvas_proxy = shape_to_canvas_proxy
                self.groups.append(group)
                diffvg_groups.append(group)
                fill_clr = None
            else:
                fill_clr = get_color_param(_params["fill_color"], _params["opacity"])

            for i, prim in enumerate(primitives):
                group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor(list(range(ind + i, ind + i + 1))),
                    fill_color=fill_clr,
                    use_even_odd_rule=False,
                    stroke_color=get_color_param(_params["stroke_color"], _params["opacity"]),
                )
                group._fill_opt = _params["fill_color"]
                group._stroke_opt = _params["stroke_color"]
                group._opacity_opt = _params["opacity"]
                group._shape_to_canvas_proxy = shape_to_canvas_proxy
                self.groups.append(group)
                diffvg_groups.append(group)
        else:
            group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor(list(range(ind, ind + len(primitives)))),
                use_even_odd_rule=False,
                fill_color=get_color_param(_params["fill_color"], _params["opacity"]),
                stroke_color=get_color_param(_params["stroke_color"], _params["opacity"]),
            )
            group._fill_opt = _params["fill_color"]
            group._stroke_opt = _params["stroke_color"]
            group._opacity_opt = _params["opacity"]
            group._shape_to_canvas_proxy = shape_to_canvas_proxy
            self.groups.append(group)
            diffvg_groups.append(group)
            
        self.shapes += shapes
        self.primitives += primitives
        self.shape_groups.append(
            edict({
                "shapes": shapes,
                "diffvg_groups": diffvg_groups,
                "fill_color": _params["fill_color"],
                "stroke_color": _params["stroke_color"],
            })
        )

        return self.shape_groups[-1]

    def render(
        self,
        background_image,
        postupdate=None,
        prefiltering=False,
        size=None,
        num_samples=2,
        seed=0,
    ):
        """Render the scene."""
        for group in self.groups:
            group.fill_color = get_color_param(group._fill_opt, group._opacity_opt)
            group.stroke_color = get_color_param(group._stroke_opt, group._opacity_opt)
            if group._shape_to_canvas_proxy is not None:
                group.shape_to_canvas = group._shape_to_canvas_proxy.shape_to_canvas()
                
        for tsm in self.transforms:
            tsm.shape_to_canvas()

        for shape in self.shapes:
            shape.update()

        if postupdate is not None:
            postupdate()

        if prefiltering:
            num_samples = 1

        if background_image is not None:
            background_image = torch.tensor(background_image, dtype=torch.float32).to(device)
            if len(background_image.shape) == 2:
                background_image = background_image[:, :, np.newaxis]
                background_image = background_image.repeat(1, 1, 3)
            h, w, _ = background_image.shape
        else:
            h, w = size

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            w, h, self.primitives, self.groups, use_prefiltering=prefiltering
        )
        img = pydiffvg.RenderFunction.apply(
            w, h, num_samples, num_samples, seed, None, *scene_args
        )
        
        if background_image is not None:
            img = img[:, :, 3:4] * img[:, :, :3] + background_image * (1 - img[:, :, 3:4])
            img = img[:, :, :3]
            return img
        else:
            return img

    def get_params(self, key, only_grad=True, numpy=False):
        """Get parameters by key."""
        params = []
        for p in self.params[key]:
            if p is not None:
                if only_grad and not p.requires_grad:
                    pass
                else:
                    params.append(p)
        if numpy:
            return [p.detach().cpu().numpy() for p in params]
        return params

    def get_points(self, only_grad=True, **kwargs):
        """Get point parameters."""
        return self.get_params("points", only_grad, **kwargs)

    def get_stroke_widths(self, only_grad=True, **kwargs):
        """Get stroke width parameters."""
        return self.get_params("stroke_width", only_grad, **kwargs)

    def get_stroke_colors(self, only_grad=True, **kwargs):
        """Get stroke color parameters."""
        return self.get_params("stroke_color", only_grad, **kwargs)

    def get_fill_colors(self, only_grad=True, **kwargs):
        """Get fill color parameters."""
        return self.get_params("fill_color", only_grad, **kwargs)


class Shape:
    """Generic shape base class."""

    def __init__(self, classtype=''):
        self.degree = 1
        self.transform = None
        self.classtype = classtype

    def get_degree(self):
        """Get effective degree of the primitive."""
        return self.degree

    def length(self):
        """Compute shape length."""
        P = self.param("points")
        D = torch.diff(P, axis=0)
        return torch.sum(torch.sqrt(D[:, 0] ** 2 + D[:, 1] ** 2))

    def set_params(self, params):
        """Set shape parameters from a dict."""
        self.params = {}
        for key, p in params.items():
            if key == "points":
                p = convert_param(p, scale=Scene.scale)
                self.params[key] = p
            else:
                p = convert_param(p)
                self.params[key] = p

    def param(self, name, numpy=False):
        """Get actual tensor from parameters."""
        if numpy:
            return self.params[name].detach().cpu().numpy()
        return self.params[name]

    def has_grad(self, name):
        """Returns true if a parameter exists and has gradients."""
        if name not in self.params:
            return False
        if self.params[name] is None:
            return False
        return self.params[name].requires_grad

    def setup(self):
        pass

    def update(self):
        pass

    def get_points(self):
        return self.param("points") * Scene.scale

    def get_stroke_width(self):
        return self.param("stroke_width")

    def shape_to_canvas(self):
        return None

    def to_dict(self):
        d = {"degree": self.degree}
        d.update({
            key: val.detach().cpu().numpy()
            for key, val in self.params.items()
            if val is not None
        })
        return d


class Path(Shape):
    """Generic path, can be used to construct a piecewise cubic path."""

    def __init__(
        self,
        points=None,
        degree=1,
        closed=False,
        use_distance_approx=False,
        postprocess=None,
        split_pieces=False,
        scale=None,
        **kwargs,
    ):
        super().__init__('Path')

        if scale is None:
            scale = Scene.scale

        params = {}
        if points is not None:
            params["points"] = (points, True)

        params.update(args_to_params(**kwargs))
        _params = {"stroke_width": (1.0, False)}
        _params.update(params)
        self.set_params(_params)

        self.degree = degree
        self.closed = closed
        self.split_pieces = split_pieces

        if postprocess is not None:
            setattr(Path, "postprocess", postprocess)

        self.postprocess(False)
        self.setup()

        points = self.get_points()
        num_segments = self.num_segments(points)
        
        sw = self.params["stroke_width"]
        if len(sw.shape) == 1 and len(sw) == 1:
            self.params["stroke_width"] = torch.tensor(
                torch.ones(points.shape[0]).to(device) * sw,
                requires_grad=sw.requires_grad,
            ).to(device)

        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + self.degree - 1
        
        self.primitives = [
            pydiffvg.Path(
                num_control_points=num_control_points,
                points=points,
                stroke_width=self.get_stroke_width(),
                is_closed=closed,
                use_distance_approx=use_distance_approx,
            )
        ]

    def postprocess(self, requires_grad=False):
        pass

    def num_segments(self, points):
        n = len(points)
        if self.closed:
            return bezier.num_bezier(n + 1, self.degree)
        return bezier.num_bezier(n, self.degree)

    def num_points(self):
        return len(self.param("points"))

    def has_varying_width(self):
        w = self.param("stroke_width")
        return len(w.shape) > 0 and len(w) > 1

    def domain(self):
        return 0.0, 1.0

    def samples(self, num_or_u, der=0, thick=True):
        """Sample the path."""
        Cp = self.get_points()
        num = bezier.num_bezier(Cp.shape[0], self.degree)
        if thick and self.has_varying_width():
            Cp = torch.hstack([Cp, self.get_stroke_width().reshape(-1, 1)])
        if self.closed:
            Cp = torch.vstack([Cp, Cp[0]])
        if type(num_or_u) == int:
            subd = num_or_u // num
            t = np.linspace(0, 1.0, subd)
        else:
            t = num_or_u

        B = torch.tensor(
            bezier.bezier_mat(self.degree, t, deriv=der), device=device, dtype=Cp.dtype
        )

        X = []
        for i in range(num):
            P = Cp[i * self.degree:i * self.degree + self.degree + 1, :]
            Y = (P.T @ B).T
            X += [Y[:-1]]
        X.append(Cp[-1])
        return torch.vstack(X)

    def update(self):
        self.setup()
        pts = self.get_points()
        
        if self.transform is not None:
            mat = self.transform.transform
            pts = (
                (mat @ torch.hstack([
                    pts,
                    torch.ones((len(pts), 1), device=device, dtype=torch.float32)
                ]).T).T[:, :2].contiguous()
            )

        if self.has_grad("points"):
            self.primitives[0].points = pts
        if self.has_grad("stroke_width"):
            self.primitives[0].stroke_width = self.get_stroke_width()


# Gramian cache for SmoothingBSpline
gramian_cache = {}


class SmoothingBSpline(Path):
    """
    Constructs a B-spline approximation of a given degree.
    Internally sampled and rendered as a sequence of linear segments.
    """

    def __init__(
        self,
        points=None,
        closed=False,
        subd=10,
        degree=5,
        multiplicity=1,
        pspline=False,
        pspline_weight=True,
        clamping=True,
        width_func=None,
        split_pieces=False,
        clamped=True,
        deriv_order=3,
        init_smooth_params={},
        **kwargs,
    ):
        self.subd = subd
        self.spline_degree = degree
        self.deriv_order = deriv_order
        self.clamping = clamping
        self.multiplicity = multiplicity
        self.point_offsets = None
        self.width_func = width_func

        self.pspline_weight = pspline_weight
        self.pspline = pspline
        self.init_smooth_params = init_smooth_params
        self.clamped = clamped
        self.width_postprocess = lambda x: x
        
        super().__init__(
            points, degree=3, closed=closed, split_pieces=split_pieces, **kwargs
        )
        self.classtype = 'SmoothingBSpline'

    def get_degree(self):
        return self.spline_degree

    def has_varying_width(self):
        w = self.param("stroke_width")
        return len(w.shape) > 0 and len(w) > 1 and self.width_func is None

    def setup(self):
        P = self.param("points")
        mult = self.multiplicity
        p = self.spline_degree
        w = self.width_postprocess(self.param("stroke_width"))
        has_width = self.has_varying_width()
        
        if has_width:
            Pw = torch.hstack([self.param("points"), w.unsqueeze(1)])
        else:
            Pw = P
            
        self.spline_points = Pw

        # Clamping or periodicity
        if self.closed:
            Q = torch.vstack([P[-1:], P, P[:p - 1]])
        elif self.clamped:
            Q = torch.vstack([Pw[0]] * (p) + [Pw[mult:-mult]] + [Pw[-1]] * (p))
        else:
            Q = Pw
        self.Q = Q

        k = p + 1
        t, _, kt = bspline.tcu(
            P.detach().cpu().numpy(), k, mult, closed=self.closed, clamped=self.clamped
        )
        t = torch.tensor(t, device=device, dtype=P.dtype)
        self.kt = kt
        self.knots = t
        
        bezier_mat = torch.tensor(
            bspline.bspline_to_bezier_chain_mat(p, len(Q)),
            device=Q.device,
            dtype=Q.dtype,
        )
        Cp = bezier_mat @ Q
        
        if self.width_func is not None:
            Cp = torch.hstack([Cp, self.width_func(Cp).reshape(-1, 1)])
            has_width = True
        self.Cp = Cp

        if p > 3:
            b3mat = torch.tensor(
                bspline.bezier_chain_reduction_mat(p, 3, len(Cp)),
                device=Q.device,
                dtype=Q.dtype,
            )
            Cp3 = b3mat @ Cp
        else:
            Cp3 = Cp
        self.Cp3 = Cp3

        self.points = Cp3[:, :2].contiguous()
        if has_width:
            self.widths = Cp3[:, 2].contiguous()
        else:
            self.widths = w

    def domain(self):
        p = self.spline_degree
        k = p + 1
        return float(self.knots[p]), float(self.knots[-k])

    def samples(self, num_or_u, der=0, no_width=False, get_dt=False, numpy=False):
        """Sample the B-spline."""
        dim = 3 if self.has_varying_width() else 2
        if no_width:
            dim = 2
        p = self.spline_degree
        k = p + 1
        n = len(self.Q)
        Q = self.Q[:, :dim]
        basis_knots = np.linspace(0, k, k + 1)
        bsp = bspline.BSpline.basis_element(basis_knots).derivative(der)
        Bk = lambda u: bsp(np.clip(u, 0, k))
        t = self.knots.detach().cpu().numpy()

        if type(num_or_u) == int:
            u = np.linspace(*self.domain(), num_or_u)
        elif geom.is_number(num_or_u):
            u = np.ones(1) * num_or_u
        else:
            u = num_or_u
            
        Bu = np.zeros((n, len(u)))
        for i in range(n):
            Bu[i, :] = Bk(u - t[i])
        Bu = torch.tensor(
            np.kron(Bu.T, np.eye(dim)), device=self.Q.device, dtype=self.Q.dtype
        )
        Qhat = Q.reshape(-1, 1)
        res = (Bu @ Qhat).reshape(len(u), dim)
        
        if numpy:
            res = res.detach().cpu().numpy()
        if get_dt:
            return res, u[1] - u[0]
        return res

    def inner(self, der, normalize=False, normalize_size=None):
        """Compute inner product for smoothing."""
        dim = 3 if self.has_varying_width() else 2
        n = len(self.Q)
        k = self.spline_degree + 1
        
        if (n, k, der) in gramian_cache:
            G = gramian_cache[(n, k, der)]
        else:
            if self.pspline:
                D = np.diff(np.eye(n), der)
                Gd = D @ D.T
                if self.pspline_weight:
                    G = bspline.uniform_gramian(n, k=k, der=der)
                    w = np.max(np.diag(G)) / np.max(np.diag(Gd))
                    Gd = Gd * w
                G = Gd
            else:
                G = bspline.uniform_gramian(n, k=k, der=der)
            G = torch.tensor(
                np.kron(G, np.eye(dim)), device=self.Q.device, dtype=self.Q.dtype
            )
            gramian_cache[(n, k, der)] = G
            
        if normalize or normalize_size is not None:
            if normalize_size is None:
                P = self.param("points")
                D = torch.diff(P, axis=0)
                l = torch.sum(torch.sqrt(torch.sum(D ** 2, axis=1)))
            else:
                l = normalize_size
        else:
            l = 1.0
            
        Qhat = self.Q.reshape(-1, 1) / l
        res = Qhat.T @ G @ Qhat
        return res

    def get_points(self):
        return self.points * Scene.scale

    def get_stroke_width(self):
        return self.widths


def args_to_params(**kwargs):
    """Convert keyword arguments to parameter format."""
    params = {}
    for key, value in kwargs.items():
        if type(value) == tuple:
            params[key] = value
        else:
            params[key] = (value, False)
    return params


def to_tensor(v, dtype=torch.float32):
    """Convert value to tensor."""
    return torch.tensor(v, dtype=dtype).contiguous().to(device)


def convert_param(p, dtype=torch.float32, scale=None):
    """Convert a (list, has_grad) param to a tensor with requires_grad set."""
    try:
        if p[0] is None:
            return None
        var = to_tensor(p[0], dtype)
        if p[1]:
            var.requires_grad = True
    except TypeError:
        var = to_tensor(p, dtype)
    return var


def get_color_param(color, opacity=1.0):
    """Get RGBA color depending on cardinality of input."""
    if color is None:
        return None
    if len(color) == 1:
        if cfg.one_channel_is_alpha:
            return torch.concat([torch.zeros(3).to(device), color * opacity])
        else:
            return torch.concat([
                torch.ones(3).to(device) * color,
                torch.ones(1).to(device) * opacity
            ])
    elif len(color) == 2:
        return torch.concat([torch.ones(3) * color[0], color[1:] * opacity])
    elif len(color) == 3:
        return torch.concat([color, torch.tensor([1.0 * opacity]).to(device)])
    return color


def cubic_bspline(P, periodic=False):
    """Naive implementation of Bohm's algorithm for knot insertion."""
    def lerp(a, b, t):
        return a + t * (b - a)

    m = len(P)
    if periodic:
        P = torch.vstack([P[-1], P, P[0], P[1]])
    else:
        P = torch.vstack([P[0], P, P[-1]])

    n = P.shape[0]
    Cp = []
    for i in range(n - 3):
        p = P[i:i + 4]
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
            
    if periodic:
        Cp = Cp[:-1]
    return torch.vstack(Cp)

