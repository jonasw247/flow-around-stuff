import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from scipy.spatial import cKDTree
from torch import nn

POS_CENTRE, POS_HALF = np.array([4.0, 0.0, 0.0]), 8.5
BOX_MIN, BOX_MAX = np.array([-4.5, -4.5, -4.5]), np.array([12.5, 4.5, 4.5])
EDGES = np.linspace(-6.0, 1.5, 31)
CHUNK = 65536
RE_MIN, RE_MAX = 0.1, 2e6
HF_REPO, WEIGHTS = "jonasw247/flow-around-stuff", "fas-large.safetensors"


class Fourier(nn.Module):
    def __init__(self, din, dout, n_freq, min_scale=1.0):
        super().__init__()
        self.register_buffer("freq", min_scale * math.pi * 2.0 ** torch.arange(n_freq).float(), persistent=False)
        self.proj = nn.Linear(din * n_freq * 2 + din, dout)

    def forward(self, x):
        a = x.float().unsqueeze(-1) * self.freq
        return self.proj(torch.cat([torch.cat([a.sin(), a.cos()], -1).flatten(-2), x.float()], -1))


def rope(x, cos, sin):
    h = cos.size(-1)
    a, b = x[..., :h].float(), x[..., h:2 * h].float()
    return torch.cat([(a * cos - b * sin).to(x.dtype), (a * sin + b * cos).to(x.dtype), x[..., 2 * h:]], -1)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x, cos, sin):
        B, N, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.norm2(x))


class PointNet(nn.Module):
    def __init__(self, din, hidden, dout):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(din, h), nn.ReLU()]
            din = h
        self.point_mlp = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.Linear(din, dout), nn.LayerNorm(dout))

    def forward(self, x):
        return self.out(self.point_mlp(x).amax(-2))


class Decoder(nn.Module):
    def __init__(self, dim, hid, depth, n_freq):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq, dtype=torch.float32), persistent=False)
        self.z_proj = nn.Linear(dim, hid)
        self.p_proj = nn.Linear(3 + 6 * n_freq + 12, hid)
        layers = []
        for _ in range(depth - 1):
            layers += [nn.GELU(), nn.Linear(hid, hid)]
        self.mlp = nn.Sequential(*layers, nn.GELU(), nn.Linear(hid, 9))

    def forward(self, hz, raw):
        raw = raw.float()
        local, w, r = raw[..., :3], raw[..., 3:4], raw[..., 4:5]
        ang = (local[..., None] * self.freqs).flatten(-2)
        feat = torch.cat([local, ang.sin(), ang.cos(), torch.log1p(100.0 * w), (w / r).clamp(0.0, 4.0), torch.log10(r), raw[..., 5:]], -1)
        return self.mlp(hz + self.p_proj(feat))


class FlowAroundStuff(nn.Module):
    def __init__(self, dim, num_heads, depth, mlp_ratio, geo_hidden, geo_dim, decoder_dim, decoder_depth, fourier, n_freq):
        super().__init__()
        self.geo = PointNet(15, geo_hidden, geo_dim)
        self.tok = nn.Sequential(nn.Linear(geo_dim + 4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = Fourier(3, dim, n_freq)
        self.cond = nn.Sequential(Fourier(2, 256, 6, min_scale=0.25), nn.GELU(), nn.Linear(256, dim))
        f = dim // num_heads // 6
        self.register_buffer("omega", 10000.0 ** (-torch.arange(f, dtype=torch.float32) / f), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.decoder = Decoder(dim, decoder_dim, decoder_depth, fourier)

    def forward(self, raw, tok, centre, cond):
        raw = raw.float()
        w, r = raw[..., 3:4], raw[..., 4:5]
        feat = torch.cat([raw[..., :3], torch.log1p(100.0 * w), raw[..., 5:13], (w / r).clamp(0.0, 4.0), torch.log10(r), raw[..., 13:]], -1)
        g = self.geo(feat)
        x = self.tok(torch.cat([g, tok.to(g.dtype)], -1)) + self.pos(centre) + self.cond(cond).unsqueeze(1)
        ang = (((centre.float() + 1.0) * 500.0)[..., None] * self.omega).flatten(-2)
        cos, sin = ang.cos()[:, None], ang.sin()[:, None]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.decoder.z_proj(self.norm(x))


def find_weights():
    for f in (Path(WEIGHTS), Path(__file__).with_name(WEIGHTS)):
        if f.is_file():
            return f
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(HF_REPO, WEIGHTS)
    except Exception as e:
        raise FileNotFoundError(f"no {WEIGHTS} in the working directory and the download from huggingface.co/{HF_REPO} failed "
                                f"({type(e).__name__}): request access on that page and run `hf auth login`, or pass the path of the weights") from e


def load(path=None, device=None):
    path = find_weights() if path is None else path
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata()["fas"])
    model = FlowAroundStuff(**meta["model"])
    model.load_state_dict(load_file(str(path)))
    model.meta = meta
    return model.to(device).eval()


def unit(n):
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def wall_distance(x, sp, sn, tree, h):
    d, near = tree.query(x, workers=-1)
    signed = ((x - sp[near]) * sn[near]).sum(1)
    return np.where(d < 3.0 * h, np.abs(signed), d), near, signed


def spacing(sp, tree):
    return float(np.median(tree.query(sp[:: max(1, len(sp) // 4096)], k=2)[0][:, 1]))


def curvature(sp, sn, tree, k=12):
    nb = tree.query(sp, k=min(k + 1, len(sp)))[1][:, 1:]
    dp, dn = sp[nb] - sp[:, None], sn[nb] - sn[:, None]
    same = ((sn[nb] * sn[:, None]).sum(-1) > 0).astype(np.float64)
    kappa = ((dn * dp).sum(-1) * same).sum(1) / np.maximum(((dp ** 2).sum(-1) * same).sum(1), 1e-30)
    return np.sign(kappa) * np.log1p(np.abs(kappa))


def floor_height(z, gz):
    if gz is None:
        return np.ones(len(z))
    return np.clip(np.log1p(100.0 * np.maximum(z - gz, 0.0)) / math.log1p(400.0), 0.0, 1.0)


def resample(wall, count, hist, rng):
    b = np.clip(np.digitize(np.log10(np.maximum(wall, 1e-6)), EDGES) - 1, 0, len(hist) - 1)
    have = np.bincount(b, minlength=len(hist))
    q = np.where(have > 0, np.asarray(hist, np.float64), 0.0)
    want = np.floor(count * q / q.sum()).astype(np.int64)
    want[np.argmax(q)] += count - want.sum()
    order, start = np.argsort(b, kind="stable"), np.r_[0, np.cumsum(have)]
    idx = []
    for j in np.flatnonzero(want):
        mem = order[start[j]:start[j + 1]]
        idx.append(rng.choice(mem, want[j], replace=False) if len(mem) >= want[j] else mem)
    return np.concatenate(idx)


def volume_points(p, n, w, area, gz, rng, tree, h, count, hist):
    lo, hi = p.min(0), p.max(0)
    mid = 0.5 * (lo + hi)
    dlo, dhi = np.maximum(lo - [4.0, 6.0, 4.0], BOX_MIN), np.minimum(hi + [12.0, 6.0, 4.0], BOX_MAX)
    rb = 0.5 * float(np.linalg.norm(hi - lo))
    boxes = [(dlo, dhi, 110.0, 0.0),
             (np.array([mid[0] - rb - 0.25, mid[1] - 2, mid[2] - 2]), np.array([mid[0] + rb + 8, mid[1] + 2, mid[2] + 2]), 1600.0, 110.0),
             (lo - 0.8, hi + 0.8, 1600.0, 110.0), (lo - 0.5, hi + 0.5, 8000.0, 1600.0)]
    parts = []
    for a, b, r, q in boxes:
        a, b = np.maximum(a, dlo), np.minimum(b, dhi)
        parts.append((float(np.prod(np.clip(b - a, 0, None))) * (r - q), (a, b)))
    parts += [(area * 0.15 * 92000.0, 0.15), (area * 0.045 * 700000.0, 0.045)]
    if gz is not None:
        gap = max(float(lo[2] - gz), 1e-4)
        cell = 0.2 / 2 ** int(np.clip(math.ceil(math.log2(0.9 / gap)), 0, 5))
        a, b = np.array([lo[0] - 0.05, lo[1] - 0.05, gz]), np.array([hi[0] + 0.05, hi[1] + 0.05, gz + gap + 0.02])
        parts.append((float(np.prod(b - a)) * (cell ** -3 - 8000.0), (a, b)))
    tot = sum(c for c, _ in parts)
    x = []
    for c, spec in parts:
        m = rng.poisson(6 * count * max(c, 0.0) / tot)
        if isinstance(spec, tuple):
            x.append(rng.uniform(spec[0], spec[1], (m, 3)))
        else:
            i = rng.choice(len(p), m, p=w)
            d = rng.uniform(0.0, spec, m)
            d[:int(0.08 * m)] = 0.004 * rng.uniform(0, 1, int(0.08 * m)) ** 1.5
            x.append(p[i] + n[i] * d[:, None])
    x = np.concatenate(x)
    x = x[np.all((x > BOX_MIN) & (x < BOX_MAX), 1) & (x[:, 2] > (-np.inf if gz is None else gz))]
    wall, _, signed = wall_distance(x, p, n, tree, h)
    x, wall = x[signed > 0], wall[signed > 0]
    return x[resample(wall, count, hist, rng)]


def cover(pts, k, rng, chunk=4096):
    tree = cKDTree(pts)
    covered = np.zeros(len(pts), bool)
    order = rng.permutation(len(pts))
    cen, nbr, rad = [], [], []
    for s in range(0, len(pts), chunk):
        cand = order[s:s + chunk]
        cand = cand[~covered[cand]]
        if not len(cand):
            continue
        dist, nb = tree.query(pts[cand], k=k)
        for i, c in enumerate(cand):
            if not covered[c]:
                covered[nb[i]] = True
                cen.append(c)
                nbr.append(nb[i])
                rad.append(dist[i, -1])
    return np.asarray(cen), np.stack(nbr), np.maximum(np.asarray(rad), 1e-6)


def assign_balls(points, centres, radius, m=4):
    n = len(points)
    hits = cKDTree(points).query_ball_point(centres, radius)
    ball = np.repeat(np.arange(len(centres)), [len(h) for h in hits])
    pt = np.concatenate([np.asarray(h, np.int64) for h in hits])
    rel = np.linalg.norm(points[pt] - centres[ball], axis=1) / radius[ball]
    order = np.lexsort((rel, pt))
    pt, ball, rel = pt[order], ball[order], rel[order]
    first = np.r_[True, pt[1:] != pt[:-1]]
    rank = np.arange(len(pt)) - np.maximum.accumulate(np.where(first, np.arange(len(pt)), 0))
    keep = rank < m
    nb, out = np.zeros((n, m), np.int64), np.full((n, m), np.inf)
    nb[pt[keep], rank[keep]], out[pt[keep], rank[keep]] = ball[keep], rel[keep]
    un = np.flatnonzero(~np.isfinite(out[:, 0]))
    if len(un):
        d, j = cKDTree(centres).query(points[un], k=min(32, len(centres)))
        d, j = d.reshape(len(un), -1), j.reshape(len(un), -1)
        b, r = (d / radius[j]).argmin(1), np.arange(len(un))
        nb[un, 0], out[un, 0] = j[r, b], (d / radius[j])[r, b]
    return nb, out


@torch.no_grad()
def draw(model, p, n, w, area, re, gz, vq, seed, pool=None):
    cfg, st, dev = model.meta["pool"], model.meta["stats"], next(model.parameters()).device
    rng = np.random.default_rng(seed)
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
    tree_all = cKDTree(p)
    h_all = spacing(p, tree_all)
    if pool is None:
        vp = volume_points(p, n, w, area, gz, rng, tree_all, h_all, cfg["pool_volume"], cfg["hist"])
    else:
        vp = pool[np.all((pool > BOX_MIN) & (pool < BOX_MAX), 1) & (pool[:, 2] > (-np.inf if gz is None else gz))]
        wall = wall_distance(vp, p, n, tree_all, h_all)[0]
        vp = vp[resample(wall, cfg["pool_volume"], cfg["hist"], rng)]
    si = np.sort(rng.choice(len(p), cfg["pool_surface"], replace=False, p=w)) if len(p) > cfg["pool_surface"] else np.arange(len(p))
    sp, sn = p[si], n[si]
    tree = cKDTree(sp)
    h, curv = spacing(sp, tree), curvature(sp, sn, tree)

    def records(x, surface, normals=None):
        if surface:
            near = tree.query(x)[1]
            wall, direction = np.zeros(len(x)), normals
        else:
            wall, near, _ = wall_distance(x, sp, sn, tree, h)
            direction = unit(x - sp[near])
        nrm = normals if surface else sn[near]
        return np.concatenate([wall[:, None], np.zeros((len(x), 1)), np.full((len(x), 1), float(surface)), direction, nrm,
                               curv[near, None], floor_height(x[:, 2], gz)[:, None]], 1)

    pts = np.concatenate([vp, sp])
    rec = np.concatenate([records(vp, False), np.concatenate([np.zeros((len(sp), 2)), np.ones((len(sp), 1)), sn, sn, curv[:, None],
                                                              floor_height(sp[:, 2], gz)[:, None]], 1)])
    cen, nbr, rad = cover(pts, cfg["k"], rng)
    cpos, enc = pts[cen], nbr[:, ::2]
    raw = np.concatenate([(pts[enc] - cpos[:, None]) / rad[:, None, None], rec[enc]], -1)
    raw[..., 4] = rad[:, None]
    tok = np.stack([np.log10(rad), np.log1p(100.0 * rec[cen, 0]), rec[cen, 2], rec[cen, 10]], -1)
    cond = [(math.log10(re) - 3.0) / 2.5, -1.0 if gz is None else 1.0]
    half = dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=half):
        hz = model(T(raw)[None], T(tok)[None], T((cpos - POS_CENTRE) / POS_HALF)[None], T(cond)[None])[0]
    qp = np.concatenate([p, vq])
    qrec = np.concatenate([records(p, True, n), records(vq, False)]) if len(vq) else records(p, True, n)
    nb, rel = assign_balls(qp, cpos, rad)
    cpos_t, rad_t = T(cpos), T(rad)
    out = []
    for s in range(0, len(qp), CHUNK):
        b = torch.as_tensor(nb[s:s + CHUNK], device=dev)
        rl, q, r = T(rel[s:s + CHUNK]), T(qrec[s:s + CHUNK]), rad_t[b]
        local = (T(qp[s:s + CHUNK])[:, None] - cpos_t[b]) / r[..., None]
        x = torch.cat([local, q[:, None].expand(-1, b.shape[1], -1)], -1)
        x[..., 4] = r
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=half):
            f = model.decoder(hz[b], x).float()
        wt = torch.isfinite(rl).float() / (rl.clamp(max=1e6) + 1e-5)
        out.append(((f * wt[..., None]).sum(1) / wt.sum(1, keepdim=True)).cpu().numpy().astype(np.float64))
    f = np.concatenate(out)
    s, v = f[:len(p), :4] * st["s_std"] + st["s_mean"], f[len(p):, 4:] * st["v_std"] + st["v_mean"]
    p_scale, f_scale = 1.0 + 14.0 / re, 2.9 / re + re ** -0.48
    s[:, 0] *= p_scale
    s[:, 1:] *= f_scale
    v[:, 0] *= p_scale
    v[:, 4] = np.sign(v[:, 4]) * np.expm1(np.minimum(np.abs(v[:, 4]), 20.0))
    return s, v


def outward(c, n, a):
    c0 = (c * a[:, None]).sum(0) / a.sum()
    return n if (a * (((c - c0) * n).sum(1) > 0)).sum() >= 0.5 * a.sum() else -n


def predict(model, centres, normals, areas, re, floor=None, points=None, draws=1):
    c, a = np.asarray(centres, np.float64), np.asarray(areas, np.float64)
    if not re > 0:
        raise ValueError(f"the Reynolds number must be positive, got {re}")
    if not RE_MIN <= re <= RE_MAX:
        warnings.warn(f"Re {re:g} is outside the trained range {RE_MIN:g} to {RE_MAX:g}")
    n = outward(c, unit(np.asarray(normals, np.float64)), a)
    c0 = (c * a[:, None]).sum(0) / a.sum()
    L = float(np.ptp((c - c0) @ np.linalg.eigh(np.cov((c - c0).T, aweights=a))[1], 0).max())
    p, w = (c - c0) / L, a / a.sum()
    gz = None if floor is None else (floor - c0[2]) / L
    if gz is not None and gz >= p[:, 2].max():
        raise ValueError(f"the floor (z = {floor:g}) is above the body (z = {c[:, 2].min():g} to {c[:, 2].max():g})")
    if gz is not None and gz > p[:, 2].min() + 0.05:
        warnings.warn(f"the floor (z = {floor:g}) cuts the body well above its lowest point (z = {c[:, 2].min():g})")
    vq = np.zeros((0, 3)) if points is None else (np.asarray(points, np.float64).reshape(-1, 3) - c0) / L
    ok = np.all((vq > BOX_MIN) & (vq < BOX_MAX), 1) & (vq[:, 2] > (-np.inf if gz is None else gz))
    if ok.any():
        ok[ok] = wall_distance(vq[ok], p, n, cKDTree(p), 0.0)[2] > 0
    if not ok.all():
        warnings.warn(f"{int((~ok).sum())} of {len(ok)} points are inside the body, below the floor or outside the model's box "
                      f"(x {BOX_MIN[0]:g} to {BOX_MAX[0]:g}, y and z within {BOX_MAX[1]:g} body lengths of the body's centre): NaN")
    vq = vq[ok]
    s, v = 0.0, 0.0
    for seed in range(draws):
        ds, dv = draw(model, p, n, w, a.sum() / L ** 2, re, gz, vq, seed)
        s, v = s + ds / draws, v + dv / draws
    f = -(s[:, :1] * n + s[:, 1:]) * a[:, None]
    out = {"cp": s[:, 0], "cf": s[:, 1:], "normals": n, "force_area": f.sum(0)}
    if points is not None:
        full = np.full((len(ok), 5), np.nan)
        full[ok] = v
        out.update(point_cp=full[:, 0], point_u=full[:, 1:4], point_vorticity=full[:, 4])
    return out


def mesh_faces(path):
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    mesh.fix_normals()
    if not mesh.is_watertight:
        warnings.warn("the mesh is not watertight: the model expects a closed surface with consistent outward normals")
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, 0.01 * np.ptp(mesh.vertices, 0).max())
    mesh = trimesh.Trimesh(v, f, process=False)
    keep = mesh.area_faces > 0
    return mesh.triangles_center[keep], mesh.face_normals[keep], mesh.area_faces[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--aref", type=float, default=None)
    ap.add_argument("--points", default=None)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default="result.npz")
    args = ap.parse_args()
    for f in (args.mesh, args.weights, args.points):
        if f is not None and not Path(f).is_file():
            ap.error(f"no such file: {f}")
    if not args.re > 0:
        ap.error("--re must be positive")
    c, n, a = mesh_faces(args.mesh)
    model = load(args.weights)
    pts = None if args.points is None else np.load(args.points)
    r = predict(model, c, n, a, args.re, args.floor, pts, args.draws)
    np.savez(args.out, centres=c, areas=a, **{k: x for k, x in r.items() if k != "force_area"})
    fx, fy, fz = r["force_area"]
    res = {"drag_area": fx, "side_area": fy, "lift_area": fz}
    if args.aref:
        res.update(Cd=fx / args.aref, Cs=fy / args.aref, Cl=fz / args.aref)
    print(json.dumps({k: round(float(x), 6) for k, x in res.items()}))


if __name__ == "__main__":
    main()
