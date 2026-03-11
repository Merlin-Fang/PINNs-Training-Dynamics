"""
Offline (non-training) precompute for Corrective-PINNs.

This script:
  1) loads two base checkpoints ("early" and "late")
  2) evaluates u_pred(t,x) on the reference grid
  3) forms signed delta: Δ = u_late - u_early
  4) builds:
       - teacher_map  : gauss(|Δ|)-gate + tanh compression (signed)
       - wconf_map    : low-pass Δ gate (in [0,1])
  5) saves a single .npz with {t_grid, x_grid, teacher_map, wconf_map}

Usage example:

  python -m src.precompute_corr_assets \
      --pde burgers \
      --ckptdir /path/to/ckpts/burgers_uniform_sampling \
      --early_step 101000 \
      --late_step 199000

Outputs by default to:
  pdes/<pde_name>/corr_assets/corr_assets_early<early>_late<late>.npz

python -m src.precompute_corr_assets --pde allen_cahn --ckptdir /scratch/merlinf/repos/PINNs-Training-Dynamics/ckpts/allen_cahn/allen_cahn_uniform_sampling --early_step 101000 --late_step 199000
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map

from pdes.burgers.model import Burgers
from pdes.allen_cahn.model import Allen_Cahn

from src.configs.burgers_unif import get_config as get_config_burgers
from src.configs.allen_cahn_unif import get_config as get_config_allen_cahn
from src.utils import load_dataset, load_checkpoint


# --------------------------------------------------------------------------------------
# Registries
# --------------------------------------------------------------------------------------

_PDE_MODELS = {
    "burgers": Burgers,
    "allen_cahn": Allen_Cahn,
}

_PDE_CONFIGS = {
    "burgers": get_config_burgers,
    "allen_cahn": get_config_allen_cahn,
}


# --------------------------------------------------------------------------------------
# JAX prediction loading
# --------------------------------------------------------------------------------------

def _get_u_pred(model, params, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate model solution on the full (t,x) grid.

    Returns:
      u_pred: (Nt, Nx) numpy float32
    """
    t_j = jnp.asarray(t)
    x_j = jnp.asarray(x)

    # vmap over x then over t, matching the orientation you used in plots.
    u_pred = vmap(
        vmap(model.get_solution, in_axes=(None, 0, None)),
        in_axes=(None, None, 0),
    )(params, t_j, x_j)
    u_pred = u_pred.T  # (Nt, Nx)
    return np.asarray(u_pred, dtype=np.float32)


def _load_pred_and_grid(
    pde_name: str,
    ckptdir: str,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load u_pred(step) along with u_ref and grids from the dataset."""
    if pde_name not in _PDE_MODELS:
        raise ValueError(f"Unknown pde '{pde_name}'. Known: {sorted(_PDE_MODELS)}")
    if pde_name not in _PDE_CONFIGS:
        raise ValueError(f"Missing config getter for pde '{pde_name}'.")

    config = _PDE_CONFIGS[pde_name]()
    data_dir = os.path.join(config.pde.name, "data", f"{config.pde.name}.mat")
    u_ref, t, x = load_dataset(data_dir)

    ModelClass = _PDE_MODELS[pde_name]
    model = ModelClass(config, IC=(u_ref[0, :], jnp.full_like(jnp.asarray(x), t[0]), jnp.asarray(x)))

    # model.state is replicated; get a host copy of device 0 state structure.
    state_host = jax.device_get(tree_map(lambda y: y[0], model.state))
    state = load_checkpoint(state_host, ckptdir, step=step)
    params = state.params

    u_pred = _get_u_pred(model, params, t, x)
    return (
        u_pred,
        np.asarray(u_ref, dtype=np.float32),
        np.asarray(t, dtype=np.float32),
        np.asarray(x, dtype=np.float32),
    )


# --------------------------------------------------------------------------------------
# NumPy filtering utilities (offline)
# --------------------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _quantile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x.reshape(-1), q))


def _winsorize_abs(x: np.ndarray, clip_q: float = 0.995, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    """Cap |x| at quantile(|x|, clip_q), keep sign."""
    A = np.abs(x)
    cap = max(_quantile(A, clip_q), eps)
    return np.clip(x, -cap, cap), cap


def _tanh_compress(delta: np.ndarray, scale_q: float = 0.95, beta: float = 2.0, eps: float = 1e-12) -> np.ndarray:
    """Dynamic range compression: s * tanh(beta * delta/s)."""
    s = max(_quantile(np.abs(delta), scale_q), eps)
    return s * np.tanh(beta * (delta / s))


def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """1D Gaussian kernel normalized; sigma in *grid index* units."""
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    if radius is None:
        radius = int(np.ceil(3.0 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k /= (k.sum() + 1e-12)
    return k


def _convolve1d_reflect(A: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Reflect-pad 1D convolution along axis for 2D arrays."""
    if k.size == 1:
        return A.astype(np.float64, copy=False)
    pad = k.size // 2
    if axis == 0:
        Ap = np.pad(A, ((pad, pad), (0, 0)), mode="reflect")
        out = np.empty_like(A, dtype=np.float64)
        for i in range(A.shape[0]):
            out[i, :] = (Ap[i : i + k.size, :] * k[:, None]).sum(axis=0)
        return out
    if axis == 1:
        Ap = np.pad(A, ((0, 0), (pad, pad)), mode="reflect")
        out = np.empty_like(A, dtype=np.float64)
        for j in range(A.shape[1]):
            out[:, j] = (Ap[:, j : j + k.size] * k[None, :]).sum(axis=1)
        return out
    raise ValueError("axis must be 0 or 1")


def gaussian_blur_2d(A: np.ndarray, sigma_t: float = 1.5, sigma_x: float = 1.0) -> np.ndarray:
    """Separable Gaussian blur on (Nt,Nx) arrays; sigmas in *grid index* units."""
    kt = _gaussian_kernel1d(sigma_t)
    kx = _gaussian_kernel1d(sigma_x)
    B = _convolve1d_reflect(A, kt, axis=0)
    C = _convolve1d_reflect(B, kx, axis=1)
    return C


# --------------------------------------------------------------------------------------
# Two recipes for ∆
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TeacherRecipeParams:
    # gauss|Δ| gate + tanh
    q: float = 0.92
    sigma_t: float = 1.5
    sigma_x: float = 1.0
    k: float = 20.0
    p: float = 1.0
    clip_q: float = 0.995
    tanh_beta: float = 2.0
    tanh_scale_q: float = 0.95


@dataclass(frozen=True)
class WConfRecipeParams:
    # lowpass Δ gate
    q: float = 0.92
    sigma_t: float = 2.5
    sigma_x: float = 1.8
    k: float = 20.0
    p: float = 1.0
    clip_q: float = 0.995


def make_teacher_map_from_delta(
    delta: np.ndarray,
    params: TeacherRecipeParams,
    eps: float = 1e-12,
) -> np.ndarray:
    """Teacher map: gauss(|Δ|) -> sigmoid gate -> signed delta * gate -> tanh compress."""
    d, _ = _winsorize_abs(delta, clip_q=params.clip_q, eps=eps)
    A = np.abs(d)
    A_s = gaussian_blur_2d(A, sigma_t=params.sigma_t, sigma_x=params.sigma_x)

    # normalize so q behaves similarly across PDEs
    s = max(_quantile(A_s, 0.95), eps)
    A_n = A_s / s

    tau = _quantile(A_n, params.q)
    gate = _sigmoid(params.k * (A_n - tau))
    if params.p != 1.0:
        gate = gate ** params.p

    Z = d * gate
    # Z = _tanh_compress(Z, scale_q=params.tanh_scale_q, beta=params.tanh_beta, eps=eps)
    return Z.astype(np.float32)


def make_wconf_map_from_delta(
    delta: np.ndarray,
    params: WConfRecipeParams,
    eps: float = 1e-12,
) -> np.ndarray:
    """wconf map: low-pass Δ, then sigmoid gate from |low-pass Δ|. Returns [0,1]."""
    d, _ = _winsorize_abs(delta, clip_q=params.clip_q, eps=eps)

    # d_lp = gaussian_blur_2d(d, sigma_t=params.sigma_t, sigma_x=params.sigma_x)
    # A = np.abs(d_lp)

    # s = max(_quantile(A, 0.95), eps)
    # A_n = A / s
    # tau = _quantile(A_n, params.q)

    # gate = _sigmoid(params.k * (A_n - tau))
    # if params.p != 1.0:
    #     gate = gate ** params.p

    # return np.clip(gate, 0.0, 1.0).astype(np.float32)

    d_lp = gaussian_blur_2d(d, sigma_t=params.sigma_t, sigma_x=params.sigma_x)

    A = np.abs(d_lp)

    # optional robust cap (recommended)
    cap = max(_quantile(A, params.clip_q), 1e-12)
    A = np.minimum(A, cap)

    # linear normalization
    wconf = A / (cap + 1e-12)

    return wconf.astype(np.float32)


# --------------------------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------------------------

def _default_out_path(pde_name: str, early_step: int, late_step: int) -> str:
    out_dir = os.path.join("pdes", pde_name, "corr_assets")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"corr_assets_early{early_step}_late{late_step}.npz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute teacher_map and wconf_map for Corrective PINNs.")
    p.add_argument("--pde", type=str, required=True, choices=sorted(_PDE_MODELS.keys()))
    p.add_argument("--ckptdir", type=str, required=True, help="Base checkpoint directory.")
    p.add_argument("--early_step", type=int, required=True)
    p.add_argument("--late_step", type=int, required=True)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz path. Default: pdes/<pde>/corr_assets/corr_assets_early<E>_late<L>.npz",
    )

    # Teacher params (gauss|Δ| gate + tanh)
    p.add_argument("--teacher_q", type=float, default=TeacherRecipeParams.q)
    p.add_argument("--teacher_sigma_t", type=float, default=TeacherRecipeParams.sigma_t)
    p.add_argument("--teacher_sigma_x", type=float, default=TeacherRecipeParams.sigma_x)
    p.add_argument("--teacher_k", type=float, default=TeacherRecipeParams.k)
    p.add_argument("--teacher_p", type=float, default=TeacherRecipeParams.p)
    p.add_argument("--teacher_clip_q", type=float, default=TeacherRecipeParams.clip_q)
    p.add_argument("--teacher_tanh_beta", type=float, default=TeacherRecipeParams.tanh_beta)
    p.add_argument("--teacher_tanh_scale_q", type=float, default=TeacherRecipeParams.tanh_scale_q)

    # wconf params (lowpass Δ gate)
    p.add_argument("--wconf_q", type=float, default=WConfRecipeParams.q)
    p.add_argument("--wconf_sigma_t", type=float, default=WConfRecipeParams.sigma_t)
    p.add_argument("--wconf_sigma_x", type=float, default=WConfRecipeParams.sigma_x)
    p.add_argument("--wconf_k", type=float, default=WConfRecipeParams.k)
    p.add_argument("--wconf_p", type=float, default=WConfRecipeParams.p)
    p.add_argument("--wconf_clip_q", type=float, default=WConfRecipeParams.clip_q)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    early_step = int(args.early_step)
    late_step = int(args.late_step)
    out_path = args.out or _default_out_path(args.pde, early_step, late_step)

    u_early, _u_ref, t_grid, x_grid = _load_pred_and_grid(args.pde, args.ckptdir, early_step)
    u_late, _, _, _ = _load_pred_and_grid(args.pde, args.ckptdir, late_step)

    # Signed delta as in your plotting code: Δ = u_early - u_late.
    delta = (u_late - u_early).astype(np.float32)

    teacher_params = TeacherRecipeParams(
        q=float(args.teacher_q),
        sigma_t=float(args.teacher_sigma_t),
        sigma_x=float(args.teacher_sigma_x),
        k=float(args.teacher_k),
        p=float(args.teacher_p),
        clip_q=float(args.teacher_clip_q),
        tanh_beta=float(args.teacher_tanh_beta),
        tanh_scale_q=float(args.teacher_tanh_scale_q),
    )
    wconf_params = WConfRecipeParams(
        q=float(args.wconf_q),
        sigma_t=float(args.wconf_sigma_t),
        sigma_x=float(args.wconf_sigma_x),
        k=float(args.wconf_k),
        p=float(args.wconf_p),
        clip_q=float(args.wconf_clip_q),
    )

    teacher_map = make_teacher_map_from_delta(delta, teacher_params)
    wconf_map = make_wconf_map_from_delta(delta, wconf_params)

    meta: Dict[str, object] = {
        "pde": args.pde,
        "ckptdir": args.ckptdir,
        "early_step": early_step,
        "late_step": late_step,
        "teacher_recipe": "gauss|Δ| gate + tanh",
        "teacher_params": asdict(teacher_params),
        "wconf_recipe": "lowpass Δ gate",
        "wconf_params": asdict(wconf_params),
        "delta_definition": "delta = u_late - u_early",
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        t_grid=t_grid.astype(np.float32),
        x_grid=x_grid.astype(np.float32),
        teacher_map=teacher_map.astype(np.float32),
        wconf_map=wconf_map.astype(np.float32),
        meta=json.dumps(meta),
    )

    print(f"[precompute_corr_assets] saved: {out_path}")
    print(f"  teacher_map shape: {teacher_map.shape}, dtype={teacher_map.dtype}")
    print(
        f"  wconf_map   shape: {wconf_map.shape}, dtype={wconf_map.dtype}, "
        f"range=({float(wconf_map.min()):.4f},{float(wconf_map.max()):.4f})"
    )


if __name__ == "__main__":
    main()