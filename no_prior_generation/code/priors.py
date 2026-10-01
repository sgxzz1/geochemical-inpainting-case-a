from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fractal_cascade import fractal_cascade_prior_stats
from utils import normalize_01


def _pad_to_even(array: np.ndarray) -> np.ndarray:
    rows, cols = array.shape
    pad_rows = rows % 2
    pad_cols = cols % 2
    if pad_rows == 0 and pad_cols == 0:
        return array
    padded = np.full((rows + pad_rows, cols + pad_cols), np.nan, dtype=np.float32)
    padded[:rows, :cols] = array
    return padded


def _aggregate_2x2(block: np.ndarray, rng: np.random.Generator, fallback: str) -> float:
    top_left = block[0, 0]
    if np.isfinite(top_left):
        return float(top_left)

    known = block[np.isfinite(block)]
    if known.size == 0:
        return float("nan")

    if fallback == "random_known":
        return float(known[int(rng.integers(0, known.size))])
    if fallback == "median":
        return float(np.median(known))
    if fallback == "mean":
        return float(np.mean(known))
    if fallback == "p75":
        return float(np.percentile(known, 75))
    if fallback == "max":
        return float(np.max(known))
    raise ValueError(f"Unsupported fallback: {fallback}")


def fractal_downsample_once(
    array: np.ndarray,
    rng: np.random.Generator,
    fallback: str = "random_known",
) -> np.ndarray:
    padded = _pad_to_even(array)
    rows, cols = padded.shape
    out = np.full((rows // 2, cols // 2), np.nan, dtype=np.float32)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = padded[2 * r:2 * r + 2, 2 * c:2 * c + 2]
            out[r, c] = _aggregate_2x2(block, rng=rng, fallback=fallback)
    return out


def nearest_upsample_to_shape(array: np.ndarray, target_shape: tuple[int, int], level: int) -> np.ndarray:
    factor = 2 ** level
    up = np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)
    rows, cols = target_shape
    if up.shape[0] < rows or up.shape[1] < cols:
        padded = np.full((max(rows, up.shape[0]), max(cols, up.shape[1])), np.nan, dtype=np.float32)
        padded[:up.shape[0], :up.shape[1]] = up
        up = padded
    return up[:rows, :cols].astype(np.float32)


def _neighbor_mean(array: np.ndarray, r: int, c: int, radius: int = 1) -> float:
    rows, cols = array.shape
    r0, r1 = max(0, r - radius), min(rows, r + radius + 1)
    c0, c1 = max(0, c - radius), min(cols, c + radius + 1)
    patch = array[r0:r1, c0:c1]
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _direct_sampling_realization(
    known_values: np.ndarray,
    known_mask: np.ndarray,
    target_mask: np.ndarray,
    coarse_prior: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    rng: np.random.Generator,
    candidate_count: int = 64,
    coord_weight: float = 0.20,
    neighborhood_weight: float = 0.35,
) -> np.ndarray:
    rows, cols = known_values.shape
    sim = coarse_prior.astype(np.float32).copy()
    sim[known_mask > 0] = known_values[known_mask > 0]

    donor_rc = np.argwhere(known_mask > 0)
    if donor_rc.size == 0:
        fill = float(np.nanmean(coarse_prior)) if np.any(np.isfinite(coarse_prior)) else 0.0
        out = np.full_like(known_values, fill, dtype=np.float32)
        return out

    target_rc = np.argwhere(target_mask > 0)
    # Fill from boundary inward: pixels with more known neighbors are simulated first.
    scores = []
    for r, c in target_rc:
        r0, r1 = max(0, r - 1), min(rows, r + 2)
        c0, c1 = max(0, c - 1), min(cols, c + 2)
        scores.append(int(np.sum(known_mask[r0:r1, c0:c1])))
    order = np.argsort(-np.asarray(scores))
    target_rc = target_rc[order]

    for r, c in target_rc:
        n = min(candidate_count, len(donor_rc))
        candidate_ids = rng.choice(len(donor_rc), size=n, replace=False)
        best_score = float("inf")
        best_value = sim[r, c]
        dst_center = coarse_prior[r, c]
        dst_neigh = _neighbor_mean(sim, int(r), int(c), radius=1)

        for idx in candidate_ids:
            dr, dc = donor_rc[idx]
            src_center = coarse_prior[dr, dc]
            src_neigh = _neighbor_mean(sim, int(dr), int(dc), radius=1)
            value_score = abs(float(src_center) - float(dst_center)) if np.isfinite(src_center) and np.isfinite(dst_center) else 0.0
            neigh_score = abs(float(src_neigh) - float(dst_neigh)) if np.isfinite(src_neigh) and np.isfinite(dst_neigh) else 0.0
            xy_score = abs(float(x_coord[dr, dc] - x_coord[r, c])) + abs(float(y_coord[dr, dc] - y_coord[r, c]))
            score = value_score + neighborhood_weight * neigh_score + coord_weight * xy_score
            if score < best_score:
                best_score = score
                best_value = known_values[dr, dc]

        if not np.isfinite(best_value):
            best_value = dst_center if np.isfinite(dst_center) else float(np.nanmean(known_values[known_mask > 0]))
        sim[r, c] = float(best_value)

    return sim.astype(np.float32)


def ds_fractal_prior_stats(
    condition_values: np.ndarray,
    known_mask: np.ndarray,
    target_mask: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    level: int = 3,
    n_realizations: int = 16,
    seed: int = 0,
    fallback: str = "random_known",
) -> dict[str, np.ndarray]:
    """Build a lightweight DS/MPS-style prior in normalized log space."""
    rng = np.random.default_rng(seed)
    source = condition_values.astype(np.float32).copy()
    source[known_mask <= 0] = np.nan
    fill_value = float(np.nanmean(source)) if np.any(np.isfinite(source)) else 0.0

    sims = []
    for i in range(n_realizations):
        local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + i)
        coarse = source.copy()
        for _ in range(level):
            coarse = fractal_downsample_once(coarse, rng=local_rng, fallback=fallback)
        coarse_prior = nearest_upsample_to_shape(coarse, source.shape, level=level)
        coarse_prior[~np.isfinite(coarse_prior)] = fill_value
        sim = _direct_sampling_realization(
            known_values=condition_values,
            known_mask=known_mask,
            target_mask=target_mask,
            coarse_prior=coarse_prior,
            x_coord=x_coord,
            y_coord=y_coord,
            rng=local_rng,
        )
        sims.append(sim)

    stack = np.stack(sims, axis=0).astype(np.float32)
    return {
        "ds_mean": np.mean(stack, axis=0).astype(np.float32),
        "ds_p75": np.percentile(stack, 75, axis=0).astype(np.float32),
        "ds_std": np.std(stack, axis=0).astype(np.float32),
    }


@dataclass
class KrigingResult:
    prior: np.ndarray
    variance: np.ndarray
    backend: str


def _idw_prior(
    values: np.ndarray,
    known_mask: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    power: float = 2.0,
    max_points: int = 256,
) -> KrigingResult:
    known_rc = np.argwhere(known_mask > 0)
    known_vals = values[known_mask > 0].astype(np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    var = np.zeros_like(values, dtype=np.float32)
    if known_vals.size == 0:
        return KrigingResult(out, var, backend="idw_empty")

    coords = np.column_stack([xx[known_mask > 0], yy[known_mask > 0]]).astype(np.float32)
    all_rc = np.argwhere(np.ones_like(values, dtype=bool))
    for r, c in all_rc:
        dx = coords[:, 0] - float(xx[r, c])
        dy = coords[:, 1] - float(yy[r, c])
        dist = np.sqrt(dx * dx + dy * dy)
        if max_points and len(dist) > max_points:
            ids = np.argpartition(dist, max_points)[:max_points]
            dist_use = dist[ids]
            vals_use = known_vals[ids]
        else:
            dist_use = dist
            vals_use = known_vals
        if np.any(dist_use < 1e-8):
            out[r, c] = vals_use[np.argmin(dist_use)]
            var[r, c] = 0.0
        else:
            weights = 1.0 / np.maximum(dist_use, 1e-8) ** power
            weights = weights / np.sum(weights)
            pred = float(np.sum(weights * vals_use))
            out[r, c] = pred
            var[r, c] = float(np.sum(weights * (vals_use - pred) ** 2))
    return KrigingResult(out, normalize_01(var), backend="idw")


def ordinary_kriging_prior(
    values: np.ndarray,
    known_mask: np.ndarray,
    x_coords_1d: np.ndarray,
    y_coords_1d: np.ndarray,
    backend: str = "auto",
    variogram_model: str = "spherical",
    nlags: int = 6,
) -> KrigingResult:
    """Ordinary Kriging prior in normalized log space.

    backend="auto" uses PyKrige when available and IDW otherwise. Use
    backend="pykrige" if you want the script to fail when PyKrige is missing.
    """
    xx, yy = np.meshgrid(x_coords_1d.astype(np.float64), y_coords_1d.astype(np.float64))
    if backend not in {"auto", "pykrige", "idw"}:
        raise ValueError("--kriging-backend must be auto, pykrige, or idw")
    if backend == "idw":
        return _idw_prior(values, known_mask, xx, yy)

    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError:
        if backend == "pykrige":
            raise
        return _idw_prior(values, known_mask, xx, yy)

    xs = xx[known_mask > 0]
    ys = yy[known_mask > 0]
    zs = values[known_mask > 0].astype(np.float64)
    if zs.size < 4:
        return _idw_prior(values, known_mask, xx, yy)

    try:
        ok = OrdinaryKriging(
            xs,
            ys,
            zs,
            variogram_model=variogram_model,
            nlags=nlags,
            verbose=False,
            enable_plotting=False,
        )
        z_grid, ss_grid = ok.execute("grid", x_coords_1d.astype(np.float64), y_coords_1d.astype(np.float64))
        prior = np.asarray(z_grid, dtype=np.float32)
        variance = normalize_01(np.asarray(ss_grid, dtype=np.float32))
        prior[known_mask > 0] = values[known_mask > 0]
        variance[known_mask > 0] = 0.0
        return KrigingResult(prior=prior.astype(np.float32), variance=variance.astype(np.float32), backend="pykrige")
    except Exception:
        if backend == "pykrige":
            raise
        return _idw_prior(values, known_mask, xx, yy)


def build_prior_channels(
    condition_values: np.ndarray,
    known_mask: np.ndarray,
    target_mask: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    x_coords_1d: np.ndarray,
    y_coords_1d: np.ndarray,
    ds_level: int = 3,
    ds_ensemble: int = 16,
    ds_method: str = "cascade",
    seed: int = 0,
    kriging_backend: str = "auto",
    variogram_model: str = "spherical",
) -> dict[str, np.ndarray | str]:
    if ds_method == "legacy":
        ds = ds_fractal_prior_stats(
            condition_values=condition_values,
            known_mask=known_mask,
            target_mask=target_mask,
            x_coord=x_coord,
            y_coord=y_coord,
            level=ds_level,
            n_realizations=ds_ensemble,
            seed=seed,
        )
        ds_meta = {"ds_method": "legacy"}
    elif ds_method == "cascade":
        cascade_condition = condition_values.astype(np.float32).copy()
        cascade_condition[known_mask <= 0] = np.nan
        cascade = fractal_cascade_prior_stats(
            condition=cascade_condition,
            original_known_values=condition_values.astype(np.float32),
            original_known_mask=known_mask.astype(np.uint8),
            n_realizations=ds_ensemble,
            seed=seed,
            max_downsample_levels=max(ds_level, 1),
            downsample_fallback="random_known",
            distance_threshold=0.25,
            max_scan=256,
            max_radius=2,
            max_neighbors=12,
        )
        ds = {
            "ds_mean": cascade.mean,
            "ds_p75": cascade.p75,
            "ds_std": cascade.std,
        }
        ds_meta = {
            "ds_method": "cascade",
            "ds_closed_level": str(cascade.closed_level),
            "ds_level_shapes": str(cascade.level_shapes),
        }
    else:
        raise ValueError("--ds-method must be cascade or legacy")

    kriging = ordinary_kriging_prior(
        values=condition_values,
        known_mask=known_mask,
        x_coords_1d=x_coords_1d,
        y_coords_1d=y_coords_1d,
        backend=kriging_backend,
        variogram_model=variogram_model,
    )
    out: dict[str, np.ndarray | str] = {
        **ds,
        "kriging_prior": kriging.prior,
        "kriging_variance": kriging.variance,
        "kriging_backend": kriging.backend,
        **ds_meta,
    }
    return out
