from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CascadeResult:
    realizations: np.ndarray
    mean: np.ndarray
    p75: np.ndarray
    std: np.ndarray
    level_shapes: list[tuple[int, int]]
    closed_level: int
    filled_nan_count: int


def _pad_to_even(array: np.ndarray) -> np.ndarray:
    rows, cols = array.shape
    pad_rows = rows % 2
    pad_cols = cols % 2
    if pad_rows == 0 and pad_cols == 0:
        return array
    out = np.full((rows + pad_rows, cols + pad_cols), np.nan, dtype=np.float32)
    out[:rows, :cols] = array
    return out


def _pick_2x2_value(block: np.ndarray, rng: np.random.Generator, fallback: str) -> float:
    top_left = block[0, 0]
    if np.isfinite(top_left):
        return float(top_left)
    known = block[np.isfinite(block)]
    if known.size == 0:
        return float("nan")
    if fallback == "random_known":
        return float(known[int(rng.integers(0, known.size))])
    if fallback == "mean":
        return float(np.mean(known))
    if fallback == "median":
        return float(np.median(known))
    if fallback == "p75":
        return float(np.percentile(known, 75))
    if fallback == "max":
        return float(np.max(known))
    raise ValueError(f"Unsupported fallback: {fallback}")


def downsample_2x2_left_top(
    array: np.ndarray,
    rng: np.random.Generator,
    fallback: str = "random_known",
) -> np.ndarray:
    padded = _pad_to_even(array.astype(np.float32))
    rows, cols = padded.shape
    out = np.full((rows // 2, cols // 2), np.nan, dtype=np.float32)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = padded[2 * r:2 * r + 2, 2 * c:2 * c + 2]
            out[r, c] = _pick_2x2_value(block, rng=rng, fallback=fallback)
    return out


def make_closed_coarse_pyramid(
    condition: np.ndarray,
    max_levels: int,
    rng: np.random.Generator,
    fallback: str = "random_known",
) -> tuple[list[np.ndarray], int]:
    """Reverse step: coarsen an incomplete image until it is fully informed."""
    levels = [condition.astype(np.float32).copy()]
    current = levels[0]
    for level in range(1, max_levels + 1):
        if np.all(np.isfinite(current)):
            break
        current = downsample_2x2_left_top(current, rng=rng, fallback=fallback)
        levels.append(current)
        if np.all(np.isfinite(current)):
            break
    closed_level = len(levels) - 1
    return levels, closed_level


def _data_event(
    grid: np.ndarray,
    r: int,
    c: int,
    max_radius: int,
    max_neighbors: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    rows, cols = grid.shape
    offsets: list[tuple[int, int]] = []
    values: list[float] = []
    candidates: list[tuple[int, int, float, int]] = []
    for dr in range(-max_radius, max_radius + 1):
        for dc in range(-max_radius, max_radius + 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            value = grid[rr, cc]
            if not np.isfinite(value):
                continue
            dist2 = dr * dr + dc * dc
            candidates.append((dr, dc, float(value), dist2))
    candidates.sort(key=lambda item: item[3])
    for dr, dc, value, _ in candidates[:max_neighbors]:
        offsets.append((dr, dc))
        values.append(value)
    return offsets, np.asarray(values, dtype=np.float32)


def _event_distance(
    ti: np.ndarray,
    y_r: int,
    y_c: int,
    offsets: list[tuple[int, int]],
    values: np.ndarray,
) -> float | None:
    if len(offsets) == 0:
        return 0.0
    rows, cols = ti.shape
    diffs = []
    for idx, (dr, dc) in enumerate(offsets):
        rr, cc = y_r + dr, y_c + dc
        if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
            return None
        ti_value = ti[rr, cc]
        if not np.isfinite(ti_value):
            return None
        diffs.append(abs(float(values[idx]) - float(ti_value)))
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def _random_ti_value(ti: np.ndarray, rng: np.random.Generator) -> float:
    finite = ti[np.isfinite(ti)]
    if finite.size == 0:
        return 0.0
    return float(finite[int(rng.integers(0, finite.size))])


def direct_sampling_fill(
    sim_grid: np.ndarray,
    training_image: np.ndarray,
    rng: np.random.Generator,
    distance_threshold: float = 0.25,
    max_scan: int = 256,
    max_radius: int = 2,
    max_neighbors: int = 12,
) -> np.ndarray:
    """Fill NaNs in sim_grid using a Direct Sampling style data-event match."""
    sim = sim_grid.astype(np.float32).copy()
    ti = training_image.astype(np.float32)
    missing = np.argwhere(~np.isfinite(sim))
    if missing.size == 0:
        return sim

    path = rng.permutation(len(missing))
    ti_rows, ti_cols = ti.shape
    for idx in path:
        r, c = missing[idx]
        offsets, values = _data_event(sim, int(r), int(c), max_radius=max_radius, max_neighbors=max_neighbors)

        if len(offsets) == 0:
            sim[r, c] = _random_ti_value(ti, rng)
            continue

        best_dist = float("inf")
        best_value = None
        for _ in range(max_scan):
            y_r = int(rng.integers(0, ti_rows))
            y_c = int(rng.integers(0, ti_cols))
            dist = _event_distance(ti, y_r, y_c, offsets, values)
            if dist is None:
                continue
            if dist < best_dist:
                best_dist = dist
                best_value = float(ti[y_r, y_c])
            if dist <= distance_threshold:
                best_value = float(ti[y_r, y_c])
                break

        if best_value is None:
            best_value = _random_ti_value(ti, rng)
        sim[r, c] = best_value
    return sim.astype(np.float32)


def superresolve_one_level(
    coarse: np.ndarray,
    target_shape: tuple[int, int],
    conditioning_at_target_level: np.ndarray | None,
    rng: np.random.Generator,
    distance_threshold: float,
    max_scan: int,
    max_radius: int,
    max_neighbors: int,
) -> np.ndarray:
    """Paper-style step: parent top-left migration, then DS fills children."""
    coarse = coarse.astype(np.float32)
    rows, cols = coarse.shape
    fine = np.full((rows * 2, cols * 2), np.nan, dtype=np.float32)
    fine[0::2, 0::2] = coarse

    tr, tc = target_shape
    fine = fine[:tr, :tc]
    if conditioning_at_target_level is not None:
        cond = conditioning_at_target_level.astype(np.float32)
        finite = np.isfinite(cond)
        fine[finite] = cond[finite]

    return direct_sampling_fill(
        sim_grid=fine,
        training_image=coarse,
        rng=rng,
        distance_threshold=distance_threshold,
        max_scan=max_scan,
        max_radius=max_radius,
        max_neighbors=max_neighbors,
    )


def fractal_cascade_realization(
    condition: np.ndarray,
    original_known_values: np.ndarray,
    original_known_mask: np.ndarray,
    rng: np.random.Generator,
    max_downsample_levels: int = 6,
    downsample_fallback: str = "random_known",
    distance_threshold: float = 0.25,
    max_scan: int = 256,
    max_radius: int = 2,
    max_neighbors: int = 12,
) -> tuple[np.ndarray, list[tuple[int, int]], int, int]:
    levels, closed_level = make_closed_coarse_pyramid(
        condition=condition,
        max_levels=max_downsample_levels,
        rng=rng,
        fallback=downsample_fallback,
    )
    filled_nan_count = 0
    coarse = levels[-1].astype(np.float32).copy()
    if np.any(~np.isfinite(coarse)):
        finite = coarse[np.isfinite(coarse)]
        fill = float(np.mean(finite)) if finite.size else 0.0
        filled_nan_count = int(np.sum(~np.isfinite(coarse)))
        coarse[~np.isfinite(coarse)] = fill

    for level_idx in range(len(levels) - 2, -1, -1):
        coarse = superresolve_one_level(
            coarse=coarse,
            target_shape=levels[level_idx].shape,
            conditioning_at_target_level=levels[level_idx],
            rng=rng,
            distance_threshold=distance_threshold,
            max_scan=max_scan,
            max_radius=max_radius,
            max_neighbors=max_neighbors,
        )

    out = coarse[: condition.shape[0], : condition.shape[1]].astype(np.float32)
    out[original_known_mask > 0] = original_known_values[original_known_mask > 0]
    return out, [arr.shape for arr in levels], closed_level, filled_nan_count


def fractal_cascade_prior_stats(
    condition: np.ndarray,
    original_known_values: np.ndarray,
    original_known_mask: np.ndarray,
    n_realizations: int = 32,
    seed: int = 0,
    max_downsample_levels: int = 6,
    downsample_fallback: str = "random_known",
    distance_threshold: float = 0.25,
    max_scan: int = 256,
    max_radius: int = 2,
    max_neighbors: int = 12,
) -> CascadeResult:
    sims = []
    level_shapes: list[tuple[int, int]] = []
    closed_level = 0
    filled_total = 0
    base_rng = np.random.default_rng(seed)
    for i in range(n_realizations):
        rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)) + i)
        sim, shapes, level, filled = fractal_cascade_realization(
            condition=condition,
            original_known_values=original_known_values,
            original_known_mask=original_known_mask,
            rng=rng,
            max_downsample_levels=max_downsample_levels,
            downsample_fallback=downsample_fallback,
            distance_threshold=distance_threshold,
            max_scan=max_scan,
            max_radius=max_radius,
            max_neighbors=max_neighbors,
        )
        sims.append(sim)
        if not level_shapes:
            level_shapes = shapes
            closed_level = level
        filled_total += filled
    stack = np.stack(sims, axis=0).astype(np.float32)
    return CascadeResult(
        realizations=stack,
        mean=np.mean(stack, axis=0).astype(np.float32),
        p75=np.percentile(stack, 75, axis=0).astype(np.float32),
        std=np.std(stack, axis=0).astype(np.float32),
        level_shapes=level_shapes,
        closed_level=closed_level,
        filled_nan_count=filled_total,
    )
