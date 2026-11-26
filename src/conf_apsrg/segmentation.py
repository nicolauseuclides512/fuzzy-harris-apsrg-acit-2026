from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from typing import Any, Optional

import cv2
import numpy as np
from skimage.morphology import remove_small_holes

from .io import normalize_uint8


@dataclass(frozen=True)
class APSRGConfig:
    vesselness_kernel_sizes: tuple[int, ...] = (7, 11, 15)
    seed_selection_method: str = "selective_fuzzy_harris"
    seed_percentile: float = 92.0
    candidate_percentile: float = 78.0
    target_seed_count: int = 77
    min_seed_count: int = 7
    max_seed_count: int = 77
    seed_dilate_radius: int = 1
    min_seed_distance: int = 3
    fuzzy_weight: float = 0.35
    harris_weight: float = 0.40
    vesselness_weight: float = 0.25
    region_growing_mode: str = "edge_delayed"
    connectivity: int = 8
    max_vesselness_difference: float = 18.0
    max_iterations: int = 500000
    priority_mode: str = "kang_product"
    acceptance_mode: str = "local_or_region"
    region_mean_tolerance_multiplier: float = 2.0
    region_mean_scope: str = "connected_seed_component"
    reconsider_rejected_candidates: bool = True
    kang_minimum_edge_factor: float = 0.05
    hybrid_local_weight: float = 0.65
    hybrid_edge_weight: float = 0.35
    min_component_area: int = 8
    fill_small_holes_area: int = 8

    @property
    def local_tolerance(self) -> float:
        return self.max_vesselness_difference

    @property
    def region_tolerance(self) -> float:
        return (
            self.max_vesselness_difference
            * self.region_mean_tolerance_multiplier
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "APSRGConfig":
        data = data or {}
        hybrid = data.get("hybrid_seed", {}) or {}
        edge = data.get("edge_delayed_region_growing", {}) or {}
        kernels = tuple(
            int(value)
            for value in data.get("vesselness_kernel_sizes", (7, 11, 15))
        )
        return cls(
            vesselness_kernel_sizes=kernels,
            seed_selection_method=str(
                data.get("seed_selection_method", "selective_fuzzy_harris")
            ),
            seed_percentile=float(data.get("seed_percentile", 92)),
            candidate_percentile=float(data.get("candidate_percentile", 78)),
            target_seed_count=int(
                hybrid.get(
                    "target_seed_count",
                    data.get("target_seed_count", 77),
                )
            ),
            min_seed_count=int(hybrid.get("min_seed_count", 7)),
            max_seed_count=int(hybrid.get("max_seed_count", 77)),
            seed_dilate_radius=int(hybrid.get("seed_dilate_radius", 1)),
            min_seed_distance=int(hybrid.get("min_seed_distance", 3)),
            fuzzy_weight=float(hybrid.get("fuzzy_weight", 0.35)),
            harris_weight=float(hybrid.get("harris_weight", 0.40)),
            vesselness_weight=float(hybrid.get("vesselness_weight", 0.25)),
            region_growing_mode=str(
                data.get("region_growing_mode", "edge_delayed")
            ),
            connectivity=int(data.get("region_growing_connectivity", 8)),
            max_vesselness_difference=float(
                data.get("max_vesselness_difference", 18)
            ),
            max_iterations=int(data.get("max_iterations", 500000)),
            priority_mode=str(edge.get("priority_mode", "kang_product")),
            acceptance_mode=str(
                edge.get("acceptance_mode", "local_or_region")
            ),
            region_mean_tolerance_multiplier=float(
                edge.get("region_mean_tolerance_multiplier", 2.0)
            ),
            region_mean_scope=str(
                edge.get(
                    "region_mean_scope",
                    "connected_seed_component",
                )
            ),
            reconsider_rejected_candidates=bool(
                edge.get("reconsider_rejected_candidates", True)
            ),
            kang_minimum_edge_factor=float(
                edge.get("kang_minimum_edge_factor", 0.05)
            ),
            hybrid_local_weight=float(
                edge.get("hybrid_local_weight", 0.65)
            ),
            hybrid_edge_weight=float(
                edge.get("hybrid_edge_weight", 0.35)
            ),
            min_component_area=int(data.get("min_component_area", 8)),
            fill_small_holes_area=int(
                data.get("fill_small_holes_area", 8)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _odd(value: int) -> int:
    value = max(int(value), 3)
    return value if value % 2 else value + 1


def enhance_vessels(
    gray: np.ndarray,
    kernel_sizes: tuple[int, ...],
) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    responses = []
    for size in kernel_sizes:
        kernel_size = _odd(size)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        responses.append(
            cv2.morphologyEx(
                gray,
                cv2.MORPH_BLACKHAT,
                kernel,
            ).astype(np.float32)
        )
    return normalize_uint8(np.max(np.stack(responses), axis=0))


def percentile_maps(
    vesselness: np.ndarray,
    fov: Optional[np.ndarray],
    seed_percentile: float,
    candidate_percentile: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    valid = (
        np.ones(vesselness.shape, dtype=bool)
        if fov is None
        else np.asarray(fov, dtype=bool)
    )
    values = vesselness[valid]
    values = values[values > 0]
    if values.size == 0:
        empty = np.zeros(vesselness.shape, dtype=bool)
        return empty, empty, 255.0, 255.0
    seed_threshold = float(np.percentile(values, seed_percentile))
    candidate_threshold = min(
        float(np.percentile(values, candidate_percentile)),
        seed_threshold,
    )
    seeds = (vesselness >= seed_threshold) & valid
    candidate = (vesselness >= candidate_threshold) & valid
    return seeds, candidate, seed_threshold, candidate_threshold


def fuzzy_support(
    vesselness: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    normalized = vesselness.astype(np.float32) / 255.0
    local_mean = cv2.blur(normalized, (5, 5))
    similarity = 1.0 - np.abs(normalized - local_mean)
    gradient_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    edge = edge / (float(edge.max()) + 1e-8)
    support = 0.55 * similarity + 0.45 * (1.0 - edge)
    support *= candidate.astype(np.float32)
    return np.clip(support, 0.0, 1.0)


def harris_response(vesselness: np.ndarray) -> np.ndarray:
    image = vesselness.astype(np.float32) / 255.0
    response = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
    response = cv2.dilate(response, None)
    return np.clip(
        response / (float(response.max()) + 1e-8),
        0.0,
        1.0,
    )


def _select_spaced_points(
    score: np.ndarray,
    valid: np.ndarray,
    count: int,
    min_distance: int,
) -> np.ndarray:
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return np.zeros(valid.shape, dtype=bool)
    values = score[ys, xs]
    order = np.argsort(values)[::-1]
    chosen: list[tuple[int, int]] = []
    result = np.zeros(valid.shape, dtype=bool)
    minimum_squared_distance = max(min_distance, 0) ** 2
    for index in order:
        y, x = int(ys[index]), int(xs[index])
        if minimum_squared_distance and any(
            (y - chosen_y) ** 2 + (x - chosen_x) ** 2
            < minimum_squared_distance
            for chosen_y, chosen_x in chosen
        ):
            continue
        chosen.append((y, x))
        result[y, x] = True
        if len(chosen) >= count:
            break
    return result


def select_hybrid_seeds(
    vesselness: np.ndarray,
    candidate: np.ndarray,
    fov: Optional[np.ndarray],
    cfg: APSRGConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = candidate.copy()
    if fov is not None:
        valid &= np.asarray(fov, dtype=bool)
    fuzzy = fuzzy_support(vesselness, candidate)
    harris = harris_response(vesselness)
    vessel = vesselness.astype(np.float32) / 255.0
    score = (
        cfg.fuzzy_weight * fuzzy
        + cfg.harris_weight * harris
        + cfg.vesselness_weight * vessel
    )
    target = min(
        max(cfg.target_seed_count, cfg.min_seed_count),
        cfg.max_seed_count,
    )
    initial_seeds = _select_spaced_points(
        score,
        valid,
        target,
        cfg.min_seed_distance,
    )
    initial_seed_count = int(initial_seeds.sum())
    seeds = initial_seeds.copy()

    if cfg.seed_dilate_radius > 0:
        radius = cfg.seed_dilate_radius
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        seeds = (
            cv2.dilate(seeds.astype(np.uint8), kernel).astype(bool)
            & candidate
        )

    final_seed_pixels = int(seeds.sum())
    return seeds, {
        "target_seed_count": target,
        "initial_seed_count": initial_seed_count,
        "final_seed_pixels_after_dilation": final_seed_pixels,
        "seed_dilate_radius": cfg.seed_dilate_radius,
        "min_seed_distance": cfg.min_seed_distance,
        "fuzzy_support": fuzzy,
        "harris_response": harris,
        "hybrid_score": score,
    }


def select_vesselness_topk_seeds(
    vesselness: np.ndarray,
    candidate: np.ndarray,
    fov: Optional[np.ndarray],
    cfg: APSRGConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a count-matched vesselness-only control seed set."""
    valid = candidate.copy()
    if fov is not None:
        valid &= np.asarray(fov, dtype=bool)
    score = vesselness.astype(np.float32) / 255.0
    target = min(
        max(cfg.target_seed_count, cfg.min_seed_count),
        cfg.max_seed_count,
    )
    initial_seeds = _select_spaced_points(
        score,
        valid,
        target,
        cfg.min_seed_distance,
    )
    initial_seed_count = int(initial_seeds.sum())
    seeds = initial_seeds.copy()

    if cfg.seed_dilate_radius > 0:
        radius = cfg.seed_dilate_radius
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        seeds = (
            cv2.dilate(seeds.astype(np.uint8), kernel).astype(bool)
            & candidate
        )

    return seeds, {
        "target_seed_count": target,
        "initial_seed_count": initial_seed_count,
        "final_seed_pixels_after_dilation": int(seeds.sum()),
        "seed_dilate_radius": cfg.seed_dilate_radius,
        "min_seed_distance": cfg.min_seed_distance,
        "vesselness_topk_score": score,
    }


def neighbors(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def grow_bfs(
    candidate: np.ndarray,
    seeds: np.ndarray,
    vesselness: np.ndarray,
    cfg: APSRGConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.zeros(candidate.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for y, x in zip(*np.where(seeds & candidate)):
        result[y, x] = True
        queue.append((int(y), int(x)))

    iterations = 0
    rejected_attempts = 0
    offsets = neighbors(cfg.connectivity)
    while queue and iterations < cfg.max_iterations:
        y, x = queue.popleft()
        iterations += 1
        for delta_y, delta_x in offsets:
            yy, xx = y + delta_y, x + delta_x
            if (
                yy < 0
                or xx < 0
                or yy >= candidate.shape[0]
                or xx >= candidate.shape[1]
            ):
                continue
            if result[yy, xx] or not candidate[yy, xx]:
                continue
            difference = abs(
                float(vesselness[yy, xx]) - float(vesselness[y, x])
            )
            if difference <= cfg.max_vesselness_difference:
                result[yy, xx] = True
                queue.append((yy, xx))
            else:
                rejected_attempts += 1
    return result, {
        "mode": "bfs",
        "iterations": iterations,
        "rejected_attempts": rejected_attempts,
        "reconsider_rejected_candidates": True,
    }


def _accept(
    local_diff: float,
    region_diff: float,
    cfg: APSRGConfig,
) -> bool:
    local_ok = local_diff <= cfg.local_tolerance
    region_ok = region_diff <= cfg.region_tolerance

    if cfg.acceptance_mode == "region_mean":
        return region_ok
    if cfg.acceptance_mode == "local_path":
        return local_ok
    if cfg.acceptance_mode == "local_and_region":
        return local_ok and region_ok
    if cfg.acceptance_mode == "local_or_region":
        return local_ok or region_ok
    raise ValueError(
        f"Unsupported acceptance_mode: {cfg.acceptance_mode}"
    )


def _calculate_priority(
    local_diff: float,
    candidate_vesselness: float,
    cfg: APSRGConfig,
) -> float:
    """Calculate a frontier priority consistently for every heap insertion."""
    inverse_vesselness = 1.0 - float(candidate_vesselness) / 255.0

    if cfg.priority_mode == "kang_product":
        return float(
            local_diff
            * max(inverse_vesselness, cfg.kang_minimum_edge_factor)
        )

    if cfg.priority_mode == "hybrid":
        return float(
            cfg.hybrid_local_weight * local_diff
            + cfg.hybrid_edge_weight * (255.0 * inverse_vesselness)
        )

    raise ValueError(
        f"Unsupported priority_mode: {cfg.priority_mode}"
    )


def _push_frontier(
    heap: list[tuple[float, int, int, int, int, int]],
    counter: int,
    region_id: int,
    parent_y: int,
    parent_x: int,
    target_y: int,
    target_x: int,
    candidate: np.ndarray,
    vesselness: np.ndarray,
    accepted_region: np.ndarray,
    blocked: np.ndarray,
    cfg: APSRGConfig,
) -> int:
    if not (
        0 <= target_y < candidate.shape[0]
        and 0 <= target_x < candidate.shape[1]
    ):
        return counter
    if (
        not candidate[target_y, target_x]
        or accepted_region[target_y, target_x] != 0
        or blocked[target_y, target_x]
    ):
        return counter

    local_diff = abs(
        float(vesselness[target_y, target_x])
        - float(vesselness[parent_y, parent_x])
    )
    priority = _calculate_priority(
        local_diff,
        float(vesselness[target_y, target_x]),
        cfg,
    )
    heappush(
        heap,
        (
            priority,
            counter,
            region_id,
            target_y,
            target_x,
            int(vesselness[parent_y, parent_x]),
        ),
    )
    return counter + 1


def grow_edge_delayed(
    candidate: np.ndarray,
    seeds: np.ndarray,
    vesselness: np.ndarray,
    cfg: APSRGConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if cfg.region_mean_scope != "connected_seed_component":
        raise ValueError(
            "Unsupported region_mean_scope: "
            f"{cfg.region_mean_scope}"
        )

    result = seeds.astype(bool) & candidate
    number_of_labels, seed_labels = cv2.connectedComponents(
        result.astype(np.uint8),
        connectivity=8 if cfg.connectivity == 8 else 4,
    )
    accepted_region = seed_labels.astype(np.int32)
    blocked = np.zeros(candidate.shape, dtype=bool)
    attempt_counts = np.zeros(candidate.shape, dtype=np.int32)
    heap: list[tuple[float, int, int, int, int, int]] = []
    counter = 0

    region_sums: dict[int, float] = {}
    region_counts: dict[int, int] = {}
    initial_region_means: dict[int, float] = {}
    region_growth_counts: dict[int, int] = {}
    for region_id in range(1, number_of_labels):
        values = vesselness[seed_labels == region_id].astype(np.float64)
        region_sums[region_id] = float(values.sum())
        region_counts[region_id] = int(values.size)
        initial_region_means[region_id] = (
            float(values.mean()) if values.size else 0.0
        )
        region_growth_counts[region_id] = 0

    for y, x in zip(*np.where(result)):
        region_id = int(seed_labels[y, x])
        for delta_y, delta_x in neighbors(cfg.connectivity):
            yy, xx = y + delta_y, x + delta_x
            counter = _push_frontier(
                heap,
                counter,
                region_id,
                int(y),
                int(x),
                yy,
                xx,
                candidate,
                vesselness,
                accepted_region,
                blocked,
                cfg,
            )

    iterations = 0
    rejected_attempts = 0
    reconsidered_attempts = 0
    accepted_after_reconsideration = 0
    accepted_local_only = 0
    accepted_region_only = 0
    accepted_both = 0
    discarded_claims = 0
    while heap and iterations < cfg.max_iterations:
        _, _, region_id, y, x, parent_vesselness = heappop(heap)
        if accepted_region[y, x] != 0 or blocked[y, x]:
            discarded_claims += 1
            continue

        previous_attempts = int(attempt_counts[y, x])
        attempt_counts[y, x] += 1
        if previous_attempts > 0:
            reconsidered_attempts += 1

        iterations += 1
        local_diff = abs(
            float(vesselness[y, x]) - float(parent_vesselness)
        )
        region_mean = (
            region_sums[region_id] / region_counts[region_id]
        )
        region_diff = abs(float(vesselness[y, x]) - region_mean)
        local_ok = local_diff <= cfg.local_tolerance
        region_ok = region_diff <= cfg.region_tolerance

        if not _accept(local_diff, region_diff, cfg):
            rejected_attempts += 1
            if not cfg.reconsider_rejected_candidates:
                blocked[y, x] = True
            continue

        result[y, x] = True
        accepted_region[y, x] = region_id
        region_sums[region_id] += float(vesselness[y, x])
        region_counts[region_id] += 1
        region_growth_counts[region_id] += 1

        if previous_attempts > 0:
            accepted_after_reconsideration += 1
        if local_ok and region_ok:
            accepted_both += 1
        elif local_ok:
            accepted_local_only += 1
        else:
            accepted_region_only += 1

        for delta_y, delta_x in neighbors(cfg.connectivity):
            yy, xx = y + delta_y, x + delta_x
            counter = _push_frontier(
                heap,
                counter,
                region_id,
                y,
                x,
                yy,
                xx,
                candidate,
                vesselness,
                accepted_region,
                blocked,
                cfg,
            )

    region_statistics = []
    for region_id in sorted(region_counts):
        region_statistics.append(
            {
                "region_id": region_id,
                "seed_pixels": int(
                    np.count_nonzero(seed_labels == region_id)
                ),
                "grown_pixels": region_growth_counts[region_id],
                "initial_mean": initial_region_means[region_id],
                "final_mean": (
                    region_sums[region_id] / region_counts[region_id]
                ),
            }
        )

    return result, {
        "mode": "edge_delayed",
        "iterations": iterations,
        "rejected_attempts": rejected_attempts,
        "reconsidered_attempts": reconsidered_attempts,
        "accepted_after_reconsideration": accepted_after_reconsideration,
        "accepted_local_only": accepted_local_only,
        "accepted_region_only": accepted_region_only,
        "accepted_both": accepted_both,
        "discarded_claims": discarded_claims,
        "unique_rejected_candidates": int(
            np.count_nonzero((attempt_counts > 0) & (accepted_region == 0))
        ),
        "n_regions_initial": number_of_labels - 1,
        "n_regions_grown": sum(
            count > 0 for count in region_growth_counts.values()
        ),
        "region_statistics": region_statistics,
        "acceptance_mode": cfg.acceptance_mode,
        "priority_mode": cfg.priority_mode,
        "local_tolerance": cfg.local_tolerance,
        "region_tolerance": cfg.region_tolerance,
        "region_mean_scope": cfg.region_mean_scope,
        "reconsider_rejected_candidates": (
            cfg.reconsider_rejected_candidates
        ),
        "connectivity": cfg.connectivity,
    }


def remove_small_components(
    mask: np.ndarray,
    min_area: int,
    connectivity: int = 8,
) -> np.ndarray:
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8 if connectivity == 8 else 4,
    )
    output = np.zeros(mask.shape, dtype=bool)
    for label in range(1, number_of_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            output[labels == label] = True
    return output


def segment_apsrg(
    preprocessed_gray: np.ndarray,
    fov_mask: Optional[np.ndarray],
    config: APSRGConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    gray = np.asarray(preprocessed_gray, dtype=np.uint8)
    fov = (
        None
        if fov_mask is None
        else np.asarray(fov_mask, dtype=bool)
    )
    vesselness = enhance_vessels(gray, config.vesselness_kernel_sizes)
    percentile_seeds, candidate, seed_threshold, candidate_threshold = (
        percentile_maps(
            vesselness,
            fov,
            config.seed_percentile,
            config.candidate_percentile,
        )
    )

    seed_selection_method = config.seed_selection_method.strip().lower()
    if seed_selection_method in {
        "percentile",
        "percentile_polling",
        "polling",
    }:
        seeds = percentile_seeds
        seed_debug = {
            "initial_seed_count": int(seeds.sum()),
            "final_seed_pixels_after_dilation": int(seeds.sum()),
            "seed_dilate_radius": 0,
        }
    elif seed_selection_method in {
        "vesselness_topk",
        "matched_vesselness_topk",
    }:
        seeds, seed_debug = select_vesselness_topk_seeds(
            vesselness,
            candidate,
            fov,
            config,
        )
    elif seed_selection_method in {
        "selective_fuzzy_harris",
        "hybrid",
    }:
        seeds, seed_debug = select_hybrid_seeds(
            vesselness,
            candidate,
            fov,
            config,
        )
    else:
        raise ValueError(
            "Unsupported seed_selection_method: "
            f"{config.seed_selection_method}"
        )

    seeds &= candidate
    growing_mode = config.region_growing_mode.strip().lower()

    if growing_mode == "bfs":
        grown, growing_debug = grow_bfs(
            candidate,
            seeds,
            vesselness,
            config,
        )
    elif growing_mode == "edge_delayed":
        grown, growing_debug = grow_edge_delayed(
            candidate,
            seeds,
            vesselness,
            config,
        )
    else:
        raise ValueError(
            "Unsupported region_growing_mode: "
            f"{config.region_growing_mode}"
        )

    cleaned = remove_small_components(
        grown,
        config.min_component_area,
        config.connectivity,
    )
    if config.fill_small_holes_area > 0:
        try:
            # scikit-image >= 0.26: max_size removes areas <= max_size.
            cleaned = remove_small_holes(
                cleaned,
                max_size=max(config.fill_small_holes_area - 1, 0),
            )
        except TypeError:
            # scikit-image 0.22–0.25: remove areas < area_threshold.
            cleaned = remove_small_holes(
                cleaned,
                area_threshold=config.fill_small_holes_area,
            )
    if fov is not None:
        cleaned &= fov

    return cleaned.astype(bool), {
        "vesselness": vesselness,
        "candidate": candidate,
        "seeds": seeds,
        "seed_threshold": seed_threshold,
        "candidate_threshold": candidate_threshold,
        "seed_debug": seed_debug,
        "growing_debug": growing_debug,
        "n_output_pixels": int(cleaned.sum()),
        "config": config.to_dict(),
    }
