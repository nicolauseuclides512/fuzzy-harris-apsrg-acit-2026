from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from yaml.nodes import MappingNode, SequenceNode

import conf_apsrg.segmentation as segmentation
from conf_apsrg.ablation import (
    PLANNED_CONTRASTS,
    SCENARIOS,
    build_scenario_configs,
    changed_config_paths,
    contrast_validation_frame,
    scenario_matrix_frame,
    summarize_results,
)
from conf_apsrg.metrics import evaluate
from conf_apsrg.segmentation import (
    APSRGConfig,
    _accept,
    _calculate_priority,
    grow_bfs,
    grow_edge_delayed,
    select_vesselness_topk_seeds,
    segment_apsrg,
)


def test_metrics_perfect():
    ground_truth = np.zeros((16, 16), dtype=bool)
    ground_truth[4:8, 4:8] = True

    metrics = evaluate(ground_truth, ground_truth)

    assert metrics.accuracy == 1.0
    assert metrics.f1_score == 1.0


def test_segmentation_shapes():
    gray = np.full((48, 48), 180, dtype=np.uint8)
    gray[8:40, 24] = 20
    fov = np.ones((48, 48), dtype=bool)

    mask, debug = segment_apsrg(
        gray,
        fov,
        APSRGConfig(max_iterations=10000),
    )

    assert mask.shape == gray.shape
    assert mask.dtype == bool
    assert debug["candidate"].shape == gray.shape


@pytest.mark.parametrize(
    ("mode", "local_diff", "region_diff", "expected"),
    [
        ("region_mean", 10.0, 40.0, False),
        ("region_mean", 25.0, 30.0, True),
        ("local_path", 10.0, 40.0, True),
        ("local_path", 25.0, 30.0, False),
        ("local_and_region", 10.0, 30.0, True),
        ("local_and_region", 10.0, 40.0, False),
        ("local_or_region", 10.0, 40.0, True),
        ("local_or_region", 25.0, 30.0, True),
        ("local_or_region", 25.0, 40.0, False),
    ],
)
def test_acceptance_modes(mode, local_diff, region_diff, expected):
    config = APSRGConfig(
        max_vesselness_difference=18.0,
        region_mean_tolerance_multiplier=2.0,
        acceptance_mode=mode,
    )

    assert _accept(local_diff, region_diff, config) is expected


def test_acceptance_tolerances():
    config = APSRGConfig(
        max_vesselness_difference=18.0,
        region_mean_tolerance_multiplier=2.0,
    )

    assert config.local_tolerance == 18.0
    assert config.region_tolerance == 36.0


def test_config_reads_vesselness_difference():
    config = APSRGConfig.from_dict(
        {"max_vesselness_difference": 23.0}
    )

    assert config.max_vesselness_difference == 23.0
    assert config.local_tolerance == 23.0


def test_invalid_acceptance_mode():
    config = APSRGConfig(acceptance_mode="invalid_mode")

    with pytest.raises(ValueError, match="Unsupported acceptance_mode"):
        _accept(10.0, 10.0, config)


def test_priority_formulas():
    kang = APSRGConfig(
        priority_mode="kang_product",
        kang_minimum_edge_factor=0.05,
    )
    hybrid = APSRGConfig(
        priority_mode="hybrid",
        hybrid_local_weight=0.65,
        hybrid_edge_weight=0.35,
    )

    assert _calculate_priority(10.0, 255.0, kang) == pytest.approx(0.5)
    assert _calculate_priority(10.0, 0.0, kang) == pytest.approx(10.0)
    assert _calculate_priority(10.0, 255.0, hybrid) == pytest.approx(6.5)
    assert _calculate_priority(10.0, 0.0, hybrid) == pytest.approx(95.75)


def test_invalid_priority_mode():
    config = APSRGConfig(priority_mode="invalid_priority")

    with pytest.raises(ValueError, match="Unsupported priority_mode"):
        _calculate_priority(10.0, 100.0, config)


def test_edge_delayed_propagates_beyond_initial_frontier():
    candidate = np.ones((3, 3), dtype=bool)
    vesselness = np.full((3, 3), 100, dtype=np.uint8)
    seeds = np.zeros((3, 3), dtype=bool)
    seeds[1, 1] = True
    config = APSRGConfig(
        connectivity=4,
        acceptance_mode="local_or_region",
        max_vesselness_difference=1.0,
    )

    result, debug = grow_edge_delayed(
        candidate,
        seeds,
        vesselness,
        config,
    )

    assert result.all()
    assert debug["accepted_both"] == 8
    assert debug["n_regions_initial"] == 1


def test_region_means_are_kept_per_seed_component():
    candidate = np.zeros((5, 5), dtype=bool)
    seeds = np.zeros((5, 5), dtype=bool)
    vesselness = np.zeros((5, 5), dtype=np.uint8)

    candidate[1, 1] = True
    candidate[3, 3] = True
    seeds[1, 1] = True
    seeds[3, 3] = True
    vesselness[1, 1] = 10
    vesselness[3, 3] = 200

    _, debug = grow_edge_delayed(
        candidate,
        seeds,
        vesselness,
        APSRGConfig(connectivity=4),
    )

    means = sorted(
        item["initial_mean"]
        for item in debug["region_statistics"]
    )
    assert debug["n_regions_initial"] == 2
    assert means == [10.0, 200.0]


def test_rejected_candidate_can_pass_from_another_parent(monkeypatch):
    candidate = np.zeros((5, 4), dtype=bool)
    vesselness = np.zeros((5, 4), dtype=np.uint8)
    seeds = np.zeros((5, 4), dtype=bool)

    values = {
        (2, 1): 0,
        (1, 1): 20,
        (3, 1): 5,
        (3, 2): 10,
        (2, 2): 15,
        (1, 2): 20,
    }
    for (y, x), value in values.items():
        candidate[y, x] = True
        vesselness[y, x] = value
    seeds[2, 1] = True

    monkeypatch.setattr(
        segmentation,
        "_calculate_priority",
        lambda *_args, **_kwargs: 0.0,
    )
    config = APSRGConfig(
        connectivity=4,
        acceptance_mode="local_path",
        max_vesselness_difference=6.0,
        reconsider_rejected_candidates=True,
    )

    result, debug = grow_edge_delayed(
        candidate,
        seeds,
        vesselness,
        config,
    )

    assert result[1, 1]
    assert debug["rejected_attempts"] >= 1
    assert debug["accepted_after_reconsideration"] >= 1


def test_bfs_reconsiders_candidate_from_another_parent():
    candidate = np.zeros((5, 4), dtype=bool)
    vesselness = np.zeros((5, 4), dtype=np.uint8)
    seeds = np.zeros((5, 4), dtype=bool)

    values = {
        (2, 1): 0,
        (1, 1): 20,
        (3, 1): 5,
        (3, 2): 10,
        (2, 2): 15,
        (1, 2): 20,
    }
    for (y, x), value in values.items():
        candidate[y, x] = True
        vesselness[y, x] = value
    seeds[2, 1] = True

    result, debug = grow_bfs(
        candidate,
        seeds,
        vesselness,
        APSRGConfig(
            connectivity=4,
            max_vesselness_difference=6.0,
        ),
    )

    assert result[1, 1]
    assert debug["rejected_attempts"] >= 1


def test_invalid_region_mean_scope():
    candidate = np.ones((3, 3), dtype=bool)
    seeds = np.zeros((3, 3), dtype=bool)
    seeds[1, 1] = True
    vesselness = np.full((3, 3), 100, dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported region_mean_scope"):
        grow_edge_delayed(
            candidate,
            seeds,
            vesselness,
            APSRGConfig(region_mean_scope="global"),
        )


def _assert_unique_yaml_keys(node):
    if isinstance(node, MappingNode):
        keys = [key.value for key, _ in node.value]
        assert len(keys) == len(set(keys)), f"Duplicate YAML key in {keys}"
        for key, value in node.value:
            _assert_unique_yaml_keys(key)
            _assert_unique_yaml_keys(value)
    elif isinstance(node, SequenceNode):
        for item in node.value:
            _assert_unique_yaml_keys(item)


def test_stage2_yaml_is_unique_and_loads_algorithm_parameters():
    config_path = Path("configs/conference_base.yaml")
    yaml_text = config_path.read_text(encoding="utf-8")
    _assert_unique_yaml_keys(yaml.compose(yaml_text))

    raw_config = yaml.safe_load(yaml_text)["apsrg_baseline"]
    config = APSRGConfig.from_dict(raw_config)

    assert config.max_vesselness_difference == 18.0
    assert config.region_mean_scope == "connected_seed_component"
    assert config.reconsider_rejected_candidates is True
    assert config.kang_minimum_edge_factor == 0.05
    assert config.hybrid_local_weight == 0.65
    assert config.hybrid_edge_weight == 0.35


def _load_stage3_base_config():
    config_path = Path("configs/isolated_ablation_base.yaml")
    yaml_text = config_path.read_text(encoding="utf-8")
    _assert_unique_yaml_keys(yaml.compose(yaml_text))
    return yaml.safe_load(yaml_text)


def test_stage3_defines_legacy_reference_and_eight_controlled_scenarios():
    assert tuple(SCENARIOS) == (
        "L00",
        "A00",
        "A01",
        "A02",
        "A03",
        "A04",
        "A05",
        "A06",
        "A07",
    )
    assert len(PLANNED_CONTRASTS) == 8


def test_vesselness_topk_control_matches_requested_seed_count():
    vesselness = np.arange(100, dtype=np.uint8).reshape(10, 10)
    candidate = np.ones((10, 10), dtype=bool)
    config = APSRGConfig(
        target_seed_count=7,
        min_seed_count=1,
        max_seed_count=20,
        seed_dilate_radius=0,
        min_seed_distance=0,
    )

    seeds, debug = select_vesselness_topk_seeds(
        vesselness,
        candidate,
        None,
        config,
    )

    assert int(seeds.sum()) == 7
    assert debug["initial_seed_count"] == 7
    assert debug["final_seed_pixels_after_dilation"] == 7


def test_invalid_seed_selection_method_is_rejected():
    gray = np.full((16, 16), 128, dtype=np.uint8)
    fov = np.ones((16, 16), dtype=bool)

    with pytest.raises(ValueError, match="Unsupported seed_selection_method"):
        segment_apsrg(
            gray,
            fov,
            APSRGConfig(seed_selection_method="unknown"),
        )


def test_stage3_all_planned_contrasts_change_one_path():
    configs = build_scenario_configs(_load_stage3_base_config())
    validation = contrast_validation_frame(configs)

    assert validation["validation_passed"].all()
    assert (validation["n_changed_paths"] == 1).all()


def test_stage3_primary_acceptance_contrasts_hold_other_factors_constant():
    configs = build_scenario_configs(_load_stage3_base_config())
    expected = (
        "apsrg_baseline.edge_delayed_region_growing.acceptance_mode",
    )

    assert changed_config_paths(configs["A04"], configs["A06"]) == expected
    assert changed_config_paths(configs["A05"], configs["A06"]) == expected


def test_stage3_contrast_validator_rejects_hidden_second_change():
    configs = build_scenario_configs(_load_stage3_base_config())
    configs["A06"]["preprocessing"]["clahe_clip_limit"] = 3.0

    with pytest.raises(ValueError, match="Non-isolated contrast detected"):
        contrast_validation_frame(configs)


def test_stage3_scenario_matrix_records_effective_settings():
    configs = build_scenario_configs(_load_stage3_base_config())
    matrix = scenario_matrix_frame(configs).set_index("scenario")

    assert matrix.loc["L00", "analysis_role"] == "context_only"
    assert matrix.loc["A00", "seed_selection_method"] == "vesselness_topk"
    assert matrix.loc["A01", "target_seed_count"] == 35
    assert matrix.loc["A02", "target_seed_count"] == 77
    assert matrix.loc["A03", "seed_dilate_radius"] == 1
    assert matrix.loc["A04", "acceptance_mode"] == "local_path"
    assert matrix.loc["A05", "acceptance_mode"] == "region_mean"
    assert matrix.loc["A06", "acceptance_mode"] == "local_or_region"
    assert matrix.loc["A07", "priority_mode"] == "hybrid"


def test_stage3_summary_reports_mean_std_and_median():
    frame = np.array([0.6, 0.8])
    data = {
        metric: frame.copy()
        for metric in (
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1_score",
            "iou",
            "balanced_accuracy",
        )
    }
    summary = summarize_results({"A00": pd.DataFrame(data)})

    assert summary.loc[0, "f1_score_mean"] == pytest.approx(0.7)
    assert summary.loc[0, "f1_score_std"] == pytest.approx(2 ** 0.5 / 10)
    assert summary.loc[0, "f1_score_median"] == pytest.approx(0.7)


def test_invalid_region_growing_mode_is_rejected():
    gray = np.full((16, 16), 128, dtype=np.uint8)
    fov = np.ones((16, 16), dtype=bool)

    with pytest.raises(ValueError, match="Unsupported region_growing_mode"):
        segment_apsrg(
            gray,
            fov,
            APSRGConfig(region_growing_mode="invalid_mode"),
        )