"""Phase 1 end-to-end verification tests for Q-SAR Sentinel backend."""

from __future__ import annotations

import time
import unittest

import numpy as np
import torch

from backend.data.generate_demo_patches import (
    _CUSTOM_SCENARIOS_PATH,
    SCENARIOS,
    generate_realistic_sar_scene,
    register_custom_scenario,
)
from backend.main import (
    AnalyzeRequest,
    PrepareLocationRequest,
    VerificationUpdateRequest,
    _VERIFICATION_PATH,
    _build_location_bbox,
    _parse_lat_lon_query,
    _run_full_analysis,
    _slugify_location_id,
    geocode_location,
    get_quantum_circuit,
    get_scenarios,
    get_verifications,
    health,
    prepare_custom_location,
    update_verification,
)
from backend.quantum_model import DEFAULT_WEIGHTS_PATH, QDEV_NAME, load_or_init_hybrid_model


class TestPhase1Backend(unittest.TestCase):
    def test_sar_scene_shapes_and_physics(self) -> None:
        for sid in SCENARIOS:
            scene = generate_realistic_sar_scene(sid, look_count=4, seed=42)
            self.assertEqual(scene["post_patches"].shape, (36, 2, 64, 64))
            self.assertEqual(scene["pre_patches"].shape, (36, 2, 64, 64))
            self.assertTrue(0.0 <= float(scene["post_patches"].min()) <= float(scene["post_patches"].max()) <= 1.0)
            self.assertEqual(len(scene["tile_metrics"]), 36)
            self.assertIn("asf_rgb", scene["images"])
            self.assertIn("change_rgb", scene["images"])

    def test_hybrid_model_weights_and_inference(self) -> None:
        model = load_or_init_hybrid_model(DEFAULT_WEIGHTS_PATH)
        self.assertTrue(DEFAULT_WEIGHTS_PATH.exists())
        weight_kb = DEFAULT_WEIGHTS_PATH.stat().st_size / 1024.0
        self.assertLess(weight_kb, 5120.0, "Weight file must stay strictly < 5MB")

        scene = generate_realistic_sar_scene("nagpur_flood", look_count=4, seed=42)
        x = torch.from_numpy(scene["post_patches"])
        t0 = time.perf_counter()
        with torch.no_grad():
            q_logits, c_logits, angles, expvals = model(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(q_logits.shape, (36, 4))
        self.assertEqual(c_logits.shape, (36, 4))
        self.assertEqual(angles.shape, (36, 4))
        self.assertEqual(expvals.shape, (36, 8))
        self.assertTrue(float(angles.min()) >= -np.pi - 1e-4 and float(angles.max()) <= np.pi + 1e-4)

        sv_probs, bloch, entropy = model.extract_quantum_telemetry(angles[:2])
        self.assertEqual(sv_probs.shape, (2, 16))
        self.assertEqual(bloch.shape, (2, 4, 3))
        self.assertEqual(entropy.shape, (2,))
        self.assertTrue(np.all((entropy >= 0.0) & (entropy <= 1.0)))
        self.assertTrue(np.allclose(sv_probs.sum(axis=-1), 1.0, atol=1e-4))
        _, _, _, exp_fast, _, _, _ = model.infer_with_telemetry(x[:2])
        self.assertTrue(torch.allclose(expvals[:2], exp_fast, atol=1e-4))
        print(f"\n[Model] device={QDEV_NAME}, weights={weight_kb:.1f}KB, 36-patch forward={elapsed_ms:.1f}ms")

    def test_full_pipeline_endpoints_and_qaoa(self) -> None:
        h = health()
        self.assertEqual(h["status"], "ok")
        sc = get_scenarios()
        self.assertGreaterEqual(len(sc["scenarios"]), 2)
        qc = get_quantum_circuit()
        self.assertEqual(qc["n_qubits"], 4)
        self.assertEqual(qc["weight_shape"], [2, 4, 3])

        for sid in SCENARIOS:
            t0 = time.perf_counter()
            res = _run_full_analysis(
                AnalyzeRequest(scenario_id=sid, look_count=4, data_source="real_s1_rtc")
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.assertEqual(len(res["tiles"]), 36)
            self.assertTrue(res["data_provenance"]["is_real_satellite_data"])
            self.assertTrue(res["model_status"]["ground_truth_available"])
            self.assertFalse(res["model_status"]["is_verified_manual_ground_truth"])
            self.assertIsNotNone(res["model_status"]["scene_quantum_accuracy"])
            self.assertIn("humanitarian_impact", res)
            self.assertIn("explainable_brief", res)
            self.assertIn("routing_comparison", res["qaoa_routing"])
            self.assertIn("selected_strategy_name", res["qaoa_routing"])
            for t in res["tiles"]:
                self.assertIn("quantum_entropy", t)
                self.assertTrue(0.0 <= t["quantum_entropy"] <= 1.0)
            qaoa = res["qaoa_routing"]
            self.assertEqual(qaoa["n_qubits"], 4)
            self.assertEqual(len(qaoa["basis_state_probs"]), 16)
            self.assertEqual(qaoa["optimal_path_nodes"][0], qaoa["start_node"])
            self.assertEqual(qaoa["optimal_path_nodes"][-1], qaoa["end_node"])
            self.assertEqual(qaoa["geojson"]["type"], "FeatureCollection")
            for cand in qaoa["corridor_candidates"]:
                self.assertGreaterEqual(len(cand["path_coords"]), 2)
                self.assertGreaterEqual(len(cand["path_coords"]), len(cand["path_nodes"]))
            print(
                f"[RealGeoTIFF:{sid}] total={elapsed_ms:.1f}ms "
                f"mean_HQ={res['model_status']['mean_quantum_entropy']:.3f} "
                f"hazard_reduction={qaoa['hazard_reduction_pct']}% "
                f"approx_ratio={qaoa['approximation_ratio']}"
            )

        # Also verify labeled NISAR simulation mode, custom endpoints, and Pareto risk_aversion
        sim_res = _run_full_analysis(
            AnalyzeRequest(
                scenario_id="nagpur_flood",
                look_count=4,
                data_source="nisar_sim",
                start_node=(1, 0),
                end_node=(4, 5),
                risk_aversion=1.6,
            )
        )
        self.assertTrue(sim_res["model_status"]["ground_truth_available"])
        self.assertIsNotNone(sim_res["model_status"]["scene_quantum_accuracy"])
        self.assertEqual(sim_res["qaoa_routing"]["start_node"], [1, 0])
        self.assertEqual(sim_res["qaoa_routing"]["end_node"], [4, 5])
        print(
            f"[NISAR-Sim:nagpur_flood] q_acc={sim_res['model_status']['scene_quantum_accuracy']:.2f} "
            f"hazard_reduction={sim_res['qaoa_routing']['hazard_reduction_pct']}%"
        )

        # Verify dynamic risk_aversion corridor switching, impact schema, and warm latency
        low_ra = _run_full_analysis(
            AnalyzeRequest(scenario_id="vidarbha_crop_stress", data_source="real_s1_rtc", risk_aversion=0.25)
        )
        high_ra = _run_full_analysis(
            AnalyzeRequest(scenario_id="vidarbha_crop_stress", data_source="real_s1_rtc", risk_aversion=1.95)
        )
        self.assertLess(
            low_ra["qaoa_routing"]["selected_corridor_index"],
            high_ra["qaoa_routing"]["selected_corridor_index"],
        )
        self.assertLess(high_ra["timing_ms"]["total_pipeline_ms"], 1200.0)
        for key in (
            "flooded_area_km2",
            "burn_scar_area_km2",
            "agricultural_area_ha",
            "roads_at_risk_km",
            "bridges_at_risk_count",
            "facilities_total",
            "facilities_exposed_count",
            "priority_queue",
        ):
            self.assertIn(key, high_ra["humanitarian_impact"])
        for key in ("headline", "what_changed", "next_verification_action"):
            self.assertTrue(len(str(high_ra["explainable_brief"][key])) > 10)
        self.assertIn("routing_comparison", high_ra["qaoa_routing"])

        # Verify XY-mixer constraint preservation, dual labels, data quality, and verifications
        self.assertIn("data_quality", high_ra)
        self.assertEqual(high_ra["data_quality"]["status"], "PASS")
        self.assertTrue(high_ra["data_quality"]["crs_aligned"])
        self.assertTrue(high_ra["data_quality"]["shape_aligned"])
        self.assertIn("radiometric_reference_label", high_ra["tiles"][0])
        self.assertIsNone(high_ra["tiles"][0]["verified_ground_truth_label"])
        q2 = high_ra["qaoa_routing"]
        self.assertGreaterEqual(q2["valid_onehot_prob_mass"], 0.99)
        self.assertLessEqual(q2["invalid_bitstring_prob"], 0.01)
        self.assertGreater(q2["x_mixer_invalid_leakage"], 0.05)
        self.assertGreater(q2["quantum_opt_time_ms"], 0.0)
        self.assertGreater(q2["candidate_gen_time_ms"], 0.0)
        self.assertEqual(q2["approval_status"], "pending_human_approval")
        self.assertGreater(high_ra["humanitarian_impact"]["facilities_total"], 0)
        self.assertGreater(len(high_ra["humanitarian_impact"]["priority_queue"]), 0)

        prev_verif = _VERIFICATION_PATH.read_text(encoding="utf-8") if _VERIFICATION_PATH.exists() else None
        try:
            update_verification(
                VerificationUpdateRequest(
                    scenario_id="vidarbha_crop_stress",
                    tile_id=0,
                    tile_status="confirmed_hazard",
                    route_approval="approved_for_recon",
                )
            )
            fetched = get_verifications(scenario_id="vidarbha_crop_stress")
            self.assertEqual(fetched["tiles"]["0"]["status"], "confirmed_hazard")
            self.assertEqual(fetched["route_approval"]["status"], "approved_for_recon")
        finally:
            if prev_verif is None:
                _VERIFICATION_PATH.unlink(missing_ok=True)
            else:
                _VERIFICATION_PATH.write_text(prev_verif, encoding="utf-8")

    def test_custom_location_geocode_and_prepare(self) -> None:
        from unittest.mock import patch

        geo = geocode_location("20.92, 79.60")
        self.assertEqual(len(geo["results"]), 1)
        self.assertAlmostEqual(geo["results"][0]["lat"], 20.92, places=4)
        self.assertAlmostEqual(geo["results"][0]["lon"], 79.60, places=4)

        self.assertEqual(_parse_lat_lon_query("29.7604N, 95.3698W"), (29.7604, -95.3698))
        bbox = _build_location_bbox(29.7604, -95.3698, 0.24)
        self.assertEqual(len(bbox), 4)
        self.assertLess(bbox[0], bbox[2])
        self.assertLess(bbox[1], bbox[3])
        sid = _slugify_location_id("Houston, TX", 29.7604, -95.3698, "2025-08-01", "2025-10-01")
        self.assertTrue(sid.startswith("loc_houston_"))

        prev_custom = (
            _CUSTOM_SCENARIOS_PATH.read_text(encoding="utf-8")
            if _CUSTOM_SCENARIOS_PATH.exists()
            else None
        )
        created_id = None
        try:
            mock_stac = {
                "orbit": 34,
                "pre": {"datetime": "2025-08-12T12:00:00Z", "item_ids": ["S1A_PRE"]},
                "post": {"datetime": "2025-10-11T12:00:00Z", "item_ids": ["S1A_POST"]},
            }
            mock_osm = {
                "available": False,
                "roads": [],
                "bridges": [],
                "settlements": [],
                "facilities": [],
            }
            with (
                patch("backend.main.fetch_custom_sar_scene", return_value=mock_stac),
                patch("backend.main.ensure_osm_cache_for_bbox", return_value=mock_osm),
            ):
                prep = prepare_custom_location(
                    PrepareLocationRequest(
                        lat=29.7604,
                        lon=-95.3698,
                        label="Houston Test Basin, USA",
                        pre_date="2025-08-01",
                        post_date="2025-10-01",
                    )
                )
            created_id = prep["scenario"]["scenario_id"]
            self.assertEqual(prep["status"], "ready")
            self.assertEqual(prep["scenario"]["pre_date"], "2025-08-12")
            self.assertEqual(prep["scenario"]["post_date"], "2025-10-11")
            self.assertIn(created_id, SCENARIOS)

            # Verify full pipeline works on dynamic location with empty/unavailable OSM fallback
            dyn_res = _run_full_analysis(
                AnalyzeRequest(scenario_id=created_id, data_source="nisar_sim", look_count=4)
            )
            self.assertEqual(len(dyn_res["tiles"]), 36)
            self.assertEqual(dyn_res["qaoa_routing"]["optimal_path_nodes"][0], [0, 0])
            self.assertEqual(dyn_res["qaoa_routing"]["optimal_path_nodes"][-1], [5, 5])
        finally:
            if created_id and created_id in SCENARIOS:
                SCENARIOS.pop(created_id, None)
            if prev_custom is None:
                _CUSTOM_SCENARIOS_PATH.unlink(missing_ok=True)
            else:
                _CUSTOM_SCENARIOS_PATH.write_text(prev_custom, encoding="utf-8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
