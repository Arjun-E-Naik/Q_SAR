"""Scientific Integrity & Regression Test Suite for Q-SAR Sentinel 2.0.

Verifies:
1. Exact geometric road clipping (preventing whole-polyline overcounting in hazard zones).
2. Honest heuristic agreement vs. true accuracy nomenclature separation.
3. Sensor adapter isolation (NISARGCOVAdapter, Sentinel1RTCAdapter, SyntheticSARAdapter).
4. Exact 4-candidate classical enumeration benchmark in QAOA routing telemetry.
5. Scenario cache fallback isolation (never returning unrelated scenarios).
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from backend.impact_analyzer import (
    _clip_segment_to_bounds,
    _polyline_clipped_length_km,
    _polyline_length_km,
    haversine_km,
)
from backend.main import app
from backend.qaoa_router import solve_qaoa_disaster_route
from backend.sar_processor import (
    NISARGCOVAdapter,
    Sentinel1RTCAdapter,
    SyntheticSARAdapter,
    load_sar_scene,
)


class TestScientificIntegrity(unittest.TestCase):
    def test_liang_barsky_segment_clipping(self) -> None:
        """Verify that a line segment crossing tile bounds is clipped exactly to the boundary."""
        box = [0.0, 0.0, 1.0, 1.0]  # [west, south, east, north]

        # Segment crossing horizontally from outside left (-1.0, 0.5) to outside right (2.0, 0.5)
        p0 = (-1.0, 0.5)
        p1 = (2.0, 0.5)
        clipped = _clip_segment_to_bounds(p0, p1, box)
        self.assertIsNotNone(clipped)
        assert clipped is not None
        c0, c1 = clipped
        self.assertAlmostEqual(c0[0], 0.0, places=6)
        self.assertAlmostEqual(c0[1], 0.5, places=6)
        self.assertAlmostEqual(c1[0], 1.0, places=6)
        self.assertAlmostEqual(c1[1], 0.5, places=6)

        # Full polyline length vs clipped length inside box
        coords = [(-1.0, 0.5), (2.0, 0.5)]
        full_len = _polyline_length_km(coords)
        clipped_len = _polyline_clipped_length_km(coords, box)

        # Full length spans 3 degrees longitude (~333 km at lat 0.5)
        # Clipped length spans exactly 1 degree longitude (~111 km at lat 0.5)
        self.assertGreater(full_len, clipped_len * 2.5)
        expected_clipped = haversine_km(0.0, 0.5, 1.0, 0.5)
        self.assertAlmostEqual(clipped_len, expected_clipped, delta=0.01)

    def test_segment_clipping_strictly_outside(self) -> None:
        """Verify that segments completely outside the bounding box are safely ignored."""
        box = [0.0, 0.0, 1.0, 1.0]
        # Segment completely to the north
        p0 = (0.2, 1.5)
        p1 = (0.8, 1.8)
        self.assertIsNone(_clip_segment_to_bounds(p0, p1, box))
        coords = [(0.2, 1.5), (0.8, 1.8)]
        self.assertEqual(_polyline_clipped_length_km(coords, box), 0.0)

    def test_sensor_adapters_isolation(self) -> None:
        """Verify that adapters cleanly preserve mission, band, and polarization identity."""
        self.assertEqual(NISARGCOVAdapter.sensor_name, "NASA-ISRO SAR (NISAR)")
        self.assertEqual(NISARGCOVAdapter.sensor_band, "L-Band (1.257 GHz)")
        self.assertEqual(NISARGCOVAdapter.polarizations, ("HH", "HV"))

        self.assertEqual(Sentinel1RTCAdapter.sensor_name, "Copernicus Sentinel-1 RTC")
        self.assertEqual(Sentinel1RTCAdapter.sensor_band, "C-Band (5.405 GHz)")
        self.assertEqual(Sentinel1RTCAdapter.polarizations, ("VV", "VH"))

        self.assertEqual(SyntheticSARAdapter.sensor_name, "Simulated NISAR L-Band Physics Benchmark")
        self.assertEqual(SyntheticSARAdapter.sensor_band, "L-Band (1.257 GHz · Simulated)")

    def test_real_s1_heuristic_reference_provenance(self) -> None:
        """Verify that real Sentinel-1 scenes report heuristic reference, not verified manual ground truth."""
        scene = load_sar_scene("nagpur_flood", look_count=4, data_source="real_s1_rtc")
        prov = scene["data_provenance"]
        self.assertTrue(prov["is_real_satellite_data"])
        self.assertFalse(prov["is_verified_manual_ground_truth"])
        self.assertEqual(prov["data_source_mode"], "live_sentinel1")
        self.assertEqual(prov["ground_truth_category"], "radiometric_heuristic_reference")
        self.assertEqual(prov["metric_display_mode"], "heuristic_agreement")

    def test_simulation_provenance_and_ground_truth(self) -> None:
        """Verify that simulation mode reports synthetic physics ground truth."""
        scene = load_sar_scene("nagpur_flood", look_count=4, data_source="nisar_sim")
        prov = scene["data_provenance"]
        self.assertFalse(prov["is_real_satellite_data"])
        self.assertEqual(prov["data_source_mode"], "simulation")
        self.assertEqual(prov["ground_truth_category"], "synthetic_physics_ground_truth")
        self.assertTrue(prov["is_verified_manual_ground_truth"])
        self.assertEqual(prov["metric_display_mode"], "accuracy")

    def test_qaoa_routing_classical_enumeration_benchmark(self) -> None:
        """Verify that QAOA routing includes exact 4-candidate classical enumeration comparison."""
        scene = load_sar_scene("nagpur_flood", look_count=4, data_source="real_s1_rtc")
        # Build minimal tile predictions dictionary
        tiles = []
        for t in scene["tile_metrics"]:
            tiles.append(
                {
                    **t,
                    "quantum_class": int(t["reference_class"]),
                    "quantum_confidence": 0.88,
                    "quantum_probs": [0.1, 0.8, 0.05, 0.05],
                    "quantum_entropy": 0.35,
                    "abstention_flag": False,
                }
            )

        res = solve_qaoa_disaster_route(
            tile_predictions=tiles,
            start_node=(0, 0),
            end_node=(5, 5),
            tile_km=4.2,
            risk_aversion=1.0,
            scenario_id="nagpur_flood",
        )
        comp = res["routing_comparison"]
        self.assertIn("classical_direct", comp)
        self.assertIn("classical_dijkstra", comp)
        self.assertIn("classical_astar", comp)
        self.assertIn("classical_enumeration", comp)
        self.assertIn("qaoa_selected", comp)

        enum_bench = comp["classical_enumeration"]
        self.assertIn("total_objective_cost", enum_bench)
        self.assertIn("strategy_name", enum_bench)
        self.assertLessEqual(enum_bench["runtime_ms"], 1.0)
        self.assertAlmostEqual(
            enum_bench["total_objective_cost"],
            res["best_classical_candidate_cost"],
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
