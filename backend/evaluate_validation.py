"""Independent Multi-Track Scientific Validation Suite for Q-SAR Sentinel 2.0.

Evaluates Hybrid Q-SAR (4-qubit PennyLane VQC) vs. Matched Classical CNN Baseline across
strictly separated evaluation tracks (zero real-satellite training leakage):
  Track A: Held-Out Synthetic Dual-Pol Physics (L=4 & L=1 Gamma Speckle Stress)
  Track B: Cross-Region Backscatter Shift & Spatial Holdout (360 patches across L=4/L=1)
  Track C: Zero-Shot Real Sentinel-1 C-Band RTC Scenes (108 tiles across Nagpur Flood,
           California Park Fire, Vidarbha Crop Stress vs. OPERA Radiometric Reference)

Outputs comprehensive metrics (Accuracy, Macro-F1, Per-Class Precision/Recall/IoU,
4x4 Confusion Matrix, 10-bin ECE, Normal-Tile FPR, Abstention Rate, Selective Accuracy,
and inference latency) to `backend/models/benchmark_metrics.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from backend.data.generate_demo_patches import (
    CLASS_NAMES,
    SCENARIOS,
    generate_bitemporal_training_batch,
)
from backend.quantum_model import (
    DEFAULT_BENCHMARK_PATH,
    N_CLASSES,
    ClassicalSARNet,
    HybridQSARNet,
    QDEV_NAME,
    _init_sar_matched_conv_filters,
    build_bitemporal_input,
    compute_uncertainty_and_abstention,
    load_or_init_hybrid_model,
)
from backend.sar_processor import load_sar_scene


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (10 equal-width confidence bins)."""
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(y_true))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if np.any(mask):
            acc_b = float(np.mean(correct[mask]))
            conf_b = float(np.mean(conf[mask]))
            ece += (float(np.sum(mask)) / n) * abs(acc_b - conf_b)
    return round(ece, 4)


def compute_classification_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    entropies: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute Accuracy, Macro-F1, Per-Class P/R/IoU, Confusion Matrix, ECE, FPR, and Abstention."""
    y_true = np.asarray(y_true, dtype=np.int64)
    preds = np.argmax(probs, axis=1)
    n = max(1, len(y_true))
    acc = round(float(np.mean(preds == y_true)), 4)

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for yt, yp in zip(y_true, preds):
        if 0 <= yt < N_CLASSES and 0 <= yp < N_CLASSES:
            cm[yt, yp] += 1

    per_class: List[Dict[str, Any]] = []
    f1s: List[float] = []
    ious: List[float] = []
    for c in range(N_CLASSES):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        support = int(np.sum(cm[c, :]))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        if support > 0 or fp > 0:
            f1s.append(f1)
            ious.append(iou)
        per_class.append(
            {
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "support": support,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "iou": round(iou, 4),
            }
        )

    # Normal-tile False Positive Rate: hazard predicted when true class is Normal (0)
    normal_mask = y_true == 0
    fpr_normal = (
        round(float(np.mean(preds[normal_mask] != 0)), 4)
        if np.any(normal_mask)
        else 0.0
    )

    abstention_rate = 0.0
    selective_acc = acc
    if entropies is not None:
        abstain_flags = np.array(
            [
                compute_uncertainty_and_abstention(probs[i], float(entropies[i]))["abstention_flag"]
                for i in range(len(y_true))
            ],
            dtype=bool,
        )
        abstention_rate = round(float(np.mean(abstain_flags)), 4)
        decisive = ~abstain_flags
        if np.any(decisive):
            selective_acc = round(float(np.mean(preds[decisive] == y_true[decisive])), 4)

    return {
        "samples": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
        "ece": compute_ece(probs, y_true),
        "normal_tile_fpr": fpr_normal,
        "abstention_rate": abstention_rate,
        "selective_accuracy": selective_acc,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def _train_standalone_classical(seed: int = 42) -> ClassicalSARNet:
    """Train an independent ClassicalSARNet with identical init & schedule for fair ablation."""
    torch.manual_seed(seed)
    c_model = ClassicalSARNet(n_qubits=4, in_channels=4)
    _init_sar_matched_conv_filters(c_model.stem)
    pre_np, post_np, y_np = generate_bitemporal_training_batch(
        num_samples=640, look_count=(4, 2, 1), seed=seed + 100, include_cband=True
    )
    x_tr = build_bitemporal_input(pre_np, post_np)
    y_tr = torch.from_numpy(y_np)
    opt = torch.optim.AdamW(
        [
            {"params": c_model.stem.parameters(), "lr": 6e-3},
            {"params": c_model.head.parameters(), "lr": 2.8e-2},
        ],
        weight_decay=4e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=24)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    rng = np.random.default_rng(seed)
    c_model.train()
    for _epoch in range(24):
        perm = rng.permutation(len(y_tr))
        for i in range(0, len(y_tr), 128):
            idx = torch.from_numpy(perm[i : i + 128]).long()
            opt.zero_grad()
            loss = loss_fn(c_model(x_tr[idx]), y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(c_model.parameters(), 2.0)
            opt.step()
        scheduler.step()
    c_model.eval()
    return c_model


def _eval_batch(
    hybrid: HybridQSARNet,
    standalone_c: ClassicalSARNet,
    pre_np: np.ndarray,
    post_np: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    x = build_bitemporal_input(pre_np, post_np)
    with torch.no_grad():
        t0 = time.perf_counter()
        q_logits, c_logits, _, _, _, _, entropy_np = hybrid.infer_with_telemetry(x)
        q_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        sc_logits = standalone_c(x)
        sc_ms = (time.perf_counter() - t1) * 1000.0

        q_probs = torch.softmax(q_logits, dim=-1).numpy()
        c_probs = torch.softmax(c_logits, dim=-1).numpy()
        sc_probs = torch.softmax(sc_logits, dim=-1).numpy()

    return {
        "hybrid_quantum": {
            **compute_classification_report(y_true, q_probs, entropies=entropy_np),
            "latency_ms": round(q_ms, 2),
        },
        "matched_classical_head": compute_classification_report(y_true, c_probs),
        "standalone_classical_cnn": {
            **compute_classification_report(y_true, sc_probs),
            "latency_ms": round(sc_ms, 2),
        },
    }


def run_validation_suite(output_path: Path = DEFAULT_BENCHMARK_PATH) -> Dict[str, Any]:
    """Run all 3 validation tracks and persist results to benchmark_metrics.json."""
    hybrid = load_or_init_hybrid_model()
    standalone_c = _train_standalone_classical(seed=42)

    # Track A: Synthetic L-Band Physics Scenarios (108 tiles per look regime)
    track_a: Dict[str, Any] = {}
    for looks_key, looks_val in (("l4", 4), ("l1_speckle_stress", 1)):
        pre_l, post_l, y_l = [], [], []
        for sc_id in SCENARIOS:
            scene = load_sar_scene(sc_id, look_count=looks_val, data_source="nisar_sim", seed=77)
            pre_l.append(scene["pre_patches"])
            post_l.append(scene["post_patches"])
            y_l.append(np.array([t["reference_class"] for t in scene["tile_metrics"]]))
        track_a[looks_key] = _eval_batch(
            hybrid,
            standalone_c,
            np.concatenate(pre_l, axis=0),
            np.concatenate(post_l, axis=0),
            np.concatenate(y_l, axis=0),
        )

    # Track B: Cross-Region Backscatter Shift Holdout (360 patches, +/-1.5 dB bias stress, L=4 & L=1)
    track_b: Dict[str, Any] = {}
    for looks_key, looks_val in (("l4_domain_shift", 4), ("l1_domain_shift", 1)):
        pre_h, post_h, y_h = generate_bitemporal_training_batch(
            num_samples=360, look_count=looks_val, seed=9091, include_cband=True
        )
        rng = np.random.default_rng(314 + looks_val)
        # Apply uniform regional backscatter shift identically to pre & post (+/-0.06 norm ~ +/-1.5 dB)
        # testing true bi-temporal change invariance across unseen terrain offsets
        shift = rng.uniform(-0.06, 0.06, size=(360, 2, 1, 1)).astype(np.float32)
        pre_shifted = np.clip(pre_h + shift, 0.0, 1.0)
        post_shifted = np.clip(post_h + shift, 0.0, 1.0)
        track_b[looks_key] = _eval_batch(hybrid, standalone_c, pre_shifted, post_shifted, y_h)

    # Track C: Zero-Shot Real Sentinel-1 RTC Scenes (108 tiles across Nagpur, California, Vidarbha)
    pre_r, post_r, y_r = [], [], []
    per_scene_real: Dict[str, Any] = {}
    for sc_id in SCENARIOS:
        scene = load_sar_scene(sc_id, look_count=4, data_source="real_s1_rtc", seed=42)
        y_sc = np.array([t["reference_class"] for t in scene["tile_metrics"]])
        per_scene_real[sc_id] = _eval_batch(
            hybrid, standalone_c, scene["pre_patches"], scene["post_patches"], y_sc
        )
        pre_r.append(scene["pre_patches"])
        post_r.append(scene["post_patches"])
        y_r.append(y_sc)

    track_c_all = _eval_batch(
        hybrid,
        standalone_c,
        np.concatenate(pre_r, axis=0),
        np.concatenate(post_r, axis=0),
        np.concatenate(y_r, axis=0),
    )

    t_l4 = track_a["l4"]
    t_l1 = track_a["l1_speckle_stress"]
    summary = {
        "bitemporal_version": 2,
        "source": "evaluate_validation_suite_v2",
        "quantum_device": QDEV_NAME,
        "weight_source": hybrid.weight_source,
        "note": (
            "Honest 3-track evaluation (synthetic L=4/L=1, domain-shift holdout, and zero-shot "
            "real Sentinel-1 C-band RTC vs. OPERA radiometric reference). Real S1 reference is a "
            "heuristic baseline, not manually digitized ground truth."
        ),
        "is_verified_manual_ground_truth": False,
        "ground_truth_category": "multi_track_physics_and_radiometric_heuristic",
        "quantum_accuracy": t_l4["hybrid_quantum"]["accuracy"],
        "classical_accuracy": t_l4["matched_classical_head"]["accuracy"],
        "standalone_classical_accuracy": t_l4["standalone_classical_cnn"]["accuracy"],
        "quantum_macro_f1": t_l4["hybrid_quantum"]["macro_f1"],
        "classical_macro_f1": t_l4["matched_classical_head"]["macro_f1"],
        "quantum_1look_accuracy": t_l1["hybrid_quantum"]["accuracy"],
        "classical_1look_accuracy": t_l1["matched_classical_head"]["accuracy"],
        "standalone_classical_1look_accuracy": t_l1["standalone_classical_cnn"]["accuracy"],
        "real_s1_quantum_heuristic_agreement": track_c_all["hybrid_quantum"]["accuracy"],
        "real_s1_classical_heuristic_agreement": track_c_all["standalone_classical_cnn"]["accuracy"],
        "real_s1_quantum_heuristic_macro_f1": track_c_all["hybrid_quantum"]["macro_f1"],
        "real_s1_classical_heuristic_macro_f1": track_c_all["standalone_classical_cnn"]["macro_f1"],
        # Backward-compatible aliases
        "real_s1_quantum_accuracy": track_c_all["hybrid_quantum"]["accuracy"],
        "real_s1_classical_accuracy": track_c_all["standalone_classical_cnn"]["accuracy"],
        "real_s1_quantum_macro_f1": track_c_all["hybrid_quantum"]["macro_f1"],
        "real_s1_classical_macro_f1": track_c_all["standalone_classical_cnn"]["macro_f1"],
        "quantum_params": int(
            hybrid.q_layer.weights.numel()
            + sum(p.numel() for p in hybrid.classifier_head.parameters())
        ),
        "total_params": int(sum(p.numel() for p in hybrid.parameters())),
        "tracks": {
            "synthetic_nisar_sim": track_a,
            "domain_shift_holdout": track_b,
            "real_s1_rtc_zeroshot": {
                "overall": track_c_all,
                "per_scenario": per_scene_real,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    t0 = time.perf_counter()
    res = run_validation_suite()
    print(f"Validation completed in {time.perf_counter()-t0:.2f}s")
    print(f"Track A (NISAR Sim L=4): Q={res['quantum_accuracy']:.4f} | HeadC={res['classical_accuracy']:.4f} | StandaloneC={res['standalone_classical_accuracy']:.4f}")
    print(f"Track A (NISAR Sim L=1): Q={res['quantum_1look_accuracy']:.4f} | HeadC={res['classical_1look_accuracy']:.4f} | StandaloneC={res['standalone_classical_1look_accuracy']:.4f}")
    print(f"Track B (Domain Shift L=4): Q={res['tracks']['domain_shift_holdout']['l4_domain_shift']['hybrid_quantum']['accuracy']:.4f} | StandaloneC={res['tracks']['domain_shift_holdout']['l4_domain_shift']['standalone_classical_cnn']['accuracy']:.4f}")
    print(f"Track C (Real S1 Zero-Shot vs OPERA Heuristic Ref): Q Agreement={res['real_s1_quantum_heuristic_agreement']:.4f} (F1={res['real_s1_quantum_heuristic_macro_f1']:.4f}) | StandaloneC Agreement={res['real_s1_classical_heuristic_agreement']:.4f} (F1={res['real_s1_classical_heuristic_macro_f1']:.4f})")
    for sc, sr in res["tracks"]["real_s1_rtc_zeroshot"]["per_scenario"].items():
        print(f"  Real S1 [{sc:20s}]: Q Agreement={sr['hybrid_quantum']['accuracy']:.4f} | HeadC={sr['matched_classical_head']['accuracy']:.4f} | StandaloneC={sr['standalone_classical_cnn']['accuracy']:.4f}")
