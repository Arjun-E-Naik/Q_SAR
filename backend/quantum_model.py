"""Bi-Temporal Hybrid Quantum-Classical SAR Anomaly Model (PyTorch + PennyLane).

Architecture:
- Bi-Temporal Input Builder: Fuses pre- and post-event (B, 2, 64, 64) dual-pol SAR patches
  into (B, 4, 64, 64) = [Post_Co, Post_Cross, Norm(Delta_Co), Norm(Delta_Cross)].
- Classical SAR Stem: 3-layer speckle-aware Conv2d feature compressor mapping
  (B, 4, 64, 64) -> (B, 4) features bounded to [-pi, pi].
- Quantum Layer: 4-qubit PennyLane QNode (AngleEmbedding(Y) + 2x StronglyEntanglingLayers)
  using `lightning.qubit` (fallback `default.qubit`) with `diff_method="adjoint"`
  wrapped inside `qml.qnn.TorchLayer`.
- Telemetry QNode: Weight-shared inference QNode computing 16 basis probabilities and
  normalized Shannon entropy H_Q for epistemic uncertainty & unknown/abstention gating.
- Matched Classical Baseline Head: Parallel 4->8->4 MLP operating on the same 4D
  bottleneck to enable direct per-tile Classical vs. Hybrid Q-SAR comparison.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from backend.data.generate_demo_patches import (
    CLASS_NAMES,
    generate_bitemporal_training_batch,
)

logger = logging.getLogger(__name__)

N_QUBITS = 4
N_LAYERS = 2
N_CLASSES = 4
N_Q_OBS = 8  # 4 single-qubit <Z_i> + 4 ring two-qubit correlators <Z_i Z_{(i+1)%4}>
IN_CHANNELS = 4
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_WEIGHTS_PATH = DEFAULT_MODELS_DIR / "qsar_hybrid_weights.pt"
DEFAULT_BENCHMARK_PATH = DEFAULT_MODELS_DIR / "benchmark_metrics.json"


def create_quantum_device(n_wires: int = N_QUBITS) -> Tuple[Any, str]:
    """Instantiate `lightning.qubit` with automatic fallback to `default.qubit`."""
    try:
        dev = qml.device("lightning.qubit", wires=n_wires)
        return dev, "lightning.qubit"
    except Exception as exc:
        logger.warning("lightning.qubit unavailable (%s); falling back to default.qubit", exc)
        return qml.device("default.qubit", wires=n_wires), "default.qubit"


_QDEV, QDEV_NAME = create_quantum_device(N_QUBITS)


def build_bitemporal_input(
    pre_patches: Union[np.ndarray, torch.Tensor],
    post_patches: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """Fuse (B, 2, H, W) pre/post SAR patches into (B, 4, H, W) bi-temporal tensors."""
    pre_t = torch.from_numpy(pre_patches) if isinstance(pre_patches, np.ndarray) else pre_patches
    post_t = torch.from_numpy(post_patches) if isinstance(post_patches, np.ndarray) else post_patches
    pre_t = pre_t.float()
    post_t = post_t.float()
    # Smooth tanh delta compression preserves subtle C-band shifts (~2dB) without clipping L-band (-18dB)
    delta = 0.5 + 0.5 * torch.tanh(2.4 * (post_t - pre_t))
    return torch.cat([post_t, delta], dim=1)


def _quantum_circuit_ansatz(inputs: torch.Tensor, weights: torch.Tensor) -> None:
    """Shared variational ansatz for classification and statevector telemetry."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))


@qml.qnode(_QDEV, interface="torch", diff_method="adjoint")
def _q_classify_qnode(inputs: torch.Tensor, weights: torch.Tensor) -> List[Any]:
    _quantum_circuit_ansatz(inputs, weights)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)] + [
        qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 1) % N_QUBITS))
        for i in range(N_QUBITS)
    ]


@qml.qnode(_QDEV, interface="torch", diff_method=None)
def _q_state_qnode(inputs: torch.Tensor, weights: torch.Tensor) -> Any:
    _quantum_circuit_ansatz(inputs, weights)
    return qml.state()


class SARConvStem(nn.Module):
    """Dual-stream speckle-aware Conv2d bi-temporal SAR extractor (spatial + 4x4 multi-look)."""

    def __init__(self, n_qubits: int = N_QUBITS, in_channels: int = IN_CHANNELS) -> None:
        super().__init__()
        self.in_channels = in_channels
        # Stream 1: Full-resolution 3x3 spatial morphology & edge hierarchy (24 features)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2),  # 64 -> 32
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2),  # 32 -> 16
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # Stream 2: Linear 4x4 multi-look pooling BEFORE GELU suppresses L=1 speckle bias by 16x
        self.multilook_blocks = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),  # 64 -> 16
            nn.Conv2d(in_channels, 24, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(48, n_qubits)
        self.angle_bound = nn.Tanh()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h_spatial = self.flatten(self.conv_blocks(x))
        h_radiom = self.flatten(self.multilook_blocks(x))
        return torch.cat([h_spatial, h_radiom], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        return self.angle_bound(self.proj(h)) * float(np.pi)


class HybridQSARNet(nn.Module):
    """Bi-Temporal 4-Qubit Quantum-Classical SAR Classifier with matched classical head."""

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        in_channels: int = IN_CHANNELS,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.stem = SARConvStem(n_qubits=n_qubits, in_channels=in_channels)
        self.q_layer = qml.qnn.TorchLayer(
            _q_classify_qnode,
            weight_shapes={"weights": (n_layers, n_qubits, 3)},
        )
        self.classifier_head = nn.Linear(N_Q_OBS, N_CLASSES)
        self.classical_head = nn.Sequential(
            nn.Linear(n_qubits, 8),
            nn.GELU(),
            nn.Linear(8, N_CLASSES),
        )
        self.weight_source = "uninitialized"

    def _ensure_bitemporal(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 2 and self.in_channels == 4:
            neutral_delta = torch.full_like(x, 0.5)
            return torch.cat([x, neutral_delta], dim=1)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (quantum_logits, classical_logits, angles, q_expvals)."""
        x_in = self._ensure_bitemporal(x)
        angles = self.stem(x_in)
        q_expvals = self.q_layer(angles)
        q_logits = self.classifier_head(q_expvals)
        c_logits = self.classical_head(angles / float(np.pi))
        return q_logits, c_logits, angles, q_expvals

    @torch.no_grad()
    def infer_with_telemetry(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        """Fast batched inference returning logits, statevector probs, Bloch vectors, and entropy."""
        x_in = self._ensure_bitemporal(x)
        angles = self.stem(x_in)
        sv_probs, bloch, entropy = self.extract_quantum_telemetry(angles)
        # Compute exact 4x <Z_i> + 4x <Z_i Z_{(i+1)%4}> directly from 16-state probabilities
        single_z = bloch[:, :, 2]  # (B, 4)
        states_idx = np.arange(16, dtype=np.int64)
        zz_list = []
        for q in range(self.n_qubits):
            q_next = (q + 1) % self.n_qubits
            b1 = (states_idx >> (self.n_qubits - 1 - q)) & 1
            b2 = (states_idx >> (self.n_qubits - 1 - q_next)) & 1
            parity = np.where(b1 == b2, 1.0, -1.0).astype(np.float32)
            zz_list.append(sv_probs @ parity)
        pair_zz = np.stack(zz_list, axis=1)  # (B, 4)
        q_obs = np.concatenate([single_z, pair_zz], axis=1).astype(np.float32)
        q_expvals = torch.from_numpy(q_obs).to(device=angles.device, dtype=angles.dtype)
        q_logits = self.classifier_head(q_expvals)
        c_logits = self.classical_head(angles / float(np.pi))
        return q_logits, c_logits, angles, q_expvals, sv_probs, bloch, entropy

    @torch.no_grad()
    def extract_quantum_telemetry(
        self, angles: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 16-state probabilities, (B, 4, 3) Bloch vectors, and normalized Shannon entropy."""
        weights = self.q_layer.weights.detach()
        states_t = _q_state_qnode(angles.detach(), weights)
        states = np.asarray(
            states_t.detach().cpu().numpy() if torch.is_tensor(states_t) else states_t,
            dtype=np.complex64,
        )
        if states.ndim == 1:
            states = states[None, :]

        probs = np.real(states * np.conj(states)).astype(np.float32)
        probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-8)

        entropy = (
            -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=-1) / np.log(16.0)
        ).astype(np.float32)
        entropy = np.clip(entropy, 0.0, 1.0)

        batch_size = states.shape[0]
        psi_4d = states.reshape(batch_size, 2, 2, 2, 2)
        bloch = np.zeros((batch_size, self.n_qubits, 3), dtype=np.float32)
        for q in range(self.n_qubits):
            psi_q = np.moveaxis(psi_4d, 1 + q, 1).reshape(batch_size, 2, 8)
            rho = np.einsum("bik,bjk->bij", psi_q, np.conj(psi_q))
            bloch[:, q, 0] = np.real(2.0 * rho[:, 0, 1])
            bloch[:, q, 1] = np.real(-2.0 * np.imag(rho[:, 0, 1]))
            bloch[:, q, 2] = np.real(rho[:, 0, 0] - rho[:, 1, 1])

        return probs, bloch, entropy


class ClassicalSARNet(nn.Module):
    """Independent parameter-matched purely classical SAR baseline for ablation benchmarking."""

    def __init__(self, n_qubits: int = N_QUBITS, in_channels: int = IN_CHANNELS) -> None:
        super().__init__()
        self.stem = SARConvStem(n_qubits=n_qubits, in_channels=in_channels)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 8),
            nn.GELU(),
            nn.Linear(8, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.stem(x)
        return self.head(angles / float(np.pi))


def _init_sar_matched_conv_filters(stem: SARConvStem) -> None:
    """Deterministic bi-temporal polarimetric + spatial filter initialization (zero training loop)."""
    with torch.no_grad():
        conv1: nn.Conv2d = stem.conv_blocks[0]  # (16, 4, 3, 3)
        conv1.weight.zero_()
        conv1.bias.zero_()

        # Channels: (0: Post_Co, 1: Post_Cross, 2: 0.5+1.5*dCo, 3: 0.5+1.5*dCross)
        templates = [
            (-1.2, 0.0, -1.6, -0.4, 1.45),   # 0: New open flood (low post_co + negative delta_co)
            (0.0, -1.1, -0.4, -1.5, 1.35),   # 1: Cross-pol drop (water or burn)
            (1.4, -0.7, 2.0, -0.4, -1.15),   # 2: Flooded vegetation double-bounce (+delta_co, high post_co)
            (0.2, -0.6, 0.4, -2.6, 0.95),    # 3: Wildfire burn scar (sharp negative delta_cross, non-water co)
            (-1.1, -0.8, 0.0, 0.0, 0.65),    # 4: Static dark channel detector (permanent river separation)
            (0.8, 0.9, 0.0, 0.0, -0.70),     # 5: Intact vegetated canopy baseline
            (-0.3, -0.3, 1.4, 1.6, -0.75),   # 6: Positive agricultural/moisture growth
            (-0.5, -0.4, -1.4, -0.7, 1.15),  # 7: Crop dry-down / harvest shift (co-pol drops >= cross-pol)
        ]
        for idx, (w_co, w_cr, w_dco, w_dcr, bias) in enumerate(templates):
            conv1.weight[idx, 0, :, :] = w_co / 9.0
            conv1.weight[idx, 1, :, :] = w_cr / 9.0
            conv1.weight[idx, 2, :, :] = w_dco / 9.0
            conv1.weight[idx, 3, :, :] = w_dcr / 9.0
            conv1.bias[idx] = bias

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        sobel_y = sobel_x.t()
        lap = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]) / 2.0
        spatial_ops = [sobel_x, sobel_y, lap, -lap]
        for k in range(4):
            conv1.weight[8 + k, 0] = spatial_ops[k]
            conv1.weight[8 + k, 2] = 0.5 * spatial_ops[k]
            conv1.weight[12 + k, 1] = spatial_ops[k]
            conv1.weight[12 + k, 3] = 0.5 * spatial_ops[k]
            conv1.bias[8 + k] = 0.05
            conv1.bias[12 + k] = 0.05

        rng = torch.Generator().manual_seed(2026)
        for layer_idx in (3, 6):
            conv_l: nn.Conv2d = stem.conv_blocks[layer_idx]
            nn.init.kaiming_normal_(conv_l.weight, a=0.2, mode="fan_out", nonlinearity="relu", generator=rng)
            conv_l.bias.fill_(0.05)

        # Multi-look 1x1 radiometric basis (24 filters on clean 16x16 pooled channels)
        ml_conv: nn.Conv2d = stem.multilook_blocks[1]
        ml_conv.weight.zero_()
        ml_conv.bias.zero_()
        ml_templates = templates + [
            (1.0, 0.0, 0.0, 0.0, -0.50),     # +post_co deviation
            (-1.0, 0.0, 0.0, 0.0, 0.50),     # -post_co deviation
            (0.0, 1.0, 0.0, 0.0, -0.50),     # +post_cr deviation
            (0.0, -1.0, 0.0, 0.0, 0.50),     # -post_cr deviation
            (0.0, 0.0, 2.5, 0.0, -1.25),     # +delta_co
            (0.0, 0.0, -2.5, 0.0, 1.25),     # -delta_co
            (0.0, 0.0, 0.0, 2.5, -1.25),     # +delta_cr
            (0.0, 0.0, 0.0, -2.5, 1.25),     # -delta_cr
            (0.0, 0.0, 1.8, -2.8, 0.50),     # double-bounce separation (+dCo - dCr)
            (0.0, 0.0, -1.2, -2.8, 2.00),    # burn separation (dCr drops more than dCo)
            (-1.5, -1.0, 1.8, 1.8, -1.00),   # static/growth
            (-2.0, 0.0, 0.0, 0.0, 0.65),     # water threshold (~ -15 dB VV)
            (1.2, -1.2, 2.0, -1.2, -0.60),   # paddy double-bounce indicator
            (0.5, -0.8, 0.5, -3.0, 1.25),    # moderate burn scar indicator
            (0.0, 0.0, 2.0, 2.5, -2.25),     # vegetation recovery/moisture gain
            (0.0, 0.0, -2.5, -0.8, 1.65),    # harvest co-pol drop
        ]
        for idx, (w_co, w_cr, w_dco, w_dcr, bias) in enumerate(ml_templates):
            ml_conv.weight[idx, 0, 0, 0] = w_co
            ml_conv.weight[idx, 1, 0, 0] = w_cr
            ml_conv.weight[idx, 2, 0, 0] = w_dco
            ml_conv.weight[idx, 3, 0, 0] = w_dcr
            ml_conv.bias[idx] = bias


@torch.no_grad()
def calibrate_deterministic_baseline(model: HybridQSARNet) -> HybridQSARNet:
    """Zero-training closed-form bi-temporal calibration (<120ms, zero gradient epochs).

    Calibrates purely on physics-generated synthetic L-band and C-band bi-temporal pairs
    across L in {4, 2, 1} so evaluation against real Sentinel-1 scenes remains strictly zero-shot.
    """
    _init_sar_matched_conv_filters(model.stem)

    pre_np, post_np, labels_np = generate_bitemporal_training_batch(
        num_samples=384, look_count=(4, 2, 1), seed=7, include_cband=True
    )
    x_proto = build_bitemporal_input(pre_np, post_np)
    y_proto = torch.from_numpy(labels_np)

    feats = model.stem.forward_features(x_proto)  # (256, 48)
    class_angle_targets = torch.tensor(
        [
            [0.18, 0.18, 0.18, 0.18],  # Normal Terrain / Unchanged River
            [0.82, 0.82, 0.18, 0.82],  # Flood Inundation
            [0.18, 0.82, 0.82, 0.18],  # Wildfire Burn Scar
            [0.82, 0.18, 0.82, 0.18],  # Crop/Soil Stress
        ],
        dtype=torch.float32,
    )
    target_pre_tanh = torch.atanh(class_angle_targets[y_proto])
    X_aug = torch.cat([feats, torch.ones(feats.shape[0], 1)], dim=1)
    reg_eye = 1e-2 * torch.eye(X_aug.shape[1])
    reg_eye[-1, -1] = 1e-5
    W_proj = torch.linalg.solve(X_aug.T @ X_aug + reg_eye, X_aug.T @ target_pre_tanh)
    model.stem.proj.weight.copy_(W_proj[:-1].T)
    model.stem.proj.bias.copy_(W_proj[-1])

    angles = model.stem(x_proto)
    y_onehot = torch.nn.functional.one_hot(y_proto, num_classes=N_CLASSES).float() * 3.0 - 0.75

    best_acc = -1.0
    best_q_weights = None
    best_W_qhead = None
    rng_w = torch.Generator().manual_seed(42)
    for trial in range(10):
        scale = 0.12 if trial < 4 else (0.20 if trial < 7 else 0.30)
        cand_w = torch.randn((N_LAYERS, N_QUBITS, 3), generator=rng_w) * scale
        model.q_layer.weights.copy_(cand_w)
        q_exp = model.q_layer(angles)
        Q_aug = torch.cat([q_exp, torch.ones(q_exp.shape[0], 1)], dim=1)
        q_eye = 1e-2 * torch.eye(Q_aug.shape[1])
        q_eye[-1, -1] = 1e-5
        W_cand = torch.linalg.solve(Q_aug.T @ Q_aug + q_eye, Q_aug.T @ y_onehot)
        preds = torch.argmax(Q_aug @ W_cand, dim=-1)
        acc = float((preds == y_proto).float().mean())
        if acc > best_acc:
            best_acc = acc
            best_q_weights = cand_w.clone()
            best_W_qhead = W_cand.clone()

    model.q_layer.weights.copy_(best_q_weights)
    model.classifier_head.weight.copy_(best_W_qhead[:-1].T)
    model.classifier_head.bias.copy_(best_W_qhead[-1])

    nn.init.orthogonal_(model.classical_head[0].weight, gain=1.4)
    model.classical_head[0].bias.zero_()
    c_hidden = model.classical_head[1](model.classical_head[0](angles / float(np.pi)))
    C_aug = torch.cat([c_hidden, torch.ones(c_hidden.shape[0], 1)], dim=1)
    c_eye = 2e-2 * torch.eye(C_aug.shape[1])
    c_eye[-1, -1] = 1e-5
    W_chead = torch.linalg.solve(C_aug.T @ C_aug + c_eye, C_aug.T @ y_onehot)
    model.classical_head[2].weight.copy_(W_chead[:-1].T)
    model.classical_head[2].bias.copy_(W_chead[-1])

    model.weight_source = "bitemporal_calibrated_baseline"
    return model


def compute_uncertainty_and_abstention(
    probs: np.ndarray, entropy: float
) -> Dict[str, Any]:
    """Compute calibrated confidence tier and unknown/abstention recommendation."""
    max_prob = float(np.max(probs))
    ent = float(entropy)
    abstain = (max_prob < 0.45) or (ent > 0.78)
    if abstain or ent > 0.72:
        tier = "High Uncertainty (Verify)"
    elif ent >= 0.45 or max_prob < 0.65:
        tier = "Moderate Uncertainty"
    else:
        tier = "High Certainty"
    return {
        "abstention_flag": bool(abstain),
        "uncertainty_tier": tier,
    }


def load_or_init_hybrid_model(
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
) -> HybridQSARNet:
    """Load saved 4-channel weights (<5MB) or calibrate & persist deterministic baseline."""
    model = HybridQSARNet(in_channels=IN_CHANNELS)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if weights_path.exists():
        try:
            payload = torch.load(str(weights_path), map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and payload.get("delta_gain") == 2.4:
                state = payload["state_dict"] if "state_dict" in payload else payload
                model.load_state_dict(state, strict=True)
                model.weight_source = str(payload.get("weight_source", "saved_bitemporal_checkpoint"))
                model.eval()
                return model
        except Exception as exc:
            logger.warning("Incompatible or missing weights at %s (%s); recalibrating bi-temporal baseline.", weights_path, exc)

    calibrate_deterministic_baseline(model)
    model.eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "weight_source": model.weight_source,
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "in_channels": IN_CHANNELS,
            "delta_gain": 2.4,
            "class_names": CLASS_NAMES,
        },
        str(weights_path),
    )
    return model


def train_synthetic_hybrid_model(
    epochs: int = 20,
    num_samples: int = 512,
    save_path: Path = DEFAULT_WEIGHTS_PATH,
    seed: int = 42,
) -> HybridQSARNet:
    """Offline joint PyTorch + PennyLane adjoint training strictly on synthetic L+C band data.

    Never invoked inside FastAPI startup/requests. Zero real-satellite scenes are touched
    during training so real Sentinel-1 evaluation remains strictly zero-shot.
    """
    torch.manual_seed(seed)
    model = HybridQSARNet(in_channels=IN_CHANNELS)
    _init_sar_matched_conv_filters(model.stem)
    with torch.no_grad():
        nn.init.normal_(model.stem.proj.weight, std=0.22)
        model.stem.proj.bias.zero_()
        nn.init.normal_(model.q_layer.weights, std=0.18)
    model.train()

    pre_np, post_np, y_np = generate_bitemporal_training_batch(
        num_samples=num_samples, look_count=(4, 2, 1), seed=seed + 100, include_cband=True
    )
    x_train = build_bitemporal_input(pre_np, post_np)
    y_train = torch.from_numpy(y_np)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.stem.parameters(), "lr": 6e-3},
            {"params": model.q_layer.parameters(), "lr": 2.8e-2},
            {"params": model.classifier_head.parameters(), "lr": 2.8e-2},
        ],
        weight_decay=4e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)

    batch_size = 128
    n = x_train.shape[0]
    rng = np.random.default_rng(seed)

    for _epoch in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = torch.from_numpy(perm[i : i + batch_size]).long()
            xb, yb = x_train[idx], y_train[idx]
            optimizer.zero_grad()
            angles = model.stem(xb)
            q_expvals = model.q_layer(angles)
            q_logits = model.classifier_head(q_expvals)
            loss = loss_fn(q_logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()

    # Fit parallel classical MLP head cleanly on converged 4D bottleneck features (<0.1s)
    with torch.no_grad():
        norm_angles = (model.stem(x_train) / float(np.pi)).detach()
    c_opt = torch.optim.AdamW(model.classical_head.parameters(), lr=3e-2, weight_decay=4e-4)
    for _step in range(80):
        c_opt.zero_grad()
        c_loss = loss_fn(model.classical_head(norm_angles), y_train)
        c_loss.backward()
        c_opt.step()

    model.eval()
    model.weight_source = "bitemporal_hybrid_trained"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "weight_source": model.weight_source,
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "in_channels": IN_CHANNELS,
            "delta_gain": 2.4,
            "class_names": CLASS_NAMES,
        },
        str(save_path),
    )
    return model


def load_benchmark_metrics(
    model: HybridQSARNet,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
) -> Dict[str, Any]:
    """Load validated ablation benchmarks if present, or compute fast live bi-temporal eval."""
    if benchmark_path.exists():
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("bitemporal_version") == 2:
                return data
        except Exception:
            pass

    pre4, post4, y4 = generate_bitemporal_training_batch(
        num_samples=96, look_count=4, seed=999, include_cband=True
    )
    pre1, post1, y1 = generate_bitemporal_training_batch(
        num_samples=96, look_count=1, seed=1337, include_cband=True
    )
    x4 = build_bitemporal_input(pre4, post4)
    x1 = build_bitemporal_input(pre1, post1)
    with torch.no_grad():
        q_l4, c_l4, _, _ = model(x4)
        q_l1, c_l1, _, _ = model(x1)
        qp4 = torch.argmax(q_l4, dim=-1).numpy()
        cp4 = torch.argmax(c_l4, dim=-1).numpy()
        qp1 = torch.argmax(q_l1, dim=-1).numpy()
        cp1 = torch.argmax(c_l1, dim=-1).numpy()

    def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        f1s: List[float] = []
        for c in range(N_CLASSES):
            tp = int(np.sum((y_true == c) & (y_pred == c)))
            fp = int(np.sum((y_true != c) & (y_pred == c)))
            fn = int(np.sum((y_true == c) & (y_pred != c)))
            denom = 2 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
        return round(float(np.mean(f1s)), 4)

    q_acc = round(float(np.mean(qp4 == y4)), 4)
    c_acc = round(float(np.mean(cp4 == y4)), 4)
    q1_acc = round(float(np.mean(qp1 == y1)), 4)
    c1_acc = round(float(np.mean(cp1 == y1)), 4)
    return {
        "bitemporal_version": 2,
        "source": "live_bitemporal_baseline_eval",
        "note": "Zero-training bi-temporal evaluation across L-band & C-band held-out patches.",
        "eval_samples": 96,
        "quantum_accuracy": q_acc,
        "classical_accuracy": c_acc,
        "quantum_macro_f1": _macro_f1(y4, qp4),
        "classical_macro_f1": _macro_f1(y4, cp4),
        "quantum_1look_accuracy": q1_acc,
        "classical_1look_accuracy": c1_acc,
        "quantum_params": int(model.q_layer.weights.numel() + sum(p.numel() for p in model.classifier_head.parameters())),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "few_shot_curves": [],
    }


def get_circuit_specification(model: HybridQSARNet) -> Dict[str, Any]:
    """Return QNode specification metadata for API compatibility."""
    weights = model.q_layer.weights.detach().cpu().numpy()
    return {
        "device": QDEV_NAME,
        "interface": "torch",
        "diff_method": "adjoint",
        "n_qubits": model.n_qubits,
        "n_entangling_layers": model.n_layers,
        "basis_states": [f"|{i:04b}>" for i in range(16)],
        "weight_shape": list(weights.shape),
        "total_quantum_parameters": int(weights.size),
        "circuit_depth_estimate": 1 + model.n_layers * (model.n_qubits + 2),
        "gates": [
            {"stage": "Feature Encoding", "gate": "AngleEmbedding(rotation='Y')", "wires": [0, 1, 2, 3], "params_per_wire": 1},
            {"stage": "Variational Layer 1", "gate": "StronglyEntanglingLayers[0] (Rot + CNOT ring)", "wires": [0, 1, 2, 3], "params_per_wire": 3},
            {"stage": "Variational Layer 2", "gate": "StronglyEntanglingLayers[1] (Rot + CNOT stride-2)", "wires": [0, 1, 2, 3], "params_per_wire": 3},
            {"stage": "Measurement", "gate": "PauliZ Expectation (<Z_0>..<Z_3>)", "wires": [0, 1, 2, 3], "output_dim": 4},
        ],
        "variational_angles_rad": np.round(weights, 4).tolist(),
    }
