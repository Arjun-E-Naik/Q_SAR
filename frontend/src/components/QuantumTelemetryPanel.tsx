"use client";

import React, { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Compass,
  Navigation,
  Radar,
  Route,
  ShieldAlert,
  ShieldCheck,
  Users,
} from "lucide-react";
import type {
  AnalysisResponse,
  CorridorCandidate,
  RouteApprovalStatus,
  TilePrediction,
  TileVerificationRecord,
  TileVerificationStatus,
} from "@/types/sar";

interface QuantumTelemetryPanelProps {
  analysis: AnalysisResponse | null;
  selectedTile: TilePrediction | null;
  selectedCorridorIdx: number;
  onSelectCorridor: (idx: number) => void;
  onSelectTileById?: (tileId: number) => void;
  tileVerification?: TileVerificationRecord;
  onVerifyTile?: (tileId: number, status: TileVerificationStatus) => void;
  routeApprovalStatus?: RouteApprovalStatus;
  onRouteApprovalChange?: (status: RouteApprovalStatus) => void;
}

const CLASS_META: Array<{
  id: number;
  label: string;
  color: string;
  barClass: string;
  pillClass: string;
}> = [
  {
    id: 0,
    label: "Normal Terrain",
    color: "#22d3ee",
    barClass: "bg-cyan-400",
    pillClass: "bg-cyan-500/15 text-cyan-300 border-cyan-500/35",
  },
  {
    id: 1,
    label: "Flood Inundation",
    color: "#3b82f6",
    barClass: "bg-blue-500",
    pillClass: "bg-blue-500/15 text-blue-300 border-blue-500/35",
  },
  {
    id: 2,
    label: "Wildfire Burn Scar",
    color: "#f97316",
    barClass: "bg-orange-500",
    pillClass: "bg-orange-500/15 text-orange-300 border-orange-500/35",
  },
  {
    id: 3,
    label: "Crop/Soil Stress",
    color: "#eab308",
    barClass: "bg-amber-400",
    pillClass: "bg-amber-500/15 text-amber-300 border-amber-500/35",
  },
];

export default function QuantumTelemetryPanel({
  analysis,
  selectedTile,
  selectedCorridorIdx,
  onSelectCorridor,
  onSelectTileById,
  tileVerification,
  onVerifyTile,
  routeApprovalStatus,
  onRouteApprovalChange,
}: QuantumTelemetryPanelProps) {
  const [activeTab, setActiveTab] = useState<"tile" | "routes" | "validation">("tile");

  const tile = useMemo(() => {
    if (!analysis || analysis.tiles.length === 0) return null;
    if (selectedTile) return selectedTile;
    return (
      analysis.tiles.find((t) => t.quantum_class === 1 || t.quantum_class === 2) ??
      analysis.tiles[Math.floor(analysis.tiles.length / 2)]
    );
  }, [analysis, selectedTile]);

  if (!analysis || !tile) {
    return (
      <div className="glass-panel flex h-full items-center justify-center rounded-xl p-6 text-xs text-slate-400">
        Awaiting dual-polarization SAR analysis...
      </div>
    );
  }

  const coLabel = analysis.data_provenance?.co_pol_label ?? "VV";
  const crLabel = analysis.data_provenance?.cross_pol_label ?? "VH";
  const qClass = CLASS_META[tile.quantum_class] ?? CLASS_META[0];
  const cClass = CLASS_META[tile.classical_class] ?? CLASS_META[0];
  const qConf = tile.quantum_confidence ?? Math.max(...tile.quantum_probs);
  const cConf = tile.classical_confidence ?? Math.max(...tile.classical_probs);
  const modelsDisagree = tile.models_disagree ?? tile.quantum_class !== tile.classical_class;
  const routing = analysis.qaoa_routing;
  const impact = analysis.humanitarian_impact;
  const brief = analysis.explainable_brief;
  const tracks = analysis.benchmarks.tracks;

  return (
    <div className="glass-panel flex h-full flex-col overflow-hidden rounded-xl border border-slate-800/80 bg-slate-900/55 backdrop-blur-sm">
      {/* Outcome-Focused Panel Navigation */}
      <div className="flex items-center justify-between border-b border-slate-800/90 bg-slate-950/60 px-3 py-2">
        <div className="flex items-center gap-1 rounded-lg bg-slate-900/90 p-0.5 border border-slate-800">
          <button
            type="button"
            onClick={() => setActiveTab("tile")}
            className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition ${
              activeTab === "tile"
                ? "bg-cyan-500/20 text-cyan-200 border border-cyan-400/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Radar className="h-3.5 w-3.5" />
            <span>Tile Inspector</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("routes")}
            className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition ${
              activeTab === "routes"
                ? "bg-emerald-500/20 text-emerald-200 border border-emerald-400/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Route className="h-3.5 w-3.5" />
            <span>Corridors</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("validation")}
            className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition ${
              activeTab === "validation"
                ? "bg-violet-500/20 text-violet-200 border border-violet-400/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <BarChart3 className="h-3.5 w-3.5" />
            <span>Impact & Benchmarks</span>
          </button>
        </div>

        <span className="font-mono text-[10px] text-slate-400">
          {analysis.timing_ms.total_pipeline_ms.toFixed(0)}ms
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-3.5 space-y-3 text-xs">
        {/* Tab 1: Tile Inspector */}
        {activeTab === "tile" && (
          <>
            {/* Tile Header & Uncertainty / Abstention Pill */}
            <div className="rounded-lg border border-slate-800 bg-slate-900/75 p-3">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] font-semibold text-cyan-300">
                      TILE T{tile.tile_id}
                    </span>
                    <span className="text-[11px] text-slate-400">
                      Row {tile.row} · Col {tile.col}
                    </span>
                    {tile.area_km2 && (
                      <span className="font-mono text-[10px] text-slate-500">
                        ({tile.area_km2.toFixed(1)} km²)
                      </span>
                    )}
                  </div>
                  <div className="mt-0.5 font-mono text-[10px] text-slate-500">
                    {tile.center_lat.toFixed(4)}°, {tile.center_lon.toFixed(4)}° ·{" "}
                    {tile.osm_road_count ?? 0} roads · {tile.osm_bridge_count ?? 0} bridges
                    {(tile.osm_facility_count ?? 0) > 0
                      ? ` · ${tile.osm_facility_count} facilities`
                      : ""}
                  </div>
                </div>

                {tile.abstention_flag ? (
                  <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/40 bg-amber-500/15 px-2 py-0.5 text-[10px] font-semibold text-amber-300">
                    <AlertTriangle className="h-3 w-3" />
                    Unknown / Verify
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/35 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
                    <ShieldCheck className="h-3 w-3" />
                    {tile.uncertainty_tier ?? "Decisive"}
                  </span>
                )}
              </div>

              {/* Operator Field Verification Triage */}
              {onVerifyTile && (
                <div className="mt-2.5 flex flex-wrap items-center justify-between gap-1.5 rounded-md border border-slate-800/80 bg-slate-950/60 px-2.5 py-1.5">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                    Field Verification
                  </span>
                  <div className="flex flex-wrap items-center gap-1">
                    {(
                      [
                        { id: "confirmed_hazard", label: "Confirmed", active: "border-emerald-500/60 bg-emerald-500/20 text-emerald-200" },
                        { id: "needs_inspection", label: "Inspect", active: "border-amber-500/60 bg-amber-500/20 text-amber-200" },
                        { id: "false_alarm", label: "False Alarm", active: "border-slate-500/60 bg-slate-700/60 text-slate-200" },
                      ] as const
                    ).map((btn) => {
                      const isCur = tileVerification?.status === btn.id;
                      return (
                        <button
                          key={btn.id}
                          type="button"
                          onClick={() => onVerifyTile(tile.tile_id, isCur ? "unreviewed" : btn.id)}
                          className={`rounded border px-2 py-0.5 font-mono text-[9.5px] transition ${
                            isCur
                              ? btn.active
                              : "border-slate-800 bg-slate-900 text-slate-400 hover:border-slate-600 hover:text-slate-200"
                          }`}
                        >
                          {btn.label}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Classical vs Hybrid Q-SAR Side-by-Side Decision Cards */}
              <div className="mt-3 grid grid-cols-2 gap-2">
                <div className="rounded-md border border-slate-800 bg-slate-950/70 p-2">
                  <div className="text-[10px] uppercase tracking-wider text-slate-400">
                    Classical CNN
                  </div>
                  <div className="mt-1 flex items-center justify-between gap-1">
                    <span
                      className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold ${cClass.pillClass}`}
                    >
                      {cClass.label}
                    </span>
                    <span className="font-mono text-[11px] font-semibold text-slate-200">
                      {(cConf * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div className="rounded-md border border-cyan-500/30 bg-cyan-950/20 p-2">
                  <div className="text-[10px] uppercase tracking-wider text-cyan-300/90">
                    Hybrid Q-SAR
                  </div>
                  <div className="mt-1 flex items-center justify-between gap-1">
                    <span
                      className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold ${qClass.pillClass}`}
                    >
                      {qClass.label}
                    </span>
                    <span className="font-mono text-[11px] font-semibold text-cyan-200">
                      {(qConf * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Combined Agreement / Disagreement Status */}
              <div
                className={`mt-2 flex items-center justify-between rounded-md border px-2.5 py-1.5 text-[11px] ${
                  modelsDisagree
                    ? "border-amber-500/40 bg-amber-950/25 text-amber-200"
                    : "border-slate-800 bg-slate-950/60 text-slate-300"
                }`}
              >
                <span className="flex items-center gap-1.5">
                  {modelsDisagree ? (
                    <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
                  ) : (
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                  )}
                  {modelsDisagree
                    ? "Classical vs. Quantum Disagreement"
                    : "Models Consensus"}
                </span>
                <span className="font-mono text-[10px] text-slate-400">
                  Confidence Δ {(tile.confidence_delta * 100 >= 0 ? "+" : "")}
                  {(tile.confidence_delta * 100).toFixed(1)}% · H_Q {tile.quantum_entropy.toFixed(2)}
                </span>
              </div>
            </div>

            {/* 4-Class Probability Distribution: Hybrid Q-SAR vs Classical */}
            <div className="rounded-lg border border-slate-800 bg-slate-900/75 p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[11px] font-semibold text-slate-200">
                  Class Probabilities (Hybrid Q-SAR vs. Classical)
                </span>
                <span className="text-[10px] text-slate-400">Cyan=Q · Slate=C</span>
              </div>
              <div className="space-y-2">
                {CLASS_META.map((cls) => {
                  const qp = (tile.quantum_probs[cls.id] ?? 0) * 100;
                  const cp = (tile.classical_probs[cls.id] ?? 0) * 100;
                  return (
                    <div key={cls.id}>
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-slate-300">{cls.label}</span>
                        <span className="font-mono text-[10px] text-slate-300">
                          <span className="text-cyan-300">{qp.toFixed(1)}%</span>
                          <span className="text-slate-500"> / {cp.toFixed(1)}%</span>
                        </span>
                      </div>
                      <div className="mt-1 space-y-1">
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                          <div
                            className={`h-full rounded-full ${cls.barClass} transition-all`}
                            style={{ width: `${Math.min(100, Math.max(2, qp))}%` }}
                          />
                        </div>
                        <div className="h-1 w-full overflow-hidden rounded-full bg-slate-800/70">
                          <div
                            className="h-full rounded-full bg-slate-400/70 transition-all"
                            style={{ width: `${Math.min(100, Math.max(1, cp))}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Bi-Temporal Radar Signature & Surface Decomposition */}
            <div className="rounded-lg border border-slate-800 bg-slate-900/75 p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-200">
                  <Activity className="h-3.5 w-3.5 text-cyan-400" />
                  Bi-Temporal SAR Signature
                </span>
                <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                  {tile.physical_anomaly}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 font-mono text-[11px]">
                <div className="rounded bg-slate-950/75 p-2 border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">Δ{coLabel} Shift</div>
                  <div className="mt-0.5 font-semibold text-cyan-300">
                    {(tile.delta_hh_db ?? 0) >= 0 ? "+" : ""}
                    {(tile.delta_hh_db ?? 0).toFixed(2)} dB
                  </div>
                  <div className="text-[9px] text-slate-500">Post {tile.hh_db.toFixed(1)} dB</div>
                </div>
                <div className="rounded bg-slate-950/75 p-2 border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">Δ{crLabel} Shift</div>
                  <div className="mt-0.5 font-semibold text-amber-300">
                    {(tile.delta_hv_db ?? 0) >= 0 ? "+" : ""}
                    {(tile.delta_hv_db ?? 0).toFixed(2)} dB
                  </div>
                  <div className="text-[9px] text-slate-500">Post {tile.hv_db.toFixed(1)} dB</div>
                </div>
                <div className="rounded bg-slate-950/75 p-2 border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">ΔRVI / RVI</div>
                  <div className="mt-0.5 font-semibold text-emerald-300">
                    {(tile.delta_rvi ?? 0) >= 0 ? "+" : ""}
                    {(tile.delta_rvi ?? 0).toFixed(2)}
                  </div>
                  <div className="text-[9px] text-slate-500">Post RVI {tile.rvi.toFixed(2)}</div>
                </div>
              </div>

              <div className="mt-2 grid grid-cols-4 gap-1.5 text-center font-mono text-[10px]">
                <div className="rounded bg-slate-950/60 p-1.5 border border-slate-800/60">
                  <div className="text-slate-500">New Flood</div>
                  <div className="font-semibold text-blue-300">
                    {(tile.new_flood_pct ?? tile.water_pct).toFixed(1)}%
                  </div>
                </div>
                <div className="rounded bg-slate-950/60 p-1.5 border border-slate-800/60">
                  <div className="text-slate-500">Perm River</div>
                  <div className="font-semibold text-cyan-300">
                    {(tile.permanent_water_pct ?? 0).toFixed(1)}%
                  </div>
                </div>
                <div className="rounded bg-slate-950/60 p-1.5 border border-slate-800/60">
                  <div className="text-slate-500">Veg DblBnc</div>
                  <div className="font-semibold text-indigo-300">
                    {tile.double_bounce_pct.toFixed(1)}%
                  </div>
                </div>
                <div className="rounded bg-slate-950/60 p-1.5 border border-slate-800/60">
                  <div className="text-slate-500">Canopy Loss</div>
                  <div className="font-semibold text-orange-300">
                    {tile.burn_pct.toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Plain-Language Operational Explanation */}
            {tile.tile_explanation && (
              <div className="rounded-lg border border-cyan-500/25 bg-slate-900/80 p-3 space-y-1.5">
                <div className="text-[11px] font-semibold text-cyan-200">
                  Operational Radar Interpretation
                </div>
                <p className="text-[11px] leading-relaxed text-slate-200">
                  {tile.tile_explanation.what_changed}
                </p>
                <p className="font-mono text-[10px] text-slate-400">
                  {tile.tile_explanation.supporting_evidence}
                </p>
                <p className="text-[10px] text-slate-300/90">
                  {tile.tile_explanation.confidence_note}
                </p>
              </div>
            )}
          </>
        )}

        {/* Tab 2: Corridors & Routing */}
        {activeTab === "routes" && (
          <>
            {/* Routing Engine Badge + Human-in-the-Loop Approval */}
            <div className="rounded-lg border border-slate-800 bg-slate-900/80 p-3 space-y-2.5">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Compass className="h-4 w-4 text-emerald-400" />
                  <div>
                    <div className="text-[11px] font-semibold text-slate-100">
                      {routing.routing_mode === "osm_road_network"
                        ? `OpenStreetMap Road Network (${routing.osm_roads_used_count ?? 0} roads)`
                        : "6×6 Tactical Terrain Grid Fallback"}
                    </div>
                    <div className="text-[10px] text-slate-400">
                      Lowest-Modeled-Risk Candidate:{" "}
                      <span className="font-semibold text-emerald-300">
                        {routing.selected_strategy_name ?? "High-Ground Convoy"}
                      </span>{" "}
                      (α={routing.risk_aversion.toFixed(2)})
                    </div>
                  </div>
                </div>
                <span className="rounded-full border border-emerald-500/40 bg-emerald-500/15 px-2 py-0.5 font-mono text-[10px] font-semibold text-emerald-300">
                  -{routing.hazard_reduction_pct.toFixed(1)}% Risk
                </span>
              </div>

              {/* Honest Constraint-Preserving XY-Mixer vs Standard X-Mixer Telemetry */}
              <div className="grid grid-cols-3 gap-1.5 rounded border border-slate-800/80 bg-slate-950/65 p-2 font-mono text-[9.5px]">
                <div>
                  <div className="text-slate-400">XY-Mixer Valid</div>
                  <div className="text-emerald-300 font-semibold">
                    {((routing.valid_onehot_prob_mass ?? 1.0) * 100).toFixed(1)}% in |W₄⟩
                  </div>
                </div>
                <div>
                  <div className="text-slate-400">X-Mixer Leakage</div>
                  <div className="text-rose-300 font-semibold">
                    {((routing.x_mixer_invalid_leakage ?? 0) * 100).toFixed(1)}% invalid
                  </div>
                </div>
                <div>
                  <div className="text-slate-400">Gen / Q-Opt</div>
                  <div className="text-cyan-300 font-semibold">
                    {(routing.candidate_gen_time_ms ?? 0).toFixed(0)} / {(routing.quantum_opt_time_ms ?? 0).toFixed(0)}ms
                  </div>
                </div>
              </div>

              {onRouteApprovalChange && (
                <div className="flex flex-wrap items-center justify-between gap-1.5 rounded border border-slate-800 bg-slate-950/75 px-2.5 py-1.5">
                  <span className="text-[10px] font-mono text-slate-300">
                    Dispatch Sign-Off:
                  </span>
                  <div className="flex items-center gap-1 font-mono text-[9.5px]">
                    {(
                      [
                        { id: "pending_human_approval", label: "Pending", cls: "border-amber-500/50 bg-amber-500/15 text-amber-200" },
                        { id: "approved_for_recon", label: "Approve Recon", cls: "border-emerald-500/60 bg-emerald-500/20 text-emerald-200" },
                        { id: "rejected", label: "Reject", cls: "border-rose-500/60 bg-rose-500/20 text-rose-200" },
                      ] as const
                    ).map((opt) => {
                      const active = (routeApprovalStatus ?? routing.approval_status ?? "pending_human_approval") === opt.id;
                      return (
                        <button
                          key={opt.id}
                          type="button"
                          onClick={() => onRouteApprovalChange(opt.id)}
                          className={`rounded border px-2 py-0.5 transition ${
                            active ? opt.cls : "border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200"
                          }`}
                        >
                          {opt.label}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Classical Candidate Gen vs XY-QAOA Selector Benchmark Table */}
            {routing.routing_comparison && (
              <div className="rounded-lg border border-slate-800 bg-slate-900/75 p-3">
                <div className="mb-2 text-[11px] font-semibold text-slate-200">
                  {routing.routing_formulation ?? "Hybrid Candidate Gen + 4-Qubit XY-QAOA Selector"}
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-[10px]">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="py-1 pr-2">Solver</th>
                        <th className="py-1 px-1.5 text-right">Dist</th>
                        <th className="py-1 px-1.5 text-right">Modeled Haz</th>
                        <th className="py-1 px-1.5 text-right">Bridges</th>
                        <th className="py-1 pl-1.5 text-right">Time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      <tr>
                        <td className="py-1.5 pr-2 text-rose-300">Classical Blind Direct</td>
                        <td className="px-1.5 text-right">{routing.routing_comparison.classical_direct.distance_km.toFixed(1)} km</td>
                        <td className="px-1.5 text-right text-rose-300">{routing.routing_comparison.classical_direct.hazard_exposure.toFixed(1)}</td>
                        <td className="px-1.5 text-right">{routing.routing_comparison.classical_direct.bridge_crossings}</td>
                        <td className="pl-1.5 text-right text-slate-400">{routing.routing_comparison.classical_direct.runtime_ms.toFixed(1)}ms</td>
                      </tr>
                      <tr>
                        <td className="py-1.5 pr-2 text-slate-300">Classical Dijkstra</td>
                        <td className="px-1.5 text-right">{routing.routing_comparison.classical_dijkstra.distance_km.toFixed(1)} km</td>
                        <td className="px-1.5 text-right">{routing.routing_comparison.classical_dijkstra.hazard_exposure.toFixed(1)}</td>
                        <td className="px-1.5 text-right">{routing.routing_comparison.classical_dijkstra.bridge_crossings}</td>
                        <td className="pl-1.5 text-right text-slate-400">{routing.routing_comparison.classical_dijkstra.runtime_ms.toFixed(1)}ms</td>
                      </tr>
                      <tr>
                        <td className="py-1.5 pr-2 text-slate-300">Classical A* Heuristic</td>
                        <td className="px-1.5 text-right">{routing.routing_comparison.classical_astar.distance_km.toFixed(1)} km</td>
                        <td className="px-1.5 text-right">{routing.routing_comparison.classical_astar.hazard_exposure.toFixed(1)}</td>
                        <td className="px-1.5 text-right">{routing.routing_comparison.classical_astar.bridge_crossings}</td>
                        <td className="pl-1.5 text-right text-slate-400">{routing.routing_comparison.classical_astar.runtime_ms.toFixed(1)}ms</td>
                      </tr>
                      <tr className="bg-emerald-950/25 font-semibold text-emerald-200">
                        <td className="py-1.5 pr-2">XY-QAOA Selector</td>
                        <td className="px-1.5 text-right">{routing.routing_comparison.qaoa_selected.distance_km.toFixed(1)} km</td>
                        <td className="px-1.5 text-right text-emerald-300">{routing.routing_comparison.qaoa_selected.hazard_exposure.toFixed(1)}</td>
                        <td className="px-1.5 text-right">{routing.routing_comparison.qaoa_selected.bridge_crossings}</td>
                        <td className="pl-1.5 text-right text-emerald-300">{routing.routing_comparison.qaoa_selected.runtime_ms.toFixed(1)}ms</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* 4 Strategy Corridor Cards */}
            <div className="space-y-2">
              <div className="text-[11px] font-semibold text-slate-300">
                4 Candidate Relief Corridors (Click to Inspect on Map)
              </div>
              {routing.corridor_candidates.map((cand: CorridorCandidate, idx: number) => {
                const isSelected = idx === selectedCorridorIdx;
                const isOptimal = idx === routing.selected_corridor_index;
                return (
                  <button
                    key={cand.corridor_id}
                    type="button"
                    onClick={() => onSelectCorridor(idx)}
                    className={`w-full rounded-lg border p-2.5 text-left transition ${
                      isSelected
                        ? "border-emerald-400/70 bg-emerald-950/30"
                        : "border-slate-800 bg-slate-900/70 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <Navigation className="h-3.5 w-3.5 text-emerald-400" />
                        <span className="font-semibold text-slate-100">
                          {cand.name ?? `Corridor Strategy ${idx + 1}`}
                        </span>
                        {isOptimal && (
                          <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[9px] font-semibold text-emerald-300">
                            LOWEST RISK
                          </span>
                        )}
                      </div>
                      <span className="font-mono text-[11px] text-emerald-300">
                        {((cand.conditional_onehot_prob ?? cand.qaoa_prob) * 100).toFixed(0)}% |W₄⟩
                      </span>
                    </div>

                    <div className="mt-1.5 grid grid-cols-4 gap-1 font-mono text-[10px] text-slate-300">
                      <span>{cand.distance_km.toFixed(1)} km</span>
                      <span>Haz: {(cand.hazard_exposure ?? cand.hazard_cost).toFixed(1)}</span>
                      <span>Unc: {(cand.uncertainty_exposure ?? 0).toFixed(1)}</span>
                      <span>Br: {cand.bridge_crossings ?? 0}</span>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Operational Field Caution */}
            {routing.operational_caution && (
              <div className="rounded-lg border border-amber-500/35 bg-amber-950/20 p-2.5 text-[11px] text-amber-200/90 flex items-start gap-2">
                <ShieldAlert className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
                <span>{routing.operational_caution}</span>
              </div>
            )}
          </>
        )}

        {/* Tab 3: Impact & Validation */}
        {activeTab === "validation" && (
          <>
            {/* Explainable Scene Brief */}
            {brief && (
              <div className="rounded-lg border border-cyan-500/30 bg-slate-900/85 p-3 space-y-1.5">
                <div className="text-[11px] font-semibold text-cyan-200">
                  {brief.headline}
                </div>
                <p className="text-[11px] leading-relaxed text-slate-200">{brief.what_changed}</p>
                <div className="font-mono text-[10px] text-slate-400">{brief.supporting_channels}</div>
                <div className="rounded bg-slate-950/75 px-2 py-1.5 text-[10px] text-emerald-300 border border-slate-800">
                  <strong>Next Action:</strong> {brief.next_verification_action}
                </div>
              </div>
            )}

            {/* Raster Geo-Alignment & Data Quality Gate */}
            {analysis.data_quality && (
              <div className="rounded-lg border border-slate-800 bg-slate-900/75 px-3 py-2 flex flex-wrap items-center justify-between gap-2 font-mono text-[10px]">
                <div className="flex items-center gap-1.5">
                  <span
                    className={`px-1.5 py-0.5 rounded font-semibold border ${
                      analysis.data_quality.status === "PASS"
                        ? "border-emerald-500/45 bg-emerald-500/15 text-emerald-300"
                        : "border-amber-500/45 bg-amber-500/15 text-amber-300"
                    }`}
                  >
                    QC {analysis.data_quality.status}
                  </span>
                  <span className="text-slate-300">
                    {analysis.data_quality.crs} · {analysis.data_quality.valid_pixel_pct.toFixed(1)}% valid
                  </span>
                </div>
                <span className="text-slate-400">
                  [{analysis.data_quality.db_range[0]}..{analysis.data_quality.db_range[1]} dB]
                </span>
              </div>
            )}

            {/* Geospatial & Humanitarian Infrastructure Exposure */}
            {impact && (
              <div className="rounded-lg border border-slate-800 bg-slate-900/75 p-3 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-200">
                    <Users className="h-3.5 w-3.5 text-amber-400" />
                    OSM Infrastructure & Critical Facilities
                  </span>
                  <span className="font-mono text-[10px] text-slate-400">
                    Scene {impact.total_scene_area_km2.toFixed(0)} km²
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-1.5 font-mono text-[10px]">
                  <div className="rounded bg-slate-950/75 p-2 border border-slate-800">
                    <div className="text-slate-400">Roads Flagged</div>
                    <div className="mt-0.5 text-xs font-semibold text-amber-300">
                      {impact.roads_at_risk_km.toFixed(1)} km
                    </div>
                    <div className="text-[9px] text-slate-500">
                      of {impact.total_osm_roads_km.toFixed(0)} km
                    </div>
                  </div>
                  <div className="rounded bg-slate-950/75 p-2 border border-slate-800">
                    <div className="text-slate-400">Facilities Exposed</div>
                    <div className="mt-0.5 text-xs font-semibold text-teal-300">
                      {impact.facilities_exposed_count ?? 0} / {impact.facilities_total ?? 0}
                    </div>
                    <div className="text-[9px] text-slate-500">
                      Hosp / Shelter / Sch
                    </div>
                  </div>
                  <div className="rounded bg-slate-950/75 p-2 border border-slate-800">
                    <div className="text-slate-400">Est. Exposed Pop.</div>
                    <div className="mt-0.5 text-xs font-semibold text-rose-300">
                      {impact.estimated_exposed_population.toLocaleString()}
                    </div>
                    <div className="text-[9px] text-slate-500">
                      {impact.bridges_at_risk_count} br to inspect
                    </div>
                  </div>
                </div>

                {impact.critical_chokepoint_tiles.length > 0 && (
                  <div>
                    <div className="mb-1 text-[10px] font-semibold text-slate-400">
                      Critical Chokepoints (Click to Inspect Tile)
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {impact.critical_chokepoint_tiles.map((cp) => (
                        <button
                          key={cp.tile_id}
                          type="button"
                          onClick={() => {
                            onSelectTileById?.(cp.tile_id);
                            setActiveTab("tile");
                          }}
                          className="rounded border border-rose-500/40 bg-rose-950/30 px-2 py-0.5 font-mono text-[10px] text-rose-200 hover:bg-rose-900/40"
                        >
                          T{cp.tile_id} ({cp.osm_bridges} br / {cp.osm_roads} rd)
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Live Scene Accuracy vs Reference */}
            <div className="rounded-lg border border-slate-800 bg-slate-900/75 p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-slate-200">
                  Live Scene: Hybrid Q-SAR vs. Classical CNN
                </span>
                <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[9px] text-slate-300">
                  {analysis.model_status.is_verified_manual_ground_truth
                    ? "Verified GT"
                    : "Heuristic Ref (Not Manual GT)"}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 font-mono text-[10px]">
                <div className="rounded bg-slate-950/75 p-2 border border-slate-800">
                  <div className="text-slate-400">Hybrid Q-SAR</div>
                  <div className="mt-0.5 text-sm font-semibold text-cyan-300">
                    {analysis.model_status.scene_quantum_accuracy !== null
                      ? `${(analysis.model_status.scene_quantum_accuracy * 100).toFixed(1)}%`
                      : "N/A"}
                  </div>
                  <div className="text-[9px] text-slate-500">
                    F1 {((analysis.model_status.scene_quantum_macro_f1 ?? 0) * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="rounded bg-slate-950/75 p-2 border border-slate-800">
                  <div className="text-slate-400">Classical CNN</div>
                  <div className="mt-0.5 text-sm font-semibold text-slate-200">
                    {analysis.model_status.scene_classical_accuracy !== null
                      ? `${(analysis.model_status.scene_classical_accuracy * 100).toFixed(1)}%`
                      : "N/A"}
                  </div>
                  <div className="text-[9px] text-slate-500">
                    F1 {((analysis.model_status.scene_classical_macro_f1 ?? 0) * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="rounded bg-slate-950/75 p-2 border border-slate-800">
                  <div className="text-slate-400">Uncertain / Diff</div>
                  <div className="mt-0.5 text-sm font-semibold text-amber-300">
                    {analysis.model_status.abstention_count ?? 0} / 36
                  </div>
                  <div className="text-[9px] text-slate-500">
                    Disagree: {analysis.model_status.disagreement_count ?? 0}
                  </div>
                </div>
              </div>
            </div>

            {/* Separated 3-Track Scientific Benchmarks */}
            <div className="rounded-lg border border-slate-800 bg-slate-900/75 p-3 space-y-2">
              <div className="text-[11px] font-semibold text-slate-200">
                Multi-Track Validation (Zero Real-Scene Leakage)
              </div>
              <div className="space-y-1.5 font-mono text-[10px]">
                <div className="flex items-center justify-between rounded bg-slate-950/70 px-2.5 py-1.5 border border-slate-800/70">
                  <span className="text-slate-300">Synthetic L=4 Clean</span>
                  <span>
                    <span className="text-cyan-300">
                      Q {(analysis.benchmarks.quantum_accuracy * 100).toFixed(1)}%
                    </span>
                    <span className="text-slate-500"> vs </span>
                    <span className="text-slate-200">
                      C {((analysis.benchmarks.standalone_classical_accuracy ?? analysis.benchmarks.classical_accuracy) * 100).toFixed(1)}%
                    </span>
                  </span>
                </div>
                <div className="flex items-center justify-between rounded bg-slate-950/70 px-2.5 py-1.5 border border-slate-800/70">
                  <span className="text-slate-300">L=1 Speckle Stress</span>
                  <span>
                    <span className="text-cyan-300">
                      Q {((analysis.benchmarks.quantum_1look_accuracy ?? 0) * 100).toFixed(1)}%
                    </span>
                    <span className="text-slate-500"> vs </span>
                    <span className="text-slate-200">
                      C {((analysis.benchmarks.standalone_classical_1look_accuracy ?? analysis.benchmarks.classical_1look_accuracy ?? 0) * 100).toFixed(1)}%
                    </span>
                  </span>
                </div>
                {tracks?.domain_shift_holdout?.l4_domain_shift && (
                  <div className="flex items-center justify-between rounded bg-slate-950/70 px-2.5 py-1.5 border border-slate-800/70">
                    <span className="text-slate-300">Regional Shift Holdout</span>
                    <span>
                      <span className="text-cyan-300">
                        Q {(tracks.domain_shift_holdout.l4_domain_shift.hybrid_quantum.accuracy * 100).toFixed(1)}%
                      </span>
                      <span className="text-slate-500"> vs </span>
                      <span className="text-slate-200">
                        C {(tracks.domain_shift_holdout.l4_domain_shift.standalone_classical_cnn.accuracy * 100).toFixed(1)}%
                      </span>
                    </span>
                  </div>
                )}
                {tracks?.real_s1_rtc_zeroshot?.overall && (
                  <div className="flex items-center justify-between rounded bg-slate-950/70 px-2.5 py-1.5 border border-slate-800/70">
                    <span className="text-slate-300">Real S1 Zero-Shot (108)</span>
                    <span>
                      <span className="text-cyan-300">
                        Q {(tracks.real_s1_rtc_zeroshot.overall.hybrid_quantum.accuracy * 100).toFixed(1)}%
                      </span>
                      <span className="text-slate-500"> vs </span>
                      <span className="text-slate-200">
                        C {(tracks.real_s1_rtc_zeroshot.overall.standalone_classical_cnn.accuracy * 100).toFixed(1)}%
                      </span>
                    </span>
                  </div>
                )}
              </div>
              {analysis.benchmarks.note && (
                <p className="text-[10px] leading-snug text-slate-400">
                  {analysis.benchmarks.note}
                </p>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
