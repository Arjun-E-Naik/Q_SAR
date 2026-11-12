"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Crosshair,
  Flag,
  Globe,
  Layers,
  LocateFixed,
  Maximize2,
  Minimize2,
  RotateCcw,
  Route,
  SlidersHorizontal,
} from "lucide-react";
import type {
  AnalyzeResponse,
  LayerMode,
  MapClassificationMode,
  MapClickMode,
  OsmInfrastructureSummary,
  TilePrediction,
  TileVerificationRecord,
} from "@/types/sar";

interface SarViewportProps {
  analysis: AnalyzeResponse;
  layerMode: LayerMode;
  onLayerModeChange: (mode: LayerMode) => void;
  selectedTileId: number;
  onSelectTile: (tileId: number) => void;
  clickMode: MapClickMode;
  onClickModeChange: (mode: MapClickMode) => void;
  customOrigin: [number, number] | null;
  customTarget: [number, number] | null;
  onPickOrigin: (node: [number, number]) => void;
  onPickTarget: (node: [number, number]) => void;
  onResetRoute: () => void;
  hoveredCorridorId: number | null;
  classificationMode?: MapClassificationMode;
  osmInfrastructure?: OsmInfrastructureSummary | null;
  tileVerifications?: Record<string, TileVerificationRecord>;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

const CLASS_STYLE: Record<
  number,
  { name: string; stroke: string; fill: string; badge: string }
> = {
  0: {
    name: "Normal Terrain",
    stroke: "rgba(100, 116, 139, 0.45)",
    fill: "rgba(15, 23, 42, 0.14)",
    badge: "#94a3b8",
  },
  1: {
    name: "Flood Inundation",
    stroke: "rgba(56, 189, 248, 0.85)",
    fill: "rgba(14, 165, 233, 0.34)",
    badge: "#38bdf8",
  },
  2: {
    name: "Wildfire Burn Scar",
    stroke: "rgba(249, 115, 22, 0.90)",
    fill: "rgba(239, 68, 68, 0.34)",
    badge: "#fb923c",
  },
  3: {
    name: "Crop/Soil Stress",
    stroke: "rgba(234, 179, 8, 0.80)",
    fill: "rgba(234, 179, 8, 0.25)",
    badge: "#facc15",
  },
};

const LAYER_OPTIONS: { id: LayerMode; label: string; hint: string }[] = [
  { id: "segmentation", label: "Hazard Seg", hint: "4-Class Hazard & Corridor Map" },
  { id: "change", label: "Change Δσ⁰", hint: "Pre/Post Temporal Anomaly" },
  { id: "hh", label: "Co-Pol", hint: "VV/HH Specular & Double-Bounce" },
  { id: "hv", label: "Cross-Pol", hint: "VH/HV Canopy Volume Scatter" },
  { id: "asf_rgb", label: "ASF RGB", hint: "R:Co G:Cross B:Ratio" },
  { id: "swipe", label: "Swipe", hint: "Before / After Temporal SAR Swipe" },
];

interface ScreenBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  w: number;
  h: number;
}

export default function SarViewport({
  analysis,
  layerMode,
  onLayerModeChange,
  selectedTileId,
  onSelectTile,
  clickMode,
  onClickModeChange,
  customOrigin,
  customTarget,
  onPickOrigin,
  onPickTarget,
  onResetRoute,
  hoveredCorridorId,
  classificationMode = "hybrid",
  osmInfrastructure = null,
  tileVerifications = {},
  isExpanded = false,
  onToggleExpand,
}: SarViewportProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<import("maplibre-gl").Map | null>(null);

  const [mapReady, setMapReady] = useState(false);
  const [mapError, setMapError] = useState(false);
  const [basemapMode, setBasemapMode] = useState<"dark" | "satellite">("satellite");
  const [tileDisplayMode, setTileDisplayMode] = useState<"hazards" | "all" | "off">("hazards");
  const [swipePct, setSwipePct] = useState(50);
  const [draggingSwipe, setDraggingSwipe] = useState(false);
  const [showNaiveRoute, setShowNaiveRoute] = useState(true);
  const [showOsmOverlay, setShowOsmOverlay] = useState(true);
  const [hoveredTile, setHoveredTile] = useState<TilePrediction | null>(null);
  const [cursorCoords, setCursorCoords] = useState<{ lon: number; lat: number } | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 580 });
  const [projVersion, setProjVersion] = useState(0);

  const bbox = analysis.scenario.bbox; // [west, south, east, north]
  const [west, south, east, north] = bbox;
  const initialBoundsRef = useRef<[number, number, number, number]>(bbox);
  const latestBoundsRef = useRef<[number, number, number, number]>(bbox);
  latestBoundsRef.current = bbox;

  // Initialize a single persistent MapLibre GL instance with keyless Esri basemaps
  useEffect(() => {
    let cancelled = false;
    let mapInstance: import("maplibre-gl").Map | null = null;
    const [iw, is, ie, inorth] = initialBoundsRef.current;

    async function initMap() {
      if (!mapContainerRef.current) return;
      try {
        const maplibregl = (await import("maplibre-gl")).default;
        if (cancelled || !mapContainerRef.current) return;

        mapInstance = new maplibregl.Map({
          container: mapContainerRef.current,
          style: {
            version: 8,
            sources: {
              "esri-sat": {
                type: "raster",
                tiles: [
                  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                ],
                tileSize: 256,
                maxzoom: 18,
                attribution: "© Esri, Maxar, Earthstar Geographics",
              },
              "esri-dark": {
                type: "raster",
                tiles: [
                  "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                ],
                tileSize: 256,
                maxzoom: 16,
                attribution: "© Esri, HERE, Garmin, OpenStreetMap",
              },
              "esri-dark-ref": {
                type: "raster",
                tiles: [
                  "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                ],
                tileSize: 256,
                maxzoom: 16,
              },
            },
            layers: [
              {
                id: "esri-sat-layer",
                type: "raster",
                source: "esri-sat",
                layout: { visibility: "visible" },
                paint: { "raster-opacity": 0.9 },
              },
              {
                id: "esri-dark-layer",
                type: "raster",
                source: "esri-dark",
                layout: { visibility: "none" },
                paint: { "raster-opacity": 0.9 },
              },
              {
                id: "esri-dark-ref-layer",
                type: "raster",
                source: "esri-dark-ref",
                layout: { visibility: "visible" },
                paint: { "raster-opacity": 0.85 },
              },
            ],
          },
          bounds: [
            [iw, is],
            [ie, inorth],
          ],
          fitBoundsOptions: { padding: 24 },
          attributionControl: { compact: true },
        });

        mapInstance.addControl(
          new maplibregl.NavigationControl({ showCompass: true, showZoom: true, visualizePitch: false }),
          "top-right"
        );
        mapInstance.addControl(
          new maplibregl.ScaleControl({ maxWidth: 110, unit: "metric" }),
          "bottom-right"
        );

        mapInstance.on("load", () => {
          if (cancelled || !mapInstance) return;
          const [lw, ls, le, ln] = latestBoundsRef.current;
          mapInstance.resize();
          mapInstance.fitBounds(
            [
              [lw, ls],
              [le, ln],
            ],
            { padding: 28, duration: 0 }
          );
          setMapReady(true);
          setProjVersion((v) => v + 1);
        });
        mapInstance.on("move", () => setProjVersion((v) => v + 1));
        mapInstance.on("zoom", () => setProjVersion((v) => v + 1));
        mapInstance.on("resize", () => setProjVersion((v) => v + 1));
        mapInstance.on("error", () => {
          setProjVersion((v) => v + 1);
        });
        mapRef.current = mapInstance;
      } catch {
        if (!cancelled) setMapError(true);
      }
    }

    initMap();
    return () => {
      cancelled = true;
      if (mapInstance) {
        mapInstance.remove();
        mapRef.current = null;
      }
      setMapReady(false);
    };
  }, []);

  // Toggle dark basemap vs optical satellite reference (always keep labels visible)
  useEffect(() => {
    const m = mapRef.current;
    if (!m || !mapReady) return;
    const isSat = basemapMode === "satellite";
    if (m.getLayer("esri-sat-layer")) {
      m.setLayoutProperty("esri-sat-layer", "visibility", isSat ? "visible" : "none");
    }
    if (m.getLayer("esri-dark-layer")) {
      m.setLayoutProperty("esri-dark-layer", "visibility", isSat ? "none" : "visible");
    }
    if (m.getLayer("esri-dark-ref-layer")) {
      m.setLayoutProperty("esri-dark-ref-layer", "visibility", "visible");
      m.setPaintProperty("esri-dark-ref-layer", "raster-opacity", isSat ? 0.85 : 0.72);
    }
  }, [basemapMode, mapReady]);

  // Fit bounds cleanly when scenario switches, viewport expands, or map loads
  const fitRoiInstant = useCallback(() => {
    const m = mapRef.current;
    if (!m) return;
    m.resize();
    m.fitBounds(
      [
        [west, south],
        [east, north],
      ],
      { padding: 28, duration: 0 }
    );
    setProjVersion((v) => v + 1);
  }, [west, south, east, north]);

  const recenterMap = useCallback(() => {
    const m = mapRef.current;
    if (!m) return;
    m.resize();
    m.fitBounds(
      [
        [west, south],
        [east, north],
      ],
      { padding: 28, duration: 300 }
    );
  }, [west, south, east, north]);

  useEffect(() => {
    if (!mapReady) return;
    fitRoiInstant();
    const raf = window.requestAnimationFrame(() => fitRoiInstant());
    return () => window.cancelAnimationFrame(raf);
  }, [fitRoiInstant, isExpanded, mapReady]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
        if (mapRef.current) {
          mapRef.current.resize();
        }
        setProjVersion((v) => v + 1);
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Project lon/lat coordinates onto overlay container pixels (MapLibre or deterministic fallback)
  const projectPoint = useCallback(
    (lon: number, lat: number): { x: number; y: number } => {
      const m = mapRef.current;
      if (m && mapReady && !mapError) {
        const pt = m.project([lon, lat]);
        return { x: pt.x, y: pt.y };
      }
      const pad = 24;
      const availW = Math.max(containerSize.width - pad * 2, 120);
      const availH = Math.max(containerSize.height - pad * 2, 120);
      const size = Math.min(availW, availH);
      const left = (containerSize.width - size) * 0.5;
      const top = (containerSize.height - size) * 0.5;
      const fx = (lon - west) / Math.max(east - west, 1e-6);
      const fy = (north - lat) / Math.max(north - south, 1e-6);
      return { x: left + fx * size, y: top + fy * size };
    },
    [mapReady, mapError, containerSize, west, south, east, north, projVersion]
  );

  const screenBox: ScreenBox = useMemo(() => {
    const nw = projectPoint(west, north);
    const se = projectPoint(east, south);
    return {
      x0: nw.x,
      y0: nw.y,
      x1: se.x,
      y1: se.y,
      w: Math.max(se.x - nw.x, 1),
      h: Math.max(se.y - nw.y, 1),
    };
  }, [projectPoint, west, north, east, south]);

  const activeRasterUrl = useMemo(() => {
    if (layerMode === "hh") return analysis.images.hh_post;
    if (layerMode === "hv") return analysis.images.hv_post;
    if (layerMode === "asf_rgb") return analysis.images.asf_rgb;
    if (layerMode === "swipe") return analysis.images.hh_post;
    if (layerMode === "change") return analysis.images.change_rgb;
    return analysis.images.asf_rgb;
  }, [layerMode, analysis.images]);

  const runnerUpCorridors = useMemo(() => {
    const selectedIdx = analysis.qaoa_routing.selected_corridor_index;
    return analysis.qaoa_routing.corridor_candidates
      .filter(
        (cand) =>
          cand.corridor_id !== selectedIdx &&
          cand.path_coords &&
          cand.path_coords.length > 0
      )
      .map((cand) => ({
        corridor_id: cand.corridor_id,
        basis_state: cand.basis_state,
        points: (cand.path_coords ?? [])
          .map(([lon, lat]) => {
            const p = projectPoint(lon, lat);
            return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
          })
          .join(" "),
      }));
  }, [analysis.qaoa_routing.corridor_candidates, analysis.qaoa_routing.selected_corridor_index, projectPoint]);

  const qaoaPolyPoints = useMemo(() => {
    return analysis.qaoa_routing.optimal_path_coords
      .map(([lon, lat]) => {
        const p = projectPoint(lon, lat);
        return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
      })
      .join(" ");
  }, [analysis.qaoa_routing.optimal_path_coords, projectPoint]);

  const naivePolyPoints = useMemo(() => {
    return analysis.qaoa_routing.naive_path_coords
      .map(([lon, lat]) => {
        const p = projectPoint(lon, lat);
        return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
      })
      .join(" ");
  }, [analysis.qaoa_routing.naive_path_coords, projectPoint]);

  const handleSwipeMove = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const relX = clientX - rect.left - screenBox.x0;
      const pct = Math.max(5, Math.min(95, (relX / Math.max(screenBox.w, 1)) * 100));
      setSwipePct(pct);
    },
    [screenBox.x0, screenBox.w]
  );

  useEffect(() => {
    if (!draggingSwipe) return;
    const onMove = (e: MouseEvent) => handleSwipeMove(e.clientX);
    const onTouch = (e: TouchEvent) => {
      if (e.touches[0]) handleSwipeMove(e.touches[0].clientX);
    };
    const onUp = () => setDraggingSwipe(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("touchmove", onTouch);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("touchmove", onTouch);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchend", onUp);
    };
  }, [draggingSwipe, handleSwipeMove]);

  const atRiskBridgeKeys = useMemo(() => {
    const set = new Set<string>();
    for (const br of analysis.humanitarian_impact?.bridges_at_risk ?? []) {
      set.add(`${br.lon.toFixed(4)}:${br.lat.toFixed(4)}`);
    }
    return set;
  }, [analysis.humanitarian_impact?.bridges_at_risk]);

  const visibleSettlements = useMemo(() => {
    const raw = osmInfrastructure?.settlements ?? [];
    if (!showOsmOverlay || raw.length === 0) return [];
    const rank = (place: string) => {
      if (place === "city") return 0;
      if (place === "town") return 1;
      if (place === "suburb") return 2;
      if (place === "village") return 3;
      return 4;
    };
    const sorted = [...raw]
      .filter((st) => st.name && st.name.trim().length > 0)
      .sort((a, b) => rank(a.place) - rank(b.place));
    const placed: Array<{ osm_id: number; name: string; place: string; x: number; y: number }> = [];
    for (const st of sorted) {
      const p = projectPoint(st.lon, st.lat);
      if (p.x < 24 || p.x > containerSize.width - 24 || p.y < 24 || p.y > containerSize.height - 24) {
        continue;
      }
      const tooClose = placed.some(
        (existing) => Math.abs(existing.x - p.x) < 74 && Math.abs(existing.y - p.y) < 18
      );
      if (!tooClose) {
        placed.push({ osm_id: st.osm_id, name: st.name, place: st.place, x: p.x, y: p.y });
      }
      if (placed.length >= 14) break;
    }
    return placed;
  }, [osmInfrastructure?.settlements, showOsmOverlay, projectPoint, containerSize.width, containerSize.height]);

  const startCoord = analysis.qaoa_routing.optimal_path_coords[0];
  const endCoord =
    analysis.qaoa_routing.optimal_path_coords[
      analysis.qaoa_routing.optimal_path_coords.length - 1
    ];
  const startPt = startCoord ? projectPoint(startCoord[0], startCoord[1]) : null;
  const endPt = endCoord ? projectPoint(endCoord[0], endCoord[1]) : null;

  return (
    <div className="flex flex-col h-full bg-[#0b111a] border border-slate-800/90 rounded-lg overflow-hidden shadow-xl">
      {/* Compact Responsive Viewport Control Toolbar (zero right-edge clipping) */}
      <div className="flex flex-wrap items-center justify-between gap-1.5 px-2.5 py-1.5 bg-[#0e1623] border-b border-slate-800/90">
        <div className="flex items-center gap-1 min-w-0">
          <Layers className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <div className="flex items-center flex-wrap bg-[#070b12] border border-slate-800 rounded-sm p-0.5">
            {LAYER_OPTIONS.map((opt) => {
              const active = layerMode === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  title={opt.hint}
                  onClick={() => onLayerModeChange(opt.id)}
                  className={`px-1.5 py-0.5 text-[10px] font-mono rounded-[2px] whitespace-nowrap transition-colors ${
                    active
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                      : "text-slate-400 hover:text-slate-200 border border-transparent"
                  }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center flex-wrap gap-1">
          {/* Interactive Tile Click Mode */}
          <div className="flex items-center bg-[#070b12] border border-slate-800 rounded-sm p-0.5">
            <button
              type="button"
              onClick={() => onClickModeChange("inspect")}
              className={`flex items-center gap-1 px-1.5 py-0.5 text-[9.5px] font-mono rounded-[2px] whitespace-nowrap ${
                clickMode === "inspect"
                  ? "bg-slate-800 text-slate-100"
                  : "text-slate-400 hover:text-slate-200"
              }`}
              title="Click any tile to inspect radar signature"
            >
              <Crosshair className="w-2.5 h-2.5" />
              Inspect
            </button>
            <button
              type="button"
              onClick={() => onClickModeChange("origin")}
              className={`flex items-center gap-1 px-1.5 py-0.5 text-[9.5px] font-mono rounded-[2px] whitespace-nowrap ${
                clickMode === "origin"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                  : "text-slate-400 hover:text-cyan-200"
              }`}
              title="Pick custom rescue origin tile"
            >
              Origin
            </button>
            <button
              type="button"
              onClick={() => onClickModeChange("target")}
              className={`flex items-center gap-1 px-1.5 py-0.5 text-[9.5px] font-mono rounded-[2px] whitespace-nowrap ${
                clickMode === "target"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                  : "text-slate-400 hover:text-emerald-200"
              }`}
              title="Pick custom rescue destination tile"
            >
              <Flag className="w-2.5 h-2.5" />
              Target
            </button>
            {(customOrigin || customTarget) && (
              <button
                type="button"
                onClick={onResetRoute}
                title="Reset origin/target to default corridor endpoints"
                className="flex items-center gap-1 px-1 py-0.5 text-[9.5px] font-mono text-amber-300 hover:text-amber-200"
              >
                <RotateCcw className="w-2.5 h-2.5" />
              </button>
            )}
          </div>

          {/* Tile Grid Density Control: Hazards Only | All | Off */}
          <div
            className="flex items-center bg-[#070b12] border border-slate-800 rounded-sm p-0.5 text-[9.5px] font-mono"
            title="Show hazard-only highlights to keep satellite & roads clear, or all 36 grid tiles"
          >
            {(
              [
                { id: "hazards", label: "Hazards" },
                { id: "all", label: "Grid" },
                { id: "off", label: "Clear" },
              ] as const
            ).map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setTileDisplayMode(opt.id)}
                className={`px-1.5 py-0.5 rounded-[2px] whitespace-nowrap transition ${
                  tileDisplayMode === opt.id
                    ? "bg-slate-700 text-slate-100"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => setBasemapMode((m) => (m === "dark" ? "satellite" : "dark"))}
            title="Switch between Esri Optical Satellite + Place Labels and Dark Map"
            className={`flex items-center gap-1 px-1.5 py-0.5 text-[9.5px] font-mono border rounded-sm whitespace-nowrap transition-colors ${
              basemapMode === "satellite"
                ? "border-cyan-500/50 bg-cyan-500/15 text-cyan-200"
                : "border-slate-700 bg-slate-900/70 text-slate-300 hover:border-slate-500"
            }`}
          >
            <Globe className="w-2.5 h-2.5" />
            {basemapMode === "satellite" ? "Satellite" : "Dark"}
          </button>

          <button
            type="button"
            onClick={() => setShowOsmOverlay((v) => !v)}
            className={`flex items-center gap-1 px-1.5 py-0.5 text-[9.5px] font-mono border rounded-sm whitespace-nowrap transition-colors ${
              showOsmOverlay && osmInfrastructure?.available
                ? "border-cyan-500/45 bg-cyan-500/15 text-cyan-200"
                : "border-slate-700 bg-slate-900/60 text-slate-400"
            }`}
            title="Toggle OpenStreetMap roads, bridges, facilities & town names"
          >
            OSM
          </button>

          <button
            type="button"
            onClick={() => setShowNaiveRoute((v) => !v)}
            className={`flex items-center gap-1 px-1.5 py-0.5 text-[9.5px] font-mono border rounded-sm whitespace-nowrap transition-colors ${
              showNaiveRoute
                ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
                : "border-slate-700 bg-slate-900/60 text-slate-400"
            }`}
            title="Toggle Direct Hazard-Blind baseline route"
          >
            <Route className="w-2.5 h-2.5" />
            Direct
          </button>

          <button
            type="button"
            onClick={recenterMap}
            title="Fit map onto NASA SAR ROI footprint"
            className="flex items-center gap-1 px-1.5 py-0.5 text-[9.5px] font-mono border border-slate-700 bg-slate-900/70 text-slate-300 hover:border-cyan-500/50 hover:text-cyan-200 rounded-sm whitespace-nowrap"
          >
            <LocateFixed className="w-2.5 h-2.5" />
            Fit
          </button>

          {onToggleExpand && (
            <button
              type="button"
              onClick={onToggleExpand}
              title={isExpanded ? "Return to split view" : "Expand map across full width"}
              className={`flex items-center gap-1 px-1.5 py-0.5 text-[9.5px] font-mono border rounded-sm whitespace-nowrap transition-colors ${
                isExpanded
                  ? "border-emerald-500/50 bg-emerald-500/15 text-emerald-200"
                  : "border-slate-700 bg-slate-900/70 text-slate-200 hover:border-cyan-500/50"
              }`}
            >
              {isExpanded ? <Minimize2 className="w-2.5 h-2.5" /> : <Maximize2 className="w-2.5 h-2.5" />}
              {isExpanded ? "Split" : "Expand"}
            </button>
          )}
        </div>
      </div>

      {/* Interactive Geo-Canvas */}
      <div
        ref={containerRef}
        onMouseMove={(e) => {
          const m = mapRef.current;
          const el = containerRef.current;
          if (!m || !mapReady || !el) return;
          const rect = el.getBoundingClientRect();
          const ll = m.unproject([e.clientX - rect.left, e.clientY - rect.top]);
          setCursorCoords({ lon: ll.lng, lat: ll.lat });
        }}
        onMouseLeave={() => setCursorCoords(null)}
        className="relative flex-1 min-h-0 sar-grid-backdrop select-none overflow-hidden"
      >
        {/* MapLibre GL Basemap */}
        <div ref={mapContainerRef} className="absolute inset-0 w-full h-full" />

        {/* Projected SAR Raster Overlay */}
        <div
          className="absolute pointer-events-none"
          style={{
            left: `${screenBox.x0}px`,
            top: `${screenBox.y0}px`,
            width: `${screenBox.w}px`,
            height: `${screenBox.h}px`,
          }}
        >
          {/* Post-event or selected SAR composite */}
          <img
            src={activeRasterUrl}
            alt="SAR Backscatter Layer"
            className="absolute inset-0 w-full h-full object-fill"
            style={{
              opacity:
                layerMode === "segmentation"
                  ? tileDisplayMode === "off"
                    ? 0.12
                    : basemapMode === "satellite"
                      ? 0.28
                      : 0.42
                  : 0.88,
              imageRendering: "auto",
            }}
          />

          {/* Pre-event HH layer clipped by swipe percentage */}
          {layerMode === "swipe" && (
            <>
              <img
                src={analysis.images.hh_pre}
                alt="Pre-Event SAR Backscatter"
                className="absolute inset-0 w-full h-full object-fill"
                style={{
                  opacity: 0.94,
                  clipPath: `inset(0 ${100 - swipePct}% 0 0)`,
                }}
              />
              {/* Swipe Divider Line & Drag Handle */}
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.9)] pointer-events-auto cursor-ew-resize z-30"
                style={{ left: `${swipePct}%` }}
                onMouseDown={() => setDraggingSwipe(true)}
                onTouchStart={() => setDraggingSwipe(true)}
              >
                <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-7 h-7 rounded-full bg-[#070b12] border border-cyan-400 flex items-center justify-center shadow-lg">
                  <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-300" />
                </div>
              </div>
              <div className="absolute top-2 left-2 px-1.5 py-0.5 bg-[#070b12]/85 border border-slate-700 text-[10px] font-mono text-slate-200 rounded-[2px]">
                PRE: {analysis.scenario.pre_date}
              </div>
              <div className="absolute top-2 right-2 px-1.5 py-0.5 bg-[#070b12]/85 border border-cyan-500/50 text-[10px] font-mono text-cyan-300 rounded-[2px]">
                POST: {analysis.scenario.post_date}
              </div>
            </>
          )}
        </div>

        {/* Synchronized 6x6 Interactive Grid + QAOA Corridor SVG Overlay */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none z-20">
          <defs>
            <filter id="emerald-glow" x="-25%" y="-25%" width="150%" height="150%">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Cached OpenStreetMap Infrastructure Overlay (Roads, Bridges & Named Settlements) */}
          {showOsmOverlay && osmInfrastructure?.available && (
            <g opacity={0.92}>
              {/* Dark casing underneath OSM roads so roads read crisply over satellite imagery */}
              {osmInfrastructure.roads.map((road) => {
                if (!road.coords || road.coords.length < 2) return null;
                const pts = road.coords
                  .map(([lon, lat]) => {
                    const p = projectPoint(lon, lat);
                    return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
                  })
                  .join(" ");
                return (
                  <g key={`osm-r-${road.osm_id}`}>
                    <polyline
                      points={pts}
                      fill="none"
                      stroke="rgba(6, 9, 17, 0.72)"
                      strokeWidth={road.bridge ? 3.8 : 2.1}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <polyline
                      points={pts}
                      fill="none"
                      stroke={road.bridge ? "#f43f5e" : "rgba(203, 213, 225, 0.56)"}
                      strokeWidth={road.bridge ? 2.1 : 1.05}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </g>
                );
              })}

              {/* OSM Bridges (highlight flooded / at-risk bridges) */}
              {osmInfrastructure.bridges.map((br) => {
                const p = projectPoint(br.lon, br.lat);
                const isAtRisk = atRiskBridgeKeys.has(`${br.lon.toFixed(4)}:${br.lat.toFixed(4)}`);
                return (
                  <g key={`osm-br-${br.osm_id}`}>
                    {isAtRisk && (
                      <circle cx={p.x} cy={p.y} r={8} fill="rgba(244, 63, 94, 0.22)" stroke="#fb7185" strokeWidth={0.8} />
                    )}
                    <polygon
                      points={`${p.x},${p.y - (isAtRisk ? 5 : 3.8)} ${p.x + (isAtRisk ? 5 : 3.8)},${p.y} ${p.x},${p.y + (isAtRisk ? 5 : 3.8)} ${p.x - (isAtRisk ? 5 : 3.8)},${p.y}`}
                      fill={isAtRisk ? "#f43f5e" : "#fb923c"}
                      stroke="#070b12"
                      strokeWidth={1}
                    />
                  </g>
                );
              })}

              {/* OSM Critical Facilities (Hospitals, Clinics, Shelters, Fire Stations, Schools) */}
              {(osmInfrastructure.facilities ?? []).map((fac) => {
                const p = projectPoint(fac.lon, fac.lat);
                const isMed = fac.amenity === "hospital" || fac.amenity === "clinic" || fac.amenity === "doctors";
                const isEmergency = fac.amenity === "fire_station" || fac.amenity === "police" || fac.amenity === "shelter";
                const color = isMed ? "#2dd4bf" : isEmergency ? "#fb7185" : "#a78bfa";
                return (
                  <g key={`osm-fac-${fac.osm_id}`}>
                    <rect
                      x={p.x - 3.2}
                      y={p.y - 3.2}
                      width={6.4}
                      height={6.4}
                      rx={1.4}
                      fill={color}
                      stroke="#060911"
                      strokeWidth={1.1}
                    />
                  </g>
                );
              })}

              {/* Named Towns & Villages from OSM (haloed, collision-filtered) */}
              {visibleSettlements.map((st) => {
                const isTownOrCity = st.place === "city" || st.place === "town";
                return (
                  <g key={`osm-st-${st.osm_id}`}>
                    <circle
                      cx={st.x}
                      cy={st.y}
                      r={isTownOrCity ? 3.2 : 2.2}
                      fill={isTownOrCity ? "#f8fafc" : "#cbd5e1"}
                      stroke="#060911"
                      strokeWidth={1.1}
                    />
                    <text
                      x={st.x + 6}
                      y={st.y + 3}
                      fill={isTownOrCity ? "#f8fafc" : "#cbd5e1"}
                      fontSize={isTownOrCity ? "10.5" : "9"}
                      fontWeight={isTownOrCity ? "600" : "500"}
                      fontFamily="var(--font-sans), sans-serif"
                      stroke="#060911"
                      strokeWidth="3"
                      paintOrder="stroke"
                    >
                      {st.name}
                    </text>
                  </g>
                );
              })}
            </g>
          )}

          {/* 36 Interactive SAR Tiles */}
          {analysis.tiles.map((tile) => {
            const [tw, ts, te, tn] = tile.bounds;
            const p0 = projectPoint(tw, tn);
            const p1 = projectPoint(te, ts);
            const x = p0.x;
            const y = p0.y;
            const w = Math.max(p1.x - p0.x, 1);
            const h = Math.max(p1.y - p0.y, 1);

            const activeCls =
              classificationMode === "classical" ? tile.classical_class : tile.quantum_class;
            const disagrees = tile.quantum_class !== tile.classical_class;
            const st =
              classificationMode === "disagreement"
                ? disagrees
                  ? {
                      name: "Model Disagreement",
                      stroke: "rgba(251, 191, 36, 0.95)",
                      fill: "rgba(244, 63, 94, 0.36)",
                      badge: "#fbbf24",
                    }
                  : {
                      name: "Consensus",
                      stroke: "rgba(71, 85, 105, 0.28)",
                      fill: "rgba(15, 23, 42, 0.04)",
                      badge: "#64748b",
                    }
                : CLASS_STYLE[activeCls] ?? CLASS_STYLE[0];

            const isSelected = tile.tile_id === selectedTileId;
            const isHoveredTile = hoveredTile?.tile_id === tile.tile_id;
            const isSegMode = layerMode === "segmentation";
            const isNominalTile =
              classificationMode === "disagreement" ? !disagrees : activeCls === 0;
            const hideNonHazardFill = tileDisplayMode === "hazards" && isNominalTile;
            const hideAllTiles = tileDisplayMode === "off";
            const highEntropy = Boolean(tile.abstention_flag) || (tile.quantum_entropy ?? 0) >= 0.72;
            const hasPhysicalHazard =
              (tile.water_pct ?? 0) >= 15.0 || (tile.burn_pct ?? 0) >= 18.0;

            const fillColor = hideAllTiles
              ? "transparent"
              : !isSegMode
                ? "rgba(7, 11, 18, 0.02)"
                : hideNonHazardFill
                  ? isHoveredTile
                    ? "rgba(148, 163, 184, 0.14)"
                    : "rgba(15, 23, 42, 0.01)"
                  : isNominalTile
                    ? "rgba(15, 23, 42, 0.09)"
                    : st.fill;

            const strokeColor = isSelected
              ? "#22d3ee"
              : hideAllTiles
                ? isHoveredTile
                  ? "rgba(148, 163, 184, 0.55)"
                  : "transparent"
                : hideNonHazardFill
                  ? "rgba(148, 163, 184, 0.22)"
                  : st.stroke;

            return (
              <g
                key={tile.tile_id}
                className={`pointer-events-auto ${
                  clickMode === "inspect" ? "cursor-pointer" : "cursor-crosshair"
                }`}
                onClick={() => {
                  if (clickMode === "origin") {
                    onPickOrigin([tile.row, tile.col]);
                    onClickModeChange("inspect");
                  } else if (clickMode === "target") {
                    onPickTarget([tile.row, tile.col]);
                    onClickModeChange("inspect");
                  } else {
                    onSelectTile(tile.tile_id);
                  }
                }}
                onMouseEnter={() => setHoveredTile(tile)}
                onMouseLeave={() => setHoveredTile(null)}
              >
                <rect
                  x={x}
                  y={y}
                  width={w}
                  height={h}
                  fill={fillColor}
                  stroke={strokeColor}
                  strokeWidth={isSelected ? 2 : hideNonHazardFill ? 0.65 : isSegMode ? 1 : 0.6}
                  strokeOpacity={isSelected ? 1 : isSegMode ? 0.85 : 0.45}
                />

                {/* Physical flood/burn anomaly accent bar */}
                {!hideAllTiles && hasPhysicalHazard && (
                  <rect
                    x={x + 1.5}
                    y={y + 1.5}
                    width={Math.max(w - 3, 1)}
                    height={2}
                    fill={(tile.water_pct ?? 0) >= 15.0 ? "#38bdf8" : "#f97316"}
                    opacity={0.82}
                  />
                )}

                {/* Tile coordinate label (shown cleanly on active/selected/hazard tiles) */}
                {!hideAllTiles && w > 50 && (!hideNonHazardFill || isSelected || isHoveredTile) && (
                  <text
                    x={x + 5}
                    y={y + 13}
                    fill="#cbd5e1"
                    fontSize="8.5"
                    fontFamily="var(--font-mono), monospace"
                    stroke="#060911"
                    strokeWidth="2.5"
                    paintOrder="stroke"
                    opacity={isSelected ? 1 : 0.8}
                  >
                    T{tile.tile_id}
                  </text>
                )}

                {!hideAllTiles && highEntropy && w > 38 && (
                  <circle
                    cx={x + w - 7}
                    cy={y + 8}
                    r={2.5}
                    fill="#fbbf24"
                    opacity={0.9}
                  />
                )}

                {(() => {
                  const ver = tileVerifications?.[String(tile.tile_id)];
                  if (!ver || ver.status === "unreviewed" || hideAllTiles) return null;
                  const badgeBg =
                    ver.status === "confirmed_hazard"
                      ? "#10b981"
                      : ver.status === "needs_inspection"
                        ? "#f59e0b"
                        : "#64748b";
                  const badgeMark =
                    ver.status === "confirmed_hazard" ? "✓" : ver.status === "needs_inspection" ? "?" : "✕";
                  return (
                    <g transform={`translate(${x + 4}, ${y + Math.max(18, h - 15)})`}>
                      <rect x="0" y="0" width="12" height="11" rx="2" fill={badgeBg} stroke="#060911" strokeWidth="1" />
                      <text x="6" y="8.2" textAnchor="middle" fill="#060911" fontSize="8" fontWeight="800" fontFamily="monospace">
                        {badgeMark}
                      </text>
                    </g>
                  );
                })()}

                {/* Active target reticle corner markers */}
                {isSelected && (
                  <path
                    d={`M ${x + 3} ${y + 10} L ${x + 3} ${y + 3} L ${x + 10} ${y + 3}
                        M ${x + w - 10} ${y + 3} L ${x + w - 3} ${y + 3} L ${x + w - 3} ${y + 10}
                        M ${x + 3} ${y + h - 10} L ${x + 3} ${y + h - 3} L ${x + 10} ${y + h - 3}
                        M ${x + w - 10} ${y + h - 3} L ${x + w - 3} ${y + h - 3} L ${x + w - 3} ${y + h - 10}`}
                    fill="none"
                    stroke="#22d3ee"
                    strokeWidth="2"
                  />
                )}
              </g>
            );
          })}

          {/* Runner-up QAOA Corridor Candidates */}
          {runnerUpCorridors.map((cand) => {
            const isHovered = hoveredCorridorId === cand.corridor_id;
            return (
              <polyline
                key={`cand-${cand.corridor_id}`}
                points={cand.points}
                fill="none"
                stroke={isHovered ? "#38bdf8" : "#22d3ee"}
                strokeWidth={isHovered ? "2.8" : "1.5"}
                strokeDasharray={isHovered ? "none" : "4 4"}
                strokeOpacity={isHovered ? "0.95" : "0.42"}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            );
          })}

          {/* Hazard-Blind Direct Baseline Route (Dashed Warning) */}
          {showNaiveRoute && naivePolyPoints && (
            <polyline
              points={naivePolyPoints}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="2.1"
              strokeDasharray="6 5"
              strokeOpacity="0.88"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* QAOA Optimal Emergency Corridor (Double-stroked clean road line, no dot spam) */}
          {qaoaPolyPoints && (
            <>
              <polyline
                points={qaoaPolyPoints}
                fill="none"
                stroke="#042f2e"
                strokeWidth="6.5"
                strokeOpacity="0.88"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={qaoaPolyPoints}
                fill="none"
                stroke="#10b981"
                strokeWidth="3.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                filter="url(#emerald-glow)"
              />
            </>
          )}

          {/* Clean Start & Destination Beacons + Clamped Labels */}
          {startPt && (
            <g>
              <circle cx={startPt.x} cy={startPt.y} r={9} fill="rgba(34, 211, 238, 0.2)" stroke="#22d3ee" strokeWidth={1.2} />
              <circle cx={startPt.x} cy={startPt.y} r={4.5} fill="#22d3ee" stroke="#060911" strokeWidth={1.5} />
              <g
                transform={`translate(${Math.max(28, Math.min(containerSize.width - 28, startPt.x))}, ${Math.max(20, startPt.y - 14)})`}
              >
                <rect x="-22" y="-10" width="44" height="14" rx="2" fill="#070b12" stroke="#22d3ee" strokeWidth="1" />
                <text x="0" y="0" textAnchor="middle" fill="#67e8f9" fontSize="8.5" fontFamily="monospace" fontWeight="600">
                  ORIGIN
                </text>
              </g>
            </g>
          )}
          {endPt && (
            <g>
              <circle cx={endPt.x} cy={endPt.y} r={9} fill="rgba(16, 185, 129, 0.22)" stroke="#10b981" strokeWidth={1.2} />
              <circle cx={endPt.x} cy={endPt.y} r={4.5} fill="#10b981" stroke="#060911" strokeWidth={1.5} />
              <g
                transform={`translate(${Math.max(30, Math.min(containerSize.width - 30, endPt.x))}, ${Math.min(containerSize.height - 14, endPt.y + 19)})`}
              >
                <rect x="-25" y="-10" width="50" height="14" rx="2" fill="#070b12" stroke="#10b981" strokeWidth="1" />
                <text x="0" y="0" textAnchor="middle" fill="#6ee7b7" fontSize="8.5" fontFamily="monospace" fontWeight="600">
                  RESCUE
                </text>
              </g>
            </g>
          )}
        </svg>

        {/* Compact Top-Left 1-Line ROI Pill */}
        <div className="absolute top-2.5 left-2.5 z-30 flex flex-col gap-1.5 pointer-events-none">
          <div className="flex items-center gap-2 bg-[#070b12]/88 border border-slate-800/90 px-2.5 py-1 rounded-sm backdrop-blur-sm">
            <span className="text-[10px] font-mono uppercase tracking-wider text-cyan-300 font-semibold">
              {analysis.scenario.region}
            </span>
            <span
              className={`px-1 py-0.2 rounded-[2px] text-[9px] font-mono border ${
                analysis.data_provenance?.is_real_satellite_data !== false
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                  : "border-amber-500/40 bg-amber-500/10 text-amber-300"
              }`}
            >
              {analysis.data_provenance?.is_real_satellite_data !== false ? "REAL S1 RTC" : "NISAR SIM"}
            </span>
            <span className="text-[10px] font-mono text-slate-300 tabular-nums hidden sm:inline">
              · L={analysis.scenario.look_count} · α={(analysis.qaoa_routing.risk_aversion ?? 1.0).toFixed(2)}
            </span>
          </div>

          {clickMode !== "inspect" && (
            <div className="bg-cyan-950/90 border border-cyan-400/60 px-2.5 py-1 rounded-sm text-[10px] font-mono text-cyan-200">
              Click any tile to set {clickMode === "origin" ? "ORIGIN" : "TARGET"} waypoint
            </div>
          )}
        </div>

        {/* Non-blocking Bottom-Left Floating Hovered Tile Inspector + Compact Class Legend */}
        <div className="absolute bottom-2.5 left-2.5 z-30 flex flex-col gap-1.5 pointer-events-none max-w-[92%]">
          {hoveredTile && (
            <div className="bg-[#070b12]/95 border border-cyan-500/45 px-2.5 py-1.5 rounded-sm text-[10px] font-mono tabular-nums backdrop-blur-sm shadow-lg w-fit">
              <div className="text-cyan-300 font-semibold">
                TILE T{hoveredTile.tile_id} [{hoveredTile.row},{hoveredTile.col}] ·{" "}
                {classificationMode === "classical"
                  ? `C: ${hoveredTile.classical_class_name}`
                  : `Q: ${hoveredTile.quantum_class_name}`}
                {hoveredTile.abstention_flag ? " · VERIFY" : ""}
              </div>
              <div className="text-slate-300">
                {analysis.data_provenance?.co_pol_label ?? "Co"} {hoveredTile.hh_db} dB |{" "}
                {analysis.data_provenance?.cross_pol_label ?? "Cr"} {hoveredTile.hv_db} dB | Δ{" "}
                {(hoveredTile.delta_hh_db ?? 0) > 0 ? "+" : ""}
                {hoveredTile.delta_hh_db ?? 0} dB
              </div>
              {((hoveredTile.water_pct ?? 0) > 0 || (hoveredTile.burn_pct ?? 0) > 0) && (
                <div className="text-slate-400">
                  Water {hoveredTile.water_pct ?? 0}% | Burn {hoveredTile.burn_pct ?? 0}% |{" "}
                  {hoveredTile.physical_anomaly ?? "Nominal"}
                </div>
              )}
            </div>
          )}

          <div className="bg-[#070b12]/88 border border-slate-800/90 px-2.5 py-1.5 rounded-sm backdrop-blur-sm">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[9.5px] font-mono">
              {Object.entries(CLASS_STYLE).map(([cls, info]) => (
                <div key={cls} className="flex items-center gap-1">
                  <span
                    className="w-2 h-2 rounded-[1px] inline-block"
                    style={{ backgroundColor: info.badge }}
                  />
                  <span className="text-slate-300">{info.name}</span>
                </div>
              ))}
              <span className="text-slate-700">|</span>
              <div className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-emerald-400 inline-block" />
                <span className="text-emerald-300">Route</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-3 h-0.5 border-t border-dashed border-cyan-400 inline-block" />
                <span className="text-cyan-300">Alt</span>
              </div>
              {showNaiveRoute && (
                <div className="flex items-center gap-1">
                  <span className="w-3 h-0.5 border-t border-dashed border-amber-400 inline-block" />
                  <span className="text-amber-300">Direct</span>
                </div>
              )}
              {showOsmOverlay && (osmInfrastructure?.facilities?.length ?? 0) > 0 && (
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-[1px] bg-teal-400 inline-block" />
                  <span className="text-teal-300">Facilities</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Live Cursor Coordinates Readout (clears bottom-right ScaleControl & Attribution) */}
        {cursorCoords && (
          <div className="hidden sm:block absolute bottom-[74px] right-2.5 z-30 pointer-events-none bg-[#070b12]/90 border border-slate-800 px-2 py-0.5 rounded-sm text-[10px] font-mono text-slate-300 tabular-nums shadow">
            {cursorCoords.lat.toFixed(4)}°N, {cursorCoords.lon.toFixed(4)}°E
          </div>
        )}
      </div>
    </div>
  );
}
