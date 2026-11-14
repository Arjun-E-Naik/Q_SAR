export type SarClassId = 0 | 1 | 2 | 3;

export type MapClassificationMode = "hybrid" | "classical" | "disagreement";

export type LayerMode = "segmentation" | "change" | "hh" | "hv" | "asf_rgb" | "swipe";

export type MapClickMode = "inspect" | "origin" | "target";

export interface TileExplanation {
  what_changed: string;
  supporting_evidence: string;
  confidence_note: string;
}

export interface TilePrediction {
  tile_id: number;
  row: number;
  col: number;
  center_lon: number;
  center_lat: number;
  bounds: [number, number, number, number];
  hh_db: number;
  hv_db: number;
  pre_hh_db?: number;
  pre_hv_db?: number;
  delta_hh_db?: number;
  delta_hv_db?: number;
  cross_pol_ratio_db: number;
  rvi: number;
  delta_rvi?: number;
  water_pct: number;
  permanent_water_pct?: number;
  new_flood_pct?: number;
  burn_pct: number;
  double_bounce_pct: number;
  crop_change_pct?: number;
  reference_class: number;
  radiometric_reference_label?: number;
  verified_ground_truth_label?: number | null;
  reference_type?: string;
  quantum_class: SarClassId;
  quantum_label?: string;
  quantum_class_name: string;
  quantum_confidence?: number;
  quantum_probs: [number, number, number, number];
  classical_class: SarClassId;
  classical_label?: string;
  classical_class_name: string;
  classical_confidence?: number;
  classical_probs: [number, number, number, number];
  models_disagree?: boolean;
  confidence_delta: number;
  quantum_entropy: number;
  abstention_flag?: boolean;
  uncertainty_tier?: string;
  physical_anomaly: string;
  area_km2?: number;
  osm_road_count?: number;
  osm_bridge_count?: number;
  osm_facility_count?: number;
  tile_explanation?: TileExplanation;
  // Optional legacy backend fields (kept optional for API compatibility; not rendered in UI)
  quantum_angles_rad?: number[];
  pauli_z_expvals?: number[];
  statevector_probs?: number[];
  bloch_vectors?: number[][];
}

export interface CorridorCandidate {
  corridor_id: number;
  name?: string;
  strategy_tag?: string;
  basis_state?: string;
  qaoa_prob: number;
  conditional_onehot_prob?: number;
  candidate_normalized_score?: number;
  hazard_cost: number;
  hazard_exposure?: number;
  uncertainty_exposure?: number;
  bridge_crossings?: number;
  hops: number;
  distance_km: number;
  path_nodes: [number, number][];
  path_coords: [number, number][];
}

export interface RouteAlgorithmBenchmark {
  distance_km: number;
  hazard_exposure: number;
  uncertainty_exposure: number;
  bridge_crossings: number;
  total_objective_cost: number;
  runtime_ms: number;
  strategy_name?: string;
}

export interface QaoaRoutingResult {
  qaoa_device: string;
  routing_mode?: "osm_road_network" | "tactical_grid_fallback";
  routing_formulation?: string;
  mixer_type?: string;
  approval_status?: RouteApprovalStatus;
  osm_roads_used_count?: number;
  p_layers: number;
  n_qubits: number;
  start_node: [number, number];
  end_node: [number, number];
  optimal_path_nodes: [number, number][];
  optimal_path_coords: [number, number][];
  naive_path_nodes: [number, number][];
  naive_path_coords: [number, number][];
  dijkstra_path_coords?: [number, number][];
  selected_corridor_index: number;
  selected_strategy_name?: string;
  optimal_bitstring?: string;
  expectation_energy: number;
  approximation_ratio: number;
  valid_onehot_prob_mass?: number;
  invalid_bitstring_prob?: number;
  x_mixer_invalid_leakage?: number;
  conditional_onehot_probs?: number[];
  candidate_gen_time_ms?: number;
  quantum_opt_time_ms?: number;
  best_classical_candidate_index?: number;
  best_classical_candidate_cost?: number;
  risk_aversion: number;
  qaoa_hazard_cost: number;
  exact_dijkstra_cost: number;
  naive_hazard_cost: number;
  hazard_reduction_pct: number;
  optimal_distance_km: number;
  routing_comparison?: {
    classical_direct: RouteAlgorithmBenchmark;
    classical_dijkstra: RouteAlgorithmBenchmark;
    classical_astar: RouteAlgorithmBenchmark;
    classical_enumeration?: RouteAlgorithmBenchmark;
    qaoa_selected: RouteAlgorithmBenchmark;
  };
  operational_caution?: string;
  corridor_candidates: CorridorCandidate[];
  geojson: GeoJSON.FeatureCollection;
}

export interface DataProvenance {
  data_source: "real_s1_rtc" | "nisar_sim";
  data_source_mode?: "live_nisar" | "live_sentinel1" | "simulation" | "cached_demo";
  data_source_tier?: string;
  is_real_satellite_data: boolean;
  ground_truth_available?: boolean;
  ground_truth_category?: string;
  ground_truth_type?: string;
  is_verified_manual_ground_truth?: boolean;
  metric_display_mode?: "accuracy" | "heuristic_agreement";
  sensor_label: string;
  sensor_band?: string;
  polarization_channels?: string[];
  co_pol_label?: string;
  cross_pol_label?: string;
  polarization_basis: string;
  crs: string;
  relative_orbit: number | null;
  pre_item_ids: string[];
  post_item_ids: string[];
  event_note: string;
  speckle_mode: string;
}

export interface ValidationClassReport {
  class_id: number;
  class_name: string;
  support: number;
  precision: number;
  recall: number;
  f1: number;
  iou: number;
}

export interface ValidationModelReport {
  samples: number;
  accuracy: number;
  macro_f1: number;
  mean_iou: number;
  ece: number;
  normal_tile_fpr: number;
  abstention_rate: number;
  selective_accuracy: number;
  confusion_matrix: number[][];
  per_class: ValidationClassReport[];
  latency_ms?: number;
}

export interface ValidationTrackPair {
  hybrid_quantum: ValidationModelReport;
  matched_classical_head: ValidationModelReport;
  standalone_classical_cnn: ValidationModelReport;
}

export interface BenchmarkMetrics {
  bitemporal_version?: number;
  source: string;
  quantum_device?: string;
  weight_source?: string;
  note?: string;
  is_verified_manual_ground_truth?: boolean;
  quantum_accuracy: number;
  classical_accuracy: number;
  standalone_classical_accuracy?: number;
  quantum_macro_f1?: number;
  classical_macro_f1?: number;
  quantum_1look_accuracy?: number;
  classical_1look_accuracy?: number;
  standalone_classical_1look_accuracy?: number;
  real_s1_quantum_heuristic_agreement?: number;
  real_s1_classical_heuristic_agreement?: number;
  real_s1_quantum_heuristic_macro_f1?: number;
  real_s1_classical_heuristic_macro_f1?: number;
  real_s1_quantum_accuracy?: number;
  real_s1_classical_accuracy?: number;
  real_s1_quantum_macro_f1?: number;
  real_s1_classical_macro_f1?: number;
  quantum_params: number;
  total_params: number;
  tracks?: {
    synthetic_nisar_sim: Record<string, ValidationTrackPair>;
    domain_shift_holdout: Record<string, ValidationTrackPair>;
    real_s1_rtc_zeroshot: {
      overall: ValidationTrackPair;
      per_scenario: Record<string, ValidationTrackPair>;
    };
  };
}

export type TileVerificationStatus = "unreviewed" | "confirmed_hazard" | "false_alarm" | "needs_inspection";

export type RouteApprovalStatus = "pending_human_approval" | "approved_for_recon" | "rejected";

export interface TileVerificationRecord {
  status: TileVerificationStatus;
  verified_label?: number | null;
  note?: string;
  updated_at?: string;
}

export interface ScenarioVerificationState {
  scenario_id?: string;
  tiles: Record<string, TileVerificationRecord>;
  route_approval: {
    status: RouteApprovalStatus;
    note?: string;
    updated_at?: string | null;
  };
}

export interface DataQualityReport {
  status: "PASS" | "WARN" | "UNSUITABLE";
  crs: string;
  crs_aligned: boolean;
  shape_aligned: boolean;
  transform_aligned: boolean;
  temporal_pairs_complete: boolean;
  georeferenced_reprojection: boolean;
  valid_pixel_pct: number;
  radiometric_in_bounds: boolean;
  db_range: [number, number];
  unsuitable_reason: string | null;
}

export interface PriorityQueueItem {
  tile_id: number;
  row: number;
  col: number;
  priority_tier: "CRITICAL" | "HIGH" | "MEDIUM" | "MONITOR";
  priority_score: number;
  hazard_class: number;
  hazard_label: string;
  osm_bridges: number;
  osm_roads: number;
  osm_facilities: number;
  estimated_pop: number;
  abstention_flag: boolean;
  reason: string;
}

export interface HumanitarianImpact {
  total_scene_area_km2: number;
  affected_area_km2: number;
  flooded_area_km2: number;
  burn_scar_area_km2: number;
  crop_stress_area_km2: number;
  agricultural_area_ha: number;
  new_flood_sub_km2: number;
  permanent_water_sub_km2: number;
  double_bounce_sub_km2: number;
  osm_available: boolean;
  total_osm_roads_km: number;
  roads_flood_exposed_km: number;
  roads_fire_exposed_km: number;
  roads_at_risk_km: number;
  bridges_total: number;
  bridges_at_risk_count: number;
  bridges_at_risk: Array<{
    name: string;
    highway: string;
    tile_id: number;
    hazard_class: number;
    lon: number;
    lat: number;
  }>;
  settlements_exposed_count: number;
  settlements_exposed: Array<{
    name: string;
    place: string;
    tile_id: number;
    hazard_class: number;
    estimated_pop: number;
    lon: number;
    lat: number;
  }>;
  facilities_total?: number;
  facilities_exposed_count?: number;
  facilities_exposed?: Array<{
    name: string;
    amenity: string;
    tile_id: number;
    hazard_class: number;
    lon: number;
    lat: number;
  }>;
  estimated_exposed_population: number;
  population_method: string;
  critical_chokepoint_tiles: Array<{
    tile_id: number;
    row: number;
    col: number;
    hazard_class: number;
    hazard_label: string;
    osm_roads: number;
    osm_bridges: number;
    confidence: number;
  }>;
  priority_queue?: PriorityQueueItem[];
  uncertain_tiles_count: number;
  dominant_scattering_shift: string;
}

export interface ExplainableBrief {
  headline: string;
  what_changed: string;
  supporting_channels: string;
  confidence_summary: string;
  next_verification_action: string;
  flagged_roads_sample?: Array<{
    name: string;
    highway: string;
    bridge: boolean;
    hazard_class: number;
    length_km: number;
    tile_ids: number[];
  }>;
}

export interface OsmRoadSegment {
  osm_id: number;
  name: string;
  highway: string;
  bridge: boolean;
  coords: [number, number][];
}

export interface OsmInfrastructureSummary {
  available: boolean;
  source: string;
  counts: {
    roads?: number;
    bridges?: number;
    settlements?: number;
    waterways?: number;
    facilities?: number;
  };
  roads: OsmRoadSegment[];
  bridges: Array<{
    osm_id: number;
    name: string;
    highway: string;
    lon: number;
    lat: number;
  }>;
  settlements: Array<{
    osm_id: number;
    name: string;
    place: string;
    lon: number;
    lat: number;
    population?: string | null;
  }>;
  facilities?: Array<{
    osm_id: number;
    name: string;
    amenity: string;
    lon: number;
    lat: number;
  }>;
}

export interface AnalysisResponse {
  scenario: {
    scenario_id: string;
    title: string;
    subtitle: string;
    region: string;
    sensor_mode: string;
    nasa_collections: string[];
    bbox: [number, number, number, number];
    pre_date: string;
    post_date: string;
    tile_km: number;
    look_count: number;
    data_source: "real_s1_rtc" | "nisar_sim";
    risk_aversion: number;
  };
  data_provenance?: DataProvenance;
  data_quality?: DataQualityReport;
  nasa_metadata: {
    status: string;
    queried_collection: string;
    granule_count: number;
    granules: Array<{
      id: string;
      time_start: string;
      time_end: string;
      size_mb?: number;
      collection?: string;
    }>;
  };
  timing_ms: {
    quantum_inference_ms: number;
    statevector_telemetry_ms: number;
    qaoa_routing_ms: number;
    total_pipeline_ms: number;
  };
  model_status: {
    quantum_device: string;
    weight_source: string;
    ground_truth_available?: boolean;
    ground_truth_category?: string;
    ground_truth_type?: string;
    is_verified_manual_ground_truth?: boolean;
    metric_label?: string;
    is_heuristic_agreement?: boolean;
    scene_quantum_heuristic_agreement?: number | null;
    scene_classical_heuristic_agreement?: number | null;
    scene_quantum_accuracy: number | null;
    scene_classical_accuracy: number | null;
    scene_quantum_macro_f1?: number | null;
    scene_classical_macro_f1?: number | null;
    accuracy_delta: number | null;
    quantum_classical_agreement: number;
    disagreement_count?: number;
    abstention_count?: number;
    mean_quantum_entropy?: number;
  };
  class_names: Record<string, string>;
  affected_area_km2: Record<string, number>;
  humanitarian_impact?: HumanitarianImpact;
  explainable_brief?: ExplainableBrief;
  osm_infrastructure?: OsmInfrastructureSummary;
  tiles: TilePrediction[];
  qaoa_routing: QaoaRoutingResult;
  images: {
    hh_pre: string;
    hh_post: string;
    hv_post: string;
    asf_rgb: string;
    change_rgb: string;
    [key: string]: string;
  };
  benchmarks: BenchmarkMetrics;
}

export type AnalyzeResponse = AnalysisResponse;

export interface ScenarioSummary {
  scenario_id: string;
  title: string;
  subtitle: string;
  region: string;
  sensor_mode: string;
  nasa_collections: string[];
  bbox: [number, number, number, number];
  pre_date: string;
  post_date: string;
  start_node: [number, number];
  end_node: [number, number];
  start_label: string;
  end_label: string;
  tile_km: number;
  has_real_s1_cache?: boolean;
  is_custom?: boolean;
}

export interface GeocodeCandidate {
  label: string;
  short_name: string;
  lat: number;
  lon: number;
  type: string;
  source: string;
}

export interface PrepareLocationResponse {
  status: string;
  scenario: ScenarioSummary;
  osm_counts: {
    roads: number;
    bridges: number;
    settlements: number;
    facilities: number;
    available: boolean;
  };
  stac_meta: {
    orbit: number | string | null;
    pre_datetime: string | null;
    post_datetime: string | null;
    pre_items: string[];
    post_items: string[];
  };
}
