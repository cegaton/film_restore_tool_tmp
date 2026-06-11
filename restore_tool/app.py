"""
app_full_replacement_v10_faint_scratch_large_specks.py

Full replacement app.py for the film_restore_tool GUI.

This version keeps the compact diagnostic workflow, including:
    - Dust
    - Scratch
    - Sharp Width
    - Soft Scratch
    - Soft Scratch Repair toggle
    - Max Soft Width
    - Frames
    - Resume / rewrite render mode
    - Overlay checkbox with cached temporal preview
    - Session settings persistence
    - Current frame name display
    - Rendered / remaining counters

And adds these 5 practical GUI controls:
    1. Detail Protection
    2. Gap Bridging
    3. Temporal Confidence
    4. Max Scratch Area %
    5. Overlay View: Sharp / Soft / Dust / All

Overlay View supports category-specific preview masks when restore.py returns
those masks. If the current restore.py only returns a single combined mask,
this app gracefully falls back:
    - All    -> combined mask
    - Sharp  -> combined scratch mask fallback
    - Soft   -> combined scratch mask fallback
    - Dust   -> empty unless provided by restore.py
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import *

from .cache import FrameCache
from .io_utils import load_default_dir, save_default_dir
from .processing.mask import load_preset_mask
from .processing.restore import restore_frame
from .viewer import Viewer


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.files = []
        self.idx = 0
        self.cache = FrameCache()
        self.preset_mask = None
        self.preset_mask_path = None
        self.preset_mask_mode = None
        self.cancel_flag = False

        # Cached temporal preview. Overlay checkbox and Overlay View can redraw
        # this cache without recomputing restore.
        self.preview_cache = None

        # Save/restore GUI settings between sessions.
        self.settings = QSettings("OpenAI", "film_restore_tool")

        self.p = {
            # Diagnostics / workflow
            "enable_dust_repair": False,
            "debug_repair": True,
            "debug_timing": True,

            # Valid area / borders (user-preferred gentler edge behavior)
            "ignore_frame_border_px": 12,
            "ignore_frame_border_x_px": 4,
            "ignore_frame_border_y_px": 4,
            "mask_edge_erode": 10,
            "mask_edge_erode_x_px": 36,
            "mask_edge_erode_y_px": 2,
            "mask_boundary_safety_px": 12,
            "mask_boundary_safety_x_px": 4,
            "mask_boundary_safety_y_px": 4,

            # Negative film: scratches are bright / white.
            "scratch_rel_floor": 0.025,
            "profile_noise_bg_width": 31,
            "profile_local_noise_box": 51,
            "density_box": 121,

            # Thin/sharp profile test.
            "thin_widths": (1, 3, 5),
            "thin_side_width": 5,
            "thin_side_gap": 2,
            "thin_abs": 0.0070,
            "thin_rel": 0.110,
            "thin_noise_mul": 1.65,
            "thin_side_coherence": 0.018,
            "thin_side_coherence_rel": 0.30,
            "thin_side_noise_mul": 1.65,
            "thin_peak_width": 9,
            "thin_max_seed_pixels": 240000,
            "thin_max_seed_fraction": 0.018,

            # Soft/wide profile test.
            "enable_soft_detection": True,
            "soft_widths": (7, 11, 15, 21, 27),
            "soft_side_width": 13,
            "soft_side_gap": 4,
            "soft_abs": 0.0045,
            "soft_rel": 0.055,
            "soft_noise_mul": 1.10,
            "soft_side_coherence": 0.030,
            "soft_side_coherence_rel": 0.55,
            "soft_side_noise_mul": 2.50,
            "soft_peak_width": 23,
            "soft_max_seed_pixels": 280000,
            "soft_max_seed_fraction": 0.022,

            # Thin segment grouping / acceptance
            "thin_connect_height": 25,
            "thin_connect_width": 3,
            "thin_min_line_length": 55,
            "thin_max_line_gap": 16,
            "thin_hough_threshold": 28,
            "thin_max_abs_slope": 0.10,
            "thin_eval_half_width": 1,
            "thin_min_length": 55,
            "thin_min_coverage": 0.10,
            "thin_min_density_excess": 0.030,
            "thin_min_log_nfa": 14.0,
            "thin_min_mean_response": 0.0045,
            "thin_repair_width": 3,
            "thin_max_hough_lines": 900,
            "thin_max_lines": 120,
            "thin_nms_x": 10,
            "thin_max_x_mad": 2.0,
            "thin_max_x_span": 10.0,
            "thin_max_repair_width": 7,
            "thin_bridge_gap": 18,

            # Soft segment grouping / acceptance
            "soft_connect_height": 101,
            "soft_connect_width": 5,
            "soft_min_line_length": 55,
            "soft_max_line_gap": 160,
            "soft_hough_threshold": 26,
            "soft_max_abs_slope": 0.18,
            "soft_eval_half_width": 5,
            "soft_min_length": 55,
            "soft_min_coverage": 0.020,
            "soft_min_density_excess": 0.004,
            "soft_min_log_nfa": 7.0,
            "soft_min_mean_response": 0.0011,
            "soft_repair_width": 13,
            "soft_max_hough_lines": 600,
            "soft_max_lines": 160,
            "soft_nms_x": 6,
            "soft_max_x_mad": 7.0,
            "soft_max_x_span": 28.0,
            "soft_max_repair_width": 25,
            "soft_repair_extra": 2.0,
            "soft_bridge_gap": 240,

            # Soft line fill / lane / track recovery
            "enable_soft_line_fill": True,
            "soft_line_fill_max_anchor_gap": 420,
            "soft_line_fill_abs": 0.00035,
            "soft_line_fill_rel_factor": 0.14,
            "soft_line_fill_solid_abs": 0.00018,
            "soft_line_fill_solid_max_gap": 260,
            "soft_line_fill_min_rows": 8,
            "soft_line_min_repair_width": 23,

            "enable_soft_track_bridging": True,
            "soft_track_group_x_tol": 42.0,
            "soft_track_min_segments": 1,
            "soft_track_min_span": 160,
            "soft_track_single_min_span": 260,
            "soft_track_min_anchor_rows": 14,
            "soft_track_max_x_mad": 14.0,
            "soft_track_max_x_span": 96.0,
            "soft_track_bridge_gap": 1400,
            "soft_track_max_tracks": 12,
            "soft_track_repair_width": 31,
            "enable_soft_track_fill": True,
            "soft_track_group_slope_tol": 0.22,
            "soft_track_group_y_gap": 1400,
            "soft_track_gap_x_slope_allowance": 0.025,
            "soft_track_bin_size": 18,
            "soft_track_min_repair_width": 27,
            "soft_track_use_thin_anchors": True,
            "soft_track_max_abs_slope": 0.18,

            # Edge cleanup / artifact rejection
            "enable_edge_artifact_cleanup": True,
            "edge_artifact_boundary_dist": 140.0,
            "edge_artifact_cleanup_top_bottom": False,
            "edge_artifact_frame_strip": 48,
            "edge_artifact_frame_strip_x": 48,
            "edge_artifact_frame_strip_y": 12,
            "edge_artifact_force_side_reject_px": 0,
            "edge_artifact_side_min_area": 80,
            "edge_artifact_side_max_width": 34,
            "edge_artifact_side_max_area_keep": 500,
            "edge_artifact_side_min_valid_dist": 24.0,
            "edge_artifact_max_width": 54,
            "edge_artifact_min_area": 250,
            "edge_artifact_max_fill": 0.28,
            "edge_artifact_min_aspect_keep": 4.0,
            "thin_final_shrink_px": 0,
            "soft_final_shrink_px": 1,
            "long_scratch_final_shrink_px": 2,

            # Irregular scratched-emulsion fragments.
            "enable_irregular_emulsion_damage": True,
            "debug_emulsion_damage": True,
            "emulsion_abs": 0.0026,
            "emulsion_rel": 0.020,
            "emulsion_noise_mul": 0.60,
            "emulsion_side_coherence": 0.055,
            "emulsion_side_coherence_rel": 0.90,
            "emulsion_side_noise_mul": 3.50,
            "emulsion_connect_width": 9,
            "emulsion_connect_height": 71,
            "emulsion_dilate_width": 3,
            "emulsion_min_area": 28,
            "emulsion_max_area": 16000,
            "emulsion_min_height": 35,
            "emulsion_max_width": 82,
            "emulsion_min_aspect": 1.35,
            "emulsion_min_active_rows": 14,
            "emulsion_min_active_fraction": 0.035,
            "emulsion_min_mean_response": 0.00075,
            "emulsion_min_peak_response": 0.0020,
            "emulsion_max_x_mad": 16.0,
            "emulsion_max_x_span": 88.0,
            "emulsion_max_abs_slope": 0.22,
            "emulsion_max_lines": 48,
            "emulsion_min_repair_width": 25,
            "emulsion_repair_width": 31,

            # Projection-based irregular emulsion track detector.
            "enable_emulsion_vertical_track": True,
            "enable_fast_residual_track": False,
            "enable_emulsion_blob_track": False,
            "debug_emulsion_track": True,
            "emulsion_track_abs": 0.0018,
            "emulsion_track_rel": 0.010,
            "emulsion_track_noise_mul": 0.28,
            "emulsion_track_side_coherence": 0.095,
            "emulsion_track_side_coherence_rel": 1.35,
            "emulsion_track_side_noise_mul": 5.50,
            "emulsion_track_projection_width": 21,
            "emulsion_track_min_col_rows": 18,
            "emulsion_track_min_col_response": 0.018,
            "emulsion_track_ignore_border_x": 12,
            "emulsion_track_min_lane_width": 1,
            "emulsion_track_max_lane_width": 120,
            "emulsion_track_lane_pad": 14,
            "emulsion_track_min_active_rows": 12,
            "emulsion_track_min_span": 120,
            "emulsion_track_min_active_fraction": 0.012,
            "emulsion_track_bridge_gap": 1200,
            "emulsion_track_solid_gap": 300,
            "emulsion_track_fill_abs": 0.00020,
            "emulsion_track_fill_rel_factor": 0.10,
            "emulsion_track_fill_half_width": 18,
            "emulsion_track_max_x_mad": 22.0,
            "emulsion_track_max_x_span": 130.0,
            "emulsion_track_max_abs_slope": 0.28,
            "emulsion_track_max_tracks": 24,
            "emulsion_track_min_repair_width": 25,
            "emulsion_track_repair_width": 31,

            # Faint/wide intermittent scratch recovery.
            "enable_soft_lane_recovery": True,
            "soft_lane_connect_height": 241,
            "soft_lane_connect_width": 3,
            "soft_lane_dilate_width": 3,
            "soft_lane_x_pad": 6,
            "soft_lane_min_span": 120,
            "soft_lane_max_width": 56,
            "soft_lane_min_aspect": 3.5,
            "soft_lane_min_active_rows": 18,
            "soft_lane_min_active_fraction": 0.020,
            "soft_lane_min_mean_response": 0.0007,
            "soft_lane_min_peak_response": 0.0022,
            "soft_lane_max_lines": 32,
            "enable_soft_lane_fill": True,
            "soft_lane_fill_half_width": 18,
            "soft_lane_fill_abs": 0.00035,
            "soft_lane_fill_rel_factor": 0.12,
            "soft_lane_fill_max_anchor_gap": 480,
            "soft_lane_min_repair_width": 9,
            "soft_lane_repair_width": 25,
            "soft_lane_fill_solid_abs": 0.00018,
            "soft_lane_fill_solid_max_gap": 480,
            "line_nms_y_overlap": 0.35,

            # Temporal validation over the local window.
            "temporal_min_votes": 2,
            "temporal_neighbor_radius": 2,
            "temporal_max_dx": 28.0,
            "temporal_min_y_overlap": 0.18,
            "temporal_max_slope_delta": 0.35,
            "temporal_strong_score": 34.0,
            "temporal_strong_mean_response": 0.0040,

            # Budgets / caps.
            "thin_max_mask_fraction": 0.0060,
            "soft_max_mask_fraction": 0.0350,
            "scratch_max_mask_fraction": 0.0500,

            # Window-based intermittent detection.
            "thin_ridge_noise_tolerance": 0.06,
            "soft_ridge_noise_tolerance": 0.14,
            "thin_ridge_abs_tolerance": 1e-6,
            "soft_ridge_abs_tolerance": 0.0002,
            "thin_window_length": 96,
            "thin_window_step": 32,
            "thin_window_min_passes": 1,
            "thin_window_min_active_rows": 8,
            "thin_window_min_coverage": 0.075,
            "thin_window_min_density_excess": 0.010,
            "thin_window_min_log_nfa": 7.0,
            "thin_window_min_mean_response": 0.0036,
            "soft_window_length": 140,
            "soft_window_step": 46,
            "soft_window_min_passes": 1,
            "soft_window_min_active_rows": 10,
            "soft_window_min_coverage": 0.050,
            "soft_window_min_density_excess": 0.006,
            "soft_window_min_log_nfa": 8.0,
            "soft_window_min_mean_response": 0.0017,

            # Adaptive track fill.
            "enable_adaptive_track_fill": True,
            "debug_adaptive_track_fill": True,
            "adaptive_track_half_width": 9,
            "adaptive_track_solid_gap": 18,
            "adaptive_track_bridge_gap": 80,
            "adaptive_track_abs": 0.0038,
            "adaptive_track_weak_abs": 0.0020,
            "adaptive_track_noise_mul": 0.90,
            "adaptive_track_weak_noise_mul": 0.55,
            "adaptive_track_min_fill_ratio": 0.18,
            "adaptive_track_min_added_rows": 10,
            "adaptive_track_extra_width": 1,

            # Context-edge veto.
            "enable_context_edge_veto": True,
            "enable_thin_context_edge_veto": True,
            "debug_context_edge_veto": True,
            "context_veto_sources": "thin_line,soft_line,soft_lane,soft_track,adaptive_track,emulsion_track",
            "context_veto_blur_width": 17,
            "context_veto_min_span": 110,
            "context_veto_min_rows": 24,
            "context_veto_sample_step": 3,
            "context_veto_side_width": 11,
            "context_veto_side_gap": 11,
            "context_veto_side_abs": 0.030,
            "context_veto_side_ratio": 2.6,
            "context_veto_bad_fraction": 0.55,
            "context_veto_min_center_resid": 0.0010,
            "context_veto_cluster_window_px": 36.0,
            "context_veto_cluster_max_lines": 7,
            "context_veto_skip_width_le": 3,
            "context_veto_skip_score_ge": 40.0,
            "thin_context_veto_min_span": 80,
            "thin_context_veto_min_rows": 18,
            "thin_context_veto_side_abs": 0.020,
            "thin_context_veto_side_ratio": 2.20,
            "thin_context_veto_bad_fraction": 0.50,
            "thin_context_veto_min_center_resid": 0.0010,
            "thin_context_veto_skip_width_le": 1,
            "thin_context_veto_skip_score_ge": 70.0,

            # Scratch-vs-scene verifier. This is a second-stage classifier that
            # rejects bright vertical picture elements before they reach repair.
            "enable_scratch_scene_verifier": True,
            "debug_scratch_scene_verifier": True,
            "enable_scene_motion_veto": True,
            "scene_verify_sample_step": 4,
            "scene_verify_min_rows": 12,
            "scene_verify_smooth_width": 5,
            "scene_verify_lowfreq_width": 31,
            "scene_verify_side_width": 9,
            "scene_verify_side_gap": 4,
            "scene_verify_shoulder_gap": 2,
            "scene_verify_min_center_resid": 0.0008,
            "scene_verify_lowfreq_min_peak": 0.0020,
            "scene_verify_allow_strong_bypass": False,
            "scene_verify_strong_score_keep": 999999.0,
            "scene_verify_strong_mean_keep": 999999.0,
            "scene_verify_scene_like_diff": 0.0060,
            "scene_verify_temporal_scene_fraction": 0.65,
            "scene_verify_enable_motion_competition": True,
            "scene_verify_motion_compete_fraction": 0.45,
            "scene_verify_motion_compete_ratio": 0.70,
            "scene_verify_fixed_min_diff": 0.0025,
            "scene_verify_aligned_max_diff": 0.0100,
            "scene_motion_min_shift_px": 1.25,
            "scene_motion_min_response": 0.06,
            "scene_motion_max_neighbors": 2,
            "scene_motion_phase_max_dim": 768,
            "scene_motion_phase_blur": 31,

            # Thin verifier defaults.
            "thin_scene_min_peak_abs": 0.0025,
            "thin_scene_min_narrow_abs": 0.0010,
            "thin_scene_edge_abs": 0.018,
            "thin_scene_edge_ratio": 2.0,
            "thin_scene_min_peak_fraction": 0.22,
            "thin_scene_min_narrow_fraction": 0.12,
            "thin_scene_max_edge_fraction": 0.62,
            "thin_scene_max_plateau_fraction": 0.72,
            "thin_scene_lowfreq_edge_thr": 0.018,
            "thin_scene_column_half_width": 22,
            "thin_scene_column_abs": 0.0018,
            "thin_scene_column_rel": 0.35,
            "thin_scene_column_min_width": 8,
            "thin_scene_max_column_fraction": 0.45,
            "thin_scene_detail_half_width": 18,
            "thin_scene_detail_exclude_px": 3,
            "thin_scene_detail_edge_abs": 0.0060,
            "thin_scene_detail_row_mean_abs": 0.0028,
            "thin_scene_detail_col_fraction": 0.18,
            "thin_scene_max_vertical_detail_fraction": 0.38,
            "thin_scene_vertical_detail_median_thr": 0.0030,
            "scene_rank_detail_quantile": 0.82,
            "scene_bright_bg_quantile": 0.72,
            "thin_scene_rank_detail_row_fraction": 0.22,
            "thin_scene_max_rank_detail_fraction": 0.34,
            "thin_scene_max_bright_detail_fraction": 0.30,

            # Soft verifier defaults.
            "soft_scene_min_peak_abs": 0.0018,
            "soft_scene_min_narrow_abs": 0.0006,
            "soft_scene_edge_abs": 0.024,
            "soft_scene_edge_ratio": 2.4,
            "soft_scene_min_peak_fraction": 0.18,
            "soft_scene_min_narrow_fraction": 0.06,
            "soft_scene_max_edge_fraction": 0.58,
            "soft_scene_max_plateau_fraction": 0.82,
            "soft_scene_lowfreq_edge_thr": 0.024,
            "soft_scene_column_half_width": 30,
            "soft_scene_column_abs": 0.0012,
            "soft_scene_column_rel": 0.25,
            "soft_scene_column_min_width": 14,
            "soft_scene_max_column_fraction": 0.55,
            "soft_scene_detail_half_width": 26,
            "soft_scene_detail_exclude_px": 6,
            "soft_scene_detail_edge_abs": 0.0080,
            "soft_scene_detail_row_mean_abs": 0.0035,
            "soft_scene_detail_col_fraction": 0.24,
            "soft_scene_max_vertical_detail_fraction": 0.50,
            "soft_scene_vertical_detail_median_thr": 0.0040,
            "soft_scene_rank_detail_row_fraction": 0.30,
            "soft_scene_max_rank_detail_fraction": 0.50,
            "soft_scene_max_bright_detail_fraction": 0.45,

            # Faint/intermittent scratch recovery.
            "enable_faint_scratch_recovery": True,
            "debug_faint_scratch": True,
            "faint_polarity": "both",
            "faint_bg_width": 91,
            "faint_noise_box": 61,
            "faint_abs": 0.0018,
            "faint_noise_mul": 0.55,
            "faint_connect_height": 121,
            "faint_connect_width": 1,
            "faint_hough_threshold": 18,
            "faint_min_line_length": 120,
            "faint_max_line_gap": 90,
            "faint_eval_half_width": 2,
            "faint_max_abs_slope": 0.12,
            "faint_min_active_rows": 24,
            "faint_min_span": 140,
            "faint_min_active_fraction": 0.055,
            "faint_min_mean_response": 0.0012,
            "faint_max_x_mad": 3.5,
            "faint_max_x_span": 16.0,
            "faint_max_lines": 36,
            "faint_repair_width": 3,
            "faint_max_repair_width": 7,
            "enable_faint_line_fill": True,
            "faint_line_fill_max_gap": 180,
            "faint_line_fill_abs": 0.00075,
            "faint_line_fill_rel": 0.38,

            "final_close_height": 5,
            "final_close_width": 1,

            # Repair parameters used by restore.py.
            "scratch_interp_radius": 42,
            "scratch_interp_sample": 8,
            "scratch_interp_strength": 1.0,
            "scratch_interp_max_run": 90,
            "scratch_inpaint_after_interp": False,
            "inpaint_radius": 3,

            # Dust / specks.
            "dust": 0.040,
            "dust_abs": 0.040,
            "dust_noise_mul": 2.40,
            "dust_temporal_abs": 0.014,
            "dust_temporal_max_std": 0.12,
            "dust_max_mask_fraction": 0.0020,
            "dust_inpaint_radius": 5,
            "dust_min_area": 2,
            "dust_max_area": 900,
            "dust_max_width": 56,
            "dust_max_height": 56,
            "dust_min_peak_response": 0.018,
            "dust_min_mean_response": 0.006,
            "enable_large_dust_repair": True,
            "large_dust_abs": 0.018,
            "large_dust_noise_mul": 1.20,
            "large_dust_min_area": 18,
            "large_dust_max_area": 9000,
            "large_dust_max_width": 160,
            "large_dust_max_height": 160,
            "large_dust_max_aspect": 4.0,
            "large_dust_min_fill_ratio": 0.08,
            "large_dust_min_peak_response": 0.016,
            "large_dust_min_mean_response": 0.0040,
            "large_dust_close_size": 5,
            "large_dust_dilate": 3,
            "large_dust_max_mask_fraction": 0.0030,
        }

        self.v1 = Viewer()
        self.v2 = Viewer()
        self.v1.sync = self.sync_views
        self.v2.sync = self.sync_views

        # ===== Controls =====
        self.dust = QDoubleSpinBox()
        self.dust.setDecimals(4)
        self.dust.setRange(0.0000, 0.2000)
        self.dust.setSingleStep(0.0025)
        self.dust.setValue(self.p["dust"])
        self.dust.setKeyboardTracking(False)
        self.dust.setToolTip("Dust threshold. Used only when Dust Repair is enabled.")

        self.scratch = QDoubleSpinBox()
        self.scratch.setDecimals(2)
        self.scratch.setRange(0.50, 2.00)
        self.scratch.setSingleStep(0.05)
        self.scratch.setValue(0.70)
        self.scratch.setKeyboardTracking(False)
        self.scratch.setToolTip("Thin/sharp scratch sensitivity. Lower = more detection.")

        self.soft_scratch = QDoubleSpinBox()
        self.soft_scratch.setDecimals(2)
        self.soft_scratch.setRange(0.50, 2.00)
        self.soft_scratch.setSingleStep(0.05)
        self.soft_scratch.setValue(0.75)
        self.soft_scratch.setKeyboardTracking(False)
        self.soft_scratch.setToolTip("Soft/wide scratch sensitivity. Lower = more detection.")

        self.enable_soft_scratches = QCheckBox("Soft Scratch Repair")
        self.enable_soft_scratches.setChecked(True)
        self.enable_soft_scratches.setToolTip(
            "Enable/disable all soft/wide scratch paths for diagnosis. "
            "Off = sharp/thin scratch repair only."
        )

        self.sharp_repair_width = QSpinBox()
        self.sharp_repair_width.setRange(1, 9)
        self.sharp_repair_width.setSingleStep(2)
        self.sharp_repair_width.setValue(3)
        self.sharp_repair_width.setKeyboardTracking(False)
        self.sharp_repair_width.setToolTip(
            "Repair mask width for sharp/thin scratches only. Use 1, 3, 5, 7, or 9."
        )

        self.max_repair_width = QSpinBox()
        self.max_repair_width.setRange(5, 31)
        self.max_repair_width.setSingleStep(2)
        self.max_repair_width.setValue(25)
        self.max_repair_width.setKeyboardTracking(False)
        self.max_repair_width.setToolTip("Maximum width used for soft/wide scratch repair masks.")

        self.frames = QSpinBox()
        self.frames.setRange(3, 9)
        self.frames.setSingleStep(2)
        self.frames.setValue(5)
        self.frames.setKeyboardTracking(False)
        self.frames.setToolTip("Temporal window used for local validation.")

        self.detail_protection = QComboBox()
        self.detail_protection.addItem("Low", "low")
        self.detail_protection.addItem("Medium", "medium")
        self.detail_protection.addItem("High", "high")
        self.detail_protection.setCurrentIndex(1)
        self.detail_protection.setToolTip(
            "How strongly real image detail is protected from being mistaken for a scratch. "
            "High is safer for foliage, haze, and detailed daylight scenes."
        )

        self.gap_bridging = QComboBox()
        self.gap_bridging.addItem("Off", "off")
        self.gap_bridging.addItem("Conservative", "conservative")
        self.gap_bridging.addItem("Normal", "normal")
        self.gap_bridging.addItem("Aggressive", "aggressive")
        self.gap_bridging.setCurrentIndex(2)
        self.gap_bridging.setToolTip(
            "How strongly broken vertical scratch segments are linked together."
        )

        self.temporal_confidence = QComboBox()
        self.temporal_confidence.addItem("Low", "low")
        self.temporal_confidence.addItem("Medium", "medium")
        self.temporal_confidence.addItem("High", "high")
        self.temporal_confidence.setCurrentIndex(1)
        self.temporal_confidence.setToolTip(
            "How much multi-frame agreement is required before accepting a scratch."
        )

        self.max_scratch_area = QDoubleSpinBox()
        self.max_scratch_area.setDecimals(2)
        self.max_scratch_area.setRange(0.10, 5.00)
        self.max_scratch_area.setSingleStep(0.05)
        self.max_scratch_area.setValue(0.80)
        self.max_scratch_area.setKeyboardTracking(False)
        self.max_scratch_area.setToolTip(
            "Global cap for the total scratch repair mask area, expressed as a percentage of the frame."
        )

        self.overlay_view = QComboBox()
        self.overlay_view.addItem("All", "all")
        self.overlay_view.addItem("Sharp", "sharp")
        self.overlay_view.addItem("Soft", "soft")
        self.overlay_view.addItem("Dust", "dust")
        self.overlay_view.setCurrentIndex(0)
        self.overlay_view.setToolTip(
            "Choose which detected mask category is shown in the overlay. "
            "Category-specific views are used when restore.py provides them."
        )

        self.dust_repair = QCheckBox("Dust Repair")
        self.dust_repair.setChecked(False)
        self.dust_repair.setToolTip("Keep off while tuning scratches.")

        self.overlay = QCheckBox("Overlay")
        self.overlay.setChecked(True)

        self.frame_mask_combo = QComboBox()
        self.frame_mask_combo.addItem("16mm", "16mm")
        self.frame_mask_combo.addItem("35mm", "35mm")
        self.frame_mask_combo.addItem("Custom...", "custom")
        self.frame_mask_combo.setToolTip(
            "Choose the preset frame mask. 16mm_mask.png and 35mm_mask.png are searched "
            "in the restore_tool directory. Custom opens a file browser and does not modify ~/.default_dir."
        )

        self.render_mode = QComboBox()
        self.render_mode.addItem("Render only new frames (resume / skip existing)", "skip")
        self.render_mode.addItem("Rewrite existing frames", "rewrite")
        self.render_mode.setToolTip(
            "Skip existing output files to resume an interrupted render, or rewrite all frames."
        )

        self.load_session_settings()

        # Connect stateful controls.
        value_widgets = (
            self.dust,
            self.scratch,
            self.soft_scratch,
            self.sharp_repair_width,
            self.max_repair_width,
            self.frames,
            self.max_scratch_area,
        )
        for widget in value_widgets:
            widget.valueChanged.connect(self.update_view)
            widget.valueChanged.connect(self.save_session_settings)

        combo_widgets = (
            self.detail_protection,
            self.gap_bridging,
            self.temporal_confidence,
        )
        for widget in combo_widgets:
            widget.currentIndexChanged.connect(self.update_view)
            widget.currentIndexChanged.connect(self.save_session_settings)

        self.enable_soft_scratches.stateChanged.connect(self.update_view)
        self.enable_soft_scratches.stateChanged.connect(self.save_session_settings)
        self.dust_repair.stateChanged.connect(self.update_view)
        self.dust_repair.stateChanged.connect(self.save_session_settings)
        self.overlay.stateChanged.connect(self.redraw_temporal_preview_overlay)
        self.overlay.stateChanged.connect(self.save_session_settings)
        self.overlay_view.currentIndexChanged.connect(self.redraw_temporal_preview_overlay)
        self.overlay_view.currentIndexChanged.connect(self.save_session_settings)
        self.frame_mask_combo.currentIndexChanged.connect(self.on_frame_mask_selection_changed)
        self.frame_mask_combo.currentIndexChanged.connect(self.save_session_settings)
        self.render_mode.currentIndexChanged.connect(self.update_render_stats_label)
        self.render_mode.currentIndexChanged.connect(self.save_session_settings)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # ===== Views =====
        views = QHBoxLayout()
        views.addWidget(self.v1)
        views.addWidget(self.v2)

        # ===== Navigation =====
        nav = QHBoxLayout()
        prev = QPushButton("<<")
        nextb = QPushButton(">>")
        nav.addWidget(prev)
        nav.addWidget(nextb)
        prev.clicked.connect(lambda: self.step(-1))
        nextb.clicked.connect(lambda: self.step(1))

        # ===== Buttons =====
        buttons = QHBoxLayout()
        load = QPushButton("Load")
        load_mask = QPushButton("Load Frame Mask")
        temporal_preview = QPushButton("Temporal Preview")
        process = QPushButton("Process")
        cancel = QPushButton("Cancel")
        buttons.addWidget(load)
        buttons.addWidget(load_mask)
        buttons.addWidget(temporal_preview)
        buttons.addWidget(process)
        buttons.addWidget(cancel)
        load.clicked.connect(self.load)
        load_mask.clicked.connect(self.load_selected_frame_mask)
        temporal_preview.clicked.connect(self.temporal_preview)
        process.clicked.connect(self.process)
        cancel.clicked.connect(self.cancel)

        self.progress = QProgressBar()
        self.frame_label = QLabel("Frame: —")
        self.frame_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.frame_label.setStyleSheet("font-weight: bold;")
        self.directory_label = QLabel("Directory: —")
        self.directory_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.directory_label.setWordWrap(True)
        self.mask_label = QLabel("Frame Mask: —")
        self.mask_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.mask_label.setWordWrap(True)
        self.render_stats_label = QLabel("Rendered on disk: — | Remaining: —")
        self.render_stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status = QLabel("Idle")

        left = QVBoxLayout()
        left.addLayout(views)
        left.addLayout(nav)
        left.addLayout(buttons)
        left.addWidget(self.frame_label)
        left.addWidget(self.directory_label)
        left.addWidget(self.mask_label)
        left.addWidget(self.render_stats_label)
        left.addWidget(self.progress)
        left.addWidget(self.status)

        controls = QFormLayout()
        controls.addRow("Dust", self.dust)
        controls.addRow("Scratch", self.scratch)
        controls.addRow("Sharp Width", self.sharp_repair_width)
        controls.addRow("Soft Scratch", self.soft_scratch)
        controls.addRow(self.enable_soft_scratches)
        controls.addRow("Max Soft Width", self.max_repair_width)
        controls.addRow("Frames", self.frames)
        controls.addRow("Detail Protection", self.detail_protection)
        controls.addRow("Gap Bridging", self.gap_bridging)
        controls.addRow("Temporal Confidence", self.temporal_confidence)
        controls.addRow("Max Scratch Area %", self.max_scratch_area)
        controls.addRow("Overlay View", self.overlay_view)
        controls.addRow("Frame Mask", self.frame_mask_combo)
        controls.addRow("Render Mode", self.render_mode)
        controls.addRow(self.dust_repair)
        controls.addRow(self.overlay)

        layout = QHBoxLayout()
        layout.addLayout(left, 4)
        layout.addLayout(controls, 1)
        self.setLayout(layout)
        self.setMinimumSize(1260, 820)
        self.update_render_stats_label()

    # ------------------------------------------------------------------
    # Session settings
    # ------------------------------------------------------------------

    def load_session_settings(self):
        self.dust.setValue(float(self.settings.value("ui/dust", self.dust.value(), type=float)))
        self.scratch.setValue(float(self.settings.value("ui/scratch", self.scratch.value(), type=float)))
        self.soft_scratch.setValue(float(self.settings.value("ui/soft_scratch", self.soft_scratch.value(), type=float)))
        self.sharp_repair_width.setValue(int(self.settings.value("ui/sharp_repair_width", self.sharp_repair_width.value(), type=int)))
        self.max_repair_width.setValue(int(self.settings.value("ui/max_repair_width", self.max_repair_width.value(), type=int)))
        self.frames.setValue(int(self.settings.value("ui/frames", self.frames.value(), type=int)))
        self.max_scratch_area.setValue(float(self.settings.value("ui/max_scratch_area", self.max_scratch_area.value(), type=float)))
        self.enable_soft_scratches.setChecked(bool(self.settings.value("ui/enable_soft_scratches", self.enable_soft_scratches.isChecked(), type=bool)))
        self.dust_repair.setChecked(bool(self.settings.value("ui/dust_repair", self.dust_repair.isChecked(), type=bool)))
        self.overlay.setChecked(bool(self.settings.value("ui/overlay", self.overlay.isChecked(), type=bool)))

        for key, widget in (
            ("ui/detail_protection", self.detail_protection),
            ("ui/gap_bridging", self.gap_bridging),
            ("ui/temporal_confidence", self.temporal_confidence),
            ("ui/overlay_view", self.overlay_view),
            ("ui/frame_mask_mode", self.frame_mask_combo),
            ("ui/render_mode", self.render_mode),
        ):
            val = self.settings.value(key, widget.currentData(), type=str)
            idx = widget.findData(val)
            if idx >= 0:
                widget.setCurrentIndex(idx)

    def save_session_settings(self, *args):
        self.settings.setValue("ui/dust", float(self.dust.value()))
        self.settings.setValue("ui/scratch", float(self.scratch.value()))
        self.settings.setValue("ui/soft_scratch", float(self.soft_scratch.value()))
        self.settings.setValue("ui/sharp_repair_width", int(self.sharp_repair_width.value()))
        self.settings.setValue("ui/max_repair_width", int(self.max_repair_width.value()))
        self.settings.setValue("ui/frames", int(self.frames.value()))
        self.settings.setValue("ui/max_scratch_area", float(self.max_scratch_area.value()))
        self.settings.setValue("ui/enable_soft_scratches", bool(self.enable_soft_scratches.isChecked()))
        self.settings.setValue("ui/dust_repair", bool(self.dust_repair.isChecked()))
        self.settings.setValue("ui/overlay", bool(self.overlay.isChecked()))
        self.settings.setValue("ui/detail_protection", str(self.detail_protection.currentData() or "medium"))
        self.settings.setValue("ui/gap_bridging", str(self.gap_bridging.currentData() or "normal"))
        self.settings.setValue("ui/temporal_confidence", str(self.temporal_confidence.currentData() or "medium"))
        self.settings.setValue("ui/overlay_view", str(self.overlay_view.currentData() or "all"))
        self.settings.setValue("ui/frame_mask_mode", str(self.frame_mask_combo.currentData() or "16mm"))
        self.settings.setValue("ui/render_mode", str(self.render_mode.currentData() or "skip"))
        self.settings.sync()

    def closeEvent(self, event):
        self.save_session_settings()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Render bookkeeping
    # ------------------------------------------------------------------

    def get_output_dir(self):
        if not self.files:
            return None
        return os.path.join(os.path.dirname(self.files[0]), "restored")

    def output_frame_exists(self, path):
        return os.path.isfile(path) and os.path.getsize(path) > 0

    def count_existing_outputs(self, outdir):
        count = 0
        for src in self.files:
            name = os.path.basename(src)
            path = os.path.join(outdir, name)
            if self.output_frame_exists(path):
                count += 1
        return count

    def update_render_stats_label(self):
        if not self.files:
            self.render_stats_label.setText("Rendered on disk: — | Remaining: —")
            return
        total = len(self.files)
        outdir = self.get_output_dir()
        existing = self.count_existing_outputs(outdir) if outdir else 0
        mode = self.render_mode.currentData() or "skip"
        remaining = max(0, total - existing) if mode == "skip" else total
        mode_text = "resume / skip existing" if mode == "skip" else "rewrite all"
        self.render_stats_label.setText(
            f"Rendered on disk: {existing}/{total} | Remaining in {mode_text}: {remaining}"
        )

    # ------------------------------------------------------------------
    # Parameter mapping from GUI -> self.p
    # ------------------------------------------------------------------

    def _apply_detail_protection(self):
        level = str(self.detail_protection.currentData() or "medium")

        if level == "low":
            # More permissive; best for simple / dark scenes.
            self.p["context_veto_side_abs"] = 0.036
            self.p["context_veto_side_ratio"] = 3.0
            self.p["context_veto_bad_fraction"] = 0.72
            self.p["context_veto_min_center_resid"] = 0.00075
            self.p["thin_context_veto_side_abs"] = 0.026
            self.p["thin_context_veto_side_ratio"] = 2.70
            self.p["thin_context_veto_bad_fraction"] = 0.64
            self.p["thin_context_veto_min_center_resid"] = 0.00075
            self.p["thin_max_lines"] = 120
            self.p["thin_max_seed_fraction"] = 0.018
            self.p["thin_max_seed_pixels"] = 240000
            self.p["scene_verify_allow_strong_bypass"] = False
            self.p["thin_scene_min_peak_fraction"] = 0.18
            self.p["thin_scene_min_narrow_fraction"] = 0.08
            self.p["thin_scene_max_edge_fraction"] = 0.72
            self.p["thin_scene_max_plateau_fraction"] = 0.82
            self.p["thin_scene_max_column_fraction"] = 0.62
            self.p["thin_scene_column_min_width"] = 11
            self.p["thin_scene_column_abs"] = 0.0024
            self.p["thin_scene_max_vertical_detail_fraction"] = 0.62
            self.p["thin_scene_vertical_detail_median_thr"] = 0.0045
            self.p["thin_scene_detail_row_mean_abs"] = 0.0040
            self.p["thin_scene_detail_col_fraction"] = 0.28
            self.p["scene_rank_detail_quantile"] = 0.90
            self.p["scene_bright_bg_quantile"] = 0.82
            self.p["thin_scene_rank_detail_row_fraction"] = 0.32
            self.p["thin_scene_max_rank_detail_fraction"] = 0.60
            self.p["thin_scene_max_bright_detail_fraction"] = 0.55
            self.p["scene_verify_motion_compete_fraction"] = 0.62
            self.p["scene_verify_motion_compete_ratio"] = 0.85
            self.p["scene_verify_fixed_min_diff"] = 0.0035
            self.p["scene_verify_aligned_max_diff"] = 0.0140
            self.p["soft_side_coherence_rel"] = 0.62
            self.p["soft_side_noise_mul"] = 2.20
            self.p["adaptive_track_min_fill_ratio"] = 0.14
            self.p["edge_artifact_side_min_valid_dist"] = 18.0
            # Scene verifier: permissive for simple/dark scenes.
            self.p["thin_scene_min_peak_abs"] = 0.0018
            self.p["thin_scene_min_narrow_abs"] = 0.0006
            self.p["thin_scene_min_peak_fraction"] = 0.14
            self.p["thin_scene_min_narrow_fraction"] = 0.06
            self.p["thin_scene_max_edge_fraction"] = 0.78
            self.p["thin_scene_max_plateau_fraction"] = 0.86
            self.p["thin_scene_edge_abs"] = 0.026
            self.p["thin_scene_edge_ratio"] = 2.80
            self.p["soft_scene_min_peak_fraction"] = 0.12
            self.p["soft_scene_max_edge_fraction"] = 0.72
            self.p["scene_verify_temporal_scene_fraction"] = 0.78
        elif level == "high":
            # More protective; best for foliage, haze, and bright detailed scenes.
            self.p["context_veto_side_abs"] = 0.022
            self.p["context_veto_side_ratio"] = 2.15
            self.p["context_veto_bad_fraction"] = 0.46
            self.p["context_veto_min_center_resid"] = 0.00135
            self.p["thin_context_veto_side_abs"] = 0.016
            self.p["thin_context_veto_side_ratio"] = 1.85
            self.p["thin_context_veto_bad_fraction"] = 0.42
            self.p["thin_context_veto_min_center_resid"] = 0.00125
            self.p["thin_max_lines"] = 70
            self.p["thin_max_seed_fraction"] = 0.012
            self.p["thin_max_seed_pixels"] = 160000
            self.p["thin_window_min_coverage"] = max(self.p.get("thin_window_min_coverage", 0.070), 0.105)
            self.p["scene_verify_allow_strong_bypass"] = False
            self.p["thin_scene_min_peak_fraction"] = 0.34
            self.p["thin_scene_min_narrow_fraction"] = 0.20
            self.p["thin_scene_max_edge_fraction"] = 0.38
            self.p["thin_scene_max_plateau_fraction"] = 0.46
            self.p["thin_scene_max_column_fraction"] = 0.28
            self.p["thin_scene_column_min_width"] = 7
            self.p["thin_scene_column_abs"] = 0.0012
            self.p["thin_scene_column_rel"] = 0.25
            self.p["thin_scene_max_vertical_detail_fraction"] = 0.20
            self.p["thin_scene_vertical_detail_median_thr"] = 0.0018
            self.p["thin_scene_detail_row_mean_abs"] = 0.0018
            self.p["thin_scene_detail_col_fraction"] = 0.10
            self.p["thin_scene_detail_half_width"] = 24
            self.p["scene_rank_detail_quantile"] = 0.72
            self.p["scene_bright_bg_quantile"] = 0.62
            self.p["thin_scene_rank_detail_row_fraction"] = 0.10
            self.p["thin_scene_max_rank_detail_fraction"] = 0.16
            self.p["thin_scene_max_bright_detail_fraction"] = 0.14
            self.p["scene_verify_lowfreq_min_peak"] = 0.0035
            self.p["scene_verify_motion_compete_fraction"] = 0.32
            self.p["scene_verify_motion_compete_ratio"] = 0.58
            self.p["scene_verify_fixed_min_diff"] = 0.0018
            self.p["scene_verify_aligned_max_diff"] = 0.0080
            self.p["soft_side_coherence_rel"] = 0.48
            self.p["soft_side_noise_mul"] = 2.90
            self.p["adaptive_track_min_fill_ratio"] = 0.24
            self.p["edge_artifact_side_min_valid_dist"] = 30.0
            # Scene verifier: strict for bright daylight, foliage, haze, and architecture.
            self.p["thin_scene_min_peak_abs"] = 0.0032
            self.p["thin_scene_min_narrow_abs"] = 0.0015
            self.p["thin_scene_min_peak_fraction"] = 0.32
            self.p["thin_scene_min_narrow_fraction"] = 0.20
            self.p["thin_scene_max_edge_fraction"] = 0.48
            self.p["thin_scene_max_plateau_fraction"] = 0.56
            self.p["thin_scene_edge_abs"] = 0.014
            self.p["thin_scene_edge_ratio"] = 1.65
            self.p["thin_scene_lowfreq_edge_thr"] = 0.014
            self.p["soft_scene_min_peak_fraction"] = 0.26
            self.p["soft_scene_max_edge_fraction"] = 0.46
            self.p["soft_scene_max_plateau_fraction"] = 0.66
            self.p["scene_verify_temporal_scene_fraction"] = 0.58
            self.p["scene_verify_scene_like_diff"] = 0.0075
        else:
            # Medium / default.
            self.p["context_veto_side_abs"] = 0.030
            self.p["context_veto_side_ratio"] = 2.60
            self.p["context_veto_bad_fraction"] = 0.55
            self.p["context_veto_min_center_resid"] = 0.00100
            self.p["thin_context_veto_side_abs"] = 0.020
            self.p["thin_context_veto_side_ratio"] = 2.20
            self.p["thin_context_veto_bad_fraction"] = 0.50
            self.p["thin_context_veto_min_center_resid"] = 0.00100
            self.p["thin_max_lines"] = 90
            self.p["thin_max_seed_fraction"] = 0.015
            self.p["thin_max_seed_pixels"] = 200000
            self.p["scene_verify_allow_strong_bypass"] = False
            self.p["thin_scene_min_peak_fraction"] = 0.26
            self.p["thin_scene_min_narrow_fraction"] = 0.14
            self.p["thin_scene_max_edge_fraction"] = 0.54
            self.p["thin_scene_max_plateau_fraction"] = 0.64
            self.p["thin_scene_max_column_fraction"] = 0.42
            self.p["thin_scene_column_min_width"] = 8
            self.p["thin_scene_column_abs"] = 0.0018
            self.p["thin_scene_max_vertical_detail_fraction"] = 0.38
            self.p["thin_scene_vertical_detail_median_thr"] = 0.0030
            self.p["thin_scene_detail_row_mean_abs"] = 0.0028
            self.p["thin_scene_detail_col_fraction"] = 0.18
            self.p["scene_rank_detail_quantile"] = 0.82
            self.p["scene_bright_bg_quantile"] = 0.72
            self.p["thin_scene_rank_detail_row_fraction"] = 0.22
            self.p["thin_scene_max_rank_detail_fraction"] = 0.34
            self.p["thin_scene_max_bright_detail_fraction"] = 0.30
            self.p["scene_verify_motion_compete_fraction"] = 0.45
            self.p["scene_verify_motion_compete_ratio"] = 0.70
            self.p["scene_verify_fixed_min_diff"] = 0.0025
            self.p["scene_verify_aligned_max_diff"] = 0.0100
            self.p["soft_side_coherence_rel"] = 0.55
            self.p["soft_side_noise_mul"] = 2.50
            self.p["adaptive_track_min_fill_ratio"] = 0.18
            self.p["edge_artifact_side_min_valid_dist"] = 24.0
            # Scene verifier: balanced default.
            self.p["thin_scene_min_peak_abs"] = 0.0025
            self.p["thin_scene_min_narrow_abs"] = 0.0010
            self.p["thin_scene_min_peak_fraction"] = 0.22
            self.p["thin_scene_min_narrow_fraction"] = 0.12
            self.p["thin_scene_max_edge_fraction"] = 0.62
            self.p["thin_scene_max_plateau_fraction"] = 0.72
            self.p["thin_scene_edge_abs"] = 0.018
            self.p["thin_scene_edge_ratio"] = 2.00
            self.p["soft_scene_min_peak_fraction"] = 0.18
            self.p["soft_scene_max_edge_fraction"] = 0.58
            self.p["soft_scene_max_plateau_fraction"] = 0.82
            self.p["scene_verify_temporal_scene_fraction"] = 0.65
            self.p["scene_verify_scene_like_diff"] = 0.0060

    def _apply_gap_bridging(self):
        mode = str(self.gap_bridging.currentData() or "normal")

        if mode == "off":
            vals = {
                "thin_bridge_gap": 8,
                "soft_bridge_gap": 90,
                "soft_line_fill_solid_max_gap": 120,
                "soft_line_fill_max_anchor_gap": 160,
                "soft_lane_fill_solid_max_gap": 180,
                "soft_lane_fill_max_anchor_gap": 220,
                "soft_track_bridge_gap": 420,
                "adaptive_track_bridge_gap": 24,
                "adaptive_track_solid_gap": 8,
                "emulsion_track_bridge_gap": 450,
                "emulsion_track_solid_gap": 140,
                "final_close_height": 3,
            }
        elif mode == "conservative":
            vals = {
                "thin_bridge_gap": 12,
                "soft_bridge_gap": 150,
                "soft_line_fill_solid_max_gap": 180,
                "soft_line_fill_max_anchor_gap": 260,
                "soft_lane_fill_solid_max_gap": 260,
                "soft_lane_fill_max_anchor_gap": 320,
                "soft_track_bridge_gap": 800,
                "adaptive_track_bridge_gap": 40,
                "adaptive_track_solid_gap": 12,
                "emulsion_track_bridge_gap": 700,
                "emulsion_track_solid_gap": 200,
                "final_close_height": 4,
            }
        elif mode == "aggressive":
            vals = {
                "thin_bridge_gap": 24,
                "soft_bridge_gap": 360,
                "soft_line_fill_solid_max_gap": 420,
                "soft_line_fill_max_anchor_gap": 620,
                "soft_lane_fill_solid_max_gap": 620,
                "soft_lane_fill_max_anchor_gap": 760,
                "soft_track_bridge_gap": 2200,
                "adaptive_track_bridge_gap": 120,
                "adaptive_track_solid_gap": 28,
                "emulsion_track_bridge_gap": 2000,
                "emulsion_track_solid_gap": 520,
                "final_close_height": 7,
            }
        else:
            vals = {
                "thin_bridge_gap": 18,
                "soft_bridge_gap": 240,
                "soft_line_fill_solid_max_gap": 260,
                "soft_line_fill_max_anchor_gap": 420,
                "soft_lane_fill_solid_max_gap": 480,
                "soft_lane_fill_max_anchor_gap": 480,
                "soft_track_bridge_gap": 1400,
                "adaptive_track_bridge_gap": 80,
                "adaptive_track_solid_gap": 18,
                "emulsion_track_bridge_gap": 1200,
                "emulsion_track_solid_gap": 300,
                "final_close_height": 5,
            }

        for k, v in vals.items():
            self.p[k] = v

    def _apply_temporal_confidence(self):
        level = str(self.temporal_confidence.currentData() or "medium")

        if level == "low":
            self.p["temporal_min_votes"] = 1
            self.p["temporal_max_dx"] = 34.0
            self.p["temporal_min_y_overlap"] = 0.12
            self.p["temporal_max_slope_delta"] = 0.42
            self.p["temporal_strong_score"] = 24.0
            self.p["temporal_strong_mean_response"] = 0.0030
        elif level == "high":
            self.p["temporal_min_votes"] = 3
            self.p["temporal_max_dx"] = 22.0
            self.p["temporal_min_y_overlap"] = 0.24
            self.p["temporal_max_slope_delta"] = 0.26
            self.p["temporal_strong_score"] = 42.0
            self.p["temporal_strong_mean_response"] = 0.0050
        else:
            self.p["temporal_min_votes"] = 2
            self.p["temporal_max_dx"] = 28.0
            self.p["temporal_min_y_overlap"] = 0.18
            self.p["temporal_max_slope_delta"] = 0.35
            self.p["temporal_strong_score"] = 34.0
            self.p["temporal_strong_mean_response"] = 0.0040

    def update_params_from_ui(self):
        self.p["dust"] = self.dust.value()
        self.p["dust_abs"] = self.dust.value()
        self.p["enable_dust_repair"] = self.dust_repair.isChecked()
        self.p["preset_mask"] = self.preset_mask

        s = float(self.scratch.value())
        ss = float(self.soft_scratch.value())
        soft_enabled = bool(self.enable_soft_scratches.isChecked())

        # Thin/sharp sensitivity.
        self.p["thin_abs"] = 0.0070 * s
        self.p["thin_rel"] = 0.110 * s
        self.p["thin_min_mean_response"] = 0.0045 * max(0.70, s)

        sharp_width = int(self.sharp_repair_width.value())
        if sharp_width % 2 == 0:
            sharp_width += 1
        sharp_width = max(1, min(9, sharp_width))
        self.p["thin_repair_width"] = sharp_width
        self.p["thin_max_repair_width"] = sharp_width

        # True soft on/off diagnostic.
        self.p["enable_soft_detection"] = soft_enabled
        self.p["enable_soft_lane_recovery"] = soft_enabled
        self.p["enable_soft_line_fill"] = soft_enabled
        self.p["enable_soft_track_bridging"] = soft_enabled
        self.p["enable_adaptive_track_fill"] = soft_enabled
        self.p["enable_irregular_emulsion_damage"] = soft_enabled
        self.p["enable_emulsion_vertical_track"] = soft_enabled

        # Context veto must stay enabled even when Soft Scratch Repair is off,
        # because scene vertical elements can also fool the thin/sharp detector.
        self.p["enable_context_edge_veto"] = True
        self.p["enable_thin_context_edge_veto"] = True

        # Soft sensitivity.
        self.p["soft_abs"] = 0.0045 * ss
        self.p["soft_rel"] = 0.055 * ss
        self.p["soft_min_mean_response"] = 0.0022 * max(0.50, ss)
        self.p["soft_min_coverage"] = max(0.018, 0.030 * ss)
        self.p["soft_min_density_excess"] = max(0.004, 0.008 * ss)

        # Faint recovery follows Soft Scratch sensitivity, but remains narrow.
        self.p["enable_faint_scratch_recovery"] = soft_enabled
        self.p["faint_abs"] = 0.0018 * max(0.55, ss)
        self.p["faint_noise_mul"] = 0.55 + 0.25 * max(0.0, ss - 1.0)
        self.p["faint_min_mean_response"] = 0.0012 * max(0.55, ss)
        self.p["faint_min_active_fraction"] = max(0.035, 0.055 * ss)

        width = int(self.max_repair_width.value())
        if width % 2 == 0:
            width += 1
        effective_soft_width = max(1, width)

        # Soft / adaptive / emulsion repair width caps.
        self.p["soft_max_repair_width"] = effective_soft_width
        self.p["soft_lane_repair_width"] = effective_soft_width
        self.p["soft_track_repair_width"] = effective_soft_width
        self.p["soft_line_min_repair_width"] = max(1, min(effective_soft_width, 5))
        self.p["soft_track_min_repair_width"] = max(1, min(effective_soft_width, 5))
        self.p["soft_lane_min_repair_width"] = max(1, min(effective_soft_width, 5))
        self.p["soft_repair_width"] = max(1, min(effective_soft_width, 9))
        self.p["soft_repair_extra"] = 0.0
        self.p["soft_lane_fill_half_width"] = max(2, min(10, effective_soft_width // 2 + 1))
        self.p["soft_eval_half_width"] = max(3, min(10, effective_soft_width // 4 + 1))

        self.p["emulsion_repair_width"] = effective_soft_width
        self.p["emulsion_min_repair_width"] = max(1, min(effective_soft_width, 5))
        self.p["emulsion_track_repair_width"] = effective_soft_width
        self.p["emulsion_track_min_repair_width"] = max(1, min(effective_soft_width, 5))
        self.p["emulsion_track_fill_half_width"] = max(2, min(10, effective_soft_width // 2 + 1))

        # Intermittent detection responsiveness.
        self.p["thin_ridge_noise_tolerance"] = 0.035 + 0.050 * max(0.0, 1.0 - s)
        self.p["soft_ridge_noise_tolerance"] = 0.070 + 0.160 * max(0.0, 1.0 - ss)
        self.p["thin_window_min_mean_response"] = 0.0036 * max(0.65, s)
        self.p["soft_window_min_mean_response"] = 0.0020 * max(0.55, ss)
        self.p["thin_window_min_coverage"] = max(0.070, 0.095 * s)
        self.p["soft_window_min_coverage"] = max(0.050, 0.080 * ss)
        self.p["thin_window_min_density_excess"] = max(0.010, 0.015 * s)
        self.p["soft_window_min_density_excess"] = max(0.006, 0.012 * ss)
        self.p["thin_window_min_log_nfa"] = 6.0 + 4.0 * s
        self.p["soft_window_min_log_nfa"] = 6.0 + 5.0 * ss

        self.p["adaptive_track_abs"] = 0.0042 * max(0.55, ss)
        self.p["adaptive_track_weak_abs"] = 0.0024 * max(0.55, ss)
        self.p["adaptive_track_noise_mul"] = 0.70 + 0.35 * ss
        self.p["adaptive_track_weak_noise_mul"] = 0.42 + 0.22 * ss
        self.p["adaptive_track_half_width"] = max(4, min(12, effective_soft_width // 2 + 4))
        self.p["adaptive_track_min_repair_width"] = max(1, min(effective_soft_width, 5))
        self.p["adaptive_track_extra_width"] = 0

        # Guided high-level controls.
        self._apply_detail_protection()
        self._apply_gap_bridging()
        self._apply_temporal_confidence()

        # Max scratch area % -> fractions.
        total_cap = max(0.001, float(self.max_scratch_area.value()) / 100.0)
        self.p["scratch_max_mask_fraction"] = total_cap
        self.p["thin_max_mask_fraction"] = min(0.0060, total_cap)
        self.p["soft_max_mask_fraction"] = min(0.0350, total_cap)

    # ------------------------------------------------------------------
    # Preview cache / overlay helpers
    # ------------------------------------------------------------------

    def _extract_restore_payload(self, result) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Accept multiple restore return formats.

        Supported:
            out, mask
            out, mask, extra
            {"image": out, "mask": mask, "preview_masks": {...}}
        """
        out = None
        mask = None
        extra = None

        if isinstance(result, dict):
            out = result.get("image") or result.get("out")
            mask = result.get("mask")
            extra = result.get("preview_masks") or result.get("extra") or result
        elif isinstance(result, tuple):
            if len(result) >= 2:
                out, mask = result[0], result[1]
            if len(result) >= 3:
                extra = result[2]
        else:
            raise ValueError("Unsupported restore_frame return format")

        if out is None or mask is None:
            raise ValueError("restore_frame did not return image and mask")

        preview_masks = self._normalize_preview_masks(mask, extra)
        return out, mask, preview_masks

    def _normalize_preview_masks(self, combined_mask, extra: Optional[Dict]) -> Dict[str, np.ndarray]:
        base = (combined_mask > 0.5).astype(np.float32)
        h, w = base.shape[:2]

        def as_mask(arr, default=None):
            if arr is None:
                if default is None:
                    return np.zeros((h, w), dtype=np.float32)
                return default.astype(np.float32)
            m = arr
            if m.ndim == 3:
                m = m[:, :, 0]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            return (m > 0.5).astype(np.float32)

        masks = {"all": base}
        if isinstance(extra, dict):
            src = extra.get("preview_masks", extra)
            if isinstance(src, dict):
                masks["all"] = as_mask(
                    src.get("all", src.get("combined", src.get("mask", base))),
                    default=base,
                )
                masks["sharp"] = as_mask(
                    src.get("sharp", src.get("thin", src.get("scratch", masks["all"]))),
                    default=masks["all"],
                )
                masks["soft"] = as_mask(
                    src.get("soft", src.get("soft_wide", src.get("soft_mask", masks["all"]))),
                    default=masks["all"],
                )
                masks["dust"] = as_mask(src.get("dust", None))
            else:
                masks["sharp"] = masks["all"]
                masks["soft"] = masks["all"]
                masks["dust"] = np.zeros((h, w), dtype=np.float32)
        else:
            masks["sharp"] = masks["all"]
            masks["soft"] = masks["all"]
            masks["dust"] = np.zeros((h, w), dtype=np.float32)

        # Always ensure keys exist.
        for key in ("all", "sharp", "soft", "dust"):
            if key not in masks:
                masks[key] = np.zeros((h, w), dtype=np.float32)

        return masks

    def _current_overlay_mode(self) -> str:
        return str(self.overlay_view.currentData() or "all")

    def _current_overlay_mask(self, preview_masks: Dict[str, np.ndarray]) -> np.ndarray:
        mode = self._current_overlay_mode()
        if mode not in preview_masks:
            mode = "all"
        return preview_masks.get(mode, preview_masks.get("all"))

    def _overlay_label(self) -> str:
        mapping = {
            "all": "All",
            "sharp": "Sharp",
            "soft": "Soft",
            "dust": "Dust",
        }
        return mapping.get(self._current_overlay_mode(), "All")

    def cache_temporal_preview(self, original, restored, preview_masks, percent):
        self.preview_cache = {
            "idx": int(self.idx),
            "original": original.copy(),
            "restored": restored.copy(),
            "preview_masks": {k: v.copy() for k, v in preview_masks.items()},
            "percent": float(percent),
        }

    def show_result_with_overlay(self, original, restored, preview_masks):
        active_mask = self._current_overlay_mask(preview_masks)
        mask = active_mask.astype(np.float32)

        if self.preset_mask is not None:
            mask *= self.preset_mask
            restored = original * (1.0 - self.preset_mask[..., None]) + restored * self.preset_mask[..., None]

        left = original.copy()
        if self.overlay.isChecked():
            overlay_mask = mask > 0.5
            left[overlay_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            edge = cv2.dilate(
                overlay_mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                iterations=1,
            ).astype(bool) & ~overlay_mask
            left[edge] = np.array([1.0, 1.0, 0.0], dtype=np.float32)

        self.v1.set_image(left)
        self.v2.set_image(restored)

    def redraw_temporal_preview_overlay(self):
        if not self.files:
            return

        if (
            self.preview_cache is not None
            and int(self.preview_cache.get("idx", -1)) == int(self.idx)
        ):
            self.show_result_with_overlay(
                self.preview_cache["original"],
                self.preview_cache["restored"],
                self.preview_cache["preview_masks"],
            )
            state = "with overlay" if self.overlay.isChecked() else "without overlay"
            label = self._overlay_label()
            self.status.setText(
                f"Temporal preview shown {state}. Overlay View: {label}. "
                f"Mask area: {self.preview_cache['percent']:.3f}%"
            )
            return

        self.update_view()

    # ------------------------------------------------------------------
    # View / sequence helpers
    # ------------------------------------------------------------------

    def sync_views(self, x, y):
        self.v1.x = x
        self.v1.y = y
        self.v2.x = x
        self.v2.y = y
        self.v1.update_view()
        self.v2.update_view()

    def load(self):
        folder = QFileDialog.getExistingDirectory(self, "Select", load_default_dir())
        if not folder:
            return

        save_default_dir(folder)
        self.files = sorted(glob.glob(os.path.join(folder, "*.exr")))
        self.idx = 0
        self.preset_mask = None
        self.preset_mask_path = None
        self.preset_mask_mode = None
        self.preview_cache = None
        self.update_mask_label()

        img = self.get_frame(self.idx)
        if img is not None:
            self.v1.set_image(img)
            self.v2.set_image(img)
            self.update_frame_label()
            self.update_render_stats_label()
            self.auto_load_selected_frame_mask_after_sequence_load()
        else:
            self.update_frame_label()
            self.update_render_stats_label()
            self.status.setText("No EXR frames found.")

    def restore_tool_mask_dir(self):
        """Directory where restore_tool/main.py and app.py live."""
        return os.path.dirname(os.path.abspath(__file__))

    def builtin_frame_mask_path(self, mode):
        names = {
            "16mm": "16mm_mask.png",
            "35mm": "35mm_mask.png",
        }
        name = names.get(str(mode))
        if not name:
            return None

        tool_dir = self.restore_tool_mask_dir()
        candidates = [
            os.path.join(tool_dir, name),
            os.path.join(os.path.dirname(tool_dir), name),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    def update_mask_label(self):
        if self.preset_mask_path:
            mode = self.preset_mask_mode or "custom"
            self.mask_label.setText(f"Frame Mask: {mode} — {self.preset_mask_path}")
        else:
            self.mask_label.setText("Frame Mask: —")

    def load_frame_mask_from_path(self, path, mode_label):
        if not self.files:
            QMessageBox.warning(self, "No sequence loaded", "Load an EXR sequence first.")
            return False

        if not path or not os.path.isfile(path):
            QMessageBox.warning(
                self,
                "Frame mask not found",
                f"Could not find the {mode_label} frame mask:\n{path}"
            )
            return False

        frame = self.get_frame(self.idx)
        if frame is None:
            QMessageBox.warning(self, "No frame loaded", "Could not read the current frame.")
            return False

        try:
            self.preset_mask = load_preset_mask(path, frame.shape)
            self.preset_mask_path = path
            self.preset_mask_mode = mode_label
        except Exception as e:
            QMessageBox.critical(self, "Mask load error", str(e))
            return False

        self.preview_cache = None
        self.update_mask_label()
        self.status.setText(f"Using {mode_label} frame mask: {os.path.basename(path)}")
        self.update_view()
        return True

    def load_builtin_frame_mask(self, mode):
        path = self.builtin_frame_mask_path(mode)
        label = "16mm" if mode == "16mm" else "35mm"
        return self.load_frame_mask_from_path(path, label)

    def load_custom_frame_mask(self):
        if not self.files:
            QMessageBox.warning(self, "No sequence loaded", "Load an EXR sequence first.")
            return False

        # Do not use save_default_dir() here. Custom mask selection must not
        # modify ~/.default_dir, which is reserved for EXR sequence loading.
        start_dir = os.path.dirname(self.files[self.idx]) if self.files else os.path.expanduser("~")

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select custom black/white frame mask",
            start_dir,
            "Mask images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if not path:
            self.status.setText("Custom frame mask selection cancelled. Previous mask unchanged.")
            return False

        return self.load_frame_mask_from_path(path, "custom")

    def load_selected_frame_mask(self):
        mode = self.frame_mask_combo.currentData() or "16mm"
        if mode == "custom":
            return self.load_custom_frame_mask()
        return self.load_builtin_frame_mask(mode)

    def auto_load_selected_frame_mask_after_sequence_load(self):
        mode = self.frame_mask_combo.currentData() or "16mm"

        if mode == "custom":
            self.status.setText("Sequence loaded. Custom frame mask selected; press Load Frame Mask to choose it.")
            self.update_mask_label()
            return

        ok = self.load_builtin_frame_mask(mode)
        if ok:
            self.status.setText(f"Sequence loaded. {mode} frame mask loaded. Use Temporal Preview.")
        else:
            self.status.setText(
                f"Sequence loaded, but the {mode} frame mask was not found. "
                "Choose another Frame Mask or use Custom."
            )

    def on_frame_mask_selection_changed(self):
        if not self.files:
            self.status.setText("Frame Mask selection changed. Load an EXR sequence to apply it.")
            return

        mode = self.frame_mask_combo.currentData() or "16mm"
        if mode == "custom":
            self.load_custom_frame_mask()
        else:
            self.load_builtin_frame_mask(mode)

    # Backward-compatible button/method name.
    def load_mask(self):
        return self.load_selected_frame_mask()

    def get_frame(self, i):
        if not self.files:
            return None
        i = max(0, min(int(i), len(self.files) - 1))
        return self.cache.get(self.files[i])

    def update_frame_label(self):
        if not self.files:
            self.frame_label.setText("Frame: —")
            self.directory_label.setText("Directory: —")
            self.update_mask_label()
            return
        name = os.path.basename(self.files[self.idx])
        folder = os.path.dirname(self.files[self.idx])
        self.frame_label.setText(f"Frame {self.idx + 1}/{len(self.files)}: {name}")
        self.directory_label.setText(f"Directory: {folder}")

    def get_temporal_frames(self, center_index):
        win = int(self.frames.value())
        if win % 2 == 0:
            win += 1
        half = win // 2
        return [self.get_frame(center_index + offset) for offset in range(-half, half + 1)]

    def update_view(self):
        if not self.files:
            return

        # Any parameter or frame change invalidates the cached temporal preview.
        self.preview_cache = None

        img = self.get_frame(self.idx)
        if img is not None:
            self.v1.set_image(img)
            self.v2.set_image(img)
        self.update_frame_label()
        self.status.setText("Use Temporal Preview to evaluate the real scratch repair pipeline.")

    def keyPressEvent(self, event):
        if not self.files:
            return
        if event.key() == Qt.Key_Right:
            self.step(1)
        elif event.key() == Qt.Key_Left:
            self.step(-1)

    def step(self, d):
        if not self.files:
            return
        self.idx = max(0, min(self.idx + d, len(self.files) - 1))
        self.update_view()

    # ------------------------------------------------------------------
    # Preview / process
    # ------------------------------------------------------------------

    def temporal_preview(self):
        if not self.files:
            return
        if self.preset_mask is None:
            QMessageBox.warning(self, "No preset mask loaded", "Load a black/white preset mask first.")
            return

        self.update_params_from_ui()
        self.update_frame_label()
        frames = self.get_temporal_frames(self.idx)
        if any(f is None for f in frames):
            QMessageBox.warning(self, "Preview error", "Could not load all frames for temporal preview.")
            return

        self.status.setText("Running temporal preview...")
        QApplication.processEvents()

        result = restore_frame(frames, self.p, self.idx, None)
        out, mask, preview_masks = self._extract_restore_payload(result)
        center = frames[len(frames) // 2]

        valid_area = np.sum(self.preset_mask > 0.5)
        detected_area = np.sum((preview_masks["all"] * self.preset_mask) > 0.5)
        percent = 100.0 * detected_area / valid_area if valid_area > 0 else 0.0

        self.cache_temporal_preview(center, out, preview_masks, percent)
        self.show_result_with_overlay(center, out, preview_masks)

        state = "with overlay" if self.overlay.isChecked() else "without overlay"
        self.status.setText(
            f"Temporal preview done {state}. Overlay View: {self._overlay_label()}. "
            f"Mask area: {percent:.3f}%"
        )

    def cancel(self):
        self.cancel_flag = True
        self.status.setText("Cancel requested")

    def process(self):
        if not self.files:
            return
        if self.preset_mask is None:
            QMessageBox.warning(self, "No preset mask loaded", "Load a black/white preset mask before processing.")
            return

        self.update_params_from_ui()
        self.cancel_flag = False
        self.preview_cache = None

        outdir = os.path.join(os.path.dirname(self.files[0]), "restored")
        os.makedirs(outdir, exist_ok=True)

        from .io_utils import write_exr

        total = len(self.files)
        mode = self.render_mode.currentData() or "skip"
        existing = self.count_existing_outputs(outdir)

        if mode == "skip":
            pending = total - existing
            if pending <= 0:
                self.progress.setMaximum(total)
                self.progress.setValue(total)
                self.status.setText("All output frames already exist. Nothing to do in resume mode.")
                self.update_render_stats_label()
                return
            self.status.setText(
                f"Resume mode: {existing} existing frame(s) will be skipped, {pending} frame(s) will be rendered."
            )
        else:
            self.status.setText(
                f"Rewrite mode: processing all {total} frame(s). Existing outputs will be overwritten."
            )

        self.progress.setMaximum(total)
        self.progress.setValue(0)
        QApplication.processEvents()

        written = 0
        skipped = 0

        for i in range(total):
            if self.cancel_flag:
                self.status.setText(
                    f"Cancelled. Wrote {written}, skipped {skipped}, remaining {total - i}."
                )
                self.update_render_stats_label()
                return

            name = os.path.basename(self.files[i])
            path = os.path.join(outdir, name)
            self.frame_label.setText(f"Processing {i + 1}/{total}: {name}")

            if mode == "skip" and self.output_frame_exists(path):
                skipped += 1
                print(f"Skipping existing frame {i + 1}/{total}: {name}")
                self.progress.setValue(i + 1)
                self.status.setText(
                    f"Skipping existing {i + 1}/{total}: {name}  |  skipped={skipped}, written={written}"
                )
                QApplication.processEvents()
                continue

            print(f"Processing {i + 1}/{total}: {name}")
            frames = self.get_temporal_frames(i)
            if any(f is None for f in frames):
                print("Skipping frame due to read error")
                self.progress.setValue(i + 1)
                self.status.setText(f"Read error on {i + 1}/{total}: {name}")
                QApplication.processEvents()
                continue

            result = restore_frame(frames, self.p, i, None)
            out, mask, _preview_masks = self._extract_restore_payload(result)
            center = frames[len(frames) // 2]

            out = center * (1.0 - self.preset_mask[..., None]) + out * self.preset_mask[..., None]
            mask = (mask > 0.5).astype(np.float32) * self.preset_mask

            if self.p.get("debug_repair", False):
                diff = np.mean(np.abs(out - center), axis=2)
                print(
                    f"[APP DEBUG] Before write: max diff={float(np.max(diff)):.6f}, "
                    f"mean diff={float(np.mean(diff)):.6f}, "
                    f"changed pixels={int(np.sum(diff > 1e-6))}"
                )

            self.status.setText(f"Writing {i + 1}/{total}: {name}")
            QApplication.processEvents()
            write_exr(path, out)
            written += 1

            self.progress.setValue(i + 1)
            remaining = total - (i + 1)
            self.status.setText(
                f"Wrote {i + 1}/{total}: {name}  |  written={written}, skipped={skipped}, remaining={remaining}"
            )
            QApplication.processEvents()

        if mode == "skip":
            self.status.setText(
                f"Done. Resume mode wrote {written} new frame(s) and skipped {skipped} existing frame(s)."
            )
        else:
            self.status.setText(f"Done. Rewrite mode wrote {written} frame(s).")
        self.update_render_stats_label()
