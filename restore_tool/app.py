"""
app_ac_v13.py

Clean GUI for the a-contrario-style scratch detector.

Only practical controls are exposed:
    Dust
    Scratch
    Soft Scratch
    Max Repair Width
    Frames
    Dust Repair
    Overlay
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt
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
        self.cancel_flag = False

        # Compact parameter set.  The GUI scales the important thresholds.
        self.p = {
            # Diagnostics / workflow
            "enable_dust_repair": False,
            "debug_repair": True,
            "debug_timing": True,

            # Valid area / borders
            "ignore_frame_border_px": 96,
            "mask_edge_erode": 24,
            "mask_boundary_safety_px": 128,

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

            # Hough/segment grouping.  These are not full-height columns; they
            # find partial, slanted, intermittent segments.
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
            # Fill weaker rows along accepted soft Hough lines.
            # This reduces the dotted/gapped mask without drawing full columns.
            "enable_soft_line_fill": True,
            "soft_line_fill_max_anchor_gap": 420,
            "soft_line_fill_abs": 0.00035,
            "soft_line_fill_rel_factor": 0.14,
            "soft_line_fill_solid_abs": 0.00018,
            "soft_line_fill_solid_max_gap": 260,
            "soft_line_fill_min_rows": 8,
            "soft_line_min_repair_width": 23,
            # Track-level bridging between accepted soft segments on the same
            # scratch. This fills remaining intermittent gaps after v7.
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
            "enable_edge_artifact_cleanup": True,
            "edge_artifact_boundary_dist": 140.0,
            "edge_artifact_frame_strip": 120,
            "edge_artifact_max_width": 54,
            "edge_artifact_min_area": 250,
            "edge_artifact_max_fill": 0.28,
            "edge_artifact_min_aspect_keep": 4.0,
            "soft_track_max_abs_slope": 0.18,
            # Irregular scratched-emulsion fragments:
            # thicker, softer, non-ridge pieces that the line detector rejects.
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
            # This catches vertical-ish leftover emulsion fragments that are
            # not clean enough to be detected as ridge/Hough lines.
            "enable_emulsion_vertical_track": True,
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
            # This groups accepted AC soft seed fragments into near-vertical
            # lanes, but the final mask still uses supported rows only.
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

            # Separate budgets: thin is protected, soft is secondary.
            "thin_max_mask_fraction": 0.0060,
            "soft_max_mask_fraction": 0.0380,
            "scratch_max_mask_fraction": 0.0500,
            "final_close_height": 5,
            "final_close_width": 1,

            # Repair parameters used by restore.py.
            "scratch_interp_radius": 42,
            "scratch_interp_sample": 8,
            "scratch_interp_strength": 1.0,
            "scratch_interp_max_run": 90,
            "scratch_inpaint_after_interp": False,
            "inpaint_radius": 3,

            # Dust repair remains optional.
            "dust": 0.045,
            "dust_inpaint_radius": 3,
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

        self.dust_repair = QCheckBox("Dust Repair")
        self.dust_repair.setChecked(False)
        self.dust_repair.setToolTip("Keep off while tuning scratches.")

        self.overlay = QCheckBox("Overlay")
        self.overlay.setChecked(True)

        for widget in (self.dust, self.scratch, self.soft_scratch, self.max_repair_width, self.frames):
            widget.valueChanged.connect(self.update_view)
        self.dust_repair.stateChanged.connect(self.update_view)
        self.overlay.stateChanged.connect(self.update_view)

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
        load_mask = QPushButton("Load Mask")
        temporal_preview = QPushButton("Temporal Preview")
        process = QPushButton("Process")
        cancel = QPushButton("Cancel")
        buttons.addWidget(load)
        buttons.addWidget(load_mask)
        buttons.addWidget(temporal_preview)
        buttons.addWidget(process)
        buttons.addWidget(cancel)
        load.clicked.connect(self.load)
        load_mask.clicked.connect(self.load_mask)
        temporal_preview.clicked.connect(self.temporal_preview)
        process.clicked.connect(self.process)
        cancel.clicked.connect(self.cancel)

        self.progress = QProgressBar()
        self.frame_label = QLabel("Frame: —")
        self.frame_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.frame_label.setStyleSheet("font-weight: bold;")
        self.status = QLabel("Idle")

        left = QVBoxLayout()
        left.addLayout(views)
        left.addLayout(nav)
        left.addLayout(buttons)
        left.addWidget(self.frame_label)
        left.addWidget(self.progress)
        left.addWidget(self.status)

        controls = QFormLayout()
        controls.addRow("Dust", self.dust)
        controls.addRow("Scratch", self.scratch)
        controls.addRow("Soft Scratch", self.soft_scratch)
        controls.addRow("Max Repair Width", self.max_repair_width)
        controls.addRow("Frames", self.frames)
        controls.addRow(self.dust_repair)
        controls.addRow(self.overlay)

        layout = QHBoxLayout()
        layout.addLayout(left, 4)
        layout.addLayout(controls, 1)
        self.setLayout(layout)
        self.setMinimumSize(1200, 800)

    def update_params_from_ui(self):
        self.p["dust"] = self.dust.value()
        self.p["enable_dust_repair"] = self.dust_repair.isChecked()
        self.p["preset_mask"] = self.preset_mask

        # Lower GUI value = more sensitive.
        s = float(self.scratch.value())
        ss = float(self.soft_scratch.value())

        self.p["thin_abs"] = 0.0070 * s
        self.p["thin_rel"] = 0.110 * s
        self.p["thin_min_mean_response"] = 0.0045 * max(0.70, s)

        # Soft Scratch now has lower floors so the control actually helps
        # faint/wide scratches, while slope + support geometry keep diagonals out.
        self.p["soft_abs"] = 0.0045 * ss
        self.p["soft_rel"] = 0.055 * ss
        self.p["soft_min_mean_response"] = 0.0022 * max(0.50, ss)
        self.p["soft_min_coverage"] = max(0.018, 0.030 * ss)
        self.p["soft_min_density_excess"] = max(0.004, 0.008 * ss)

        width = int(self.max_repair_width.value())
        if width % 2 == 0:
            width += 1

        # The soft detector estimates local width per accepted line, and the GUI
        # caps that width.  Do not make every accepted line blindly this wide.
        # Soft/wide scratches need a practical minimum repair width.
        # Reducing the GUI width below this should not make soft tracks only
        # 5–11 px wide; use edge cleanup and budgets to control false positives.
        effective_soft_width = max(width, 27)

        self.p["soft_max_repair_width"] = effective_soft_width
        self.p["soft_lane_repair_width"] = effective_soft_width
        self.p["soft_track_repair_width"] = effective_soft_width
        self.p["soft_track_min_repair_width"] = max(23, min(effective_soft_width, 27))
        self.p["soft_lane_fill_half_width"] = max(10, min(22, effective_soft_width // 2))
        self.p["soft_repair_width"] = min(effective_soft_width, 13)
        self.p["soft_eval_half_width"] = max(3, min(10, effective_soft_width // 4))
        self.p["soft_line_min_repair_width"] = max(17, min(effective_soft_width, 23))
        self.p["emulsion_repair_width"] = effective_soft_width
        self.p["emulsion_min_repair_width"] = max(23, min(effective_soft_width, 25))
        self.p["emulsion_track_repair_width"] = effective_soft_width
        self.p["emulsion_track_min_repair_width"] = max(23, min(effective_soft_width, 25))
        self.p["emulsion_track_fill_half_width"] = max(12, min(24, effective_soft_width // 2 + 2))

        # Keep mask caps fixed here so preview changes do not silently undo
        # the self.p values above.
        self.p["thin_max_mask_fraction"] = 0.0060
        self.p["soft_max_mask_fraction"] = 0.0600
        self.p["scratch_max_mask_fraction"] = 0.0800

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

        img = self.get_frame(self.idx)
        if img is not None:
            self.v1.set_image(img)
            self.v2.set_image(img)
            self.update_frame_label()
            self.status.setText("Sequence loaded. Load a preset mask, then use Temporal Preview.")
        else:
            self.update_frame_label()
            self.status.setText("No EXR frames found.")

    def load_mask(self):
        if not self.files:
            QMessageBox.warning(self, "No sequence loaded", "Load an EXR sequence first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select black/white preset mask",
            load_default_dir(),
            "Mask images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if not path:
            return

        frame = self.get_frame(self.idx)
        try:
            self.preset_mask = load_preset_mask(path, frame.shape)
            self.preset_mask_path = path
        except Exception as e:
            QMessageBox.critical(self, "Mask load error", str(e))
            return

        self.status.setText(f"Using preset mask: {os.path.basename(path)}")
        self.update_view()

    def get_frame(self, i):
        if not self.files:
            return None
        i = max(0, min(int(i), len(self.files) - 1))
        return self.cache.get(self.files[i])

    def update_frame_label(self):
        if not self.files:
            self.frame_label.setText("Frame: —")
            return
        name = os.path.basename(self.files[self.idx])
        self.frame_label.setText(f"Frame {self.idx + 1}/{len(self.files)}: {name}")

    def get_temporal_frames(self, center_index):
        win = int(self.frames.value())
        if win % 2 == 0:
            win += 1
        half = win // 2
        return [self.get_frame(center_index + offset) for offset in range(-half, half + 1)]

    def show_result_with_overlay(self, original, restored, mask):
        mask = mask.astype(np.float32)
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

        out, mask = restore_frame(frames, self.p, self.idx, None)
        center = frames[len(frames) // 2]
        self.show_result_with_overlay(center, out, mask)

        valid_area = np.sum(self.preset_mask > 0.5)
        detected_area = np.sum((mask * self.preset_mask) > 0.5)
        percent = 100.0 * detected_area / valid_area if valid_area > 0 else 0.0
        self.status.setText(f"Temporal preview done. Mask area: {percent:.3f}%")

    def update_view(self):
        if not self.files:
            return
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

        outdir = os.path.join(os.path.dirname(self.files[0]), "restored")
        os.makedirs(outdir, exist_ok=True)

        from .io_utils import write_exr

        total = len(self.files)
        self.progress.setMaximum(total)
        self.progress.setValue(0)
        self.status.setText("Processing...")
        QApplication.processEvents()

        for i in range(total):
            if self.cancel_flag:
                self.status.setText("Cancelled")
                return

            name = os.path.basename(self.files[i])
            self.frame_label.setText(f"Processing {i + 1}/{total}: {name}")
            print(f"Processing {i + 1}/{total}: {name}")
            frames = self.get_temporal_frames(i)
            if any(f is None for f in frames):
                print("Skipping frame due to read error")
                continue

            out, mask = restore_frame(frames, self.p, i, None)
            center = frames[len(frames) // 2]

            out = center * (1.0 - self.preset_mask[..., None]) + out * self.preset_mask[..., None]
            mask = mask * self.preset_mask

            if self.p.get("debug_repair", False):
                diff = np.mean(np.abs(out - center), axis=2)
                print(
                    f"[APP DEBUG] Before write: max diff={float(np.max(diff)):.6f}, "
                    f"mean diff={float(np.mean(diff)):.6f}, "
                    f"changed pixels={int(np.sum(diff > 1e-6))}"
                )

            path = os.path.join(outdir, name)
            self.status.setText(f"Writing {i + 1}/{total}: {name}")
            QApplication.processEvents()
            write_exr(path, out)

            self.progress.setValue(i + 1)
            self.status.setText(f"Wrote {i + 1}/{total}")
            QApplication.processEvents()

        self.status.setText("Done")
