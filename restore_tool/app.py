import os
import glob
import cv2
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from .viewer import Viewer
from .io_utils import load_default_dir, save_default_dir
from .cache import FrameCache
from .processing.restore import restore_frame
from .processing.mask import load_preset_mask


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.files = []
        self.idx = 0
        self.cache = FrameCache()
        self.preset_mask = None
        self.preset_mask_path = None
        self.cancel_flag = False

        # Keep this dictionary small. Most values are internal defaults for the
        # segment-based scratch detector; the GUI exposes only the practical ones.
        self.p = {
            # Workflow / diagnostics
            "enable_dust_repair": False,
            "debug_repair": True,
            "debug_timing": True,

            # Negative-film scratches: bright / white only.
            # The Scratch GUI multiplies scratch_abs and scratch_rel.
            "scratch_abs": 0.0075,
            "scratch_rel": 0.12,
            "scratch_rel_floor": 0.025,
            "scratch_strong_abs": 0.030,
            "scratch_strong_rel": 0.12,

            # Horizontal profile test.
            "profile_side_width": 5,
            "profile_side_gap": 2,
            "profile_widths": (1, 3, 5),
            "profile_side_coherence": 0.018,
            "profile_side_coherence_rel": 0.30,
            "profile_side_noise_mul": 1.75,
            "profile_local_noise_box": 51,
            "profile_noise_bg_width": 31,
            "scratch_noise_mul": 1.75,
            "scratch_peak_width": 9,
            "scratch_peak_eps": 1e-5,

            # Local vertical support validation.
            # This replaces the old full-column/anchor approach.
            "support_seed_x_dilate": 1,
            "support_vertical_blur": 81,
            "support_bg_width": 81,
            "support_min": 0.015,
            "support_excess": 0.007,
            "support_rel_factor": 1.90,
            "support_strong_min": 0.010,

            # Segment validation. These are local/background-normalized;
            # they should reject bright face texture better than global budgets.
            "segment_connect_height": 31,
            "segment_connect_width": 3,
            "segment_min_area": 6,
            "segment_min_height": 28,
            "segment_max_width": 22,
            "segment_min_aspect": 2.8,
            "segment_max_abs_slope": 0.75,
            "segment_min_local_z": 2.5,
            "segment_min_density_ratio": 2.0,
            "segment_bg_pad_x": 50,
            "segment_bg_pad_y": 30,
            "segment_bg_guard_x": 8,
            "segment_line_width_for_density": 5,
            "segment_strong_response": 0.030,
            "segment_max_component_fraction": 0.0008,

            # Temporal linking: local window only, with drift allowed.
            "track_min_votes": 2,
            "track_neighbor_radius": 2,
            "track_max_dx_per_frame": 18.0,
            "track_min_y_overlap": 0.15,
            "track_max_width_ratio": 4.0,
            "track_max_slope_delta": 0.60,
            "track_strong_score": 0.060,
            "track_strong_max_response": 0.035,
            "track_strong_local_z": 6.0,

            # Mask safety and repair. Keep repair narrow; restore.py does interpolation.
            "scratch_max_mask_fraction": 0.0020,
            "scratch_repair_width": 1,
            "scratch_repair_extra_width": 0,
            "scratch_repair_max_width": 3,
            "scratch_repair_hard_max_width": 3,
            "scratch_repair_close_height": 3,
            "scratch_interp_radius": 36,
            "scratch_interp_sample": 6,
            "scratch_interp_strength": 1.0,
            "scratch_inpaint_after_interp": False,
            "inpaint_radius": 3,

            # Border handling. Keep enough border to avoid film gate edges.
            "ignore_frame_border_px": 32,
            "mask_edge_erode": 12,

            # Dust repair is optional and intentionally off while scratch tuning.
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
        self.scratch.setValue(1.00)
        self.scratch.setKeyboardTracking(False)
        self.scratch.setToolTip(
            "Scratch sensitivity multiplier. Lower values detect more scratches; higher values are safer."
        )

        self.frames = QSpinBox()
        self.frames.setRange(3, 9)
        self.frames.setSingleStep(2)
        self.frames.setValue(5)
        self.frames.setKeyboardTracking(False)
        self.frames.setToolTip("Temporal window used for local scratch tracking.")

        self.dust_repair = QCheckBox("Dust Repair")
        self.dust_repair.setChecked(False)
        self.dust_repair.setToolTip("Keep off while tuning scratches.")

        self.overlay = QCheckBox("Overlay")
        self.overlay.setChecked(True)

        self.dust.valueChanged.connect(self.update_view)
        self.scratch.valueChanged.connect(self.update_view)
        self.frames.valueChanged.connect(self.update_view)
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
        self.status = QLabel("Idle")

        left = QVBoxLayout()
        left.addLayout(views)
        left.addLayout(nav)
        left.addLayout(buttons)
        left.addWidget(self.progress)
        left.addWidget(self.status)

        controls = QFormLayout()
        controls.addRow("Dust", self.dust)
        controls.addRow("Scratch", self.scratch)
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

        # Lower Scratch value = more sensitive. Only the core profile thresholds
        # are scaled; the column/segment geometry stays stable.
        s = float(self.scratch.value())
        self.p["scratch_abs"] = 0.0070 * s
        self.p["scratch_rel"] = 0.11 * s
        self.p["scratch_strong_abs"] = 0.024 * max(0.85, s)

        # Keep the column-anchor gates from getting wildly permissive at low
        # Scratch values; otherwise face texture becomes candidate scratches.
        self.p["column_min_support"] = max(0.014, min(0.030, 0.018 * s))
        self.p["column_min_excess"] = max(0.0035, min(0.010, 0.005 * s))
        self.p["column_min_score"] = max(0.006, min(0.014, 0.009 * s))

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
        self.flow_cache = {}

        img = self.get_frame(self.idx)
        if img is not None:
            self.v1.set_image(img)
            self.v2.set_image(img)
            self.status.setText("Sequence loaded. Load a preset mask, then use Temporal Preview.")
        else:
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

            print(f"Processing {i + 1}/{total}")
            frames = self.get_temporal_frames(i)
            if any(f is None for f in frames):
                print("Skipping frame due to read error")
                continue

            out, mask = restore_frame(frames, self.p, i, None)
            center = frames[len(frames) // 2]

            # Keep repair only inside the preset mask.
            out = center * (1.0 - self.preset_mask[..., None]) + out * self.preset_mask[..., None]
            mask = mask * self.preset_mask

            if self.p.get("debug_repair", False):
                diff = np.mean(np.abs(out - center), axis=2)
                print(
                    f"[APP DEBUG] Before write: max diff={float(np.max(diff)):.6f}, "
                    f"mean diff={float(np.mean(diff)):.6f}, "
                    f"changed pixels={int(np.sum(diff > 1e-6))}"
                )

            name = os.path.basename(self.files[i])
            path = os.path.join(outdir, name)
            self.status.setText(f"Writing {i + 1}/{total}: {name}")
            QApplication.processEvents()
            write_exr(path, out)

            self.progress.setValue(i + 1)
            self.status.setText(f"Wrote {i + 1}/{total}")
            QApplication.processEvents()

        self.status.setText("Done")
