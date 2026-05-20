import os, glob
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from collections import deque
import numpy as np


from .viewer import Viewer
from .io_utils import load_default_dir, save_default_dir
from .cache import FrameCache

from .processing.restore import restore_frame
from .processing.preview import restore_preview
from .processing.mask import load_preset_mask

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.files = []
        self.idx = 0
        self.cache = FrameCache()
        self.preset_mask = None
        self.preset_mask_path = None
        self.flow_cache = {}

        self.p = {
            "max_mask_fraction": 0.015,
            "max_component_fraction": 0.0025,
            "min_component_area": 3
        }

        self.v1 = Viewer()
        self.v2 = Viewer()
        
        self.v1.sync = self.sync_views
        self.v2.sync = self.sync_views

        # UI 
        
        self.dust = QDoubleSpinBox()
        self.dust.setDecimals(4)
        self.dust.setRange(0.0000, 0.2000)
        self.dust.setSingleStep(0.0025)
        self.dust.setValue(0.0700)
        self.dust.setKeyboardTracking(False)

        self.scratch = QDoubleSpinBox()
        self.scratch.setDecimals(2)
        self.scratch.setRange(0.50, 8.00)
        self.scratch.setSingleStep(0.10)
        self.scratch.setValue(4.00)
        self.scratch.setKeyboardTracking(False)

        
        self.frames = QSpinBox()
        self.frames.setRange(3, 7)
        self.frames.setSingleStep(2)
        self.frames.setValue(3)
        self.frames.setKeyboardTracking(False)
        
        
        # Connect control changes to preview refresh
        self.dust.valueChanged.connect(self.update_view)
        self.scratch.valueChanged.connect(self.update_view)
        self.frames.valueChanged.connect(self.update_view)
        
        self.overlay = QCheckBox("Overlay")
        self.overlay.setChecked(True)
        self.overlay.stateChanged.connect(self.update_view)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        
        self.cancel_flag = False
        

        # ===== VIEWS (top) =====
        views = QHBoxLayout()
        views.addWidget(self.v1)
        views.addWidget(self.v2)
        
        # ===== NAVIGATION =====
        nav = QHBoxLayout()
        prev = QPushButton("<<")
        nextb = QPushButton(">>")
        nav.addWidget(prev)
        nav.addWidget(nextb)
        
        prev.clicked.connect(lambda: self.step(-1))
        nextb.clicked.connect(lambda: self.step(1))
        
        # ===== ACTION BUTTONS =====
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
        
        # ===== PROGRESS + STATUS =====
        self.progress = QProgressBar()
        self.status = QLabel("Idle")
        
        # ===== LEFT SIDE (everything stacked vertically) =====
        left = QVBoxLayout()
        left.addLayout(views)
        left.addLayout(nav)
        left.addLayout(buttons)
        left.addWidget(self.progress)
        left.addWidget(self.status)
        
        # ===== CONTROLS (RIGHT SIDE) =====
        controls = QFormLayout()
        controls.addRow("Dust", self.dust)
        controls.addRow("Scratch", self.scratch)
        controls.addRow("Frames", self.frames)
        #controls.addRow("Length", self.length)
        controls.addRow(self.overlay)
        
        # ===== MAIN LAYOUT =====
        layout = QHBoxLayout()
        layout.addLayout(left, 4)     # give more space to images
        layout.addLayout(controls, 1) # controls narrower
        
        self.setLayout(layout)
        
        # prevent insane resizing
        self.setMinimumSize(1200, 800)
        

        
        # controls.addRow("Frames", self.frames)
        # controls.addRow("Length", self.length)
        # controls.addRow(self.overlay)
                        

    def update_params_from_ui(self):
        self.p["dust"] = self.dust.value()
        self.p["scratch"] = self.scratch.value()
        
    def sync_views(self, x, y):
        self.v1.x = x
        self.v1.y = y
        self.v2.x = x
        self.v2.y = y

        self.v1.update_view()
        self.v2.update_view()

    def load(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select", load_default_dir()
        )
        if not folder:
            return

        save_default_dir(folder)

        self.files = sorted(glob.glob(os.path.join(folder, "*.exr")))
        self.idx = 0
        
        
        # Important: no automatic mask generation.
        self.preset_mask = None
        self.preset_mask_path = None

        img = self.get_frame(self.idx)
        if img is not None:
            self.v1.set_image(img)
            self.v2.set_image(img)
            
        if hasattr(self, "status"):
            self.status.setText("Sequence loaded. Please load a preset mask before preview/process.")

     
    def load_mask(self):
       if not self.files:
           QMessageBox.warning(
               self,
               "No sequence loaded",
               "Load an EXR sequence first, then load the preset mask."
           )
           return
    #
       path, _ = QFileDialog.getOpenFileName(
           self,
           "Select black/white preset mask",
           load_default_dir(),
           "Mask images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp);;All files (*)"
       )
    #
       if not path:
           return
    #
       frame = self.get_frame(self.idx)
    #
       try:
           self.preset_mask = load_preset_mask(path, frame.shape)
           self.preset_mask_path = path
       except Exception as e:
           QMessageBox.critical(
               self,
               "Mask load error",
               str(e)
           )
           return
    #
       print("Using preset mask:", path)
    #
       if hasattr(self, "status"):
           self.status.setText(f"Using preset mask: {os.path.basename(path)}")
    #
       self.update_view()



    def get_frame(self, i):
        i = max(0, min(i, len(self.files)-1))
        return self.cache.get(self.files[i])

    def show_result_with_overlay(self, original, restored, mask):
        """
        Show original + overlay on the left, restored frame on the right.
        """
        mask = mask.astype(np.float32)
    
        if self.preset_mask is not None:
            mask *= self.preset_mask
            restored = original * (1.0 - self.preset_mask[..., None]) + restored * self.preset_mask[..., None]
    
        left = original.copy()
    
        if self.overlay.isChecked():
            overlay_mask = mask > 0.5
    
            # Red fill for detected defects.
            left[overlay_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
            # Yellow outline around the detected defect.
            edge = cv2.dilate(
                overlay_mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                iterations=1
            ).astype(bool) & ~overlay_mask
    
            left[edge] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    
        self.v1.set_image(left)
        self.v2.set_image(restored)
        
    def temporal_preview(self):
        """
        Run the real temporal restoration pipeline only for the current frame.
    
        This is slower than update_view(), but it shows the actual mask generated
        by restore_frame(), including temporal dust detection.
        """
        if not self.files:
            return
    
        if self.preset_mask is None:
            QMessageBox.warning(
                self,
                "No preset mask loaded",
                "Load a black/white preset mask before using Temporal Preview."
            )
            return
    
        self.update_params_from_ui()

    
        win = self.frames.value()
        half = win // 2
    
        frames = []
    
        for offset in range(-half, half + 1):
            frame_index = self.idx + offset
            frame_index = max(0, min(frame_index, len(self.files) - 1))
            frames.append(self.get_frame(frame_index))
    
        if any(f is None for f in frames):
            QMessageBox.warning(
                self,
                "Preview error",
                "Could not load all frames needed for temporal preview."
            )
            return
    
        if hasattr(self, "status"):
            self.status.setText("Running temporal preview...")
    
        QApplication.processEvents()
    
        # Use a separate frame_idx key so preview cache does not conflict badly
        # with final processing cache.
        preview_key = self.idx
    
        out, mask = restore_frame(
            frames,
            self.p,
            preview_key,
            self.flow_cache
        )
    
        center = frames[len(frames) // 2]
    
        mask *= self.preset_mask
        valid_area = np.sum(self.preset_mask > 0.5)
        detected_area = np.sum(mask > 0.5)
        
        if valid_area > 0:
            percent = 100.0 * detected_area / valid_area
        else:
            percent = 0.0
        
        print(f"Temporal Preview mask area: {percent:.3f}%")
        
        out = center * (1.0 - self.preset_mask[..., None]) + out * self.preset_mask[..., None]
    
        self.show_result_with_overlay(center, out, mask)
    
        if hasattr(self, "status"):
            detected_pixels = int(np.sum(mask > 0.5))
            self.status.setText(f"Temporal preview done. Detected area: {percent:.3f}%")


    def update_view(self):
        if not self.files:
            return

        self.update_params_from_ui()

        c = self.get_frame(self.idx)

        if self.preset_mask is None:
            self.v1.set_image(c)
            self.v2.set_image(c)
        
            if hasattr(self, "status"):
                self.status.setText("Load a preset mask before previewing restoration.")
        
            return
        
        out, mask = restore_preview(c, self.p)
        self.show_result_with_overlay(c, out, mask)
        
    def keyPressEvent(self, event):
        if not self.files:
            return
    
        if event.key() == Qt.Key_Right:
            self.step(1)
    
        elif event.key() == Qt.Key_Left:
            self.step(-1)
            
    
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
        print("Cancel requested")
        self.cancel_flag = True


    def process(self):
        if not self.files:
            return
    
        if self.preset_mask is None:
            QMessageBox.warning(
                self,
                "No preset mask loaded",
                "Load a black/white preset mask before processing."
            )
            return
        
        self.update_params_from_ui()
        
        print("Process Started")
    
        self.cancel_flag = False
        self.processing = True
    
        outdir = os.path.join(os.path.dirname(self.files[0]), "restored")
        os.makedirs(outdir, exist_ok=True)
    
        total = len(self.files)
    
        # UI elements (if they exist)
        if hasattr(self, "progress"):
            self.progress.setMaximum(total)
            self.progress.setValue(0)
    
        if hasattr(self, "status"):
            self.status.setText("Processing...")
    
        from .io_utils import write_exr  # ✅ correct import
        
        
        win = self.frames.value()
        half = win // 2
        
        # 👇 initialize sliding window
        frame_buffer = deque(
            [self.get_frame(j) for j in range(-half, half+1)],
            maxlen=win
)
    
        for i in range(total):

            if self.cancel_flag:
                print("Cancelled")
                if hasattr(self, "status"):
                    self.status.setText("Cancelled")
                self.processing = False
                return
        
            print(f"Processing {i+1}/{total}")
        
            # 👇 convert buffer to list
            frames = list(frame_buffer)

            # safety: skip if any failed
            if any(f is None for f in frames):
                print("Skipping frame due to read error")
            else:
                  # restore
                out, mask = restore_frame(frames, self.p, i, self.flow_cache)
    
                # Apply required preset mask.
                # Inside white mask: restored output.
                # Outside black mask: original center frame.
                center = frames[len(frames) // 2]
                
                out = center * (1.0 - self.preset_mask[..., None]) + out * self.preset_mask[..., None]
                mask *= self.preset_mask
        
                # 👇 build output path
                name = os.path.basename(self.files[i])
                path = os.path.join(outdir, name)
                
                # 👇 status BEFORE writing
                if hasattr(self, "status"):
                    self.status.setText(f"Writing {i+1}/{total} : {name}")
                
                QApplication.processEvents()
                
                # 👇 actual write
                write_exr(path, out)
            
                # 👇 console + GUI feedback AFTER write
                print(f"Wrote frame {i+1}/{total} : {name}")
                
                if hasattr(self, "progress"):
                    self.progress.setValue(i + 1)
                
                if hasattr(self, "status"):
                    self.status.setText(f"Wrote {i+1}/{total}")    
                
            # 👇 slide window (correct placement)
            next_index = i + half + 1
            frame_buffer.append(self.get_frame(next_index))
    
            # 🔴 VERY IMPORTANT: keeps UI alive
            QApplication.processEvents()
    
        if hasattr(self, "status"):
            self.status.setText("Done")
    
        self.processing = False
