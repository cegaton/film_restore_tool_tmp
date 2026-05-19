import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QImage, QPixmap


class Viewer(QLabel):
    def __init__(self):
        super().__init__()

        self.img = None

        # view state
        self.x = 0
        self.y = 0
        self.roi = 600

        # interaction
        self.drag = False
        self.last = QPoint()

        # sync callback (set by App)
        self.sync = None

    # ================= SET IMAGE =================

    def set_image(self, img):
        self.img = img
        self.update_view()

    # ================= MOUSE =================

    def mousePressEvent(self, e):
        self.drag = True
        self.last = e.pos()

    def mouseMoveEvent(self, e):
        if self.drag and self.img is not None:
            d = e.pos() - self.last

            self.x -= d.x()
            self.y -= d.y()

            self.last = e.pos()

            # sync both viewers
            if self.sync:
                self.sync(self.x, self.y)

            self.update_view()

    def mouseReleaseEvent(self, e):
        self.drag = False

    # ================= VIEW UPDATE =================

    def update_view(self):
        if self.img is None:
            return

        h, w = self.img.shape[:2]

        # clamp ROI
        self.x = max(0, min(self.x, w - self.roi))
        self.y = max(0, min(self.y, h - self.roi))

        # crop
        crop = self.img[
            self.y:self.y + self.roi,
            self.x:self.x + self.roi
        ]

        # display transform (gamma)
        disp = np.clip(crop, 0, 1) ** (1 / 2.2)
        disp = (disp * 255).astype(np.uint8)

        # convert to QImage
        q = QImage(
            disp.data,
            disp.shape[1],
            disp.shape[0],
            3 * disp.shape[1],
            QImage.Format_RGB888
        )

        self.setPixmap(QPixmap.fromImage(q))
