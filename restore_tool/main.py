
import sys
from PyQt5.QtWidgets import QApplication
from .app import App

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = App()
    w.resize(1600, 900)
    w.show()
    sys.exit(app.exec_())
