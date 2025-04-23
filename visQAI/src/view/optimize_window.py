# optimize_tab.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout


class OptimizeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        # add optimize controls here
        self.setLayout(layout)
