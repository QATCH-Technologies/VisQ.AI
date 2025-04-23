# learn_window.py
from PyQt5.QtWidgets import QWidget, QTableWidget, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
import pandas as pd


class LearnWidnow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        headers = ["ID", "Protein type", "Protein", ...]
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        add_btn = QPushButton("Add Measurement")
        add_btn.clicked.connect(self.add_row)
        rm_btn = QPushButton("Remove Measurement")
        rm_btn.clicked.connect(self.remove_row)

        self.surface_fig = Figure(figsize=(5, 4))
        self.surface_canvas = FigureCanvas(self.surface_fig)
        self.surface_ax = self.surface_fig.add_subplot(111, projection='3d')

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(rm_btn)
        layout.addLayout(btn_layout)
        layout.addWidget(self.surface_canvas)
        self.setLayout(layout)

    def add_row(self): pass
    def remove_row(self): pass
    def update_surface(self): pass
