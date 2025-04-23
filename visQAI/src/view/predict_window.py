
# predict_tab.py
from PyQt5.QtWidgets import QWidget, QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QPushButton, QVBoxLayout, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
import numpy as np
import os
from predictors.viscosity_predictor import ViscosityPredictor


class PredictWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        inputs = QGroupBox("Formulation & Model")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignLeft)
        self.model_cb = QComboBox()
        self.model_cb.addItems(["XGB", "Neural Net", "Linear", "CNN"])
        form.addRow("Model:", self.model_cb)
        # ... repeat for other entries ...
        self.predict_btn = QPushButton("Predict Viscosity")
        self.predict_btn.clicked.connect(self.on_predict)

        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(
            ["Shear Rate", "Viscosity"])

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(inputs)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.results_table)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def on_predict(self):
        # gather inputs, choose predictor, plot & table
        pass
