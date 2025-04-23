import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFormLayout, QComboBox,
    QDoubleSpinBox, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFrame, QTableWidget, QTableWidgetItem,
    QTabWidget, QStackedWidget
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from sklearn.decomposition import PCA
from scipy.interpolate import griddata

import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from linear_predictor import LinearPredictor
from xgb_predictor import XGBPredictor
from nn_predictor import NNPredictor
from cnn_predictor import CNNPredictor


class ViscosityPredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Viscosity Profile Predictor")
        self.setMinimumSize(900, 600)
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        # -- Input Group ------------------------------------------------
        inputs_group = QGroupBox("Formulation & Model")
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFormAlignment(Qt.AlignLeft)
        form_layout.setHorizontalSpacing(20)
        form_layout.setVerticalSpacing(8)

        # Model selector
        self.model_cb = QComboBox()
        self.model_cb.addItems(["XGB", "Neural Net", "Linear", "CNN"])
        form_layout.addRow("Model:", self.model_cb)

        # Protein type
        self.protein_type_cb = QComboBox()
        self.protein_type_cb.addItems(["None", "poly-hIgG", "BSA"])
        form_layout.addRow("Protein Type:", self.protein_type_cb)

        # Protein concentration
        self.protein_conc_sb = QDoubleSpinBox()
        self.protein_conc_sb.setRange(0, 1000)
        self.protein_conc_sb.setSuffix(" mg/mL")
        self.protein_conc_sb.setValue(10.0)
        form_layout.addRow("Protein Conc:", self.protein_conc_sb)

        # Temperature
        self.temperature_sb = QDoubleSpinBox()
        self.temperature_sb.setRange(0, 100)
        self.temperature_sb.setSuffix(" °C")
        self.temperature_sb.setValue(25.0)
        form_layout.addRow("Temperature:", self.temperature_sb)

        # Buffer
        self.buffer_cb = QComboBox()
        self.buffer_cb.addItems(["PBS"])
        form_layout.addRow("Buffer:", self.buffer_cb)

        # Sugar
        self.sugar_cb = QComboBox()
        self.sugar_cb.addItems(["None", "Sucrose"])
        form_layout.addRow("Sugar:", self.sugar_cb)

        # Sugar concentration
        self.sugar_conc_sb = QDoubleSpinBox()
        self.sugar_conc_sb.setRange(0, 2)
        self.sugar_conc_sb.setDecimals(3)
        self.sugar_conc_sb.setSuffix(" M")
        form_layout.addRow("Sugar Conc:", self.sugar_conc_sb)

        # Surfactant
        self.surfactant_cb = QComboBox()
        self.surfactant_cb.addItems(["None", "Tween-20", "Tween-80"])
        form_layout.addRow("Surfactant:", self.surfactant_cb)

        # Surfactant concentration
        self.surfactant_conc_sb = QDoubleSpinBox()
        self.surfactant_conc_sb.setRange(0, 100)
        self.surfactant_conc_sb.setDecimals(2)
        self.surfactant_conc_sb.setSuffix(" %w")
        form_layout.addRow("Surf. Conc:", self.surfactant_conc_sb)

        inputs_group.setLayout(form_layout)

        # -- Predict button ---------------------------------------------
        self.predict_btn = QPushButton("Predict Viscosity")
        self.predict_btn.clicked.connect(self.on_predict)

        # -- Results table ----------------------------------------------
        results_group = QGroupBox("Viscosity Profile")
        results_layout = QVBoxLayout()
        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(
            ["Shear Rate (1/s)", "Viscosity (cP)"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)

        # -- Plot area ---------------------------------------------------
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # -- Divider -----------------------------------------------------
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)

        # -- Tab Widget --------------------------------------------------
        self.tab_widget = QTabWidget()

        # Predict Tab
        predict_tab = QWidget()
        predict_layout = QVBoxLayout()
        predict_layout.addWidget(inputs_group)
        predict_layout.addWidget(self.predict_btn)
        predict_layout.addSpacing(10)
        predict_layout.addWidget(results_group)
        predict_layout.addStretch()
        predict_tab.setLayout(predict_layout)
        self.tab_widget.addTab(predict_tab, "Predict")

        learn_tab = QWidget()
        learn_layout = QVBoxLayout()

        # 1) Define headers in the user-specified order
        headers = [
            "ID", "Protein type", "Protein", "Temperature", "Buffer",
            "Sugar", "Sugar (M)", "Surfactant", "TWEEN",
            "Viscosity100", "Viscosity1000", "Viscosity10000",
            "Viscosity100000", "Viscosity15000000",
        ]

        self.learn_table = QTableWidget(0, len(headers))
        self.learn_table.setHorizontalHeaderLabels(headers)
        self.learn_table.horizontalHeader().setStretchLastSection(True)
        self.learn_table.verticalHeader().setVisible(False)
        self.learn_table.setAlternatingRowColors(True)
        self.learn_table.setShowGrid(True)
        self.learn_table.setGridStyle(Qt.SolidLine)
        self.learn_table.setStyleSheet("""
            QTableWidget { gridline-color: #888; }
            QTableWidget::item { border: 1px solid #888; }
            QHeaderView::section { border: 1px solid #888; background: #eaeaea; }
        """)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        add_btn = QPushButton("Add Measurement")
        add_btn.clicked.connect(self.add_measurement_row)
        btn_layout.addWidget(add_btn)
        remove_btn = QPushButton("Remove Measurement")
        remove_btn.clicked.connect(self.remove_measurement_row)
        btn_layout.addWidget(remove_btn)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.learn_table)
        left_layout.addLayout(btn_layout)
        left_widget.setLayout(left_layout)

        learn_layout.addWidget(left_widget)
        learn_tab.setLayout(learn_layout)
        self.tab_widget.addTab(learn_tab, "Learn")

        # Optimize Tab
        optimize_tab = QWidget()
        # ... existing optimize setup here ...
        self.tab_widget.addTab(optimize_tab, "Optimize")

        # Plot canvases
        self.profile_fig = Figure(figsize=(5, 4))
        self.profile_canvas = FigureCanvas(self.profile_fig)
        self.profile_ax = self.profile_fig.add_subplot(111)

        self.surface_fig = Figure(figsize=(5, 4))
        self.surface_canvas = FigureCanvas(self.surface_fig)
        self.surface_ax = self.surface_fig.add_subplot(111, projection='3d')

        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.profile_canvas)
        self.plot_stack.addWidget(self.surface_canvas)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.tab_widget, 1)
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        main_layout.addWidget(divider)
        main_layout.addWidget(self.plot_stack, 2)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Initialize Learn
        self.add_measurement_row()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.on_tab_changed(0)

    def on_tab_changed(self, index: int):
        # Predict tab = 0, Learn = 1, Optimize = 2
        if index == 1:
            # in Learn, show surface
            self.plot_stack.setCurrentIndex(1)
            self.update_surface_plot()
        else:
            # Predict or Optimize: show profile plot
            self.plot_stack.setCurrentIndex(0)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { font-family: Arial; font-size: 13px; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #aaa;
                border-radius: 4px;
                margin-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px; padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #4679BD;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
            }
        """)

    def on_predict(self):
        # gather inputs
        params = {
            "Protein type": self.protein_type_cb.currentText(),
            "Protein": self.protein_conc_sb.value(),
            "Temperature": self.temperature_sb.value(),
            "Buffer": self.buffer_cb.currentText(),
            "Sugar": self.sugar_cb.currentText(),
            "Sugar (M)": self.sugar_conc_sb.value(),
            "Surfactant": self.surfactant_cb.currentText(),
            "TWEEN": self.surfactant_conc_sb.value(),
        }
        model = self.model_cb.currentText()

        shear_rates, viscosities = self.predict_viscosity(params, model)

        # Update plot
        self.ax.clear()
        self.ax.scatter(shear_rates, viscosities, s=50,
                        label="Predictions", zorder=3)
        log_sr = np.log(shear_rates)
        log_visc = np.log(viscosities)
        f_cubic = interp1d(log_sr, log_visc, kind="cubic")
        x_log_smooth = np.linspace(log_sr.min(), log_sr.max(), 200)
        x_smooth = np.exp(x_log_smooth)
        y_smooth = np.exp(f_cubic(x_log_smooth))
        self.ax.plot(x_smooth, y_smooth, lw=2, label="Best Fit", zorder=2)
        self.ax.set_xscale("log")
        self.ax.set_xlabel("Shear Rate (1/s)")
        self.ax.set_ylabel("Viscosity (cP)")
        self.ax.set_title(f"Viscosity Profile ({model})")
        self.ax.grid(True, which="both", ls="--", lw=0.5)
        self.ax.legend()
        self.canvas.draw()

        # Populate results table
        self.results_table.setRowCount(len(shear_rates))
        for i, (sr, v) in enumerate(zip(shear_rates, viscosities)):
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{sr:.0f}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{v:.3f}"))

    def predict_viscosity(self, params, predictor_type: str):
        shear_rates = np.array([100, 1000, 10000, 100000, 15000000])
        df = pd.DataFrame({**params, 'shear_rate': shear_rates})

        # choose predictor
        if predictor_type == "XGB":
            predictor = XGBPredictor(os.path.join(
                'visqAI', 'objects', 'xgb_regressor'))
        elif predictor_type == "Neural Net":
            predictor = NNPredictor(os.path.join(
                'visqAI', 'objects', 'nn_regressor'))
        elif predictor_type == "CNN":
            predictor = CNNPredictor(os.path.join(
                'visqAI', 'objects', 'cnn_regressor'))
        else:
            predictor = LinearPredictor(os.path.join(
                'visqAI', 'objects', 'linear_regressor'))

        raw_visc = predictor.predict(df)
        arr = np.array(raw_visc)

        # flatten to 1D matching shear_rates
        if arr.ndim == 1:
            visc = arr
        elif arr.ndim == 2:
            visc = arr.flatten()[:len(shear_rates)]
        else:
            visc = arr.flatten()[:len(shear_rates)]
        return shear_rates, visc.tolist()

    def add_measurement_row(self):
        row = self.learn_table.rowCount()
        self.learn_table.insertRow(row)
        self.learn_table.setItem(row, 0, QTableWidgetItem(str(row+1)))
        drop_cols = {1: self.protein_type_cb, 4: self.buffer_cb,
                     5: self.sugar_cb, 7: self.surfactant_cb}
        for col, template in drop_cols.items():
            combo = QComboBox()
            for i in range(template.count()):
                combo.addItem(template.itemText(i))
            self.learn_table.setCellWidget(row, col, combo)
        for col in range(self.learn_table.columnCount()):
            if col in drop_cols or col == 0:
                continue
            self.learn_table.setItem(row, col, QTableWidgetItem(""))
        self.update_surface_plot()

    def remove_measurement_row(self):
        count = self.learn_table.rowCount()
        if count:
            self.learn_table.removeRow(count-1)
        for r in range(self.learn_table.rowCount()):
            self.learn_table.setItem(r, 0, QTableWidgetItem(str(r+1)))
        self.update_surface_plot()

    def update_surface_plot(self):
        # Gather data into DataFrame
        headers = [self.learn_table.horizontalHeaderItem(i).text()
                   for i in range(self.learn_table.columnCount())]
        records = []
        for r in range(self.learn_table.rowCount()):
            rec = {}
            for c, h in enumerate(headers):
                if c in (1, 4, 5, 7):
                    w = self.learn_table.cellWidget(r, c)
                    rec[h] = w.currentText() if w else None
                else:
                    itm = self.learn_table.item(r, c)
                    rec[h] = itm.text() if itm else None
            records.append(rec)
        df = pd.DataFrame(records)
        num_cols = ["Protein", "Temperature", "Sugar (M)", "TWEEN",
                    "Viscosity100", "Viscosity1000", "Viscosity10000",
                    "Viscosity100000", "Viscosity15000000"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
        cat_cols = ["Protein type", "Buffer", "Sugar", "Surfactant"]
        df_cat = pd.get_dummies(df[cat_cols], dummy_na=False)
        X = pd.concat([df[num_cols], df_cat], axis=1).dropna()
        if X.empty:
            self.surface_ax.clear()
            self.surface_canvas.draw()
            return
        n_pts, n_feat = X.shape
        n_comp = min(3, n_pts, n_feat)
        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(X)
        xs = coords[:, 0]
        ys = coords[:, 1] if n_comp > 1 else np.zeros_like(xs)
        zs = coords[:, 2] if n_comp > 2 else np.zeros_like(xs)
        self.surface_ax.clear()
        # Scatter all points in 3D
        self.surface_ax.scatter(xs, ys, zs, c=zs, cmap='hot', s=50)
        self.surface_ax.set_xlabel('PC1')
        self.surface_ax.set_ylabel('PC2')
        self.surface_ax.set_zlabel('PC3')
        self.surface_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ViscosityPredictorGUI()
    gui.show()
    sys.exit(app.exec_())
