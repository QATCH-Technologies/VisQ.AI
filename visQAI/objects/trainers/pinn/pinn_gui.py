import sys
import os
import pandas as pd
import tensorflow as tf
import numpy as np

from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pinn_domain import DataLoader
from pinn_validation import Validator
from scipy.interpolate import PchipInterpolator


class ViscosityPredictorGUI(QtWidgets.QWidget):
    """
    GUI application for predicting viscosity profiles based on user-defined formulation features.
    Left pane: input form; right pane: embedded plot.
    """

    def __init__(self, csv_path: str, model_path: str):
        super().__init__()
        # Load data and preprocessing pipeline
        self.loader = DataLoader(csv_path)
        self.loader.load()
        self.loader.build_preprocessor()
        # Raw features define the expected input fields
        X_df, _ = self.loader.split(preprocess=False)
        self.feature_columns = X_df.columns.tolist()
        self.target_names = self.loader.TARGET_COLUMNS

        from pinn_net import Sine
        self.model = tf.keras.models.load_model(
            "best_pinn_model.h5",
            custom_objects={"Sine": Sine},
            compile=False
        )
        self.validator = Validator(
            model=self.model,
            preprocessor=self.loader.preprocessor,
            target_names=self.target_names,
        )

        # Keep track of which QLineEdits to manage for each category
        self.autofill_cols = {'MW', 'PI_mean', 'PI_range', 'Buffer_pH'}
        self.categorical = {
            'Protein_type':    ['None', 'BSA', 'poly-hIgG', 'BGG'],
            'Buffer_type':     ['None', 'PBS', 'acetate', 'Histidine'],
            'Sugar_type':      ['None', 'Sucrose', 'Trehalose'],
            'Surfactant_type': ['None', 'Tween-20', 'Tween-80'],
        }

        self.protein_defaults = {
            'BSA':       {'MW': 66,  'PI_mean': 4.7, 'PI_range': 0.3},
            'poly-hIgG': {'MW': 150, 'PI_mean': 7.6, 'PI_range': 1.0},
            'BGG':       {'MW': 150, 'PI_mean': 6.6, 'PI_range': 1.0},
            'None':      {'MW': 0,   'PI_mean': 0.0, 'PI_range': 0.0},
        }
        self.buffer_defaults = {
            'PBS':      {'Buffer_pH': 7.4},
            'acetate':  {'Buffer_pH': 5.0},
            'Histidine': {'Buffer_pH': 6.0},
            'None':     {'Buffer_pH': 0.0},
        }

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Viscosity Predictor")
        main_layout = QtWidgets.QHBoxLayout()
        left = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout()
        self.inputs: dict[str, QtWidgets.QWidget] = {}

        for feat in self.feature_columns:
            display = feat.replace('_', ' ')
            if feat in self.categorical:
                combo = QtWidgets.QComboBox()
                combo.addItems(self.categorical[feat])
                form.addRow(display, combo)
                self.inputs[feat] = combo

                # Hook up handlers for each category
                if feat == 'Protein_type':
                    combo.currentTextChanged.connect(self.on_protein_changed)
                elif feat == 'Buffer_type':
                    combo.currentTextChanged.connect(self.on_buffer_changed)
                elif feat == 'Sugar_type':
                    combo.currentTextChanged.connect(self.on_sugar_changed)
                elif feat == 'Surfactant_type':
                    combo.currentTextChanged.connect(
                        self.on_surfactant_changed)

            else:
                line = QtWidgets.QLineEdit()
                line.setPlaceholderText("Enter numeric value")
                # no permanent read-only here
                form.addRow(display, line)
                self.inputs[feat] = line

        # Predict button
        btn = QtWidgets.QPushButton("Predict Viscosity Profile")
        btn.clicked.connect(self.on_predict)
        form.addRow(btn)

        left.setLayout(form)
        main_layout.addWidget(left)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)
        if 'Temperature' in self.inputs:
            temp_w = self.inputs['Temperature']
            if isinstance(temp_w, QtWidgets.QLineEdit):
                temp_w.setText("25")
        # Trigger the initial state based on default combo values
        self.on_protein_changed(self.inputs['Protein_type'].currentText())
        self.on_buffer_changed(self.inputs['Buffer_type'].currentText())
        self.on_sugar_changed(self.inputs['Sugar_type'].currentText())
        self.on_surfactant_changed(
            self.inputs['Surfactant_type'].currentText())

    def on_protein_changed(self, name: str):
        """Autofill MW, PI_mean, PI_range and lock/unlock."""
        for col, val in self.protein_defaults[name].items():
            w: QtWidgets.QLineEdit = self.inputs[col]  # type: ignore
            w.setText(str(val))
            # read-only if 'None', editable otherwise
            w.setReadOnly(name == 'None')

    def on_buffer_changed(self, name: str):
        """Autofill Buffer_pH and lock/unlock."""
        for col, val in self.buffer_defaults[name].items():
            w: QtWidgets.QLineEdit = self.inputs[col]  # type: ignore
            w.setText(str(val))
            w.setReadOnly(name == 'None')

    def on_sugar_changed(self, name: str):
        """Set sugar concentration to 0 & lock if None, else unlock."""
        # find any QLineEdit whose feature starts with "Sugar" but isn't the combo
        for feat, w in self.inputs.items():
            if isinstance(w, QtWidgets.QLineEdit) and feat.startswith('Sugar'):
                if name == 'None':
                    w.setText('0')
                    w.setReadOnly(True)
                else:
                    w.setReadOnly(False)

    def on_surfactant_changed(self, name: str):
        """Set surfactant concentration to 0 & lock if None, else unlock."""
        for feat, w in self.inputs.items():
            if isinstance(w, QtWidgets.QLineEdit) and feat.startswith('Surfactant'):
                if name == 'None':
                    w.setText('0')
                    w.setReadOnly(True)
                else:
                    w.setReadOnly(False)

    def on_predict(self) -> None:
        """Gather inputs, run prediction, and update the embedded plot
        (monotonic spline + linear fit on log-x scale) with uncertainty."""
        # 1) collect & predict
        data = {}
        for feat, widget in self.inputs.items():
            if isinstance(widget, QtWidgets.QLineEdit):
                try:
                    data[feat] = float(widget.text())
                except ValueError:
                    QtWidgets.QMessageBox.warning(
                        self, "Input Error",
                        f"Please enter a valid numeric value for {feat.replace('_',' ')}."
                    )
                    return
            else:
                data[feat] = widget.currentText()

        df_new = pd.DataFrame([data])

        # Point estimate & uncertainty
        mean_pred, std_pred = self.validator.predict_with_uncertainty(
            df_new, n_iter=100)
        print(std_pred)
        y_pred = mean_pred.flatten()
        uncertainty = std_pred.flatten()

        labels = [n.replace('Viscosity_', '') for n in self.target_names]
        shear = np.array([float(lbl) for lbl in labels])

        # prepare figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x_log = np.log10(shear)

        # --- spline of the mean prediction ---
        pchip = PchipInterpolator(x_log, y_pred, extrapolate=False)
        x_smooth_log = np.linspace(x_log.min(), x_log.max(), 300)
        y_smooth = pchip(x_smooth_log)
        x_smooth = 10 ** x_smooth_log

        # --- spline of the lower/upper bounds ---
        lower_pchip = PchipInterpolator(
            x_log, y_pred - uncertainty, extrapolate=False)
        upper_pchip = PchipInterpolator(
            x_log, y_pred + uncertainty, extrapolate=False)
        lower_smooth = lower_pchip(x_smooth_log)
        upper_smooth = upper_pchip(x_smooth_log)

        # draw confidence band
        ax.fill_between(
            x_smooth, lower_smooth, upper_smooth,
            alpha=0.2,
            color='#7F8C8D',
            label='±1σ uncertainty'
        )

        # draw mean spline
        ax.plot(
            x_smooth, y_smooth,
            linestyle='-',
            linewidth=1.5,
            color='#2C3E50',
            label='Monotonic spline'
        )

        # — linear regression in log‐space (straight line on log axis) —
        m, b = np.polyfit(x_log, y_pred, 1)
        y_lin = m * x_smooth_log + b
        ax.plot(
            x_smooth, y_lin,
            linestyle='-.',
            linewidth=1.5,
            color='#BDC3C7',
            label='Linear fit'
        )

        # ---- raw points + error bars + annotations ----
        ax.errorbar(
            shear, y_pred,
            yerr=uncertainty,
            fmt='o',
            capsize=4,
            color='#7F8C8D',
            zorder=5,
            label='Predicted ±1σ'
        )
        for xi, yi, ui in zip(shear, y_pred, uncertainty):
            ax.text(
                xi * 1.05, yi,
                f"{yi:.1f}±{ui:.1f}",
                fontsize=9, va='center', ha='left',
                color='#2C3E50'
            )

        # ---- log scale + formatting ----
        ax.set_xscale('log')
        ax.set_xticks(shear)
        ax.set_xticklabels(labels, rotation=45, fontsize=10)
        ax.set_xlim(shear.min() * 0.8, shear.max() * 1.2)
        ax.set_xlabel("Shear Rate (s⁻¹)", fontsize=12, color='#2C3E50')
        ax.set_ylabel("Viscosity (cP)", fontsize=12, color='#2C3E50')
        ax.set_title("Predicted Viscosity Profile",
                     fontsize=14, color='#2C3E50')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        # ax.legend(frameon=False, fontsize=10, loc='upper left')

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    data_csv = os.path.join('content', 'formulation_data_05152025.csv')
    model_file = 'best_pinn_model.h5'

    gui = ViscosityPredictorGUI(csv_path=data_csv, model_path=model_file)
    gui.show()
    sys.exit(app.exec_())
