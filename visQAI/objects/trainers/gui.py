from predictor import ViscosityPredictor, EnsembleViscosityPredictor
import sys
import os
from typing import Dict, Union
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFormLayout,
    QHBoxLayout, QComboBox, QLineEdit, QPushButton,
    QLabel, QMessageBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

# directory containing one subfolder per architecture (each with model+preprocessor)
ARCH_DIR = os.path.join("visQAI", "objects", "architectures")
# path to feature CSV for dropdown menus
FEATURE_CSV = os.path.join("content", "train_features.csv")

# target columns / shear rates
target_cols = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000"
]
shear_rates = [int(col.replace("Viscosity_", "")) for col in target_cols]

# architecture names to exclude from dropdown
EXCLUDE_ARCH = {"autoencoder", "physics_nn"}
# map type field -> corresponding concentration field
TYPE_TO_CONC = {
    'Protein_type': 'Protein_conc',
    'Buffer_type': 'Buffer_conc',
    'Stabilizer_type': 'Stabilizer_conc',
    'Surfactant_type': 'Surfactant_conc',
    'Salt_type': 'Salt_conc'
}


class PredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Viscosity Predictor")

        # load feature definitions for dropdowns
        try:
            self.feature_df = pd.read_csv(FEATURE_CSV)
        except Exception as e:
            QMessageBox.critical(
                self, "Initialization Error",
                f"Could not read feature file '{FEATURE_CSV}': {e}"
            )
            sys.exit(1)

        # scan available architectures, excluding certain ones
        try:
            self.architectures = sorted(
                d for d in os.listdir(ARCH_DIR)
                if os.path.isdir(os.path.join(ARCH_DIR, d))
                and d not in EXCLUDE_ARCH
            )
            if not self.architectures:
                raise RuntimeError(f"No valid subdirectories in {ARCH_DIR}")
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", str(e))
            sys.exit(1)

        # load first model
        self.predictor = None
        self._load_predictor(self.architectures[0])
        self.target_inputs: Dict[str, QLineEdit] = {}
        # build UI
        self._init_ui()

    def _load_predictor(self, arch_name):
        arch_path = os.path.join(ARCH_DIR, arch_name)
        model_path = os.path.join(arch_path, "model.keras")
        prep_path = os.path.join(arch_path, "preprocessor.pkl")
        if not os.path.isfile(model_path) or not os.path.isfile(prep_path):
            QMessageBox.critical(
                self, "Load Error",
                f"Missing model or preprocessor in {arch_path}"
            )
            return
        try:
            # self.predictor = ViscosityPredictor(
            #     model_path=model_path,
            #     preprocessor_path=prep_path
            # )
            self.predictor = EnsembleViscosityPredictor(base_dir=arch_path,)
        except Exception as e:
            QMessageBox.critical(self, f"Could not load '{arch_name}'", str(e))

    def _init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout()
        form = QFormLayout()
        self.inputs = {}

        # model selector
        self.model_selector = QComboBox()
        self.model_selector.addItems(self.architectures)
        self.model_selector.currentTextChanged.connect(self.on_model_change)
        form.addRow("Model:", self.model_selector)

        # categorical type inputs
        for field in TYPE_TO_CONC:
            opts = []
            if field in self.feature_df.columns:
                opts = sorted(
                    self.feature_df[field].dropna().astype(
                        str).unique().tolist()
                )
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(opts)
            combo.currentTextChanged.connect(
                lambda txt, f=field: self.on_type_change(f, txt)
            )
            self.inputs[field] = combo
            form.addRow(f"{field.replace('_',' ')}:", combo)

        # numeric inputs
        numeric_fields = [
            'MW', 'PI_mean', 'PI_range', 'Protein_conc',
            'Temperature', 'Buffer_pH', 'Buffer_conc',
            'Stabilizer_conc', 'Surfactant_conc', 'Salt_conc'
        ]
        for field in numeric_fields:
            line = QLineEdit()
            line.setPlaceholderText("Enter numeric value")
            if field == 'Temperature':
                line.setText("25")
            self.inputs[field] = line
            form.addRow(f"{field.replace('_',' ')}:", line)

        # connect autofill handlers
        self.inputs['Protein_type'].currentTextChanged.connect(
            self.on_protein_type_change)
        self.inputs['Buffer_type'].currentTextChanged.connect(
            self.on_buffer_type_change)

        form.addRow(QLabel("<b>True Viscosity (for Update)</b>"), QWidget())
        for col in target_cols:
            line = QLineEdit()
            line.setPlaceholderText(f"True {col}")
            self.target_inputs[col] = line
            form.addRow(f"{col}:", line)

        # — buttons in one row —
        btn_layout = QHBoxLayout()
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.on_predict)
        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self.on_update)
        btn_layout.addWidget(predict_btn)
        btn_layout.addWidget(update_btn)
        form.addRow(btn_layout)

        # result label
        self.result_label = QLabel("")
        form.addRow("Values:", self.result_label)

        # assemble
        form_widget = QWidget()
        form_widget.setLayout(form)
        main_layout.addWidget(form_widget, 1)
        self.figure = Figure(figsize=(4, 4))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas, 1)
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        for type_field in TYPE_TO_CONC:
            current = self.inputs[type_field].currentText()
            self.on_type_change(type_field, current)

        # And fire the two special autofill handlers once
        self.on_protein_type_change(self.inputs['Protein_type'].currentText())
        self.on_buffer_type_change(self.inputs['Buffer_type'].currentText())

    def on_type_change(self, type_field, text):
        """If a type is blank or 'none', zero & lock its conc field; otherwise unlock."""
        conc_field = TYPE_TO_CONC[type_field]
        widget = self.inputs[conc_field]
        if not text or text.strip().lower() == 'none':
            widget.setText("0")
            widget.setReadOnly(True)
        else:
            widget.setReadOnly(False)

    def on_protein_type_change(self, prot_type):
        """Auto-fill MW, PI_mean, PI_range based on selected protein type."""
        if prot_type and prot_type.strip().lower() != 'none':
            # restore normal autofill/editable behavior
            self._autofill_fields('Protein_type', prot_type, [
                'MW', 'PI_mean', 'PI_range'])
            # make sure the user can still tweak them if needed
            for f in ['MW', 'PI_mean', 'PI_range']:
                self.inputs[f].setReadOnly(False)
        else:
            # no protein → zero everything and lock the fields
            for f in ['MW', 'PI_mean', 'PI_range']:
                w = self.inputs[f]
                w.setText('0')
                w.setReadOnly(True)

    def on_buffer_type_change(self, buf_type):
        """Auto-fill Buffer_pH based on selected buffer type."""
        if buf_type and buf_type.strip().lower() != 'none':
            self._autofill_fields('Buffer_type', buf_type, ['Buffer_pH'])
        else:
            w = self.inputs['Buffer_pH']
            w.clear()
            w.setReadOnly(False)

    def _autofill_fields(self, key_field, key_value, target_fields):
        df = self.feature_df
        row = df[df[key_field].astype(str) == str(key_value)]
        for field in target_fields:
            widget = self.inputs[field]
            if not row.empty and field in row.columns:
                widget.setText(str(row[field].iloc[0]))
                widget.setReadOnly(True)

    def on_model_change(self, arch_name):
        self._load_predictor(arch_name)
        self.result_label.setText("")
        self.figure.clear()
        self.canvas.draw()

    def on_predict(self):
        try:
            data = {}
            for name, widget in self.inputs.items():
                if isinstance(widget, QLineEdit):
                    txt = widget.text().strip()
                    if not txt:
                        raise ValueError(f"Value required for '{name}'")
                    data[name] = float(txt)
                else:
                    val = widget.currentText().strip()
                    data[name] = val or None
            df_new = pd.DataFrame([data])
            preds, confs = self.predictor.predict(
                df_new, return_confidence=True)
            y_pred, y_conf = preds[0], confs[0]
            lines = [f"{col}: {y_pred[i]:.3f} ± {y_conf[i]:.3f}" for i,
                     col in enumerate(target_cols)]
            self.result_label.setText("\n".join(lines))
            self._plot_curve(
                [100, 1000, 10000, 100000, 15000000], y_pred, y_conf)
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))
            raise

    def _plot_curve(self, shear_rates, viscosities, confidences):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        sr = np.array(shear_rates)
        visc = np.array(viscosities)
        conf = np.array(confidences)
        mask = (sr > 0) & (visc > 0)
        sr, visc, conf = sr[mask], visc[mask], conf[mask]
        if sr.size == 0:
            ax.text(
                0.5, 0.5,
                "No positive data to plot",
                ha="center", va="center",
                transform=ax.transAxes
            )
            self.canvas.draw()
            return

        deg = 3
        if sr.size > deg:
            log_sr = np.log10(sr)
            log_visc = np.log10(visc)
            coeff = np.polyfit(log_sr, log_visc, deg)
            sr_fit = np.logspace(log_sr.min(), log_sr.max(), num=200)
            visc_fit = 10**np.polyval(coeff, np.log10(sr_fit))
            std = np.clip(conf, a_min=1e-8, a_max=None)
            coeff_s = np.polyfit(log_sr, np.log10(std), deg)
            std_fit = 10**np.polyval(coeff_s, np.log10(sr_fit))
            lower, upper = visc_fit - std_fit, visc_fit + std_fit
            ax.plot(sr_fit, visc_fit, label='Poly-Fit')
            ax.fill_between(sr_fit, lower, upper,
                            alpha=0.3, label=f'Error band')
        else:
            ax.plot(sr, visc, '-o', label='Data')

        ax.scatter(sr, visc, color='k', s=20, zorder=5)
        ax.set_xscale('log')
        ax.set_xlabel("Shear Rate")
        ax.set_ylabel("Viscosity")
        ax.set_title("Predicted Viscosity Curve")
        ax.grid(True, which='both', ls='--', lw=0.5)
        ax.legend()
        self.canvas.draw()

    def on_update(self):
        """
        Gather feature + true‐target values from the UI,
        call predictor.update(), and notify the user.
        """
        try:
            data: Dict[str, Union[float, str]] = {}
            for name, widget in self.inputs.items():
                if isinstance(widget, QLineEdit):
                    txt = widget.text().strip()
                    if not txt:
                        raise ValueError(f"Feature '{name}' is required")
                    data[name] = float(txt)
                else:
                    val = widget.currentText().strip()
                    data[name] = val or None

            df_new = pd.DataFrame([data])
            y_true = []
            for col, widget in self.target_inputs.items():
                txt = widget.text().strip()
                if not txt:
                    raise ValueError(f"True value for '{col}' is required")
                y_true.append(float(txt))
            # shape (1, T)
            y_true_arr = np.array([y_true])
            self.predictor.update(
                new_data=df_new,
                new_targets=y_true_arr,
                epochs=1,
                batch_size=1,
                save=True
            )

            QMessageBox.information(
                self, "Update Complete",
                "Ensemble members have been updated with your new sample."
            )

        except Exception as e:
            QMessageBox.critical(self, "Update Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictorGUI()
    window.show()
    sys.exit(app.exec_())
