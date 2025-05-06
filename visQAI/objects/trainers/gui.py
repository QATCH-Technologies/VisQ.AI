from predictor import ViscosityPredictor
import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFormLayout,
    QHBoxLayout, QComboBox, QLineEdit, QPushButton,
    QLabel, QMessageBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import keras
keras.config.enable_unsafe_deserialization()


# directory containing one subfolder per architecture (each with model+preprocessor)
ARCH_DIR = os.path.join("visQAI", "objects", "architectures")

# target columns / shear rates
target_cols = [
    "Viscosity100",
    "Viscosity1000",
    "Viscosity10000",
    "Viscosity100000",
    "Viscosity15000000"
]
shear_rates = [int(col.replace("Viscosity", "")) for col in target_cols]


class PredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Viscosity Predictor")

        # 1) scan available architectures
        try:
            self.architectures = sorted(
                d for d in os.listdir(ARCH_DIR)
                if os.path.isdir(os.path.join(ARCH_DIR, d))
            )
            if not self.architectures:
                raise RuntimeError(f"No subdirectories in {ARCH_DIR}")
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", str(e))
            sys.exit(1)

        # load the first model by default
        self.predictor = None
        self._load_predictor(self.architectures[0])

        # build UI
        self._init_ui()

    def _load_predictor(self, arch_name):
        """Instantiate ViscosityPredictor for a given architecture folder."""
        arch_path = os.path.join(ARCH_DIR, arch_name)
        model_path = os.path.join(arch_path, "model.keras")
        prep_path = os.path.join(arch_path, "preprocessor.pkl")

        if not os.path.isfile(model_path):
            QMessageBox.critical(
                self, "Load Error",
                f"Model file not found:\n  {model_path}"
            )
            return

        if not os.path.isfile(prep_path):
            QMessageBox.critical(
                self, "Load Error",
                f"Preprocessor file not found:\n  {prep_path}"
            )
            return

        try:
            # assume ViscosityPredictor can take explicit paths
            self.predictor = ViscosityPredictor(
                model_path=model_path,
                preprocessor_path=prep_path
            )
        except Exception as e:
            QMessageBox.critical(
                self, f"Could not load '{arch_name}'", str(e)
            )

    def _init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout()
        form = QFormLayout()
        self.inputs = {}

        # --- Model selector dropdown ---
        self.model_selector = QComboBox()
        self.model_selector.addItems(self.architectures)
        self.model_selector.currentTextChanged.connect(self.on_model_change)
        form.addRow("Model:", self.model_selector)

        # categorical inputs
        cat_opts = {
            'Protein type': ['None', 'poly-hIgG', 'BSA', 'BGG'],
            'Buffer':       ['Histidine', 'PBS'],
            'Sugar':        ['None', 'Trehalose', 'Sucrose'],
            'Surfactant':   ['None', 'tween-20', 'tween-80'],
        }
        for field, opts in cat_opts.items():
            combo = QComboBox()
            combo.addItems(opts)
            combo.setEditable(True)
            self.inputs[field] = combo
            form.addRow(f"{field}:", combo)

        # numeric inputs
        for field in ['MW(kDa)', 'PI_mean', 'PI_range', 'Protein',
                      'Temperature', 'Sugar(M)', 'Concentration']:
            line = QLineEdit()
            line.setPlaceholderText("Enter numeric value")
            self.inputs[field] = line
            form.addRow(f"{field}:", line)

        # predict button
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.on_predict)
        form.addRow(predict_btn)

        # results label
        self.result_label = QLabel("")
        form.addRow("Values:", self.result_label)

        # put the form on the left
        form_widget = QWidget()
        form_widget.setLayout(form)
        main_layout.addWidget(form_widget, 1)

        # matplotlib canvas on the right
        self.figure = Figure(figsize=(4, 4))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas, 1)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def on_model_change(self, arch_name):
        """Reload predictor when user picks a different architecture."""
        self._load_predictor(arch_name)
        # clear previous plot & results
        self.result_label.setText("")
        self.figure.clear()
        self.canvas.draw()

    def on_predict(self):
        try:
            # gather inputs
            data = {}
            for name, widget in self.inputs.items():
                if isinstance(widget, QLineEdit):
                    txt = widget.text().strip()
                    if not txt:
                        raise ValueError(f"Value required for '{name}'")
                    data[name] = float(txt)
                else:  # QComboBox
                    val = widget.currentText()
                    data[name] = None if val == 'None' else val

            # predict
            df_new = pd.DataFrame([data])
            y_pred = self.predictor.predict(df_new)[0]

            # show numeric results
            lines = [f"{col}: {y_pred[i]:.3f}" for i,
                     col in enumerate(target_cols)]
            self.result_label.setText("\n".join(lines))

            # plot
            self._plot_curve(y_pred)

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))

    def _plot_curve(self, viscosities):
        """Draw viscosity vs. shear-rate curve."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(shear_rates, viscosities, marker='o', linestyle='-')
        ax.set_xscale('log')
        ax.set_xlabel("Shear Rate")
        ax.set_ylabel("Viscosity")
        ax.set_title("Predicted Viscosity Curve")
        ax.grid(True, which='both', ls='--', lw=0.5)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictorGUI()
    window.show()
    sys.exit(app.exec_())
