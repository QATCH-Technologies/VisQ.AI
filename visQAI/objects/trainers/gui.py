import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFormLayout,
    QComboBox, QLineEdit, QPushButton, QLabel, QMessageBox
)
from visQAI.objects.predictors.predictor import ViscosityPredictor

# Paths — adjust as needed
save_path = 'visQAI/objects/cnn_regressor'

target_cols = [
    "Viscosity100",
    "Viscosity1000",
    "Viscosity10000",
    "Viscosity100000",
    "Viscosity15000000"
]


class PredictorGUI(QMainWindow):
    """
    Main window for viscosity prediction GUI, using ViscosityPredictor.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Viscosity Predictor")
        self._load_resources()
        self._init_ui()

    def _load_resources(self):
        # instantiate your standalone predictor
        try:
            self.predictor = ViscosityPredictor(save_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            sys.exit(1)

        # Hard-coded dropdown options
        self.cat_options = {
            'Protein type': ['None', 'poly-hIgG', 'BSA', 'BGG'],
            'Buffer':       ['Histidine', 'PBS'],
            'Sugar':        ['None', 'Trehalose', 'Sucrose'],
            'Surfactant':   ['None', 'tween-20', 'tween-80'],
        }

    def _init_ui(self):
        central = QWidget()
        form = QFormLayout()
        self.inputs = {}

        # Categorical inputs
        for field, opts in self.cat_options.items():
            combo = QComboBox()
            combo.addItems(opts)
            combo.setEditable(True)
            self.inputs[field] = combo
            form.addRow(f"{field}:", combo)

        # Numeric inputs
        for field in ['MW(kDa)', 'PI_mean', 'PI_range', 'Protein',
                      'Temperature', 'Sugar(M)', 'Concentration']:
            line = QLineEdit()
            line.setPlaceholderText("Enter numeric value")
            self.inputs[field] = line
            form.addRow(f"{field}:", line)

        # Predict button
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.on_predict)
        form.addRow(predict_btn)

        # Results display
        self.result_label = QLabel("")
        form.addRow(QLabel("Predicted viscosities:"), self.result_label)

        central.setLayout(form)
        self.setCentralWidget(central)

    def on_predict(self):
        try:
            # 1) gather inputs into dict
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

            # 2) build DF & predict via your predictor
            df_new = pd.DataFrame([data])
            y_pred = self.predictor.predict(df_new)
            y_pred = y_pred[0]
            print(y_pred)
            # 3) format output
            lines = []
            for i, col in enumerate(target_cols):
                lines.append(f"{col}: {y_pred[i]:.3f}")
            self.result_label.setText("\n".join(lines))

        except Exception as e:
            raise


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictorGUI()
    window.show()
    sys.exit(app.exec_())
