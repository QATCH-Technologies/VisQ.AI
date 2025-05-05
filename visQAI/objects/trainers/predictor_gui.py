import sys
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QComboBox, QFormLayout, QVBoxLayout, QMessageBox
)
from visQ_data_processor import VisQDataProcessor
from PyQt5.QtGui import QDoubleValidator

# Path to your saved TensorFlow model
SAVE_PATH = 'visQAI/objects/cnn_regressor/cnn_model.keras'

# TODO: Replace these lists with actual categories used during model training
PROTEIN_TYPES = ['TypeA', 'TypeB', 'TypeC']
BUFFERS = ['Buffer1', 'Buffer2', 'Buffer3']
SUGARS = ['SugarX', 'SugarY', 'SugarZ']
SURFACTANTS = ['SurfA', 'SurfB', 'SurfC']

# Example mappings for categorical encoding (one-hot or label encoding)
# Adjust according to training preprocessing


class PredictionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('VisQAI Viscosity Predictor')
        self.model = tf.keras.models.load_model(SAVE_PATH)
        self._init_ui()

    def _init_ui(self):
        form = QFormLayout()

        # Categorical inputs
        self.protein_type_cb = QComboBox()
        self.protein_type_cb.addItems(PROTEIN_TYPES)
        form.addRow('Protein type:', self.protein_type_cb)

        # Numeric inputs with validators
        self.inputs = {}
        numeric_fields = [
            'MW(kDa)', 'PI_mean', 'PI_range', 'Protein',
            'Temperature', 'Sugar(M)', 'Concentration'
        ]
        for field in numeric_fields:
            line_edit = QLineEdit()
            line_edit.setValidator(QDoubleValidator(bottom=0.0))
            form.addRow(f'{field}:', line_edit)
            self.inputs[field] = line_edit

        # Additional categorical inputs
        self.buffer_cb = QComboBox()
        self.buffer_cb.addItems(BUFFERS)
        form.addRow('Buffer:', self.buffer_cb)

        self.sugar_cb = QComboBox()
        self.sugar_cb.addItems(SUGARS)
        form.addRow('Sugar:', self.sugar_cb)

        self.surfactant_cb = QComboBox()
        self.surfactant_cb.addItems(SURFACTANTS)
        form.addRow('Surfactant:', self.surfactant_cb)

        # Predict button
        self.predict_btn = QPushButton('Predict Viscosity')
        self.predict_btn.clicked.connect(self.on_predict)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.predict_btn)
        self.setLayout(layout)

    def on_predict(self):
        try:
            # Gather numeric inputs
            values = []
            for key, widget in self.inputs.items():
                text = widget.text()
                if text == '':
                    raise ValueError(f'Missing value for {key}')
                values.append(float(text))

            # Encode categorical
            pt = self.protein_type_cb.currentText()
            buf = self.buffer_cb.currentText()
            sug = self.sugar_cb.currentText()
            surf = self.surfactant_cb.currentText()

            values.append(encode_categorical(pt, PROTEIN_TYPES))
            values.append(encode_categorical(buf, BUFFERS))
            values.append(encode_categorical(sug, SUGARS))
            values.append(encode_categorical(surf, SURFACTANTS))

            # Prepare input for model
            X = np.array(values, dtype=np.float32).reshape(1, -1)

            # Perform prediction
            VisQDataProcessor._generate_features(X)
            preds = self.model.predict(X)

            # Display results
            msg = QMessageBox(self)
            msg.setWindowTitle('Prediction Results')
            msg.setText(f'Predicted viscosities:\n{preds.flatten()}')
            msg.exec_()

        except Exception as e:
            err = QMessageBox(self)
            err.setWindowTitle('Error')
            err.setIcon(QMessageBox.Critical)
            err.setText(str(e))
            err.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = PredictionGUI()
    gui.show()
    sys.exit(app.exec_())
