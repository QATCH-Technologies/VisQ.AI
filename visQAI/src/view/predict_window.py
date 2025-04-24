from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QWidget,
    QFormLayout, QComboBox, QLineEdit, QDoubleSpinBox, QPushButton, QLabel
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.controllers.predictors_controller import PredictorsController
from src.controllers.formulations_controller import FormulationsController


class PredictWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Predictor")
        self.predictor_controller = PredictorsController()
        self.formulations_controller = FormulationsController()

        # Main layout
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # Top panel: configuration
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        form_layout = QFormLayout()
        top_layout.addLayout(form_layout)

        # Model selection
        self.model_combo = QComboBox()
        form_layout.addRow("Model:", self.model_combo)

        # Formulation selection
        self.formulation_combo = QComboBox()
        form_layout.addRow("Formulation:", self.formulation_combo)

        # Formulation fields
        self.fields = {}
        # textual fields
        for label in ["Protein type", "Buffer", "Sugar", "Surfactant"]:
            edit = QLineEdit()
            form_layout.addRow(label + ":", edit)
            self.fields[label] = edit
        # numeric fields
        for label in ["Protein", "Temperature", "Sugar (M)", "TWEEN"]:
            spin = QDoubleSpinBox()
            spin.setRange(0, 1e6)
            spin.setDecimals(4)
            form_layout.addRow(label + ":", spin)
            self.fields[label] = spin

        # Action buttons
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add Formulation")
        self.btn_predict = QPushButton("Predict Viscosity Profile")
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_predict)
        top_layout.addLayout(btn_layout)

        splitter.addWidget(top_widget)

        # Bottom panel: plot
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        bottom_layout.addWidget(self.canvas)
        splitter.addWidget(bottom_widget)

        # Signals
        self.btn_add.clicked.connect(self._on_add_formulation)
        self.btn_predict.clicked.connect(self._on_predict)
        self.formulation_combo.currentTextChanged.connect(
            self._on_formulation_selected)

        # Load initial data
        self._load_models()
        self._load_formulations()

    def _load_models(self):
        # Fetch model list from controller
        models = self._list_models()
        self.model_combo.clear()
        self.model_combo.addItems(models)

    def _list_models(self):
        predictors = []
        for p in self.predictor_controller.get_predictors():
            predictors.append(p.name)
        return predictors

    def _load_formulations(self):
        # Fetch formulations from controller
        self.formulations = {
            f.name: f for f in self.formulations_controller.get_formulations()}
        self.formulation_combo.clear()
        self.formulation_combo.addItems(self.formulations.keys())
        if models := self.model_combo.currentText():
            # ensure model selected first
            pass

    def _on_formulation_selected(self, name):
        # Autofill fields for the selected formulation
        formulation = self.formulations.get(name)
        if not formulation:
            return
        for key, widget in self.fields.items():
            value = formulation.name
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))

    def _on_add_formulation(self):
        # Open formulation creation dialog
        from view.menu_options.file.create_excipient_menu import CreateFormulationDialog
        dlg = CreateFormulationDialog(self)
        dlg.formulationCreated.connect(self._refresh_formulations)
        dlg.exec_()

    def _refresh_formulations(self, new_formulation):
        # Reload formulations after creation
        self._load_formulations()
        self.formulation_combo.setCurrentText(new_formulation['name'])

    def _on_predict(self):
        # Gather inputs
        model_name = self.model_combo.currentText()
        form_data = {key: (w.text() if isinstance(w, QLineEdit) else w.value())
                     for key, w in self.fields.items()}

        # Call controller to predict
        try:
            df_pred = self.predictor_controller.predict(model_name, form_data)
        except Exception as e:
            # handle error
            return

        # Plot prediction
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # Assuming df_pred has shear_rate index and viscosity columns
        try:
            x = df_pred.index.values
            for col in df_pred.columns:
                ax.plot(x, df_pred[col], label=col)
            ax.set_xlabel("Shear / rate")
            ax.set_ylabel("Viscosity")
            ax.legend()
        except Exception:
            # fallback: single row array
            x = list(range(len(df_pred)))
            ax.plot(x, df_pred.values.flatten())
        self.canvas.draw()
