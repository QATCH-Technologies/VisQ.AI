from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from datetime import datetime
from src.controllers.predictors_controller import PredictorsController, PredictorInfo


class PredictorDialog(QDialog):
    def __init__(self, parent=None, info: PredictorInfo = None):
        super().__init__(parent)
        self.setModal(True)
        self.info = info
        self.setWindowTitle(
            "Add Predictor" if info is None else f"Edit Predictor: {info.name}")
        self.resize(500, 300)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Name field
        self.name_edit = QLineEdit(info.name if info else "")
        form.addRow(QLabel("Name:"), self.name_edit)

        # Description field
        self.desc_edit = QTextEdit(info.description if info else "")
        self.desc_edit.setFixedHeight(50)
        form.addRow(QLabel("Description:"), self.desc_edit)

        # Predictor file picker
        self.pred_path = QLineEdit()
        btn_pred = QPushButton("Browse...")
        btn_pred.clicked.connect(lambda: self._browse(
            self.pred_path, "Predictor .pkl"))
        form.addRow(QLabel("Predictor File:"),
                    self._hbox(self.pred_path, btn_pred))

        # Model file picker
        self.model_path = QLineEdit()
        btn_model = QPushButton("Browse...")
        btn_model.clicked.connect(
            lambda: self._browse(self.model_path, "Model .pkl"))
        form.addRow(QLabel("Model File:"), self._hbox(
            self.model_path, btn_model))

        # Preprocessor file picker
        self.preproc_path = QLineEdit()
        btn_pre = QPushButton("Browse...")
        btn_pre.clicked.connect(lambda: self._browse(
            self.preproc_path, "Preprocessor .pkl"))
        form.addRow(QLabel("Preprocessor File:"),
                    self._hbox(self.preproc_path, btn_pre))

        layout.addLayout(form)

        # Save and Cancel buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def _hbox(self, widget: QWidget, button: QPushButton) -> QWidget:
        """
        Creates a horizontal widget container for a line edit and button.
        """
        container = QWidget()
        h = QHBoxLayout(container)
        h.addWidget(widget)
        h.addWidget(button)
        h.addStretch()
        container.setLayout(h)
        return container

    def _browse(self, line_edit: QLineEdit, title: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {title}", filter="Pickle Files (*.pkl)")
        if path:
            line_edit.setText(path)

    def get_data(self) -> dict:
        return {
            'name': self.name_edit.text().strip(),
            'description': self.desc_edit.toPlainText().strip(),
            'predictor_path': self.pred_path.text().strip() or None,
            'model_path': self.model_path.text().strip() or None,
            'preprocessor_path': self.preproc_path.text().strip() or None
        }


class ManagePredictorsWindow(QDialog):
    """
    Main window to list, add, edit, and delete predictors.
    Uses PredictorDialog for add/edit.

    Emits:
      modelAdded(name)
      modelEdited(old_name, new_name)
      modelDeleted(name)
    """
    modelAdded = pyqtSignal(str)
    modelEdited = pyqtSignal(str, str)
    modelDeleted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Predictors")
        self.resize(600, 400)

        self.controller = PredictorsController()

        layout = QVBoxLayout(self)

        # Top action buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self._on_add)
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.clicked.connect(self._on_edit)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Predictors table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Description", "Imported At"])
        self.table.setSelectionBehavior(self.table.SelectRows)
        layout.addWidget(self.table)

        self._load_predictors()

    def _load_predictors(self):
        self.table.setRowCount(0)
        for info in self.controller.get_predictors():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(info.name))
            self.table.setItem(row, 1, QTableWidgetItem(info.description))
            ts = info.created_at.strftime("%Y-%m-%d %H:%M")
            self.table.setItem(row, 2, QTableWidgetItem(ts))

    def _selected_name(self) -> str | None:
        items = self.table.selectedItems()
        return items[0].text() if items else None

    def _on_add(self):
        dlg = PredictorDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            data = dlg.get_data()
            if not all([data['name'], data['predictor_path'], data['model_path'], data['preprocessor_path']]):
                QMessageBox.warning(self, "Validation",
                                    "All fields required for a new predictor.")
                return
            info = self.controller.add_predictor(**data)
            self.modelAdded.emit(info.name)
            self._load_predictors()

    def _on_edit(self):
        name = self._selected_name()
        if not name:
            QMessageBox.warning(self, "Select", "Select a predictor to edit.")
            return
        info = self.controller.get_predictor(name)
        dlg = PredictorDialog(self, info)
        if dlg.exec_() == QDialog.Accepted:
            data = dlg.get_data()
            if not data['name']:
                QMessageBox.warning(self, "Validation",
                                    "Name cannot be empty.")
                return
            updated = self.controller.update_predictor(
                name=name,
                new_name=data['name'],
                description=data['description'],
                predictor_path=data['predictor_path'],
                model_path=data['model_path'],
                preprocessor_path=data['preprocessor_path']
            )
            if updated:
                self.modelEdited.emit(name, updated.name)
                self._load_predictors()

    def _on_delete(self):
        name = self._selected_name()
        if not name:
            QMessageBox.warning(
                self, "Select", "Select a predictor to delete.")
            return
        if QMessageBox.question(self, "Confirm Delete", f"Delete '{name}'?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            if self.controller.delete_predictor(name):
                self.modelDeleted.emit(name)
                self._load_predictors()
