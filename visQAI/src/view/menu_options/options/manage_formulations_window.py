from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem,
    QTextEdit, QPushButton, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt

from src.controllers.formulations_controller import FormulationsController


class ManageFormulationsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Formulations")
        self.controller = FormulationsController()
        self._setup_ui()
        self._load_formulations()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Table for existing formulations
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Viscosity Profile", "Notes"]
        )
        layout.addWidget(self.table)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add", self)
        self.edit_btn = QPushButton("Edit", self)
        self.delete_btn = QPushButton("Delete", self)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.delete_btn)
        layout.addLayout(btn_layout)

        # Connect signals
        self.add_btn.clicked.connect(self._on_add)
        self.edit_btn.clicked.connect(self._on_edit)
        self.delete_btn.clicked.connect(self._on_delete)

    def _load_formulations(self):
        self.table.setRowCount(0)
        for form in self.controller.get_formulations():
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Name
            name = getattr(form, 'name', '') or ''
            self.table.setItem(row, 0, QTableWidgetItem(name))

            # Viscosity profile (safely accessed)
            vp = getattr(form, 'viscosity_profile', None)
            vp_str = ", ".join(map(str, vp)) if vp else ""
            self.table.setItem(row, 1, QTableWidgetItem(vp_str))

            # Notes (safely accessed)
            notes = getattr(form, 'notes', None) or ""
            self.table.setItem(row, 2, QTableWidgetItem(notes))

    def _on_add(self):
        dlg = FormulationEditDialog(self.controller, parent=self)
        if dlg.exec_():
            self._load_formulations()

    def _on_edit(self):
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, 0).text()
        formulation = self.controller.get_formulation(name)
        if formulation:
            dlg = FormulationEditDialog(
                self.controller, formulation=formulation, parent=self
            )
            if dlg.exec_():
                self._load_formulations()

    def _on_delete(self):
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, 0).text()
        self.controller.delete_formulation(name)
        self._load_formulations()


class FormulationEditDialog(QDialog):
    def __init__(self, controller, formulation=None, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.formulation = formulation
        self.setWindowTitle(
            "Edit Formulation" if formulation else "Add Formulation"
        )
        self._setup_ui()
        self._load_excipients()
        if formulation:
            self._populate()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Name field
        layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(self)
        layout.addWidget(self.name_edit)

        # Excipient selector
        layout.addWidget(QLabel("Excipients:"))
        self.exc_list = QListWidget(self)
        layout.addWidget(self.exc_list)

        # Viscosity profile input
        layout.addWidget(QLabel("Viscosity Profile (comma-separated):"))
        self.vp_edit = QLineEdit(self)
        layout.addWidget(self.vp_edit)

        # Notes input
        layout.addWidget(QLabel("Notes:"))
        self.notes_edit = QTextEdit(self)
        layout.addWidget(self.notes_edit)

        # Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save", self)
        self.cancel_btn = QPushButton("Cancel", self)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.save_btn.clicked.connect(self._on_save)
        self.cancel_btn.clicked.connect(self.reject)

    def _load_excipients(self):
        self.exc_list.clear()
        for exc in self.controller.get_excipients():
            item = QListWidgetItem(exc.name)
            item.setData(Qt.UserRole, exc)
            item.setCheckState(Qt.Unchecked)
            self.exc_list.addItem(item)

    def _populate(self):
        self.name_edit.setText(self.formulation.name)

        # Populate viscosity profile
        vp = getattr(self.formulation, 'viscosity_profile', None)
        if vp:
            self.vp_edit.setText(
                ",".join(map(str, vp))
            )

        # Populate notes
        notes = getattr(self.formulation, 'notes', None) or ""
        self.notes_edit.setPlainText(notes)

        # Populate excipient selections
        selected_ids = {
            assoc.excipient_id for assoc in self.formulation.excipients}
        for i in range(self.exc_list.count()):
            item = self.exc_list.item(i)
            exc = item.data(Qt.UserRole)
            if exc.id in selected_ids:
                item.setCheckState(Qt.Checked)

    def _on_save(self):
        name = self.name_edit.text().strip()
        if not name:
            return
        selected_exc_ids = [
            self.exc_list.item(i).data(Qt.UserRole).id
            for i in range(self.exc_list.count())
            if self.exc_list.item(i).checkState() == Qt.Checked
        ]
        vp_text = self.vp_edit.text().strip()
        viscosity_profile = (
            [float(x) for x in vp_text.split(",")] if vp_text else []
        )
        notes = self.notes_edit.toPlainText().strip()
        data = {
            "name": name,
            "excipient_ids": selected_exc_ids,
            "viscosity_profile": viscosity_profile,
            "notes": notes
        }
        if self.formulation:
            self.controller.update_formulation(self.formulation.id, data)
        else:
            self.controller.create_formulation(data)
        self.accept()
