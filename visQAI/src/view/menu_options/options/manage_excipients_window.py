from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QHBoxLayout, QFormLayout, QLineEdit,
    QComboBox, QDoubleSpinBox, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.controllers.excipients_controller import ExcipientsController


class ManageExcipientsWindow(QDialog):
    """
    Dialog to manage existing excipients using ExcipientsController:
      - Lists known excipients by type
      - Add, Edit, Delete actions
    Emits:
      excipientAdded(type, name, conc, unit)
      excipientEdited(type, old_name, new_name, conc, unit)
      excipientDeleted(type, name)
    """
    excipientAdded = pyqtSignal(str, str, float, str)
    excipientEdited = pyqtSignal(str, str, str, float, str)
    excipientDeleted = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Excipients")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.controller = ExcipientsController()
        self._init_ui()
        self._load_excipients()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Tree of excipients grouped by type
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Concentration", "Unit"])
        layout.addWidget(self.tree)

        # Action buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add")
        edit_btn = QPushButton("Edit")
        del_btn = QPushButton("Delete")
        close_btn = QPushButton("Close")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(del_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # Connections
        add_btn.clicked.connect(self._on_add)
        edit_btn.clicked.connect(self._on_edit)
        del_btn.clicked.connect(self._on_delete)
        close_btn.clicked.connect(self.reject)

    def _load_excipients(self):
        self.tree.clear()
        # Fetch all excipients and group by type locally
        exc_list = self.controller.get_excipients()
        grouped = {}
        for exc in exc_list:
            grouped.setdefault(exc.type, []).append(exc)
        # Populate tree
        for etype, items in grouped.items():
            parent = QTreeWidgetItem([etype])
            self.tree.addTopLevelItem(parent)
            for exc in items:
                child = QTreeWidgetItem([
                    exc.name,
                    str(exc.concentration),
                    str(exc.unit)
                ])
                # Store the excipient ID in column 0, UserRole
                child.setData(0, Qt.UserRole, exc.id)
                parent.addChild(child)
            parent.setExpanded(True)

    def _on_add(self):
        dlg = self.CreateDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            etype, name, conc, unit = dlg.get_values()
            new_exc = self.controller.create_excipient({"name": name,
                                                        "excipient_type": etype,
                                                        "concentration": conc,
                                                        "unit": unit}
                                                       )
            self.excipientAdded.emit(
                new_exc.type, new_exc.name,
                new_exc.concentration, new_exc.unit
            )
            self._load_excipients()

    def _on_edit(self):
        item = self.tree.currentItem()
        if not item or not item.parent():
            QMessageBox.warning(
                self, "Select", "Please select an excipient to edit."
            )
            return
        exc_id = item.data(Qt.UserRole)
        etype = item.parent().text(0)
        old_name = item.text(0)
        dlg = self.CreateDialog(self)
        dlg.set_values(etype, old_name, float(item.text(1)), item.text(2))
        if dlg.exec_() == QDialog.Accepted:
            new_etype, new_name, conc, unit = dlg.get_values()
            updated = self.controller.update_excipient(
                excipient_id=exc_id,
                name=new_name,
                excipient_type=new_etype,
                concentration=conc,
                unit=unit
            )
            self.excipientEdited.emit(
                etype, old_name,
                updated.type, updated.name,
                updated.concentration, updated.unit
            )
            self._load_excipients()

    def _on_delete(self):
        item = self.tree.currentItem()
        if not item or not item.parent():
            QMessageBox.warning(
                self, "Select", "Please select an excipient to delete."
            )
            return
        exc_id = item.data(Qt.UserRole)
        exc_to_delete = next(
            (e for e in self.controller.get_excipients() if e.id == exc_id),
            None
        )
        if QMessageBox.question(
            self, "Confirm Delete", f"Delete '{exc_to_delete.name}'?"
        ) != QMessageBox.Yes:
            return
        if self.controller.delete_excipient(excipient_id=exc_id):
            self.excipientDeleted.emit(
                exc_to_delete.type,
                exc_to_delete.name
            )
            self._load_excipients()

    class CreateDialog(QDialog):
        """
        Sub-dialog to create or edit a single excipient entry.
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Excipient Details")
            self.setModal(True)
            self._build_ui()

        def _build_ui(self):
            layout = QVBoxLayout(self)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignRight)

            self.type_cb = QComboBox()
            self.type_cb.addItems(["Protein", "Sugar", "Surfactant", "Buffer"])
            form.addRow("Type:", self.type_cb)

            self.name_input = QLineEdit()
            form.addRow("Name:", self.name_input)

            self.conc_sb = QDoubleSpinBox()
            self.conc_sb.setRange(0.0, 10000.0)
            self.conc_sb.setDecimals(3)
            form.addRow("Concentration:", self.conc_sb)

            self.unit_cb = QComboBox()
            self.unit_cb.addItems(["g/L", "M", "%"])
            form.addRow("Unit:", self.unit_cb)

            layout.addLayout(form)

            btn_layout = QHBoxLayout()
            save_btn = QPushButton("Save")
            save_btn.clicked.connect(self.accept)
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(self.reject)
            btn_layout.addStretch()
            btn_layout.addWidget(save_btn)
            btn_layout.addWidget(cancel_btn)
            layout.addLayout(btn_layout)

            self.setLayout(layout)

        def set_values(self, etype, name, conc, unit):
            self.type_cb.setCurrentText(etype)
            self.name_input.setText(name)
            self.conc_sb.setValue(conc)
            self.unit_cb.setCurrentText(unit)

        def get_values(self):
            return (
                self.type_cb.currentText(),
                self.name_input.text().strip(),
                self.conc_sb.value(),
                self.unit_cb.currentText()
            )
