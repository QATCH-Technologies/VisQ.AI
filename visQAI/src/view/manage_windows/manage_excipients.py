import sys
import uuid
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QTableWidget, QTableWidgetItem, QPushButton, QDialog,
    QFormLayout, QLineEdit, QComboBox, QLabel, QMessageBox, QToolButton,
    QDialogButtonBox
)
from src.controllers.excipients_controller import ExcipientsController
from src.model.excipient import BaseExcipient, VisQExcipient, ConcentrationUnit


class BaseDialog(QDialog):
    """Dialog to add or edit a BaseExcipient."""

    def __init__(self, parent=None, base: BaseExcipient = None):
        super().__init__(parent)
        self.setWindowTitle("Base Excipient")
        layout = QFormLayout(self)

        self.name_edit = QLineEdit(self)
        self.etype_edit = QLineEdit(self)
        layout.addRow("Name:", self.name_edit)
        layout.addRow("Type:", self.etype_edit)

        if base:
            self.name_edit.setText(base.name)
            self.etype_edit.setText(base.etype)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> dict:
        return {"name": self.name_edit.text(), "etype": self.etype_edit.text()}


class VariationDialog(QDialog):
    """
    Dialog for adding or editing a VisQExcipient variation.
    """

    def __init__(self, parent: QWidget, base: BaseExcipient, variation: Optional[VisQExcipient] = None):
        super().__init__(parent)
        self.base = base
        self.variation = variation
        self.setWindowTitle(
            f"Add Variation for {base.name} ({base.etype})" if variation is None else f"Edit Variation for {base.name} ({base.etype})"
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)

        # Display base info
        self.base_label = QLabel(
            f"Excipient: {self.base.name}  |  Type: {self.base.etype}")
        layout.addRow(self.base_label)

        # Concentration input
        self.conc_input = QLineEdit(self)
        layout.addRow("Concentration:", self.conc_input)

        # Unit selector
        self.unit_combo = QComboBox(self)
        for unit in ConcentrationUnit:
            self.unit_combo.addItem(str(unit), unit)
        layout.addRow("Unit:", self.unit_combo)

        # If editing, pre-fill
        if self.variation:
            self.conc_input.setText(str(self.variation.concentration))
            idx = self.unit_combo.findData(self.variation.unit)
            if idx >= 0:
                self.unit_combo.setCurrentIndex(idx)

        # Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save", self)
        self.cancel_btn = QPushButton("Cancel", self)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addRow(btn_layout)

        # Connections
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def get_values(self) -> Optional[VisQExcipient]:
        try:
            conc = float(self.conc_input.text())
        except ValueError:
            QMessageBox.critical(
                self, "Error", "Concentration must be a number.")
            return None
        unit = self.unit_combo.currentData()
        vid = self.variation.id if self.variation else uuid.uuid4()
        return VisQExcipient(
            name=self.base.name,
            etype=self.base.etype,
            concentration=conc,
            unit=unit,
            id=vid
        )


class ExcipientsUI(QMainWindow):
    """
    Main window for managing BaseExcipients and their VisQExcipient variations.
    """

    def __init__(self):
        super().__init__()
        self.controller = ExcipientsController()
        self.setWindowTitle("Excipient Manager")
        self._build_ui()
        self._load_bases()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: Base excipients list and buttons
        left_layout = QVBoxLayout()
        self.base_list = QListWidget(self)
        left_layout.addWidget(QLabel("Base Excipients:"))
        left_layout.addWidget(self.base_list)
        btn_add_base = QPushButton("Add Base", self)
        btn_edit_base = QPushButton("Edit Base", self)
        btn_del_base = QPushButton("Delete Base", self)
        left_layout.addWidget(btn_add_base)
        left_layout.addWidget(btn_edit_base)
        left_layout.addWidget(btn_del_base)

        # Right: Variations table and buttons
        right_layout = QVBoxLayout()
        self.var_table = QTableWidget(0, 3, self)
        self.var_table.setHorizontalHeaderLabels(
            ["ID", "Concentration", "Unit"])
        right_layout.addWidget(QLabel("Variations:"))
        right_layout.addWidget(self.var_table)
        btn_add_var = QPushButton("Add Variation", self)
        btn_edit_var = QPushButton("Edit Variation", self)
        btn_del_var = QPushButton("Delete Variation", self)
        right_layout.addWidget(btn_add_var)
        right_layout.addWidget(btn_edit_var)
        right_layout.addWidget(btn_del_var)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Connections
        self.base_list.currentItemChanged.connect(self._on_base_selected)
        btn_add_base.clicked.connect(self._on_add_base)
        btn_edit_base.clicked.connect(self._on_edit_base)
        btn_del_base.clicked.connect(self._on_delete_base)
        btn_add_var.clicked.connect(self._on_add_variation)
        btn_edit_var.clicked.connect(self._on_edit_variation)
        btn_del_var.clicked.connect(self._on_delete_variation)

    def _load_bases(self) -> None:
        self.base_list.clear()
        for base in self.controller.list_base_excipients():
            item = self.base_list.addItem(
                f"{base.name} ({base.etype})|{base.id}")
        # Optionally select the first
        if self.base_list.count() > 0:
            self.base_list.setCurrentRow(0)

    def _on_base_selected(self) -> None:
        current = self.base_list.currentItem()
        if not current:
            return
        text = current.text()
        # Format: "name (etype)|id"
        try:
            _, id_str = text.rsplit("|", 1)
        except ValueError:
            return
        self._load_variations(id_str)

    def _on_delete_base(self) -> None:
        current = self.base_list.currentItem()
        if not current:
            return
        _, id_str = current.text().rsplit("|", 1)
        try:
            self.controller.delete_base_excipient(id_str)
            self._load_bases()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _load_variations(self, base_id: str) -> None:
        self.var_table.setRowCount(0)
        profile = self.controller.get_profile(base_id)
        if not profile:
            return
        for var in profile.get_variations():
            row = self.var_table.rowCount()
            self.var_table.insertRow(row)
            self.var_table.setItem(row, 0, QTableWidgetItem(str(var.id)))
            self.var_table.setItem(
                row, 1, QTableWidgetItem(str(var.concentration)))
            self.var_table.setItem(row, 2, QTableWidgetItem(str(var.unit)))

    def on_var_concentration_edit(self, item: QTableWidgetItem):
        """Handle editing of concentration value."""
        row = item.row()
        try:
            conc = float(item.text())
            var = self.controller.get_variation(str(self.var_ids[row]))
            var.concentration = conc
            self.controller.update_variation(var)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.load_variations()

    def on_var_unit_change(self, row: int):
        """Handle change of concentration unit."""
        try:
            combo: QComboBox = self.var_table.cellWidget(row, 1)
            unit: ConcentrationUnit = combo.currentData()
            var = self.controller.get_variation(str(self.var_ids[row]))
            var.unit = unit
            self.controller.update_variation(var)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self._load_variations()

    def _on_delete_variation(self, var_id: uuid.UUID):
        """Delete a variation and refresh list."""
        try:
            self.controller.delete_variation(str(var_id))
            self.load_variations()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_add_base(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Base Excipient")
        form = QFormLayout(dlg)
        name_input = QLineEdit(dlg)
        etype_input = QLineEdit(dlg)
        form.addRow("Name:", name_input)
        form.addRow("Type:", etype_input)
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save", dlg)
        cancel_btn = QPushButton("Cancel", dlg)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        form.addRow(btn_layout)
        save_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        if dlg.exec() == QDialog.Accepted:
            name = name_input.text().strip()
            etype = etype_input.text().strip()
            if not name or not etype:
                QMessageBox.critical(
                    self, "Error", "Name and type cannot be empty.")
                return
            try:
                new_base = BaseExcipient(
                    name=name, etype=etype, id=uuid.uuid4())
                self.controller.add_base_excipient(new_base)
                self._load_bases()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _on_edit_base(self) -> None:
        current = self.base_list.currentItem()
        if not current:
            QMessageBox.warning(
                self, "Warning", "Select a base excipient first.")
            return
        try:
            name_and_type, id_str = current.text().rsplit("|", 1)
            name_part, etype_part = name_and_type.rsplit(" (", 1)
            etype = etype_part.rstrip(")")
            name = name_part
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid base entry format.")
            return
        base = self.controller.get_base_excipient(id_str)
        if not base:
            QMessageBox.critical(self, "Error", "Selected base not found.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit Base Excipient ({base.name} - {base.etype})")
        form = QFormLayout(dlg)
        name_input = QLineEdit(base.name, dlg)
        etype_input = QLineEdit(base.etype, dlg)
        form.addRow("Name:", name_input)
        form.addRow("Type:", etype_input)
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save", dlg)
        cancel_btn = QPushButton("Cancel", dlg)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        form.addRow(btn_layout)
        save_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        if dlg.exec() == QDialog.Accepted:
            new_name = name_input.text().strip()
            new_etype = etype_input.text().strip()
            if not new_name or not new_etype:
                QMessageBox.critical(
                    self, "Error", "Name and type cannot be empty.")
                return
            try:
                updated = BaseExcipient(
                    name=new_name, etype=new_etype, id=base.id)
                self.controller.update_base_excipient(updated)
                self._load_bases()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _on_add_variation(self) -> None:
        current = self.base_list.currentItem()
        if not current:
            QMessageBox.warning(
                self, "Warning", "Select a base excipient first.")
            return
        _, base_id = current.text().rsplit("|", 1)
        base = self.controller.get_base_excipient(base_id)
        if not base:
            QMessageBox.critical(self, "Error", "Selected base not found.")
            return
        dlg = VariationDialog(self, base)
        if dlg.exec() == QDialog.Accepted:
            new_var = dlg.get_values()
            if new_var:
                try:
                    self.controller.add_variation(str(base.id), new_var)
                    self._load_variations(str(base.id))
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))

    def _on_edit_variation(self) -> None:
        row = self.var_table.currentRow()
        if row < 0:
            return
        var_id = self.var_table.item(row, 0).text()
        base_item = self.base_list.currentItem()
        _, base_id = base_item.text().rsplit("|", 1)
        base = self.controller.get_base_excipient(base_id)
        var = self.controller.get_variation(var_id)
        if not base or not var:
            return
        dlg = VariationDialog(self, base, var)
        if dlg.exec() == QDialog.Accepted:
            updated = dlg.get_values()
            if updated:
                try:
                    self.controller.update_variation(updated)
                    self._load_variations(base_id)
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))

    def _on_delete_variation(self) -> None:
        row = self.var_table.currentRow()
        if row < 0:
            return
        var_id = self.var_table.item(row, 0).text()
        try:
            self.controller.delete_variation(var_id)
            base_item = self.base_list.currentItem()
            _, base_id = base_item.text().rsplit("|", 1)
            self._load_variations(base_id)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ExcipientsUI()
    window.show()
    sys.exit(app.exec_())
