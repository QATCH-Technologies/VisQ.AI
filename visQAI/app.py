# visQAi/app.py
import sys
from PyQt5.QtWidgets import QApplication
from src.db.sqlite_db import SQLiteDB as init_db
from src.controllers.excipients_controller import ExcipientsController
from src.model.excipient import BaseExcipient
from src.view.main_window import MainWindow
from src.view.manage_windows import ExcipientsUI


def main():
    db = init_db()
    db.drop_database()
    excipient_controller = ExcipientsController()
    excipients = excipient_controller.list_base_excipients()
    if len(excipients) == 0:
        excipient_controller.add_base_excipient(
            BaseExcipient(name="BGG", etype='Protein'))
        excipient_controller.add_base_excipient(
            BaseExcipient(name="BSA", etype='Protein'))
        excipient_controller.add_base_excipient(
            BaseExcipient(name="poly-hlgG", etype='Protein'))
        excipient_controller.add_base_excipient(
            BaseExcipient(name="TWEEN20", etype='Surfactant'))
        excipient_controller.add_base_excipient(
            BaseExcipient(name="TWEEN80", etype='Surfactant'))
        excipient_controller.add_base_excipient(
            BaseExcipient(name="Sucrose", etype='Stabilizer'))
        excipient_controller.add_base_excipient(
            BaseExcipient(name="Trehalose", etype='Stabilizer'))
        excipients = excipient_controller.list_base_excipients()
    for e in excipients:
        print(e.name, e.etype)

    app = QApplication(sys.argv)
    window = ExcipientsUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
