# src/visqai/app.py
import sys
from PyQt5.QtWidgets import QApplication
from src.db.sqlite_db import SQLiteDB as init_db
from src.view.main_window import MainWindow


def main():
    db = init_db()

    excipients = db.list_base_excipients()

    if len(excipients) == 0:

        # Create default excipients:
        db.add_base_excipient("Protein", "BGG")
        db.add_base_excipient("Protein", "BSA")
        db.add_base_excipient("Protein", "poly-hlgG")
        db.add_base_excipient("Surfactant", "TWEEN20")
        db.add_base_excipient("Surfactant", "TWEEN80")
        db.add_base_excipient("Stabilizer", "Sucrose")
        db.add_base_excipient("Stabilizer", "Trehalose")

        excipients = db.list_base_excipients()  # reload

    proteins = []
    surfactants = []
    stabilizers = []
    for e in excipients:
        if e['etype'] == "Protein":
            proteins.append(e['name'])
        if e['etype'] == "Surfactant":
            surfactants.append(e['name'])
        if e['etype'] == "Stabilizer":
            stabilizers.append(e['name'])
    proteins.sort()
    surfactants.sort()
    stabilizers.sort()
    print("Proteins:", proteins)
    print("Surfactants:", surfactants)
    print("Stabilizers:", stabilizers)

    # app = QApplication(sys.argv)
    # window = MainWindow()
    # window.show()
    # sys.exit(app.exec_())


if __name__ == "__main__":
    main()
