# src/visqai/app.py
import sys
from PyQt5.QtWidgets import QApplication
from src.db import init_db
from src.view.main_window import MainWindow


def main():
    init_db()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
