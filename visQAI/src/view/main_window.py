# main_window.py
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QFrame, QHBoxLayout, QWidget
from predict_window import PredictWindow
from learn_window import LearnWidnow
from optimize_window import OptimizeWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Viscosity Profile Toolkit")
        self.setMinimumSize(900, 600)
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(PredictWindow(), "Predict")
        self.tab_widget.addTab(LearnWidnow(), "Learn")
        self.tab_widget.addTab(OptimizeWindow(), "Optimize")
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)

        # shared plot area handled inside sub-tabs or by main if needed
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.tab_widget, 1)
        main_layout.addWidget(divider)
        # placeholder for stacked plots if central

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def on_tab_changed(self, index):
        # hook if main needs to adjust shared elements
        pass

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { font-family: Arial; font-size: 13px; }
            QGroupBox { font-weight: bold; border: 1px solid #aaa; border-radius: 4px; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QPushButton { background-color: #4679BD; color: white; border-radius: 4px; padding: 6px 12px; }
        """
                           )
