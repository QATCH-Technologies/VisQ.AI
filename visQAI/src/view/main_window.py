# main_window.py
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QFrame, QHBoxLayout, QWidget,
    QTabBar, QStylePainter, QStyleOptionTab, QStyle, QAction
)
from PyQt5.QtCore import QSize
from .predict_window import PredictWindow
from .learn_window import LearnWindow
from .optimize_window import OptimizeWindow
from .menu_options.options.manage_predictors_window import ManagePredictorsWindow
from .menu_options.options.manage_excipients_window import ManageExcipientsWindow
from .menu_options.options.manage_formulations_window import ManageFormulationsWindow


class HorizontalTabBar(QTabBar):
    def tabSizeHint(self, index):
        sz = super().tabSizeHint(index)
        # swap height/width and pad so text has room
        return QSize(sz.height() + 20, sz.width() + 40)

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionTab()
        for idx in range(self.count()):
            self.initStyleOption(opt, idx)
            opt.shape = QTabBar.RoundedNorth    # draw as if tabs were on top
            # draw the tab “shell”
            painter.drawControl(QStyle.CE_TabBarTab, opt)
            # draw the label
            painter.drawControl(QStyle.CE_TabBarTabLabel, opt)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Viscosity Profile Toolkit")
        self.setMinimumSize(900, 600)
        self._create_menu_bar()
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabBar(HorizontalTabBar())
        self.tab_widget.setTabPosition(QTabWidget.West)

        self.tab_widget.addTab(PredictWindow(),  "Predict")
        self.tab_widget.addTab(LearnWindow(),    "Learn")
        self.tab_widget.addTab(OptimizeWindow(), "Optimize")

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.tab_widget, 1)
        main_layout.addWidget(divider)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _create_menu_bar(self):
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("File")

        # Edit menu
        options_menu = mb.addMenu("Options")
        excipient_actions = QAction("Manage Excipients", self)
        excipient_actions.triggered.connect(self.excipient_action)
        options_menu.addAction(excipient_actions)
        formulation_actions = QAction("Manage Formulations", self)
        formulation_actions.triggered.connect(self.formulation_action)
        options_menu.addAction(formulation_actions)
        predictors_actions = QAction("Manage Predictors", self)
        predictors_actions.triggered.connect(self.predictors_action)
        options_menu.addAction(predictors_actions)
        # View menu
        view_menu = mb.addMenu("View")
        # add view actions here

        # Help menu
        help_menu = mb.addMenu("Help")
        # add help actions here

    def predictors_action(self):
        self.import_window = ManagePredictorsWindow(parent=self)
        self.import_window.show()

    def excipient_action(self):
        self.excipient_window = ManageExcipientsWindow(parent=self)
        self.excipient_window.show()

    def formulation_action(self):
        self.formulation_window = ManageFormulationsWindow(parent=self)
        self.formulation_window.show()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { font-family: Arial; font-size: 13px; }
            QGroupBox { font-weight: bold; border: 1px solid #aaa; border-radius: 4px; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QPushButton { background-color: #4679BD; color: white; border-radius: 4px; padding: 6px 12px; }
        """)
