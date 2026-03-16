import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
    QLineEdit, QLabel, QDialog, QFormLayout, QTabWidget, QAbstractItemView, QComboBox, QLineEdit,
    QGroupBox
)
from PyQt5.QtWidgets import QMessageBox as QMsgBox
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
import pymysql
from datetime import datetime, timedelta
import json
import zipfile
import base64
from enum import Enum
import os

# Import ctypes for Windows API calls
import ctypes

VERSION = 1.2
RELEASED = "2025-09-08"


DB_TABLE_0 = "subscribers"
DB_COLS_0 = ",".join(
    ['id',
     'subscriber_name',
     'contact',
     'status',
     'creation_date',
     'renewed_date',
     'expiration',
     'term_days'])

DB_TABLE_1 = "licenses"
DB_COLS_1 = ",".join(
    ['license_key',
     'status',
     'creation_date',
     'expiration',
     'term_days',
     'auto_generated',
     'computer_name',
     'os_version',
     'bios_serial',
     'motherboard_serial',
     'cpu_id',
     'disk_serial',
     'system_uuid',
     'subscriber_id'])


class LicenseStatus(Enum):
    """Constants for license status types.

    Attributes:
        ADMIN (str): Administrator license with unlimited access.
        ACTIVE (str): Active paid license.
        TRIAL (str): Trial license with expiration.
        INACTIVE (str): Inactive or suspended license.
    """
    ADMIN = "admin"
    ACTIVE = "active"
    TRIAL = "trial"
    INACTIVE = "inactive"


def icon_path(path):
    try:
        full_path = os.path.join(os.path.dirname(__file__), "icons", path)
        if os.path.isfile(full_path) and full_path.endswith(".png"):
            pass
        else:
            print("ERROR (does not exist):", full_path)
            full_path = None
    except Exception as e:
        print(f"ERROR ({e}):", full_path)
        full_path = None
    finally:
        # print("icon_path:", full_path)
        return full_path


class DBManager:
    def __init__(self):
        self.conn = pymysql.connect(**DBManager._load_avn_key_store())
        self.init_tables()

    @staticmethod
    def _load_avn_key_store():
        DB_CONFIG = {}
        with zipfile.ZipFile("tools/avn_db_manager/avn_key_store.zip", 'r') as zip_key:
            pem_file = zip_key.read("db_config.pem").splitlines()
            pem_file[0] = b""  # remove begin line
            pem_file[-1] = b""  # remove end line
            pem_file[1] = pem_file[1][4:]  # remove "AVN_"
            pem_file = b"".join(pem_file)
            DB_CONFIG = json.loads(base64.b64decode(pem_file).decode()[::2])
            DB_CONFIG['cursorclass'] = pymysql.cursors.DictCursor
        return DB_CONFIG

    def init_tables(self):
        self.conn.ping(reconnect=True)
        with self.conn.cursor() as cursor:
            # cursor.execute("DROP TABLE IF EXISTS {}".format(DB_TABLE_0))
            # cursor.execute("DROP TABLE IF EXISTS {}".format(DB_TABLE_1))
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS subscribers (
                    id                  INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    subscriber_name     VARCHAR(255) NOT NULL,
                    contact             TEXT,
                    status              TEXT NOT NULL,
                    creation_date       DATETIME DEFAULT CURRENT_TIMESTAMP,
                    renewed_date        DATETIME,
                    expiration          DATETIME,
                    term_days           SMALLINT DEFAULT 365,
                    UNIQUE(id, subscriber_name)
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS licenses (
                    license_key         VARCHAR(24) NOT NULL PRIMARY KEY,
                    status              TEXT NOT NULL,
                    creation_date       DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiration          DATETIME,
                    term_days          SMALLINT DEFAULT 90,
                    auto_generated      TINYINT(1) DEFAULT 1,
                    computer_name       TEXT NOT NULL,
                    os_version          TEXT NOT NULL,
                    bios_serial         TEXT NOT NULL,
                    motherboard_serial  TEXT NOT NULL,
                    cpu_id              TEXT NOT NULL,
                    disk_serial         TEXT NOT NULL,
                    system_uuid         TEXT NOT NULL,
                    subscriber_id       INT DEFAULT NULL,
                    FOREIGN KEY (subscriber_id) REFERENCES subscribers(id) ON DELETE SET DEFAULT,
                    UNIQUE(license_key)
                );
                """
            )
        self.close()

    def fetch_subscribers(self):
        try:
            self.conn.ping(reconnect=True)
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT * FROM {}".format(DB_TABLE_0))
                return cursor.fetchall()
        except pymysql.err.ProgrammingError as e:
            if e.args[0] == 1146:  # Error code for 'table doesn't exist'
                print(f"Table \"{DB_TABLE_1}\" does not exist")
                # Handle as needed (e.g., create table, show message)
            else:
                raise
        finally:
            self.close()

    def fetch_licenses(self):
        try:
            self.conn.ping(reconnect=True)
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT * FROM {}".format(DB_TABLE_1))
                return cursor.fetchall()
        except pymysql.err.ProgrammingError as e:
            if e.args[0] == 1146:  # Error code for 'table doesn't exist'
                print(f"Table \"{DB_TABLE_1}\" does not exist")
                # Handle as needed (e.g., create table, show message)
            else:
                raise
        finally:
            self.close()

    def add_subscriber(self, id, subscriber_name, contact, status, creation_date, renewed_date, expiration, term_days):
        self.conn.ping(reconnect=True)
        with self.conn.cursor() as cursor:
            cursor.execute("INSERT INTO {} ({}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)".format(
                DB_TABLE_0, DB_COLS_0), (id, subscriber_name, contact, status, creation_date, renewed_date, expiration, term_days,))
        self.close()

    def edit_subscriber(self, old_id, new_id, subscriber_name, contact, status, creation_date, renewed_date, expiration, term_days):
        self.conn.ping(reconnect=True)
        with self.conn.cursor() as cursor:
            cursor.execute(
                "UPDATE {} SET id=%s WHERE id=%s".format(DB_TABLE_0), (new_id, old_id,))
            cursor.execute(
                "UPDATE {} SET subscriber_name=%s WHERE id=%s".format(DB_TABLE_0), (subscriber_name, new_id,))
            cursor.execute(
                "UPDATE {} SET contact=%s WHERE id=%s".format(DB_TABLE_0), (contact, new_id,))
            cursor.execute(
                "UPDATE {} SET status=%s WHERE id=%s".format(DB_TABLE_0), (status, new_id,))
            cursor.execute(
                "UPDATE {} SET creation_date=%s WHERE id=%s".format(DB_TABLE_0), (creation_date, new_id,))
            cursor.execute(
                "UPDATE {} SET renewed_date=%s WHERE id=%s".format(DB_TABLE_0), (renewed_date, new_id,))
            cursor.execute(
                "UPDATE {} SET expiration=%s WHERE id=%s".format(DB_TABLE_0), (expiration, new_id,))
            cursor.execute(
                "UPDATE {} SET term_days=%s WHERE id=%s".format(DB_TABLE_0), (term_days, new_id,))
        self.close()

    def delete_subscriber(self, id):
        self.conn.ping(reconnect=True)
        with self.conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM {} WHERE id=%s".format(DB_TABLE_0), (id,))
        self.close()

    def add_license(self, license_key, status, creation_date, expiration, term_days, auto_generated, computer_name, os_version, bios_serial, motherboard_serial, cpu_id, disk_serial, system_uuid, subscriber_id):
        self.conn.ping(reconnect=True)
        with self.conn.cursor() as cursor:
            cursor.execute("INSERT INTO {} ({}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)".format(DB_TABLE_1, DB_COLS_1), (license_key, status, creation_date, expiration,
                           term_days, auto_generated, computer_name, os_version, bios_serial, motherboard_serial, cpu_id, disk_serial, system_uuid, subscriber_id,))
        self.close()

    def edit_license(self, old_id, license_key, status, creation_date, expiration, term_days, auto_generated, computer_name, os_version, bios_serial, motherboard_serial, cpu_id, disk_serial, system_uuid, subscriber_id):
        self.conn.ping(reconnect=True)
        with self.conn.cursor() as cursor:
            cursor.execute(
                "UPDATE {} SET license_key=%s WHERE license_key=%s".format(DB_TABLE_1), (license_key, old_id,))
            cursor.execute(
                "UPDATE {} SET status=%s WHERE license_key=%s".format(DB_TABLE_1), (status, license_key,))
            cursor.execute(
                "UPDATE {} SET creation_date=%s WHERE license_key=%s".format(DB_TABLE_1), (creation_date, license_key,))
            cursor.execute(
                "UPDATE {} SET expiration=%s WHERE license_key=%s".format(DB_TABLE_1), (expiration, license_key,))
            cursor.execute(
                "UPDATE {} SET term_days=%s WHERE license_key=%s".format(DB_TABLE_1), (term_days, license_key,))
            cursor.execute(
                "UPDATE {} SET auto_generated=%s WHERE license_key=%s".format(DB_TABLE_1), (auto_generated, license_key,))
            cursor.execute(
                "UPDATE {} SET computer_name=%s WHERE license_key=%s".format(DB_TABLE_1), (computer_name, license_key,))
            cursor.execute(
                "UPDATE {} SET os_version=%s WHERE license_key=%s".format(DB_TABLE_1), (os_version, license_key,))
            cursor.execute(
                "UPDATE {} SET bios_serial=%s WHERE license_key=%s".format(DB_TABLE_1), (bios_serial, license_key,))
            cursor.execute(
                "UPDATE {} SET motherboard_serial=%s WHERE license_key=%s".format(DB_TABLE_1), (motherboard_serial, license_key,))
            cursor.execute(
                "UPDATE {} SET cpu_id=%s WHERE license_key=%s".format(DB_TABLE_1), (cpu_id, license_key,))
            cursor.execute(
                "UPDATE {} SET disk_serial=%s WHERE license_key=%s".format(DB_TABLE_1), (disk_serial, license_key,))
            cursor.execute(
                "UPDATE {} SET system_uuid=%s WHERE license_key=%s".format(DB_TABLE_1), (system_uuid, license_key,))
            cursor.execute(
                "UPDATE {} SET subscriber_id=%s WHERE license_key=%s".format(DB_TABLE_1), (subscriber_id, license_key,))
        self.close()

    def delete_license(self, id):
        self.conn.ping(reconnect=True)
        with self.conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM {} WHERE license_key=%s".format(DB_TABLE_1), (id,))
        self.close()

    def sync_subscriber(self, id):
        self.conn.ping(reconnect=True)
        with self.conn.cursor() as cursor:
            # Set subscription information for active licenses
            if id:  # sync single subscriber
                cursor.execute(
                    "SELECT id, status, expiration, term_days FROM {} WHERE id = %s".format(DB_TABLE_0), (id,))
            else:  # sync all subscriber info
                cursor.execute(
                    "SELECT id, status, expiration, term_days FROM {}".format(DB_TABLE_0))
            data = cursor.fetchall()
            # print(f"1/2 - Fetched {len(data)} rows...")
            for row in data:
                id = row['id']
                status = row['status']
                expiration = row['expiration']
                term_days = row['term_days']

                # Detect expired subscribers, update 'status' to 'expired'
                status_update_required = False
                if status != LicenseStatus.ADMIN.value:
                    if expiration < datetime.now():  # expired
                        if status != LicenseStatus.INACTIVE.value:
                            status_update_required = True
                            status = LicenseStatus.INACTIVE.value
                    else:  # set 'active' (these are subscribers, so they cannot be 'trial')
                        if status != LicenseStatus.ACTIVE.value:
                            status_update_required = True
                            status = LicenseStatus.ACTIVE.value
                # else: pass  # admin user (leave status as 'admin')
                if status_update_required:
                    cursor.execute(
                        "UPDATE {} SET status=%s WHERE id=%s".format(DB_TABLE_0), (status, id,))
                cursor.execute(
                    "UPDATE {} SET status=%s, expiration=%s, term_days=%s WHERE subscriber_id=%s".format(DB_TABLE_1), (status, expiration, term_days, id,))

            # Revert to trial (or expired) for unsubscibed licenses
            cursor.execute(
                "SELECT license_key, creation_date FROM {} WHERE subscriber_id IS NULL".format(DB_TABLE_1))
            data = cursor.fetchall()
            # print(f"2/2 - Fetched {len(data)} rows...")
            for row in data:
                key = row['license_key']
                status = LicenseStatus.TRIAL.value
                creation_date = row['creation_date']
                term_days = 90
                expiration = creation_date + timedelta(days=term_days)
                # Detect expired trials and mark 'expired'
                if expiration < datetime.now():  # expired
                    if status != LicenseStatus.INACTIVE.value:
                        status = LicenseStatus.INACTIVE.value
                # print(f"Updating key {key} to expiration date {expiration}")
                cursor.execute(
                    "UPDATE {} SET status=%s, expiration=%s, term_days=%s WHERE license_key=%s".format(DB_TABLE_1), (status, expiration, term_days, key,))
        self.close()

    def close(self):
        try:
            if self.conn.open:
                self.conn.commit()
                self.conn.close()
        except:
            pass

    def __del__(self):
        self.close()


class EntryDialog(QDialog):
    def __init__(self, title: str, fields: list, values: dict = {}):
        super().__init__()
        self.setWindowTitle(title)
        icon_name = f"{title.split()[1].lower()}s"
        self.setWindowIcon(QIcon(icon_path(f"tables/{icon_name}.png")))
        layout = QFormLayout()
        self.inputs: list[QLineEdit] = []
        self.fields = fields
        self.values = values
        for field in fields:
            self.inputs.append(QLineEdit())
            if values and values.get(field):
                self.inputs[-1].setText(str(values.get(field)))
            if field != "id":
                layout.addRow(f"{field}:", self.inputs[-1])
        btns = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addRow(btns)
        self.setLayout(layout)
        self.setMinimumSize(2*self.sizeHint().width(),
                            self.sizeHint().height())

    def get_values(self):
        vals = []
        resync = False
        self.creation_date = datetime.now().isoformat()
        for idx, input in enumerate(self.inputs):
            field = self.fields[idx]
            if field in ['status', 'expiration', 'term_days', 'term_days', 'subscriber_id']:
                if self.values.get(field) != input.text():
                    if resync == False and QMessageBox.question(self, "Re-Sync Subscription Status", "Would you like to re-sync the license status with the subscriber database?") == QMessageBox.StandardButton.Yes:
                        resync = True
            if field in ['id', 'subscriber_id']:
                if input.text().upper() in ["-1", "NONE", "NULL", ""]:
                    vals.append(None)
                    continue
            if field in ['creation_date', 'renewed_date', 'expiration']:
                if input.text().upper() in ["-1", "NONE", "NULL", ""]:
                    vals.append(self.creation_date)
                    continue
            if field in ['term_days', 'term_days']:
                if input.text().upper() in ["-1", "NONE", "NULL", ""]:
                    input.setText("90")
                # Calculate expiration date as "creation_date + term_days"
                vals[-1] = (datetime.fromisoformat(str(self.creation_date)) +
                            timedelta(days=float(input.text()))).isoformat()
            vals.append(input.text())
            if field in ['creation_date', 'renewed_date']:
                self.creation_date = vals[-1]
        return vals, resync


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.set_app_user_model_id()
        self.setWindowTitle(
            "Subscribers & Licenses Database Manager for QATCH nanovisQ\u2122 VisQ.AI\u2122")
        app_icon = QIcon(icon_path("DBManager.png"))
        self.setWindowIcon(app_icon)
        self.db = DBManager()
        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.table0 = QTableWidget()
        self.table0.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tab0 = QWidget()
        self.tab0_layout = QVBoxLayout(self.tab0)
        self.filter0 = QGroupBox()
        self.filter0_layout = QHBoxLayout(self.filter0)
        self.filter0_layout.addWidget(QLabel("Filter:"))
        self.filter0_layout.addWidget(QComboBox(), 1)
        self.filter0_layout.addWidget(QLineEdit(), 1)
        self.filter0_layout.addWidget(QPushButton(
            icon=QIcon(icon_path("buttons/filter.png")), text=""))
        self.filter0_layout.addSpacing(10)
        self.filter0_layout.addWidget(QLabel("<a href='#clear'>Clear</a>"))
        self.filter0_layout.addStretch(10)
        self.filter0.findChild(QComboBox).addItems(DB_COLS_0.split(","))
        self.filter0.findChild(QLineEdit).returnPressed.connect(
            self.filter0.findChild(QPushButton).click)
        self.filter0.findChild(QPushButton).clicked.connect(
            lambda: self.filter_table(id=0))
        self.filter0.findChildren(QLabel)[-1].linkActivated.connect(
            lambda: self.filter_table(id=0, clear=True))
        self.filter0.findChildren(QLabel)[-1].setHidden(True)
        self.tab0_layout.addWidget(self.filter0)
        self.tab0_layout.addWidget(self.table0)
        self.table1 = QTableWidget()
        self.table1.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout(self.tab1)
        self.filter1 = QGroupBox()
        self.filter1_layout = QHBoxLayout(self.filter1)
        self.filter1_layout.addWidget(QLabel("Filter:"))
        self.filter1_layout.addWidget(QComboBox(), 1)
        self.filter1_layout.addWidget(QLineEdit(), 1)
        self.filter1_layout.addWidget(QPushButton(
            icon=QIcon(icon_path("buttons/filter.png")), text=""))
        self.filter1_layout.addSpacing(10)
        self.filter1_layout.addWidget(QLabel("<a href='#clear'>Clear</a>"))
        self.filter1_layout.addStretch(10)
        self.filter1.findChild(QComboBox).addItems(DB_COLS_1.split(","))
        self.filter1.findChild(QLineEdit).returnPressed.connect(
            self.filter1.findChild(QPushButton).click)
        self.filter1.findChild(QPushButton).clicked.connect(
            lambda: self.filter_table(id=1))
        self.filter1.findChildren(QLabel)[-1].linkActivated.connect(
            lambda: self.filter_table(id=1, clear=True))
        self.filter1.findChildren(QLabel)[-1].setHidden(True)
        self.tab1_layout.addWidget(self.filter1)
        self.tab1_layout.addWidget(self.table1)
        self.about = QWidget()
        about_layout = QVBoxLayout(self.about)
        software_icon = QLabel()
        software_icon.setPixmap(app_icon.pixmap(64, 64))
        software_icon.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        software_title = QLabel(f"<b>{self.windowTitle()}</b>")
        software_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        software_version = QLabel(f"Version {VERSION} (Released {RELEASED})")
        software_version.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        icons8_credit = QLabel(
            "Icons by <a href='https://icons8.com/'>Icons8</a> (<a href='https://icons8.com/icons/all--style-3d-fluency'>3D Fluency</a> Style)")
        icons8_credit.setOpenExternalLinks(True)  # Automatically opens links
        icons8_credit.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        copyright_notice = QLabel(
            "\u00a9 2025 QATCH Technologies. All rights reserved.")
        copyright_notice.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        about_layout.addStretch()
        about_layout.addWidget(software_icon)
        about_layout.addWidget(software_title)
        about_layout.addWidget(software_version)
        about_layout.addStretch()
        about_layout.addWidget(icons8_credit)
        about_layout.addWidget(copyright_notice)
        self.tabs.addTab(self.tab0, QIcon(
            icon_path(f"tables/{DB_TABLE_0}.png")), "Subscribers")
        self.tabs.addTab(self.tab1, QIcon(
            icon_path(f"tables/{DB_TABLE_1}.png")), "Licenses")
        self.tabs.addTab(self.about, "About")
        self.layout.addWidget(self.tabs)
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton(
            QIcon(icon_path("buttons/add.png")), " Add Entry")
        self.edit_btn = QPushButton(
            QIcon(icon_path("buttons/edit.png")), " Edit Entry")
        self.del_btn = QPushButton(
            QIcon(icon_path("buttons/remove.png")), " Delete Entry")
        self.refresh_btn = QPushButton(
            QIcon(icon_path("buttons/re-sync.png")), " Re-Sync")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.del_btn)
        btn_layout.addWidget(self.refresh_btn)
        self.layout.addLayout(btn_layout)
        self.setLayout(self.layout)
        self.add_btn.clicked.connect(self.add_entry)
        self.edit_btn.clicked.connect(self.edit_entry)
        self.del_btn.clicked.connect(self.delete_entry)
        self.refresh_btn.clicked.connect(self.resynchronize)
        self.load_data(all=True)
        self.setMinimumSize(3*self.sizeHint().width(),
                            2*self.sizeHint().height())

    def set_app_user_model_id(self):
        """
        Set Windows App User Model ID to make taskbar show custom icon
        This is the most reliable solution for Windows 7+
        """
        try:
            # Set a unique App User Model ID for your application
            myappid = f'QATCH.VisQAI.DBManager.{VERSION}'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                myappid)
            return True
        except Exception as e:
            return False

    def load_data(self, all=False):
        idx = self.tabs.currentIndex()
        if all or idx == 0:
            data = self.db.fetch_subscribers()
            self.table0.setRowCount(len(data))
            self.table0.setColumnCount(DB_COLS_0.count(",") + 1)
            self.table0.setHorizontalHeaderLabels(DB_COLS_0.split(","))
            for row_idx, row in enumerate(data):
                for col_idx, col in enumerate(DB_COLS_0.split(",")):
                    self.table0.setItem(
                        row_idx, col_idx, QTableWidgetItem(str(row[col])))
                    if col == 'status':
                        status_icon = icon_path(
                            f"shields/{row[col]}.png") or icon_path(f"shields/question.png")
                        if row['expiration'] - datetime.now() < timedelta(days=90) and row[col] == 'active':
                            status_icon = icon_path(f"shields/warning.png")
                            self.table0.item(row_idx, col_idx).setToolTip(
                                "Expiring soon")
                        self.table0.item(row_idx, col_idx).setIcon(
                            QIcon(status_icon))
            self.table0.resizeColumnsToContents()
        if all or idx == 1:
            data = self.db.fetch_licenses()
            self.table1.setRowCount(len(data))
            self.table1.setColumnCount(DB_COLS_1.count(",") + 1)
            self.table1.setHorizontalHeaderLabels(DB_COLS_1.split(","))
            for row_idx, row in enumerate(data):
                for col_idx, col in enumerate(DB_COLS_1.split(",")):
                    self.table1.setItem(
                        row_idx, col_idx, QTableWidgetItem(str(row[col])))
                    if col == 'status':
                        status_icon = icon_path(
                            f"shields/{row[col]}.png") or icon_path(f"shields/question.png")
                        if row['expiration'] - datetime.now() < timedelta(days=90) and row[col] == 'active':
                            status_icon = icon_path(f"shields/warning.png")
                            self.table1.item(row_idx, col_idx).setToolTip(
                                "Expiring soon")
                        self.table1.item(row_idx, col_idx).setIcon(
                            QIcon(status_icon))
            self.table1.resizeColumnsToContents()
        if idx not in [0, 1] and not all:
            print(f"Unknown tab index: {idx}")

    def add_entry(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.add_subscriber()
        elif idx == 1:
            self.add_license()
        else:
            print(f"Unknown tab index: {idx}")

    def add_subscriber(self):
        dlg = EntryDialog("Add Subscriber", fields=DB_COLS_0.split(","))
        if dlg.exec_():
            try:
                values, resync = dlg.get_values()
                self.db.add_subscriber(values[0], values[1], values[2], values[3],
                                       values[4], values[5], values[6], values[7])
                if resync:
                    self.resynchronize(id=values[0])
                else:
                    self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                # raise e

    def add_license(self):
        dlg = EntryDialog("Add License", fields=DB_COLS_1.split(","))
        if dlg.exec_():
            try:
                values, resync = dlg.get_values()
                self.db.add_license(values[0], values[1], values[2], values[3], values[4], values[5], values[6],
                                    values[7], values[8], values[9], values[10], values[11], values[12], values[13])
                if resync:
                    self.resynchronize(id=values[13])
                else:
                    self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                # raise e

    def edit_entry(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.edit_subscriber()
        elif idx == 1:
            self.edit_license()
        else:
            print(f"Unknown tab index: {idx}")

    def edit_subscriber(self):
        selected = self.table0.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Select Row", "Select a row to edit.")
            return
        old_values = {}
        for idx, field in enumerate(DB_COLS_0.split(",")):
            old_values[field] = self.table0.item(selected, idx).text()
        dlg = EntryDialog(
            "Edit Subscriber", fields=DB_COLS_0.split(","), values=old_values)
        if dlg.exec_():
            try:
                values, resync = dlg.get_values()
                self.db.edit_subscriber(old_values['id'], values[0], values[1], values[2], values[3],
                                        values[4], values[5], values[6], values[7])
                if resync:
                    self.resynchronize(id=values[0])
                else:
                    self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                # raise e

    def edit_license(self):
        selected = self.table1.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Select Row", "Select a row to edit.")
            return
        old_values = {}
        for idx, field in enumerate(DB_COLS_1.split(",")):
            old_values[field] = self.table1.item(selected, idx).text()
        dlg = EntryDialog(
            "Edit License", fields=DB_COLS_1.split(","), values=old_values)
        if dlg.exec_():
            try:
                values, resync = dlg.get_values()
                self.db.edit_license(old_values['license_key'], values[0], values[1], values[2], values[3], values[4], values[5],
                                     values[6], values[7], values[8], values[9], values[10], values[11], values[12], values[13])
                if resync:
                    self.resynchronize(id=values[13])
                else:
                    self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                # raise e

    def delete_entry(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.delete_subscriber()
        elif idx == 1:
            self.delete_license()
        else:
            print(f"Unknown tab index: {idx}")

    def delete_subscriber(self):
        selected = self.table0.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Select Row", "Select a row to delete.")
            return
        try:
            id_val = self.table0.item(selected, 0).text()
            sub_name = self.table0.item(selected, 1).text()
            if QMessageBox.question(self, "Confirm Deletion", "Are you sure you want to delete the subscriber:\n\"{}\"?\nWARNING: This operation cannot be undone.".format(sub_name)) == QMessageBox.StandardButton.Yes:
                self.db.delete_subscriber(id_val)
                self.load_data()
            else:
                print("User declined confirmation. No action taken.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            # raise e

    def delete_license(self):
        selected = self.table1.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Select Row", "Select a row to delete.")
            return
        try:
            id_val = self.table1.item(selected, 0).text()
            if QMessageBox.question(self, "Confirm Deletion", "Are you sure you want to delete the license key:\n\"{}\"?\nWARNING: This operation cannot be undone.".format(id_val)) == QMessageBox.StandardButton.Yes:
                self.db.delete_license(id_val)
                self.load_data()
            else:
                print("User declined confirmation. No action taken.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            # raise e

    def resynchronize(self, id=None):
        self.db.sync_subscriber(id)
        self.load_data()

    def filter_table(self, id=None, clear=False):
        try:
            filter: QWidget = getattr(self, f"filter{id}")

            # Get combobox and lineedit values
            col_idx = filter.findChild(QComboBox).currentIndex()
            value = filter.findChild(QLineEdit).text()

            # Mark as clear if value if empty
            if len(value) == 0:
                clear = True

            # Show/hide the "Clear" link label accordingly
            filter.findChildren(QLabel)[-1].setHidden(clear)

            # Clear combobox and/or lineedit
            if clear:
                # filter.findChild(QComboBox).setCurrentIndex(0)
                filter.findChild(QLineEdit).setText(None)

            # Iterate table, hiding rows that do not match
            table: QTableWidget = getattr(self, f"table{id}")
            for row_idx in range(table.rowCount()):
                if clear:
                    table.showRow(row_idx)
                elif table.item(row_idx, col_idx).text().lower().find(value.lower()) == -1:
                    # case-insensitive query not found
                    table.hideRow(row_idx)
                else:
                    table.showRow(row_idx)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            # raise e

    def closeEvent(self, event):
        self.db.close()
        event.accept()


class QMessageBox(QMsgBox):

    def _configure(msg_box, title, text, icon, buttons, default_button):
        """
        Helper function for configuring custom QMessageBox windows.

        Args:
            msg_box: Message box (QMessageBox, self)
            title: Window title
            text: Message text
            icon: Custom icon (QIcon, QPixmap, or file path string)
            buttons: Button combination (same as QMessageBox standard buttons)
            default_button: Default button selection
        """
        # Populate the message box
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        # Set custom icon
        if isinstance(icon, str) or isinstance(icon, QPixmap):
            icon = QIcon(icon)
        if isinstance(icon, QIcon):
            # 64x64 is standard message box icon size
            pixmap = icon.pixmap(64, 64)
            msg_box.setIconPixmap(pixmap)
            msg_box.setWindowIcon(icon)
        # Set buttons
        msg_box.setStandardButtons(buttons)
        msg_box.setDefaultButton(default_button)

    @staticmethod
    def critical(parent=None, title="Critical", text="",
                 icon=icon_path("shields/critical.png"), buttons=QMsgBox.StandardButton.Ok,
                 default_button=QMsgBox.StandardButton.NoButton):
        """
        Custom QMessageBox.critical() that allows setting a custom icon

        Args:
            parent: Parent widget (can be None)
            title: Window title
            text: Message text
            icon: Custom icon (QIcon, QPixmap, or file path string)
            buttons: Button combination (same as QMessageBox standard buttons)
            default_button: Default button selection

        Returns:
            QMessageBox standard button constant (QMessageBox.Yes, QMessageBox.No, etc.)
        """
        # Create the message box
        msg_box = QMessageBox(parent)
        # Default critical icon
        msg_box.setIcon(QMessageBox.Icon.Critical)
        # Configure the message box
        msg_box._configure(title, text, icon, buttons, default_button)
        # Show and return result
        return msg_box.exec_()

    @staticmethod
    def warning(parent=None, title="Warning", text="",
                icon=icon_path("shields/warning.png"), buttons=QMsgBox.StandardButton.Ok,
                default_button=QMsgBox.StandardButton.NoButton):
        """
        Custom QMessageBox.warning() that allows setting a custom icon

        Args:
            parent: Parent widget (can be None)
            title: Window title
            text: Message text
            icon: Custom icon (QIcon, QPixmap, or file path string)
            buttons: Button combination (same as QMessageBox standard buttons)
            default_button: Default button selection

        Returns:
            QMessageBox standard button constant (QMessageBox.Yes, QMessageBox.No, etc.)
        """
        # Create the message box
        msg_box = QMessageBox(parent)
        # Default warning icon
        msg_box.setIcon(QMessageBox.Icon.Warning)
        # Configure the message box
        msg_box._configure(title, text, icon, buttons, default_button)
        # Show and return result
        return msg_box.exec_()

    @staticmethod
    def question(parent=None, title="Question", text="",
                 icon=icon_path("shields/question.png"), buttons=QMsgBox.StandardButton.Yes | QMsgBox.StandardButton.No,
                 default_button=QMsgBox.StandardButton.No):
        """
        Custom QMessageBox.question() that allows setting a custom icon

        Args:
            parent: Parent widget (can be None)
            title: Window title
            text: Message text
            icon: Custom icon (QIcon, QPixmap, or file path string)
            buttons: Button combination (same as QMessageBox standard buttons)
            default_button: Default button selection

        Returns:
            QMessageBox standard button constant (QMessageBox.Yes, QMessageBox.No, etc.)
        """
        # Create the message box
        msg_box = QMessageBox(parent)
        # Default question icon
        msg_box.setIcon(QMessageBox.Icon.Question)
        # Configure the message box
        msg_box._configure(title, text, icon, buttons, default_button)
        # Show and return result
        return msg_box.exec_()

    @staticmethod
    def information(parent=None, title="Information", text="",
                    icon=None, buttons=QMsgBox.StandardButton.Ok,
                    default_button=QMsgBox.StandardButton.NoButton):
        """
        Custom QMessageBox.information() that allows setting a custom icon

        Args:
            parent: Parent widget (can be None)
            title: Window title
            text: Message text
            icon: Custom icon (QIcon, QPixmap, or file path string)
            buttons: Button combination (same as QMessageBox standard buttons)
            default_button: Default button selection

        Returns:
            QMessageBox standard button constant (QMessageBox.Yes, QMessageBox.No, etc.)
        """
        # Create the message box
        msg_box = QMessageBox(parent)
        # Default information icon
        msg_box.setIcon(QMessageBox.Icon.Information)
        # Configure the message box
        msg_box._configure(title, text, icon, buttons, default_button)
        # Show and return result
        return msg_box.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
