from typing import Optional
from platformdirs import user_log_dir

from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QIcon, QPixmap, QDesktopServices
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QLabel, QPushButton, QDialogButtonBox,
)

from buzz.widgets.icon import APP_ICON_PATH
from buzz.locale import _
from buzz.settings.settings import APP_DISPLAY_NAME


class AboutDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowIcon(QIcon(APP_ICON_PATH))
        self.setWindowTitle(f'{_("About")} {APP_DISPLAY_NAME}')

        layout = QVBoxLayout(self)
        image_label = QLabel()
        image_label.setPixmap(QPixmap(APP_ICON_PATH).scaled(
            80, 80, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_label = QLabel(APP_DISPLAY_NAME)
        app_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_font = QtGui.QFont()
        app_font.setBold(True)
        app_font.setPointSize(20)
        app_label.setFont(app_font)

        description = QLabel("Vietnamese dubbing and subtitles for videos")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.show_logs_button = QPushButton(_("Show logs"), self)
        self.show_logs_button.clicked.connect(self.on_click_show_logs)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        button_box.rejected.connect(self.reject)

        layout.addWidget(image_label)
        layout.addWidget(app_label)
        layout.addWidget(description)
        layout.addWidget(self.show_logs_button)
        layout.addWidget(button_box)
        self.setMinimumWidth(350)

    def on_click_show_logs(self):
        # Preserve the existing log location used by installations on this PC.
        log_dir = user_log_dir(appname="Buzz")
        QDesktopServices.openUrl(QUrl.fromLocalFile(log_dir))
