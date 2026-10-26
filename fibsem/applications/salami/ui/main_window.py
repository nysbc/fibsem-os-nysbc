"""Top-level Salami window: a shell holding one tab per page.

Mirrors `autolamella/ui/AutoLamellaMainUI.py`. The shell owns the tab bar and
the menu; each tab is a whole page and does not know about the others:

    Microscope  the live control surface — quad view plus the widget strip
    Protocol    what the experiment will do — info, task parameters, preview

`SalamiUI` is embedded as the Microscope page with its own menu bar hidden, the
same way `AutoLamellaUI` is. The shell's File menu reuses that window's
`QAction`s rather than duplicating them, so there is one definition of "New
Experiment" and it stays wired to one handler.

The Protocol page is assembled from widgets `SalamiUI` already owns — the
experiment info group box and the task editor are reparented out of the tab
strip. Qt reparents on `addWidget`, and every handler keeps working because it
still refers to the same objects.
"""

import logging
import sys
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QWidget,
)

import fibsem
from fibsem.applications.salami.ui.protocol_page import (
    SalamiProtocolPage,
    preview_caption,
)
from fibsem.applications.salami.ui.SalamiUI import SalamiUI
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    GRAY_ICON_COLOR,
    NAPARI_STYLE,
    PRIMARY_BUTTON_STYLESHEET,
)
from fibsem.ui.widgets.notifications import NotificationBell

NO_EXPERIMENT_TOOLTIP = "Create or open an experiment to edit its protocol"
NO_EXPERIMENT_LABEL = "No Experiment"


class SalamiMainWindow(QMainWindow):
    """Shell window: Microscope and Protocol pages."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Salami v{fibsem.__version__}")
        self.resize(1600, 1000)
        self.setStyleSheet(NAPARI_STYLE)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # --- Microscope page -------------------------------------------------
        self.salami_ui = SalamiUI(parent_ui=self)
        # The shell provides the menu; a second one inside the tab would be dead
        # chrome (the same trick AutoLamellaMainUI uses).
        menu_bar = self.salami_ui.menuBar()
        if menu_bar is not None:
            menu_bar.setVisible(False)
        self.salami_ui.setMinimumWidth(550)
        self.tab_widget.addTab(
            self.salami_ui,
            fibsem_icon("mdi:microscope", color=GRAY_ICON_COLOR),
            "Microscope",
        )

        # --- Protocol page ---------------------------------------------------
        self.protocol_page = self._build_protocol_page()
        self.tab_widget.addTab(
            self.protocol_page,
            fibsem_icon("mdi:file-document-edit", color=GRAY_ICON_COLOR),
            "Protocol",
        )

        # Let the control surface push sync previews here, and refresh the tab
        # state when an experiment is created, opened or closed.
        self.salami_ui.protocol_page = self.protocol_page
        self.salami_ui.experiment_changed.connect(self._on_experiment_changed)

        self._create_menu_bar()
        self._create_corner_widget()
        self._on_experiment_changed()

    def _build_protocol_page(self) -> SalamiProtocolPage:
        """Move the protocol-editing widgets out of the tab strip onto a page."""
        ui = self.salami_ui

        # Now that its contents live on the Protocol page, the Task Config tab
        # would be an empty pane.
        index = ui.tabWidget.indexOf(ui.tabTaskConfig)
        if index != -1:
            ui.tabWidget.removeTab(index)

        return SalamiProtocolPage(
            info_widget=ui.groupBoxExpInfo,
            task_selector=ui.comboBoxTasks,
            task_parameters=ui.groupBoxTaskParameters,
            parent=self,
        )

    def _create_menu_bar(self) -> None:
        """Reuse the control surface's actions rather than redefining them."""
        menu_bar = self.menuBar()
        if menu_bar is None:
            logging.warning("No menu bar available on the Salami main window.")
            return

        ui = self.salami_ui
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(ui.actionCreate_New_Experiment)
        file_menu.addAction(ui.actionLoad_Experiment)
        file_menu.addSeparator()
        file_menu.addAction(ui.actionLoad_Config)
        file_menu.addAction(ui.actionExport_Config)

    def _create_corner_widget(self) -> None:
        """Experiment name, New/Open buttons and the notification bell, on the tab bar.

        Kept out of any one page because they apply to all of them — which
        experiment is open is context for everything, and a notification raised
        while you are on the Protocol page still needs to be visible.

        The buttons trigger the same `QAction`s as the File menu, so there is
        still one handler per operation. Unlike autolamella's, they are enabled
        without a microscope: a Salami experiment can be created and its
        protocol edited entirely offline.
        """
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(5)

        self.experiment_name_label = QLabel(NO_EXPERIMENT_LABEL)
        self.experiment_name_label.setStyleSheet("color: #d6d6d6; font-size: 12px;")

        ui = self.salami_ui
        self.btn_create_experiment = QPushButton("Create Experiment")
        self.btn_create_experiment.setToolTip("Create a new experiment")
        self.btn_create_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_create_experiment.clicked.connect(
            ui.actionCreate_New_Experiment.trigger
        )

        self.btn_load_experiment = QPushButton("Load Experiment")
        self.btn_load_experiment.setToolTip("Open an existing experiment")
        self.btn_load_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_load_experiment.clicked.connect(ui.actionLoad_Experiment.trigger)

        # The bell counts notifications raised through the toast service. The
        # toast manager lives on SalamiUI (it needs a page-sized widget to
        # position toasts against), so the bell registers with that one rather
        # than a second manager here.
        self.notification_bell = NotificationBell(self)
        ui.toast_manager.set_notification_bell(self.notification_bell)

        layout.addWidget(self.experiment_name_label)
        layout.addWidget(self.btn_create_experiment)
        layout.addWidget(self.btn_load_experiment)
        layout.addWidget(self.notification_bell)

        self.tab_widget.setCornerWidget(container)

    def _on_experiment_changed(self) -> None:
        """Enable the Protocol page only once there is a protocol to edit.

        Same treatment autolamella gives its Overview and Lamella tabs: visible
        so the workflow is discoverable, disabled with a reason until it applies.
        """
        has_experiment = self.salami_ui.salamiExperiment is not None
        index = self.tab_widget.indexOf(self.protocol_page)
        self.tab_widget.setTabEnabled(index, has_experiment)
        self.tab_widget.setTabToolTip(
            index, "" if has_experiment else NO_EXPERIMENT_TOOLTIP
        )

        name = (
            self.salami_ui.salamiExperimentConfig.name
            if self.salami_ui.salamiExperimentConfig is not None
            else None
        )
        title = f"Salami v{fibsem.__version__}"
        self.setWindowTitle(f"{title} — {name}" if name else title)
        self.experiment_name_label.setText(name or NO_EXPERIMENT_LABEL)

    def show_protocol_preview(self, result) -> None:
        """Display the image a microscope sync acquired."""
        if result.image is None:
            return
        self.protocol_page.show_preview(result.image, preview_caption(result))

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        self.salami_ui.close()
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SalamiMainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
