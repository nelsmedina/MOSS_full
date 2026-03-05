#!/usr/bin/env python3
"""
Loading overlay widget with animated GIF for MOSS.
Displays a centered loading animation with optional message text.
"""

from pathlib import Path
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMovie, QFont

from ..dpi_scaling import scaled, scaled_font


class LoadingOverlay(QWidget):
    """Semi-transparent loading overlay with animated GIF."""

    def __init__(self, parent=None, message: str = "Loading..."):
        super().__init__(parent)
        self._message = message
        self._init_ui()
        self.hide()  # Start hidden by default

    def _init_ui(self):
        """Initialize the UI."""
        # Make overlay fill parent and be semi-transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setAutoFillBackground(True)

        # Set dark semi-transparent background
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 220);
            }
            QLabel {
                background: transparent;
                color: white;
            }
        """)

        # Layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(scaled(20))

        # Loading GIF
        self._gif_label = QLabel()
        self._gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        gif_path = Path(__file__).parent.parent / "resources" / "loading.gif"
        if gif_path.exists():
            self._movie = QMovie(str(gif_path))
            self._gif_label.setMovie(self._movie)
        else:
            self._movie = None
            self._gif_label.setText("Loading...")

        layout.addWidget(self._gif_label)

        # Message label
        self._message_label = QLabel(self._message)
        self._message_label.setFont(scaled_font(14, QFont.Weight.Bold))
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._message_label)

    def set_message(self, message: str):
        """Update the loading message."""
        self._message = message
        self._message_label.setText(message)

    def start(self):
        """Start the loading animation and show the overlay."""
        if self._movie:
            self._movie.start()
        self.show()
        self.raise_()  # Bring to front

    def stop(self):
        """Stop the loading animation and hide the overlay."""
        if self._movie:
            self._movie.stop()
        self.hide()

    def showEvent(self, event):
        """Resize to fill parent when shown."""
        super().showEvent(event)
        if self.parent():
            self.setGeometry(self.parent().rect())
        if self._movie:
            self._movie.start()

    def resizeEvent(self, event):
        """Handle resize to stay full-size."""
        super().resizeEvent(event)
        if self.parent():
            self.setGeometry(self.parent().rect())
