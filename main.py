#!/usr/bin/env python3
"""
EyeShot AI - Smart OCR Text Extraction Tool
Created by: Tigran0000
Date: 2025-06-16
Last updated: 2025-06-16 15:33:58 UTC

Entry point for the application with optimized high DPI support.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from ocr.utils.tesseract_config import configure_tesseract
configure_tesseract()

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow

def main():
    """Main application entry point"""
    
    # Set high DPI attributes BEFORE creating QApplication
    # This fixes the "Qt::AA_EnableHighDpiScaling must be set before QCoreApplication" warning
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("EyeShot AI")
    app.setApplicationVersion("1.2.0")
    app.setOrganizationName("Tigran0000")
    
    # Optional: Set application style for consistent cross-platform appearance
    app.setStyle('Fusion')
    
    # Optional: Set application icon if available
    # icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'icon.png')
    # if os.path.exists(icon_path):
    #     app.setWindowIcon(QIcon(icon_path))
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()