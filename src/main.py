import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from login_screen import LoginScreen
from XrayClassifierGUI import XrayClassifierGUI
from register_screen import RegisterScreen


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Main App')
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: #0c0c0e; color: white;")

        # Create a stacked widget
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Initialize screens
        self.login_window = LoginScreen(self)
        self.register_window = RegisterScreen(self)
        self.xray_classifier = XrayClassifierGUI()

        # Add screens to the stacked widget
        self.stacked_widget.addWidget(self.login_window)
        self.stacked_widget.addWidget(self.register_window)

        # Start with the login window
        self.stacked_widget.setCurrentWidget(self.login_window)
        self.use_login_dimension()

    def show_xray_classifier(self):
        self.stacked_widget.addWidget(self.xray_classifier)
        self.stacked_widget.setCurrentWidget(self.xray_classifier)
        self.use_xray_gui_dimension()

    def use_login_dimension(self):
        self.setWindowTitle('Login')
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: #0c0c0e; color: white;")

    def use_register_dimension(self):
        self.setWindowTitle('Register')
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: #0c0c0e; color: white;")

    def use_xray_gui_dimension(self):
        self.setWindowTitle('X-ray Image Classifier')
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("background-color: #0c0c0e; color: white;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec())
