from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QLabel, QHBoxLayout


class LoginScreen(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Form
        self.form_el = QWidget(self)
        self.form_el_layout = QVBoxLayout(self.form_el)
        self.form_el_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.form_el.setStyleSheet("max-width: 450px;")

        # Input fields layout container
        self.inputs_container = QWidget(self)
        self.inputs_container_layout = QVBoxLayout(self.inputs_container)

        self.username_label = QLabel("Username", self)
        self.username_label.setStyleSheet("color: #ffffff; font-size: 14px; margin-bottom: 6px;")

        self.username_input = QLineEdit(self)
        self.username_input.setPlaceholderText("Username")
        self.username_input.setStyleSheet("border: 1px solid #1c1c1c; border-radius: 4px; padding: 11px; "
                                          "background-color: #131313; margin-bottom: 10px")

        # Create a form group for username
        self.inputs_container_layout.addWidget(self.username_label)
        self.inputs_container_layout.addWidget(self.username_input)

        self.password_label = QLabel("Password", self)
        self.password_label.setStyleSheet("color: #ffffff; font-size: 14px; margin-bottom: 6px;")

        self.password_input = QLineEdit(self)
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setStyleSheet("border: 1px solid #1c1c1c; border-radius: 4px; padding: 11px; "
                                          "background-color: #131313")

        # Create a form group for username
        # self.password_group = self.create_form_group(self.password_label, self.password_input)
        self.inputs_container_layout.addWidget(self.password_label)
        self.inputs_container_layout.addWidget(self.password_input)

        self.login_btn = QPushButton('Login', self)
        self.login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6801; 
                border: none; 
                border-radius: 6px; 
                padding: 10px 30px;  
                color: white;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #131313;
                color: #ff6801;
            }
        """)

        self.login_btn.clicked.connect(self.main_app.show_xray_classifier)

        self.register_btn = QPushButton('Register', self)
        self.register_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.register_btn.setStyleSheet("""
            QPushButton {
                text-decoration: underline;
                color: #ff6801;
                font-size: 16px;
                margin-top: 0px;
                border: none;
                outline:none;
                background-color: transparent;
            }
            QPushButton:hover {
                color: white;
            }
        """)
        self.register_btn.clicked.connect(self.show_register_screen)

        # Create Buttons Container
        self.button_container = QWidget(self)
        self.button_container_layout = QVBoxLayout(self.button_container)
        self.button_container.setStyleSheet("margin-top: 10px;")

        # Create a hr layout for the register button
        self.btn_temp_hr = QWidget(self)
        self.btn_temp_hr_layout = QHBoxLayout(self.btn_temp_hr)
        self.btn_temp_hr_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.btn_temp_hr_layout.addWidget(self.register_btn)

        # Add button to the layout
        self.button_container_layout.addWidget(self.login_btn)
        self.button_container_layout.addWidget(self.btn_temp_hr)

        self.form_el_layout.addWidget(self.inputs_container)
        self.form_el_layout.addWidget(self.button_container)
        layout.addWidget(self.form_el)

    def show_register_screen(self):
        self.main_app.stacked_widget.setCurrentWidget(self.main_app.register_window)
        self.main_app.use_register_dimension()


