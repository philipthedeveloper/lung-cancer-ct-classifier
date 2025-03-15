from PySide6.QtWidgets import QPushButton, QLabel, QFileDialog, QWidget, QVBoxLayout, \
    QHBoxLayout, QGraphicsDropShadowEffect, QGridLayout
from PySide6.QtGui import QPixmap, QColor
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from model import load_model, classify_image, get_class_name
from image_processing import preprocess_image


def display_image(image_path, label):
    pixmap = QPixmap(image_path)
    scaled_pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
    label.setPixmap(scaled_pixmap)


class XrayClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()

        # Main widget and layout
        self.main_layout = QHBoxLayout(self)

        # Left container (Image loader and original image)
        self.left_container = QWidget(self)
        self.left_layout = QVBoxLayout(self.left_container)
        self.left_container.setStyleSheet("background-color: #131313; border-radius: 6px; padding: 10px;")

        # Original image label
        self.original_lbl_title = QLabel("Original Image", self)
        self.original_lbl_title.setStyleSheet("color: #ff6801; font-size: 16px; margin-bottom: 6px;")
        self.left_layout.addWidget(self.original_lbl_title, alignment=Qt.AlignmentFlag.AlignCenter)

        self.original_lbl = QLabel(self)
        self.original_lbl.setFixedSize(400, 400)
        self.original_lbl.setStyleSheet("border: 2px solid #1c1c1c; border-radius: 6px; background-color: #0c0c0e")
        self.left_layout.addWidget(self.original_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        # Load button
        self.load_btn = QPushButton('Load Image', self)
        self.load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6801; 
                border: none; 
                border-radius: 6px; 
                padding: 10px 30px;  /* Increased horizontal padding */
                color: white;
                font-size: 16px;
                box-shadow: 0px 4px 15px red;
            }
            QPushButton:hover {
                background-color: #131313;
                color: #ff6801;
            }
        """)

        # Create a shadow effect
        self.shadow_effect = QGraphicsDropShadowEffect()
        self.shadow_effect.setBlurRadius(15)
        self.shadow_effect.setColor(QColor(0, 0, 0, 160))
        self.shadow_effect.setOffset(5, 5)
        self.load_btn.setGraphicsEffect(self.shadow_effect)

        # Animation setup
        self.anim = QPropertyAnimation(self.load_btn, b"geometry")
        self.anim.setDuration(1500)
        self.anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # Interceptor for events
        self.load_btn.installEventFilter(self)

        self.load_btn.clicked.connect(self.load_image)
        self.left_layout.addWidget(self.load_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add left container to layout
        self.main_layout.addWidget(self.left_container)

        # Right container (Processed images and classification)
        self.right_container = QWidget(self)
        self.right_layout = QVBoxLayout(self.right_container)
        # self.right_layout = QGridLayout(self.right_container)
        self.right_container.setStyleSheet("background-color: #131313; border-radius: 6px; padding: 10px;")

        # Right content container
        self.right_content_container = QWidget(self)
        self.right_content_container_layout = QGridLayout(self.right_content_container)
        self.right_content_container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_content_container_layout.setHorizontalSpacing(20)  # Add horizontal space between columns
        self.right_content_container_layout.setVerticalSpacing(20)  # Add vertical space between rows
        self.right_content_container.setStyleSheet("max-width: 650px;")

        # Loaded image label
        self.loaded_lbl_title = QLabel("Loaded Image", self)
        self.loaded_lbl_title.setStyleSheet("color: #ff6801; font-size: 16px; margin-bottom: 6px;")
        self.right_content_container_layout.addWidget(self.loaded_lbl_title, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)

        self.loaded_lbl = QLabel(self)
        self.loaded_lbl.setFixedSize(250, 250)  # Match the size to the original image
        self.loaded_lbl.setStyleSheet("border: 2px solid #1c1c1c; border-radius: 6px; background-color: #0c0c0e")
        self.right_content_container_layout.addWidget(self.loaded_lbl, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter)

        # Preprocessed image label
        self.preprocessed_lbl_title = QLabel("Pre-processed Image", self)
        self.preprocessed_lbl_title.setStyleSheet("color: #ff6801; font-size: 16px; margin-bottom: 6px;")
        self.right_content_container_layout.addWidget(self.preprocessed_lbl_title, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.preprocessed_lbl = QLabel(self)
        self.preprocessed_lbl.setFixedSize(250, 250)  # Adjusted size
        self.preprocessed_lbl.setStyleSheet("border: 2px solid #1c1c1c; border-radius: 6px; background-color: #0c0c0e")
        self.right_content_container_layout.addWidget(self.preprocessed_lbl, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        # Classification result label
        self.classification_lbl = QLabel(self)
        self.classification_lbl.setFixedSize(400, 50)  # Adjusted size to match the image labels
        self.classification_lbl.setStyleSheet("border: 2px solid #1c1c1c; border-radius: 6px; background-color: #0c0c0e")
        self.classification_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_content_container_layout.addWidget(self.classification_lbl, 2, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add right content container to right content
        self.right_layout.addWidget(self.right_content_container)

        # Add right container to layout
        self.main_layout.addWidget(self.right_container)

        # Load the model
        self.model = load_model()

    def load_image(self):
        image_path = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.jpeg)')[0]
        if image_path:
            # Preprocess the image
            processed_img, preprocessed_path = preprocess_image(image_path)

            # Classify the image
            predicted_class = classify_image(self.model, processed_img)

            # Display the original image
            display_image(image_path, self.original_lbl)

            # Display the loaded image (same as the original image)
            display_image(image_path, self.loaded_lbl)

            # Display the preprocessed image
            display_image(preprocessed_path, self.preprocessed_lbl)

            # Display the classification result
            class_name = get_class_name(predicted_class)
            self.classification_lbl.setText(f'Classification: {class_name}')
