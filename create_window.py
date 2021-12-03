from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt

class CreateWindow(object):

    def setupUi(self, Form):
        # Main Window
        Form.setObjectName("Form")
        Form.resize(1200, 900)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 127))
        palette.setBrush(QtGui.QPalette.All, QtGui.QPalette.Window, brush)
        Form.setPalette(palette)
        # Horizontal Layout
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        # Layout with 2 cameras
        self.camLayout = QtWidgets.QVBoxLayout()
        self.camLayout.setObjectName("camLayout")
        # First camera
        self.cam1 = QtWidgets.QLabel(Form)
        self.cam1.setObjectName("cam1")
        self.camLayout.addWidget(self.cam1)
        # Second camera
        self.cam2 = QtWidgets.QLabel(Form)
        self.cam2.setObjectName("cam2")
        self.camLayout.addWidget(self.cam2)

        self.horizontalLayout.addLayout(self.camLayout)

        # Layout with the base
        self.baseLayout = QtWidgets.QVBoxLayout()
        self.baseLayout.setObjectName("baseLayout")
        self.baseLayout.setAlignment(Qt.AlignTop)

        self.horizontalLayout.addLayout(self.baseLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "MyApp"))
        self.cam1.setText(_translate("Form", ""))
        self.cam2.setText(_translate("Form", ""))

    def addTextField(self, Form, text):
        self.textField = QtWidgets.QLineEdit(Form)
        self.textField.setText(text)
        self.textField.setReadOnly(True)
        self.baseLayout.addWidget(self.textField)

        
