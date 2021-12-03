import sys
import os
import PyQt5
import time

dirname = os.path.dirname(PyQt5.__file__)
plugin_path = os.path.join(dirname + "/Qt", 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import cv2
from create_window import *
from keras.models import load_model
from mtcnn import MTCNN
import numpy as np

face_detection_model = MTCNN()
face_recognition_model = load_model('faceModel.model')
#print(face_recognition_model.summary())
classes = ('Igor Zaton', 'Patryk Szczygiel')
font = cv2.FONT_HERSHEY_SIMPLEX
MARGIN = 10
IMG_HEIGHT = 128
IMG_WIDTH = 96
IMG_CHANNELS = 3

class MainWindow(QWidget):
   
    def __init__(self):
        super().__init__()
        self.ui = CreateWindow()
        self.ui.setupUi(self)
        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # take the first camera

        self.timer = QTimer() # create a timer
        self.timer.timeout.connect(self.viewCam) # set timer timeout callback function
        self.timer.start(2000) # start with 2000 msc


    def viewCam(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert image to RGB format

            box = face_detection_model.detect_faces(frame)
            if len(box):
                face = frame[
                    box[0]['box'][1] - MARGIN:box[0]['box'][1] + box[0]['box'][3] + MARGIN,
                    box[0]['box'][0] - MARGIN:box[0]['box'][0] + box[0]['box'][2] + MARGIN,
                    :]
                frame = cv2.rectangle(frame, (box[0]['box'][0] - MARGIN, box[0]['box'][1] - MARGIN),
                                    (box[0]['box'][0] + box[0]['box'][2] + MARGIN,
                                    box[0]['box'][1] + box[0]['box'][3] + MARGIN),
                                    (0, 255, 0), thickness=5)
                face = cv2.resize(face, (IMG_HEIGHT, IMG_WIDTH)).reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
                prediction = np.squeeze(face_recognition_model.predict(face))
                frame = cv2.putText(frame, classes[np.argmax(prediction)],
                                (box[0]['box'][0] - MARGIN, box[0]['box'][1] - MARGIN - 15), font, 1, (0, 255, 0),
                                thickness=5)
                print(prediction)
                self.ui.addTextField(self, classes[np.argmax(prediction)])

            height, width, channel = frame.shape # get image info
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.ui.cam1.setPixmap(QPixmap.fromImage(qImg)) # cam1
            self.ui.cam2.setPixmap(QPixmap.fromImage(qImg)) # cam2

                
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm exit',"Are you sure to exit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            self.timer.stop()
            self.cap.release()
            cv2.destroyAllWindows()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())