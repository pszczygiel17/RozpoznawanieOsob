import sys
import os
import PyQt5
import cv2 as cv
import numpy as np
from FNTR import normalize_pics
from faceallign import face_allign
import tensorflow as tf
#from mtcnn_tflite.MTCNN import MTCNN
from mtcnn import MTCNN

# dla drugiej kamery odkomentować linijkę 35, 59, 102

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


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.ui = CreateWindow()
        self.ui.setupUi(self)

        self.cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # take the first camera
        #self.cap2 = cv2.VideoCapture(2)  # take the first camera

        self.people_cam1 = []
        self.people_cam2 = []

        self.classes = ('Igor Zaton', 'Patryk Szczygiel', 'Grzegorz Szczypta',
                        'Michal Frankowicz', 'Konrad Arent', 'Marcin Krawczyk', 'Karolina Majka', 'Kinga Tokarska',
                        'Mateusz Auguscik')

        self.face_detection_model = MTCNN()

        self.interpreter = tf.lite.Interpreter(model_path="transferFaceNetv3.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.MARGIN = 10
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.IMG_HEIGHT = 160
        self.IMG_WIDTH = 160
        self.IMG_CHANNELS = 3

        self.timer = QTimer()  # create a timer
        self.timer.timeout.connect(self.viewCam1)  # set timer timeout callback function
        #self.timer.timeout.connect(self.viewCam2)  # set timer timeout callback function
        self.timer.timeout.connect(self.displayDetectedPeople)  # set timer timeout callback function
        self.timer.start(20)  # to odswiezanie oczywiscie do zmiany

    def viewCam1(self):
        ret, frame = self.cap1.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert image to RGB format
            self.people_cam1, frame = self.detectFace(frame)
            height, width, channel = frame.shape  # get image info
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.ui.cam1.setPixmap(QPixmap.fromImage(qImg))  # cam1

    def viewCam2(self):
        ret, frame = self.cap2.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert image to RGB format
            self.people_cam2, frame = self.detectFace(frame)
            height, width, channel = frame.shape  # get image info
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.ui.cam2.setPixmap(QPixmap.fromImage(qImg))  # cam2

    def displayDetectedPeople(self):
        items = self.ui.baseLayout.count()
        if items > 0:
            for i in range(items):
                self.ui.baseLayout.removeWidget(self.ui.baseLayout.itemAt(0).widget())
                self.ui.textField.setVisible(False)

        people = self.people_cam1 + self.people_cam2
        people = list(set(people))
        for p in people:
            self.ui.addTextField(self, p)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm exit', "Are you sure to exit?", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            self.timer.stop()
            self.cap1.release()
            #self.cap2.release()
            cv2.destroyAllWindows()
        else:
            event.ignore()

    def detectFace(self, frame):
        found_faces = []
        try:
            boxes = self.face_detection_model.detect_faces(frame)
            if len(boxes):
                for b in boxes:
                    face = frame[
                           b['box'][1] - self.MARGIN:b['box'][1] + b['box'][3] + self.MARGIN,
                           b['box'][0] - self.MARGIN:b['box'][0] + b['box'][2] + self.MARGIN,
                           :]
                    if len(face):
                        face = face_allign(face, b)

                        face = cv.resize(face, (self.IMG_HEIGHT, self.IMG_WIDTH))

                        face = normalize_pics([face]).reshape(-1, self.IMG_HEIGHT, self.IMG_WIDTH,
                                                              self.IMG_CHANNELS).astype(
                            np.float32)
                        self.interpreter.set_tensor(self.input_details[0]['index'], face)
                        self.interpreter.invoke()
                        prediction = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))

                        frame = cv.rectangle(frame, (b['box'][0] - self.MARGIN, b['box'][1] - self.MARGIN),
                                             (b['box'][0] + b['box'][2] + self.MARGIN,
                                              b['box'][1] + b['box'][3] + self.MARGIN),
                                             (0, 255, 0), thickness=2)
                        frame = cv.putText(frame, self.classes[np.argmax(prediction)] + ' ' + str(
                            np.round(prediction[np.argmax(prediction)] * 100, 2)) + '%',
                                           (b['box'][0] - self.MARGIN, b['box'][1] - self.MARGIN - 15),
                                           self.font, 0.5, (0, 255, 0),
                                           thickness=2)
                        found_faces.append(self.classes[np.argmax(prediction)])
            return found_faces, frame
        #except:
            #return found_faces, frame
        except RuntimeError as e:
            print(e)
            sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
