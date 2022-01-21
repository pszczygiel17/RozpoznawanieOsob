import matplotlib.pyplot as plt
import cv2 as cv
import os
from mtcnn_tflite.MTCNN import MTCNN
import numpy as np
from imutils import rotate_bound, rotate




def get_a_and_b(x1, y1, x2, y2):
    return (y1 - y2) / (x1 - x2), y1 - x1 * (y1 - y2) / (x1 - x2)


def face_allign(face_img, box):
    eyes_pts = (box['keypoints']['left_eye'], box['keypoints']['right_eye'])

    a, _ = get_a_and_b(eyes_pts[0][0], eyes_pts[0][1], eyes_pts[1][0], eyes_pts[1][1])
    alpha = np.degrees(np.arctan(a))

    img = rotate_bound(face_img, angle=-alpha)

    return img

if __name__ == '__main__':
    face_detection_model = MTCNN()
    MARGIN = 30

    image = cv.imread('more_people.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    boxes = face_detection_model.detect_faces(image)
    print('juz')
    for b in boxes:
        img = image[b['box'][1] - MARGIN:b['box'][1] + b['box'][3] + MARGIN,
              b['box'][0] - MARGIN:b['box'][0] + b['box'][2] + MARGIN, :]
        plt.imshow(img)
        plt.show()
        img = face_allign(img, b)

        # box_x1 = box[0]['box'][0] - MARGIN
        # box_y1 = box[0]['box'][1] - MARGIN
        # box_x2 = box_x1+box[0]['box'][2] + MARGIN
        # box_y2 = box_y1+box[0]['box'][3] + MARGIN

        # B = [[box_x1, box_y1],
        #      [box_x2, box_y2]]
        #
        # R = [[np.cos(alpha), -np.sin(alpha)],
        #      [np.sin(alpha), np.cos(alpha)]]
        #
        # new_box = np.dot(R, B).astype(int)

        # image = cv.rectangle(image, new_box[0], new_box[1], thickness=3, color=(0, 255, 0))

        # #
        # face = image[
        #        box[0]['box'][1] - MARGIN:box[0]['box'][1] + box[0]['box'][3] + MARGIN,
        #        box[0]['box'][0] - MARGIN:box[0]['box'][0] + box[0]['box'][2] + MARGIN,
        #        :]
        # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        plt.imshow(img)
        plt.show()
