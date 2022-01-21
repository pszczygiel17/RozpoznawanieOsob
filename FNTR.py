from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import random
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import copy
import tensorflow as tf
from tensorflow import device
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.models import Model

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from joblib import dump


from sklearn.metrics import ConfusionMatrixDisplay

def get_pics_from_directory(path, img_height, img_width, limit=10000):
    pics = []
    cnt = 0
    for root, dirs, files in os.walk(path, topdown=True):
        for f in files:
            try:
                if cnt < limit:
                    img = cv.imread(os.path.join(root, f))
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img, (img_height, img_width))
                    pics.append(copy.copy(img))
                    print(f)
                    cnt += 1
                else:
                    break
            except:
                print('file acquisition failed')
    return pics


def normalize_pics(pics_list: list):
    norm = []
    for img in pics_list:
        img_x_mean, img_x_std = img[:, :, 0].mean(), img[:, :, 0].std()
        img_y_mean, img_y_std = img[:, :, 1].mean(), img[:, :, 1].std()
        img_z_mean, img_z_std = img[:, :, 2].mean(), img[:, :, 2].std()
        img_x = (img[:, :, 0] - img_x_mean) / img_x_std
        img_y = (img[:, :, 1] - img_y_mean) / img_y_std
        img_z = (img[:, :, 2] - img_z_mean) / img_z_std
        norm.append(cv.merge([img_x, img_y, img_z]))

    return np.array(norm).squeeze()


def plot_history(history, save=True):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.legend()
    if save:
        plt.savefig('accuracy5.png')
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.legend()
    if save:
        plt.savefig('loss5.png')
    plt.show()

def one_hot(data, no_classes):
    res = np.zeros(shape=(len(data), no_classes)).astype(np.float32)
    for i in range(len(data)):
        res[i][data[i]] = 1.
    return res

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(4.5 * 1024))])


            # loading the data

            IMG_HEIGHT = 160
            IMG_WIDTH = 160
            IMG_CHANNELS = 3
            LIMIT = 350
            CLASSES = 9

            pic_directory = 'database/'

            classes = ('Igor Zaton', 'Patryk Szczygiel', 'Grzegorz Szczypta',
                       'Michal Frankowicz', 'Konrad Arent', 'Marcin Krawczyk', 'Karolina Majka', 'Kinga Tokarska',
                       'Mateusz Auguscik')

            igor_pics = get_pics_from_directory(pic_directory + 'Igor Zaton', IMG_HEIGHT, IMG_WIDTH, LIMIT)
            patryk_pics = get_pics_from_directory(pic_directory + 'Patryk Szczygiel', IMG_HEIGHT, IMG_WIDTH, LIMIT)
            grzegorz_pics = get_pics_from_directory(pic_directory + 'Grzegorz Szczypta', IMG_HEIGHT, IMG_WIDTH, LIMIT)
            michal_pics = get_pics_from_directory(pic_directory + 'Michal Frankowicz', IMG_HEIGHT, IMG_WIDTH, LIMIT)
            konrad_pics = get_pics_from_directory(pic_directory + 'Konrad Arent', IMG_HEIGHT, IMG_WIDTH, LIMIT)
            marcin_pics = get_pics_from_directory(pic_directory + 'Marcin Krawczyk', IMG_HEIGHT, IMG_WIDTH, LIMIT)
            karolina_pics = get_pics_from_directory(pic_directory + 'Karolina Majka', IMG_HEIGHT, IMG_WIDTH, LIMIT)
            kinga_pics = get_pics_from_directory(pic_directory + 'Kinga Tokarska', IMG_HEIGHT, IMG_WIDTH, LIMIT)
            mateusz_pics = get_pics_from_directory(pic_directory + 'Mateusz Auguscik', IMG_HEIGHT, IMG_WIDTH, LIMIT)

            norm_igor = normalize_pics(igor_pics)
            norm_patryk = normalize_pics(patryk_pics)
            norm_grzegorz = normalize_pics(grzegorz_pics)
            norm_michal = normalize_pics(michal_pics)
            norm_konrad = normalize_pics(konrad_pics)
            norm_marcin = normalize_pics(marcin_pics)
            norm_karolina = normalize_pics(karolina_pics)
            norm_kinga = normalize_pics(kinga_pics)
            norm_mateusz = normalize_pics(mateusz_pics)

            igor_y = np.ones(shape=norm_igor.shape[0]) * 0
            patryk_y = np.ones(shape=norm_patryk.shape[0]) * 1
            grzegorz_y = np.ones(shape=norm_grzegorz.shape[0]) * 2
            michal_y = np.ones(shape=norm_michal.shape[0]) * 3
            konrad_y = np.ones(shape=norm_konrad.shape[0]) * 4
            marcin_y = np.ones(shape=norm_marcin.shape[0]) * 5
            karolina_y = np.ones(shape=norm_karolina.shape[0]) * 6
            kinga_y = np.ones(shape=norm_kinga.shape[0]) * 7
            mateusz_y = np.ones(shape=norm_mateusz.shape[0]) * 8

            people_X = np.vstack(
                (norm_igor, norm_patryk, norm_grzegorz, norm_michal, norm_konrad, norm_marcin, norm_karolina, norm_kinga, norm_mateusz))
            people_y = np.concatenate((igor_y, patryk_y, grzegorz_y, michal_y, konrad_y, marcin_y, karolina_y, kinga_y, mateusz_y)).astype(np.int8)


            people_y = one_hot(people_y, CLASSES)

            train_X, test_X, train_y, test_y = train_test_split(people_X, people_y, test_size=0.7, random_state=1)

            with device('/GPU:0'):

                model = load_model('facenet_keras.h5')
                # plot_model(model)
                # opt = SGD(learning_rate=0.01, momentum=0.9)
                # schedule = LearningRateScheduler(scheduler)
                # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                for l in model.layers:
                    l._trainable = False
                model.summary()

                x = Activation('relu')(model.layers[-1].output)
                x = Dense(64)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Dense(CLASSES)(x)
                x = BatchNormalization()(x)
                outputs = Activation('softmax')(x)
                model = Model(inputs=model.input, outputs=outputs)
                model.summary()

                # opt = SGD(learning_rate=0.001, momentum=0.9)
                early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True)

                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                # history = model.fit(train_X, train_y, epochs=500, validation_split=0.1, batch_size=32, callbacks=[early_stop])

                # model.save('transferFaceNetv3.model')
                # plot_history(history, save=False)

                print(test_X.shape)

                model = load_model('transferFaceNetv3.model')
                predictons = model.predict(test_X)
                predictons = np.array([np.argmax(p) for p in predictons])
                test_y = np.array([np.argmax(p) for p in test_y])
                cm = confusion_matrix(test_y, predictons)
                print(cm)

                disp = ConfusionMatrixDisplay(confusion_matrix(test_y, predictons), display_labels=classes)
                disp.plot(xticks_rotation='vertical')
                plt.subplots_adjust(left=0.25, bottom=0.30)
                plt.show()
                #
                #
                # igor_embedding = model.predict(norm_igor)
                # patryk_embedding = model.predict(norm_patryk)
                # grzegorz_embedding = model.predict(norm_grzegorz)
                # michal_embedding = model.predict(norm_michal)
                # konrad_embedding = model.predict(norm_konrad)
                # marcin_embedding = model.predict(norm_marcin)
                #
                # igor_y = np.ones(shape=igor_embedding.shape[0]) * 0
                # patryk_y = np.ones(shape=patryk_embedding.shape[0]) * 1
                # grzegorz_y = np.ones(shape=patryk_embedding.shape[0]) * 2
                # michal_y = np.ones(shape=patryk_embedding.shape[0]) * 3
                # konrad_y = np.ones(shape=patryk_embedding.shape[0]) * 4
                # marcin_y = np.ones(shape=marcin_embedding.shape[0]) * 5
                #
                # people_X = np.vstack(
                #     (igor_embedding, patryk_embedding, grzegorz_embedding, michal_embedding, konrad_embedding, marcin_embedding))
                # people_y = np.concatenate((igor_y, patryk_y, grzegorz_y, michal_y, konrad_y, marcin_y))
                #
                # train_X, test_X, train_y, test_y = train_test_split(people_X, people_y, test_size=0.2, random_state=42)
                #
                # classifier = SVC()
                # classifier.fit(train_X, train_y)
                # predictions = classifier.predict(test_X)
                # print(predictions)
                # print(confusion_matrix(test_y, predictions))
                #
                # dump(classifier, 'svcClf.joblib')
                # print('Classifier saved')

        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
