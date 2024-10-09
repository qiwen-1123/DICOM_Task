import numpy as np
from keras.models import load_model
import cv2
import os.path
import keras
from datetime import datetime

name_points = ['S_Point']
size_image = 520

# name of dir with images
recognized_image_dir = 'Test_Data'
mean_x = []
mean_y = []

for name_point in name_points:
    text_mean_x_y_dir = '../Preparation/' + name_point + '/Mean_X_Y.txt'
    f = open(text_mean_x_y_dir, 'r')
    num_line = 0
    x_start_text = 'Mean_X:'
    x_end_text = ','
    y_start_text = 'Mean_Y:'
    y_end_text = ';'
    all_points = []

    for line in f:
        num_line += 1
        x_coor = line[line.find(x_start_text) + len(x_start_text): line.find(x_end_text)]
        y_coor = line[line.find(y_start_text) + len(y_start_text): line.find(y_end_text)]
        mean_x.append(int(x_coor))
        mean_y.append(int(y_coor))

num_images_all = 0
for subdir, dirs, files in os.walk(recognized_image_dir):
    for file in files:
        num_images_all += 1

num_images = 21

half_size = int(size_image/2)

for subdir, dirs, files in os.walk(recognized_image_dir):
    for file in files:
        curr_file_name = file
        name_image = recognized_image_dir + '/' + file
        if os.path.isfile(name_image):
            image = cv2.imread(name_image, 0)
        else:
            continue

        curr_image = image.copy()
        image_color = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2RGB)

        for i in range(len(name_points)):
            model = load_model(name_points[i] + '/Trained_CNNs/one' + str(size_image) + '.h5')
            roi_x_start = mean_x[i] - half_size
            roi_x_end = mean_x[i] + half_size
            roi_y_start = mean_y[i] - half_size
            roi_y_end = mean_y[i] + half_size
            roi = curr_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            roi2 = roi.copy()
            roi = np.expand_dims(roi, 3)
            roi = np.expand_dims(roi, 0)
            predicted = model.predict(roi)
            x_pred = int(predicted[0][0] * size_image) + mean_x[i] - half_size
            y_pred = int(predicted[0][1] * size_image) + mean_y[i] - half_size
            image_color = cv2.circle(image_color, (x_pred, y_pred), 3, (0, 0, 255), 5)
            keras.backend.clear_session()

        # Show the Coordinates for the specific point for all images
        print('X:' + str(x_pred) + "\n")
        print('Y:' + str(y_pred) + "\n")


        model.summary()
        print('FRS-Auswertung für', (curr_file_name), 'fertig')
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 800)
        cv2.imshow('image', image_color)
        cv2.waitKey(0)
