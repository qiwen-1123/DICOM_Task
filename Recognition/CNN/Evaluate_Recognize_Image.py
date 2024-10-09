import numpy as np
from keras.models import load_model
import cv2
import os.path
import keras
from datetime import datetime

name_points = ['S_Point'] # TODO ONLY one point at a time
size_image = 520
recognized_image_dir = 'Test_Data/'
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

half_size = int(size_image/2)
mean_distance_result = 0
num_images = 0
for i in range(len(name_points)):

    for subdir, dirs, files in os.walk(recognized_image_dir + name_points[i] + '_images_without/'):
        for file in files:
            curr_file_name = file
            name_image = recognized_image_dir + name_points[i] + '_images_without/' + file
            if os.path.isfile(name_image):
                image = cv2.imread(name_image, 0)
            else:
                continue

            curr_image = image.copy()
            image_color = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2RGB)
            f = open(recognized_image_dir + '/' + name_points[i] + '_Coordinates/' + os.path.splitext(curr_file_name)[0] + '.txt', 'r')
            num_line = 0
            x_start_text = 'X:'
            x_end_text = ','
            y_start_text = 'Y:'
            y_end_text = ';'

            for line in f:
                num_line += 1
                if num_line > 2:
                    x_coor = int(line[line.find(x_start_text) + len(x_start_text): line.find(x_end_text)])
                    y_coor = int(line[line.find(y_start_text) + len(y_start_text): line.find(y_end_text)])

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
            x_pred = int(predicted[0][0]*size_image) + mean_x[i] - half_size
            y_pred = int(predicted[0][1]*size_image) + mean_y[i] - half_size
            image_color = cv2.circle(image_color, (x_pred, y_pred), 3, (0, 0, 255), 5)
            image_color = cv2.circle(image_color, (x_coor, y_coor), 3, (0, 255, 0), 5)
            keras.backend.clear_session()

            distance = np.math.sqrt(((x_pred - x_coor) ** 2) + ((y_pred - y_coor) ** 2))
            horizontaldistance = np.math.sqrt(((x_pred - x_coor) ** 2))
            verticaldistance = np.math.sqrt(((y_pred - y_coor) ** 2))
            print(round(distance, 2))
            print(curr_file_name)

            # Write the Mean Radial Error for all images
            f = open(recognized_image_dir + 'External_Validation/Mean_Error/' + name_points[0] + '.txt', 'a+')
            f.write(curr_file_name)
            f.write(' ')
            f.write(str(round(distance, 2)) + "\n")
            f.close()

            # # # Write down the error for all images in the sagittal plane
            f = open(recognized_image_dir + 'External_Validation/X_Error/' + name_points[0] + '.txt', 'a+')
            f.write(curr_file_name)
            f.write(' ')
            f.write(str(round(horizontaldistance, 2)) + "\n")
            f.close()

            # # # # Write down the error for all images in the vertical plane
            f = open(recognized_image_dir + 'External_Validation/Y_Error/' + name_points[0] + '.txt', 'a+')
            f.write(curr_file_name)
            f.write(' ')
            f.write(str(round(verticaldistance, 2)) + "\n")
            f.close()

            num_images += 1
            mean_distance_result += distance

            # show image, point from CNN in RED, Real in GREEN
            print({curr_file_name})
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 800, 800)
            cv2.imshow('image', image_color)
            cv2.waitKey(0)

mean_distance_result /= num_images
print(round(mean_distance_result, 2))


