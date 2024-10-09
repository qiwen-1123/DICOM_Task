import sys
import os
import numpy as np
import cv2
import Additional_Functions as func
import keras
import random

# Important parameters for different points
name_point = 'S_Point'
size_image = 520
take_last_images_testing = True
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
    mean_x = int(x_coor)
    mean_y = int(y_coor)

recognized_text_dir = name_point + '/Coordinates'
recognized_image_dir = name_point + '/Images_without'
half_size = int(size_image/2)
percent_test = 0.1
cnn_types = [1]  # There are different types of CNNs manually written
filters = [30]
kernel_sizes = [4]
epochs = 400

roi_x_start = mean_x - half_size
roi_x_end = mean_x + half_size
roi_y_start = mean_y - half_size
roi_y_end = mean_y + half_size

num_images = 0
for subdir, dirs, files in os.walk(recognized_text_dir):
    for file in files:
        num_images += 1

num_texts = 0
for subdir, dirs, files in os.walk(recognized_image_dir):
    for file in files:
        num_texts += 1

assert num_images == num_texts, 'Problem occurs. The number of text files does not ' \
                                          'equal the number of images. There must be one text file for each image!'

# The arrays with the data needed for training
input_images = np.zeros((num_images, size_image, size_image), dtype=np.uint8)
output = np.zeros((num_images, 2))

print('Images are being preprocessed...')

num_img = 0
for subdir, dirs, files in os.walk(recognized_image_dir):
    for file in files:

        curr_file_name = file[:-4]
        name_image = recognized_image_dir+'/'+curr_file_name+'.png'
        image = cv2.imread(name_image, 0)
        image2 = cv2.imread(name_image)
        w, h = image.shape
        f = open(recognized_text_dir+'/'+curr_file_name+'.txt', 'r')
        print(curr_file_name)
        num_line = 0
        x_start_text = 'X:'
        x_end_text = ','
        y_start_text = 'Y:'
        y_end_text = ';'
        all_points = []

        for line in f:
            num_line += 1
            if num_line > 2:
                x_coor = line[line.find(x_start_text) + len(x_start_text): line.find(x_end_text)]
                y_coor = line[line.find(y_start_text) + len(y_start_text): line.find(y_end_text)]
                all_points.append([int(x_coor), int(y_coor)])

        roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        local_x = all_points[0][0] - roi_x_start
        local_y = all_points[0][1] - roi_y_start
        input_images[num_img] = roi
        output[num_img] = [local_x/size_image, local_y/size_image]
        num_img += 1

final_best_err = 1000
final_best_filter = 0
final_best_kernel_size = 0
final_best_cnn = 0
final_best_epoch = 0
input_images = np.expand_dims(input_images, 3)

for cnn_type in cnn_types:
    for filter in filters:
        for kernel_size in kernel_sizes:

            best_epoch = 0

            model, coeff = func.neural_network(cnn_type, output, size_image, filter, kernel_size, num_images, percent_test)

            if take_last_images_testing:
                train_input = input_images[:len(input_images) - coeff]
                test_input = input_images[len(input_images) - coeff:]
                train_output = output[:len(input_images) - coeff]
                test_output = output[len(input_images) - coeff:]
            else:
                train_input = input_images[coeff:]
                test_input = input_images[:coeff]
                train_output = output[coeff:]
                test_output = output[:coeff]

            min_err = 1000

            for epoch in range(epochs):
                err_end = 0
                print('\nNumber Epoch: ', epoch + 1)
                model.fit(train_input, train_output, epochs=1, validation_data=(test_input, test_output), shuffle=True, verbose=2, batch_size=32)
                pred_output = model.predict(test_input)

                for i in range(0, len(pred_output)):
                    err = func.distance(pred_output[i][0] * size_image, pred_output[i][1] * size_image,
                                        test_output[i][0] * size_image, test_output[i][1] * size_image)
                    err_end += err

                err_end = err_end/len(pred_output)
                print('Err is: ', round(err_end, 1), ' pixels')
                print('Best err until now: ', round(min_err, 1), ' pixels')

                if err_end < min_err:
                    min_err = err_end
                    best_epoch = epoch + 1
                    model.save(name_point + '/Trained_CNNs/one' + str(size_image) + '.h5')

            if min_err < final_best_err:
                final_best_err = min_err
                final_best_filter = filter
                final_best_kernel_size = kernel_size
                final_best_cnn = cnn_type
                final_best_epoch = best_epoch

                print('!! --> Best err now: ' + str(round(final_best_err, 1)) + ' with filters: ' + str(final_best_filter)
                      + ' kernel size: ' + str(final_best_kernel_size) + ' cnn type: ' + str(final_best_cnn)
                      + ' at epoch: ' + str(final_best_epoch))

            keras.backend.clear_session()

print('Best error is: ', round(final_best_err, 1), ' pixels')
print('For filters: ', final_best_filter)
print('For kernel size: ', final_best_kernel_size)
print('For cnn type: ', final_best_cnn)
print('At epoch: ', final_best_epoch)

f = open(name_point + '/Trained_CNNs/info_trained.txt', 'w')
f.write(name_point + '\n')
f.write('Best error is: ' + str(round(final_best_err, 1)) + ' pixels' + '\n')
f.write('Best filters: ' + str(final_best_filter) + '\n')
f.write('Best kernel size: ' + str(final_best_kernel_size) + '\n')
f.write('Best cnn type: ' + str(final_best_cnn) + '\n')
f.write('At epoch: ' + str(final_best_epoch) + '\n')
f.close()

model.summary()