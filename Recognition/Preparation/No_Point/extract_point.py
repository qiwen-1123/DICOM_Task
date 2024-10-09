import cv2
import os

point_name = 'No_Point'
pixels_diff_from_mean = 210

recognized_text_dir = '../1_Coordinates_For_Image'
recognized_image_dir = '../1_Images_Without'

save_text_dir = '../../CNN/' + point_name + '/Coordinates'
save_images_dir = '../../CNN/' + point_name + '/Images_without'
save_images_with_dir = '../../CNN/' + point_name + '/Images_with'

num_recognized = 0
for subdir, dirs, files in os.walk(recognized_text_dir):
    for file in files:
        num_recognized += 1

num_recognized2 = 0
for subdir, dirs, files in os.walk(recognized_image_dir):
    for file in files:
        num_recognized2 += 1

assert num_recognized == num_recognized2, 'Problem occurs. The number of text files does not ' \
                                          'equal the number of images. There must be one text file for each image!'
average_x = 0
average_y = 0

for subdir, dirs, files in os.walk(recognized_image_dir):
    for file in files:

        curr_file_name = file[:-4]
        print(curr_file_name)
        name_image = str(recognized_image_dir) + '/' + str(curr_file_name) + '.png'
        image = cv2.imread(name_image, 0)
        image2 = cv2.imread(name_image)

        w, h = image.shape
        f = open(str(recognized_text_dir) + '/' + str(curr_file_name) + '.txt', 'r')
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

        # Find maximum x-coordinates, which is the nose and max y-coordinates, which is the chin
        max_x = 0
        max_x_pos = 0
        max_y = 0
        max_y_pos = 0

        for i in range(len(all_points)):
            if all_points[i][0] > max_x:
                max_x = all_points[i][0]
                max_x_pos = i
            if all_points[i][1] > max_y:
                max_y = all_points[i][1]
                max_y_pos = i

        # Choose for which points to recognized
        # Keep in mind that it is counted from the beginning
        num_point_begin = 1
        num_point_end = 3

        # Specify the folder for the images with points
        image_folder = 'Images_With'

        img_num = max_x_pos

        # Extract the 'No_Point' from all points with different conditions, so that at the end only this point is saved
        # This is required as one point at a time is trained
        image2 = cv2.circle(image2, (all_points[img_num][0], all_points[img_num][1]), 2, (0, 0, 255), 3)


        cv2.imwrite(save_images_with_dir + '/' + curr_file_name + '.png', image2)  # Save image with point
        cv2.imwrite(save_images_dir + '/' + curr_file_name + '.png',
                    image)
        f = open(save_text_dir + '/' + curr_file_name + '.txt', 'w')
        # Add the name of the image at the beginning
        f.write(curr_file_name + "\n")
        f.write("Number of points: " + str(num_point_begin - num_point_end))
        # Write the coordinates for all points to the text file
        x_y_line = '\nX:' + str(all_points[img_num][0]) + ', ' + 'Y:' + str(all_points[img_num][1]) + ';'
        f.write(x_y_line)
        f.close()
        average_x += all_points[img_num][0]
        average_y += all_points[img_num][1]
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 800)
        cv2.imshow('image', image2)
        cv2.waitKey(0)

# average_x = average_x/num_recognized
# average_y = average_y/num_recognized
# print('Average x coordinates: ', average_x)
# print('Average y coordinates: ', average_y)
#
# f = open('Mean_X_Y.txt', 'w')
# f.write('Mean_X: ' + str(round(average_x)) + ', Mean_Y: ' + str(round(average_y)) + ';\n')
# f.close()