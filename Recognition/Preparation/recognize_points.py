import cv2
import os
import concurrent.futures


def process_images(file_name):
    isGood = True
    print('Recognizing', file_name[1], ' image')
    # Find the name of the current image without the prefix 'a' or 'o'
    curr_file_name = file_name[0][1:]

    image_with_name = '../../FRS/with_points/a' + curr_file_name
    image_without_name = '../../FRS/without_points/o' + curr_file_name
    image_without = cv2.imread(image_without_name)
    image = cv2.imread(image_with_name)
    image2 = image.copy()
    w, h, _ = image.shape

    final_coor_points = []

    for i in range(int(w * 0.25), w - w_point):
        for j in range(int(h * 0.25), h - h_point):

            # find specific features of boarder of point
            # Find 2 horizontal pixels that are red
            if (image[i][j][0] <= color[0] and image[i][j][1] <= color[1] and image[i][j][2] >= color[2]) and \
                    (image[i + 1][j][0] <= color[0] and image[i + 1][j][1] <= color[1] and image[i + 1][j][2] >=
                     color[2]):

                # Find 2 horizontal pixels that are white below the red ones
                if (image[i][j + 1][0] > 230 and image[i][j + 1][1] > 230 and image[i][j + 1][2] > 230) and \
                        (image[i + 1][j + 1][0] > 230 and image[i + 1][j + 1][1] > 230 and image[i + 1][j + 1][
                            2] > 230):

                    # Find 1 pixel on the left bottom that is red and one on the right bottom that is red
                    if (image[i - 1][j + 1][0] <= color[0] and image[i - 1][j + 1][1] <= color[1] and
                        image[i - 1][j + 1][2] >= color[2]) and \
                            (image[i - 2][j + 1][0] <= color[0] and image[i - 2][j + 1][1] <= color[1] and
                             image[i - 2][j + 1][2] >= color[2]) and \
                            (image[i + 2][j + 1][0] <= color[0] and image[i + 2][j + 1][1] <= color[1] and
                             image[i + 2][j + 1][2] >= color[2]) and \
                            (image[i + 3][j + 1][0] <= color[0] and image[i + 3][j + 1][1] <= color[1] and
                             image[i + 3][j + 1][2] >= color[2]):
                        x_coor = int(j + h_point / 2)
                        y_coor = int(i)

                        final_coor_points.append([x_coor, y_coor])

    text_name = 'image' + str(file_name[1]) + '.txt'
    image_name = 'image' + str(file_name[1]) + '.png'

    if len(final_coor_points) != 38:
        isGood = False
        print('bad', len(final_coor_points))
        cv2.imwrite('00_Bad_Images/' + image_name, image_without)

    if isGood:

        f = open('1_Coordinates_For_Image/' + text_name, 'w')
        f.write(image_name + "\n")
        f.write("Number of points: " + str(len(final_coor_points)))
        for i in range(len(final_coor_points)):
            x_y_line = '\nX:' + str(final_coor_points[i][0]) + ', ' + 'Y:' + str(final_coor_points[i][1]) + ';'
            f.write(x_y_line)
        f.close()
        cv2.imwrite('1_Images_Without/' + image_name, image_without)


# These are the folders where the original images with and without points are
images_with_dir = '../../FRS/with_points'
images_without_dir = '../../FRS/without_points'

# Save original names with the corresponding number of image
file_names = []
num_img = 0
for subdir, dirs, files in os.walk(images_with_dir):
    for file in files:
        num_img +=1
        file_names.append([file, num_img])

num_img2 = 0
for subdir, dirs, files in os.walk(images_without_dir):
    for file in files:
        num_img2 += 1

assert num_img == num_img2, 'The number images with points does NOT equal number of images without points. ' \
                                       'Check what is the problem in folder: ' + images_with_dir

point = cv2.imread('Point.png')  # Point from an image so we know what size it is
color = [20, 20, 180]  # Color of the point border BGR
w_point, h_point, _ = point.shape  # Size of the point image

# Iterate through all images
for file_name in file_names:
    isGood = True
    print('Recognizing', file_name[1], '/', num_img, ' image')
    # Find the name of the current image without the prefix 'a' or 'o'
    curr_file_name = file_name[0][1:]
    # Open the image with points
    image_with_name = '../../FRS/with_points/a' + curr_file_name
    # Open the image without points
    image_without_name = '../../FRS/without_points/o' + curr_file_name
    image_without = cv2.imread(image_without_name)
    image = cv2.imread(image_with_name)
    image2 = image.copy()
    w, h, _ = image.shape

    final_coor_points = []

    pixel_start = 0.25
    # Check for each pixel that after 'pixel_start' from the upper left corner
    # (because in the upper left corner there are no points)
    for i in range(int(w * pixel_start), w - w_point):
        for j in range(int(h * pixel_start), h - h_point):

            # Find specific features of boarder of point
            # Find 2 horizontal pixels that are red
            if (image[i][j][0] <= color[0] and image[i][j][1] <= color[1] and image[i][j][2] >= color[2]) and \
                    (image[i + 1][j][0] <= color[0] and image[i + 1][j][1] <= color[1] and image[i + 1][j][2] >=
                     color[2]):
                # Find 2 horizontal pixels that are white below the red ones
                if (image[i][j + 1][0] > 230 and image[i][j + 1][1] > 230 and image[i][j + 1][2] > 230) and \
                        (image[i + 1][j + 1][0] > 230 and image[i + 1][j + 1][1] > 230 and image[i + 1][j + 1][
                            2] > 230):
                    # Find 1 pixel on the left bottom that is red and one on the right bottom that is red
                    if (image[i - 1][j + 1][0] <= color[0] and image[i - 1][j + 1][1] <= color[1] and
                        image[i - 1][j + 1][2] >= color[2]) and \
                            (image[i - 2][j + 1][0] <= color[0] and image[i - 2][j + 1][1] <= color[1] and
                             image[i - 2][j + 1][2] >= color[2]) and \
                            (image[i + 2][j + 1][0] <= color[0] and image[i + 2][j + 1][1] <= color[1] and
                             image[i + 2][j + 1][2] >= color[2]) and \
                            (image[i + 3][j + 1][0] <= color[0] and image[i + 3][j + 1][1] <= color[1] and
                             image[i + 3][j + 1][2] >= color[2]):

                        # If all these conditions are through, then a point was find, so save the coordinates
                        x_coor = int(j + h_point / 2)
                        y_coor = int(i)

                        final_coor_points.append([x_coor, y_coor])

    text_name = 'image' + str(file_name[1]) + '.txt'
    image_name = 'image' + str(file_name[1]) + '.png'

    # All of the images have to have 38 points, if this is not the case something went wrong
    # and the image is saved under the '00_Bad_Images'
    if len(final_coor_points) != 38:
        isGood = False
        print('bad', len(final_coor_points))
        # Save the image without without points with the corresponding name
        cv2.imwrite('00_Bad_Images/' + image_name, image_without)

    # If all of the points in the image were recognized save the image and the corresponding coordinates
    if isGood:
        # Create new text file to write the coordinates for the corresponding image
        f = open('1_Coordinates_For_Image/' + text_name, 'w')
        # Add the name of the image at the beginning
        f.write(image_name + "\n")
        f.write("Number of points: " + str(len(final_coor_points)))
        # Write the coordinates for all points to the text file
        for i in range(len(final_coor_points)):
            x_y_line = '\nX:' + str(final_coor_points[i][0]) + ', ' + 'Y:' + str(final_coor_points[i][1]) + ';'
            f.write(x_y_line)
        f.close()

        # Save the image without without points with the corresponding name
        cv2.imwrite('1_Images_Without/' + image_name, image_without)

print('\n Finaly finished! \n')
