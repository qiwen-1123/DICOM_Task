import os

import cv2

os.path.join(os.path.dirname(__file__), '1_Coordinates_For_Image')

recognized_text_dir = os.path.join(os.path.dirname(__file__), '1_Coordinates_For_Image')
recognized_image_dir = os.path.join(os.path.dirname(__file__), '1_Images_Without')

assert os.path.exists(recognized_text_dir) and os.path.exists(recognized_image_dir), "Directory does not exist"

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


def get_point_directories(point_name):
    '''
    Create paths for point with given name
    :param point_name:
    :return: save_text_dir, save_images_dir, save_images_with_dir
    '''
    save_text_dir = os.path.join(os.path.dirname(__file__), '../CNN/' + point_name + '/Coordinates')
    save_images_dir = os.path.join(os.path.dirname(__file__), '../CNN/' + point_name + '/Images_without')
    save_images_with_dir = os.path.join(os.path.dirname(__file__), '../CNN/' + point_name + '/Images_with')

    # Create dirs if they don't exist
    for dr in [save_text_dir, save_images_dir, save_images_with_dir]:
        os.makedirs(dr, exist_ok=True)
    return save_text_dir, save_images_dir, save_images_with_dir


def read_points_file(txt_path):
    '''
    Read all points coordinates from txt file
    :param txt_name: name of txt_file
    :return: list(list(X, Y))
    '''

    with open(txt_path) as f:
        # Read text file line after line
        num_line = 0
        # Where to start and stop searching for x-coordinates
        x_start_text = 'X:'
        x_end_text = ','
        # Where to start and stop searching for y-coordinates
        y_start_text = 'Y:'
        y_end_text = ';'
        all_points = []

        for line in f:
            num_line += 1
            # Points coordinates are stored after the second line
            if num_line > 2:
                # Find coordinates for current point
                x_coor = line[line.find(x_start_text) + len(x_start_text): line.find(x_end_text)]
                y_coor = line[line.find(y_start_text) + len(y_start_text): line.find(y_end_text)]
                all_points.append([int(x_coor), int(y_coor)])
    return all_points


def save_point(point_name, point, image, curr_file_name):
    save_text_dir, save_images_dir, save_images_with_dir = get_point_directories(point_name)

    cv2.imwrite(save_images_dir + '/' + curr_file_name + '.png',
                image)  # Save original image without point in grayscale
    image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image2 = cv2.circle(image2, (point[0], point[1]), 3, (0, 0, 255), 4)
    cv2.imwrite(save_images_with_dir + '/' + curr_file_name + '.png', image2)  # Save image with point
    f = open(save_text_dir + '/' + curr_file_name + '.txt', 'w')
    # Add the name of the image at the beginning
    f.write(curr_file_name + "\n")
    f.write("Number of points: " + str(1))
    # Write the coordinates for all points to the text file
    x_y_line = '\nX:' + str(point[0]) + ', ' + 'Y:' + str(point[1]) + ';'
    f.write(x_y_line)
    f.close()

def show_points(image: object, points: object) -> object:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for point in points:
        image = cv2.circle(image, (point[0], point[1]), 3, (0, 0, 255), 4)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 800)
    cv2.imshow('image', image)
    cv2.waitKey(0)

def sort_points(points, by_x=True, ascending=True):
    idx = 0 if by_x else 1
    reverse = False if ascending else True
    return sorted(points, key=lambda l: l[idx], reverse=reverse)