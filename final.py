# importing required packages
import cv2
import argparse
import cv2
import numpy as np
import operator
from tensorflow import keras
import sys

model = keras.models.load_model('final_model.h5')

# adding the path of the file
parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",type=str,help="Path of input sudoku file")
parser.add_argument("-d","--debug",action="store_true",help="Shows output at each step")
args = parser.parse_args()

if args.debug:
    print("Debug mode is on")
#inputting the file
image_input=cv2.imread(args.path,0)
if image_input is None:
    sys.exit(" -- Provided path has no image file to read -- ")

image_copy=image_input
processing_image=cv2.GaussianBlur(image_copy,(9,9),0)
processing_image=cv2.adaptiveThreshold(processing_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY,11,2)

processing_image=cv2.bitwise_not(processing_image,processing_image)
kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
processing_image_dilated = cv2.dilate(processing_image, kernel)
if args.debug:
    cv2.imwrite("Processed_image.png",processing_image_dilated)

#finding the sudoku
contours,_ = cv2.findContours(processing_image_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    if len(approx) == 4:
        sudoku = c
        break

bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                      sudoku]), key=operator.itemgetter(1))
top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                  sudoku]), key=operator.itemgetter(1))
bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                     sudoku]), key=operator.itemgetter(1))
top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                   sudoku]), key=operator.itemgetter(1))
if args.debug:
    cv2.drawContours(image_copy, sudoku, -1, (0, 255, 0), 3)
    cv2.imwrite("Sudoku_detection.png",image_copy)

ordered_corners=[sudoku[top_left][0], sudoku[top_right][0], sudoku[bottom_right][0], sudoku[bottom_left][0]]
top_left,top_right,bottom_right,bottom_left=ordered_corners

# calculating width and height
width_A = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
width_B = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))

height_A = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
height_B = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

side = max(int(height_A), int(height_B),int(width_A), int(width_B))

# cropping the image
dimensions = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                       [0, side - 1]], dtype="float32")

ordered_corners = np.array(ordered_corners, dtype="float32")

grid_size = cv2.getPerspectiveTransform(ordered_corners, dimensions)
grid_image = cv2.warpPerspective(processing_image, grid_size, (side, side))



if args.debug:
        cv2.imwrite("sudoku_cropped.png",grid_image)

# splitting into cells

def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def remove_boundaries(img):

    # Get image shape
    h, w = img.shape

    # Draw a rectangle on the border to combine the wall to one contour
    cv2.rectangle(img,(0,0),(w,h),1,2)

    # Apply binary threshold
    _, threshold = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

    # Search for contours and sort them by size

    contours, _ = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours)<2:
        return np.zeros([h,w])
    # Draw it out with white color from biggest to second biggest contour
    cv2.drawContours(img, ((contours[0]),(contours[1])), -1, 0, -1)

    # Apply binary threshold again to the new image to remove little noises
    _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)


    return img

def image_to_num(image):
    if  max(map(max, crop_img(image,0.6))) == 0:
        return 0
    image = cv2.resize(image, (50, 50),interpolation = cv2.INTER_AREA)
    normalized_img=np.divide(image,255)
    normalized_img=normalized_img.reshape(1,50,50,1)
    return np.argmax(model.predict(normalized_img))

#initializing arrays to store the values
grid_numbers=np.zeros([9,9])
squares=[]


side_dim = np.shape(grid_image)[0]
side_dim = side_dim // 9

for j in range(9):
    for i in range(9):
        p1 = (i * side_dim, j * side_dim)  #Top left corner of a box
        p2 = ((i + 1) * side_dim, (j + 1) * side_dim)  #Bottom right corner
        rect=[p1, p2]
        cut_square=grid_image[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]
        cut_square=remove_boundaries(cut_square)
        squares.append(cut_square)
for i in range(9):
    for j in range(9):
        grid_numbers[i][j]=image_to_num(squares[i*9+j])



if args.debug:

    print(np.matrix(grid_numbers))
