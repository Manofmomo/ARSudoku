import cv2
import numpy as np
import operator
import sys



def image_procesor(image):

    processing_image_dilated=cv2.GaussianBlur(image,(9,9),0)
    processing_image_dilated=cv2.adaptiveThreshold(processing_image_dilated,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,11,2)#

    processing_image_dilated=cv2.bitwise_not(processing_image_dilated,processing_image_dilated)

    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    processing_image_dilated = cv2.dilate(processing_image_dilated, kernel)

    return processing_image_dilated,image

def sudoku_finder(processing_image_dilated,processing_image):
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
    grid_image = cv2.warpPerspective(processing_image, grid_size, (side, side),flags=cv2.INTER_AREA)

    return grid_image

def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped


def remove_boundaries(img,floodfill_count=2):
    Handle=True
    # Get image shape
    h, w = img.shape
    image_shrunk = cv2.resize(img, (50, 50),interpolation = cv2.INTER_AREA)
    image_shrunk,image_copy=image_procesor(image_shrunk)
    image_copy_ig=image_shrunk.copy()
    rows=image_shrunk.shape[0]
    if floodfill_count>=1:
        for i in range(rows):
            #Floodfilling the outermost layer
            cv2.floodFill(image_shrunk, None, (0, i), 0)
            cv2.floodFill(image_shrunk, None, (i, 0), 0)
            cv2.floodFill(image_shrunk, None, (rows-1, i), 0)
            cv2.floodFill(image_shrunk, None, (i, rows-1), 0)
                # Floodfilling the second outermost layer
            if floodfill_count==2:    
                cv2.floodFill(image_shrunk, None, (1, i), 0)
                cv2.floodFill(image_shrunk, None, (i, 1), 0)
                cv2.floodFill(image_shrunk, None, (rows - 2, i), 0)
                cv2.floodFill(image_shrunk, None, (i, rows - 2), 0)
            #Floodfilling the second outermost layer
        
        if np.sum(crop_img(image_copy_ig,0.3))>(255*0.3*0.3*rows*rows*0.09) and np.sum(crop_img(image_shrunk,0.3))<(255*0.3*0.3*rows*rows*0.04):
            return remove_boundaries(img,floodfill_count-1)

    contours,_ = cv2.findContours(image_shrunk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours)==0:
        return np.ones([h,w])*255
    x,y,w,h = cv2.boundingRect(contours[0])
    if w*h<0.05*50*50:
        return np.ones([h,w])*255
    
    image_copy=image_copy[y-1:y+h+1,x-1:x+w+1]
   
    return image_copy


# def remove_boundaries(img):
#     Handle=True
#     # Get image shape
#     h, w = img.shape
#     image_copy=img.copy()
#     image_shrunk = cv2.resize(image_copy, (28, 28),interpolation = cv2.INTER_AREA)
#     if np.sum(crop_img(image_shrunk,0.5))<(255*0.5*0.5*28*28*0.20):
#         return np.zeros([h,w])

#     cv2.floodFill
#     # Draw a rectangle on the border to combine the wall to one contour
#     cv2.rectangle(img,(0,0),(w,h),1,2)

#     # Apply binary threshold
#     _, threshold = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

#     # Search for contours and sort them by size

#     contours, _ = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     if len(contours)<2:
#         return np.zeros([h,w])
#     # Draw it out with white color from biggest to second biggest contour
#     cv2.drawContours(img, ((contours[0]),(contours[1])), -1, 0, -1)

#     # Apply binary threshold again to the new image to remove little noises
#     _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
#     if np.sum(crop_img(img,0.5))<(255*0.5*0.5*h*w*0.6):
#         return crop_img(image_copy,0.9)
#     return crop_img(img,0.9)



def grid_cropper(Sudoku_image): #function that is actually called
    squares=[]
    side_dim = np.shape(Sudoku_image)[0]
    side_dim = side_dim // 9
    side_dim = side_dim
    for j in range(9):
        for i in range(9):
            p1 = (i * side_dim, j * side_dim)  #Top left corner of a box
            p2 = ((i + 1) * side_dim, (j + 1) * side_dim)  #Bottom right corner
            rect=[p1, p2]
            cut_square=Sudoku_image[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]
            cut_square=remove_boundaries(cut_square)
            squares.append(cut_square)

    return squares
