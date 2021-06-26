import argparse
import cv2
import os
import numpy as np
from tensorflow import keras
from image_processing import image_procesor,sudoku_finder,grid_cropper
from digit_recognition import image_to_digits
from sudoku_solver import solve
import sys
# adding the path of the file

parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",type=str,help="Path of input sudoku file")
parser.add_argument("-d","--debug",action="store_true",help="Shows output at each step")
args = parser.parse_args()

try:
    model = keras.models.load_model('big_epoch.h5')
except:
    sys.exit(" -- Provided path has no model file to read -- ")

if args.debug:
    print("Debug mode is on")
    # detect the current working directory and print it
    path = os.getcwd()
    os.makedirs(path+"/Debug/Numbers",exist_ok=True)

image_input=cv2.imread(args.path,0)
if image_input is None:
    sys.exit(" -- Provided path has no image file to read -- ")

image_copy=image_input.copy()

Processed_image_dialated,Processed_image=image_procesor(image_input)

if args.debug:
    cv2.imwrite(path+"/Debug/Processed_image_dialted.png",Processed_image_dialated)
    cv2.imwrite(path+"/Debug/Processed_image.png",Processed_image)


Sudoku_image=sudoku_finder(Processed_image_dialated,Processed_image)


if args.debug:
        cv2.imwrite(path+"/Debug/sudoku_cropped.png",Sudoku_image)

Squares=grid_cropper(Sudoku_image) #also cleans the image and adds None for empty

if args.debug:
    for i in range(9):
        for j in range(9):
            cv2.imwrite(path+"/Debug/Numbers/grid_"+str(i)+str(j)+".png",Squares[i*9+j])

grid_numbers=image_to_digits(Squares,model)


if args.debug:

    print(np.matrix(grid_numbers))

solved_sudoku=solve(grid_numbers)


if args.debug:

    print(np.matrix(solved_sudoku))
