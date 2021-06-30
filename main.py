import argparse
import cv2
import os
import numpy as np
from tensorflow import keras
from image_processing import image_procesor,sudoku_finder,grid_cropper
from image_processing import write_grid
from digit_recognition import image_to_digits
from sudoku_solver import solve
from Webcam_capture import get_frame
import sys
import time
import shutil

# adding the path of the file

parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",type=str,help="Path of input sudoku file")
parser.add_argument("-d","--debug",action="store_true",help="Shows output at each step")
args = parser.parse_args()

def isdebug():
    global args
    return args.debug

try:
    model = keras.models.load_model('big_epoch.h5')
except:
    sys.exit(" -- Provided path has no model file to read -- ")

if args.debug:
    print("Debug mode is on")
    # detect the current working directory and print it
    path = os.getcwd()
    os.makedirs(path+"/Debug/Numbers",exist_ok=True)
    shutil.rmtree(path+"/Debug/NN",ignore_errors=True)
    os.makedirs(path+"/Debug/NN",exist_ok=True)


start = time.time()
if args.path:
    image_input=cv2.imread(args.path,0)
    if image_input is None:
        sys.exit(" -- Provided path has no image file to read -- ")
else:
    image_input=get_frame()

    if image_input is None:
        sys.exit(" -- Image Was not captured, Exiting... -- ")
    image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)


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

grid_numbers=image_to_digits(Squares,model,args.debug)


if args.debug:

    print(np.matrix(grid_numbers))

solved_sudoku=solve(grid_numbers)


if args.debug:
    print(np.matrix(solved_sudoku))

solved_sudoku=solved_sudoku-grid_numbers
end=time.time()

solved_sudoku_image=write_grid(solved_sudoku,Sudoku_image)

if args.debug:
    print("TIME TAKEN TO PROCESS: ",end-start)
    cv2.imwrite(path+"/Debug/solved_sudoku.png",solved_sudoku_image)
cv2.imshow("Solved Sudoku", solved_sudoku_image)
while cv2.getWindowProperty("Solved Sudoku", 0) >= 0:
    keyCode = cv2.waitKey(50)
    # ...
cv2.destroyAllWindows()
