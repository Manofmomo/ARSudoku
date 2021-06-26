
import numpy as np
import cv2

def image_to_digits(grid,model):
    grid_numbers=np.zeros([9,9])
    for i in range(9):
        for j in range(9):
            image=grid[i*9+j]
            if  max(map(max,image)) == 0:
                grid_numbers[i][j]=0
                continue

            image = cv2.resize(image, (28, 28),interpolation = cv2.INTER_NEAREST)
            image=image.reshape(1,28,28,1)
            grid_numbers[i][j]=np.argmax(model.predict(image))+1

    return grid_numbers
#
# parser = argparse.ArgumentParser()
# parser.add_argument("-t","--train",action="store_true",help="Shows output at each step")
# args = parser.parse_args()
# if args.train:
#
