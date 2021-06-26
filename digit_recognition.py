
import numpy as np
import cv2
from image_processing import image_procesor

def image_to_digits(grid,model,debug):
    grid_numbers=np.zeros([9,9])
    for i in range(9):
        for j in range(9):
            image=grid[i*9+j]
            if  min(map(min,image)) >= 127:
                grid_numbers[i][j]=0
                continue

            image = cv2.resize(image, (28, 28),interpolation = cv2.INTER_AREA)
            image,_=image_procesor(image)

            if debug:
                cv2.imwrite("Debug/NN/grid_"+str(i)+str(j)+".png",image)

            image=image.reshape(1,28,28,1)
            pred=np.array(model.predict(image))
            print(pred[0])
            index=np.argmax(pred[0])
            print(pred[0][index])

            if pred[0][index]>0.8:
                grid_numbers[i][j]=index+1
            else:
                grid_numbers[i][j]=0

    return grid_numbers
#
# parser = argparse.ArgumentParser()
# parser.add_argument("-t","--train",action="store_true",help="Shows output at each step")
# args = parser.parse_args()
# if args.train:
#
