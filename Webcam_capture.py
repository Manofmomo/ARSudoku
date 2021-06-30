import cv2

def get_frame():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Sudoku Capture")

    captured=False

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            captured=True
            
            break

    cv2.destroyAllWindows()
    cam.release()

    if captured:
        return frame
    else:
        return None

