# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def test_environment():
    import sys
    print("Python version", sys.version)

    import numpy as np
    print("Numpy version", np.version.version)

    #import matplotlib.pyplot as plt
    #img = plt.imread('venv/Exercise Files/Ch01/01_04/komodo.jpg')
    #plt.axis("off")
    #plt.imshow(img)
    #plt.show()

    import cv2
    print("OpenCV version", cv2.__version__)
    image = cv2.imread('venv/Exercise Files/Ch01/01_04/komodo.jpg')
    cv2.imshow("OpenCV", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_environment()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
