# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def test_environment():
    import sys
    print("Python version", sys.version)

    import numpy as np
    print("Numpy version", np.version.version)

    # import matplotlib.pyplot as plt
    # img = plt.imread('venv/Exercise Files/Ch01/01_04/komodo.jpg')
    # plt.axis("off")
    # plt.imshow(img)
    # plt.show()

    import cv2
    print("OpenCV version", cv2.__version__)
    image = cv2.imread('venv/Exercise Files/Ch01/01_04/komodo.jpg')
    cv2.imshow("OpenCV", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def basics_of_image_processing():
    import numpy as np
    import matplotlib.pyplot as plt
    '''PART 1 - Image Representation '''
    '''
    # Grey matrix
    img = np.array([[0, 225, 0],
                    [50, 200, 50],
                    [110, 127, 140]])
    plt.imshow(img, cmap='gray')

    # RGB matrix
    img = np.array([[[250, 0, 0], [0, 250, 0], [0, 0, 250]],
                    [[0, 255, 255], [255, 0, 255], [255, 255, 0]],
                    [[0, 0, 0], [255, 255, 255], [127, 127, 127]]])
    print(type(img[0, 0, 0]))
    # RGB Floating number matrix
    img = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[0.0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]])
    print(type(img[0, 0, 0]))
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    '''

    '''PART 2 - Color Encoding'''
    '''
    # Change pixel color of row 2 colum 0 Red and Green
    img[2, 0, 0] = 1.0  # Assign 100% to red
    img[2, 0, 1] = 0.8  # Assign 80% to green

    # Show a specific cells from "Row x to end" and from "column begin to y"
    x = 1
    y = 2
    plt.imshow(img[x:, :y])
    plt.show()

    plt.imshow(img)
    plt.show()
    '''

    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test_environment()
    basics_of_image_processing()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
