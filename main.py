# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import rcParams
import time
from EC_CV import *


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
    """PART 1 - Image Representation """
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
    '''

    '''PART 2 - Color Encoding'''
    '''
    # RGB Floating number matrix
    img = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[0.0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]])
    print(type(img[0, 0, 0]))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

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

    # Show a specific cells x and y
    plt.imshow(img[x:(x+1), y:(y+1)])
    plt.show()
    '''

    '''PART 3 - Image File Management'''

    jpeg = plt.imread("venv/Exercise Files/Ch02/02_03/stickanimals.jpg")
    plt.imshow(jpeg)
    # plt.show()

    png = plt.imread("venv/Exercise Files/Ch02/02_03/stickanimalsRGBA.png")
    plt.imshow(png)
    # plt.show()

    print(np.shape(jpeg))  # jpg has 3 color channels
    print(np.shape(png))   # png has 4 color channels

    # See the data type
    print("JPEG image type: ", type(jpeg[0, 0, 0]))
    print(jpeg[0, 0])  # see one pixel
    print("PNG  image type: ", type(png[0, 0, 0]))
    print(png[0, 0])

    img = adapt_PNG(plt.imread("venv/Exercise Files/Ch02/02_03/stickanimalsRGBA.png"))
    print('PNG image data type after convert: ', type(img[0, 0, 0]))
    print(img[0, 0])
    plt.imshow(img)
    plt.show()

    # Save a file
    horsie = img[250:600, 200:400, :]
    plt.imsave("venv/Exercise Files/Ch02/02_03/horsie.jpg", horsie)
    plt.imshow(plt.imread("venv/Exercise Files/Ch02/02_03/horsie.jpg"))
    plt.show()

    '''PART 4 - Rotations and Flips'''
    '''
    img = plt.imread("venv/Exercise Files/Ch02/02_05/street.jpg")
    plt.axis("off")
    plt.imshow(img)
    plt.show()

    # Rotate 90 degrees counterclockwise
    img = np.rot90(img, 1)
    plt.imshow(img)
    plt.show()

    # Rotate 90 degrees 3 times
    img = np.rot90(img, 3)
    plt.imshow(img)
    plt.show()

    # Horizontal Flip
    img = np.fliplr(img)
    plt.imshow(img)
    plt.show()

    # Vertical Flip
    img = np.flipud(img)
    plt.imshow(img)
    plt.show()
    '''

    return 0


def from_color_to_black_and_white():
    # PART 01 - Average Gray Scale
    '''Small picture
    plt.rcParams['figure.figsize'] = (8, 4)
    playspace = plt.imread('venv/Exercise Files/Ch03/03_01/playspace.png')
    toys = adapt_PNG(playspace)
    plt.subplot(2, 2, 1), plt.imshow(toys)
    plt.title("RGB Color")

    start_time = time.time()
    toys_gs = RGB_to_grayscale(toys)
    end_time = round(time.time() - start_time, 2)
    plt.subplot(2, 2, 2), plt.imshow(toys_gs, cmap='gray'), \
    plt.title(f'Average Grayscale Simple ({end_time} sec)')

    start_time=time.time()
    toys_gs = np.dot(toys[..., :3], [1/3, 1/3, 1/3])  # product 2 matrix with numpy.dot method
    end_time = round(time.time() - start_time, 2)
    plt.subplot(2, 2, 4), plt.imshow(toys_gs, cmap='gray'), \
    plt.title(f'Average Grayscale with numpy.dot ({end_time} sec)')
    plt.show()

    fruit = plt.imread('venv/Exercise Files/Ch03/03_01/fruit.jpg')
    plt.imshow(fruit)
    plt.show()
    np.shape(fruit)

    start_time = time.time()
    fruit_gs = np.dot(fruit[..., :3], [1/3, 1/3, 1/3])
    end_time = round(time.time()-start_time, 2)
    plt.imshow(fruit_gs, cmap='gray')
    plt.title(f'total convert time {end_time} ')
    plt.show()
    '''

    # PART 02 - Weighted Greyscale
    '''
    toys = adapt_PNG(plt.imread('venv/Exercise Files/Ch03/03_02/playspace.png'))
    plt.axis('off')
    rcParams['figure.figsize'] = (20, 8)
    # Calculate regular average and weighted average
    toys_avg = np.dot(toys[..., :3], [1/3, 1/3, 1/3])
    toys_wgt = np.dot(toys[..., :3], [0.299, 0.587, 0.114])

    # Display images
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(toys_avg, cmap='gray')
    ax[1].imshow(toys_wgt, cmap='gray')
    plt.show()
    '''

    # PART 03 - Converting Grayscale to Black and White
    '''
    rcParams['figure.figsize'] = [10, 8]
    toys = adapt_PNG(plt.imread('venv/Exercise Files/Ch03/03_03/playspace.png'))
    toys_wgt = np.dot(toys[..., :3], [0.299, 0.587, 0.114])
    toys_BW = grayscale_to_BW(toys_wgt, 127)
    plt.subplot(2, 2, 1), plt.imshow(toys_BW, cmap='gray')

    # ravel() flatten an array; bins = 256
    plt.subplot(2, 2, 2), plt.hist(toys_wgt.ravel(), 256, [0, 255])
    plt.title('Histogram')
    plt.xticks(np.arange(0, 255, 10)) 
    '''

    # PART04 - Adaptive Thresholding
    # img = plt.imread('venv/Exercise Files/Ch03/03_04/sudoku.jpg')
    # img = np.dot(img, [0.299, 0.587, 0.114])  # nhân ma trận với tỷ lệ RGB tiêu chuẩn ra ảnh xám
    # print(np.shape(img))  # từ ma trận 3 chiều còn 2 chiều

    threshold = 127
    print(threshold)
    img0 = cv2.imread('venv/Exercise Files/Ch03/03_04/sudoku.jpg', 0)  # flag {1 (load RGB), 0 (load in grayscale), -1 (load unchange)}
    img = cv2.medianBlur(img0, 17)  # Kernel là ma trận nxn với n là số lẻ để vị tri điểm làm mờ là trọng tâm của ma trận
    # plt.subplot(2, 1, 1), plt.imshow(img0)
    # plt.subplot(2, 1, 2), plt.imshow(img)
    ret, th1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)  #(source, threshold, maxvalue, threshold type
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 2)
    # blockSize (33) − A variable of the integer type representing size of the pixelneighborhood used to calculate the threshold value.
    # C (2) − A variable of double type representing the constant used in the both methods (subtracted from the mean or weighted mean).
    titles = ['Original Image', 'Global Thresholding (v = ' + str(threshold) + ')',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img0, th1, th2, th3]
    rcParams['figure.figsize'] = [10, 8]
    for i in range(4):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

    return 0


def filters():
    pic_name = ['field.jpg', 'workers.jpg', 'face.jpg', 'penguins.jpg']
    img = plt.imread("venv/Exercise Files/Ch04/04_03/" + pic_name[3])
    '''PART 01 - BLUR FILTERS
    # Average blur
    k_size = 15
    kernel = np.ones((k_size, k_size), np.float32)/(k_size*k_size)
    average = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)  # ddepth = -1 giữ nguyên data type của src

    # Median blur
    median = cv2.medianBlur(img, k_size)  # with kernel size = k_size x k_size
   '''
    # Gaussian blur
    k_size = 3
    gaussian = cv2.GaussianBlur(img, (k_size, k_size), cv2.BORDER_DEFAULT)
    img = gaussian
    '''
    # Display
    rcParams['figure.figsize'] = [30, 15]
    fig, ax = plt.subplots(2, 3)
    ax[0][0].imshow(average), ax[0][0].set_title('average blur'), ax[0][0].set(xticks=[], yticks=[])
    ax[0][1].imshow(median), ax[0][1].set_title('median blur'), ax[0][1].set(xticks=[], yticks=[])
    ax[0][2].imshow(gaussian), ax[0][2].set_title('Gaussian blur'), ax[0][2].set(xticks=[], yticks=[])

    ax[1][0].imshow(average[:400, 800:]), ax[1][0].set_title('avg right zoom'), ax[1][0].set(xticks=[], yticks=[])
    ax[1][1].imshow(median[:400, 800:]), ax[1][1].set_title('med right zoom'), ax[1][1].set(xticks=[], yticks=[])
    ax[1][2].imshow(gaussian[:400, 800:]), ax[1][2].set_title('gau right zoom'), ax[1][2].set(xticks=[], yticks=[])

    plt.show()
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ker_vr = np.matrix( [[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
    ker_vl = np.flip(ker_vr, 1)
    ker_hu = np.matrix( [[1, 2, 1],
                            [0, 0, 0],
                           [-1,-2,-1]])
    ker_hd = np.flip(ker_hu, 0)

    edges_vr = cv2.filter2D(src=img, ddepth=-1, kernel=ker_vr)
    edges_vl = cv2.filter2D(src=img, ddepth=-1, kernel=ker_vl)
    edges_ver = cv2.add(edges_vr, edges_vl)

    edges_hu = cv2.filter2D(src=img, ddepth=-1, kernel=ker_hu)
    edges_hd = cv2.filter2D(src=img, ddepth=-1, kernel=ker_hd)
    edges_hor = cv2.add(edges_hu, edges_hd)

    edges_all = cv2.add(edges_ver, edges_hor)

    # Turn it into black and white
    (thresh, blackAndWhiteImage) = cv2.threshold(edges_all, 127, 255, cv2.THRESH_BINARY)
    # blackAndWhiteImage = cv2.adaptiveThreshold(edges_all, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
    # blackAndWhiteImage = cv2.adaptiveThreshold(edges_all, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 2)

    # Display
    #rcParams['figure.figsize'] = [30, 15]
    #fig, ax = plt.subplots(2, 3)
    #ax[0][0].imshow(edges_vr, cmap='gray'), ax[0][0].set_title('vertical Sobel from Right'), ax[0][0].set(xticks=[], yticks=[])
    #ax[0][1].imshow(edges_vl, cmap='gray'), ax[0][1].set_title('vertical Sobel from Left'), ax[0][1].set(xticks=[], yticks=[])
    #ax[0][2].imshow(edges_ver, cmap='gray'), ax[0][2].set_title('vertical Sobel from R&L'), ax[0][2].set(xticks=[], yticks=[])
    #ax[1][0].imshow(edges_hu, cmap='gray'), ax[1][0].set_title('vertical Sobel from Up'), ax[1][0].set(xticks=[], yticks=[])
    #ax[1][1].imshow(edges_hd, cmap='gray'), ax[1][1].set_title('vertical Sobel from Down'), ax[1][1].set(xticks=[], yticks=[])
    #ax[1][2].imshow(edges_hor, cmap='gray'), ax[1][2].set_title('vertical Sobel from U&D'), ax[1][2].set(xticks=[], yticks=[])

    gs_kw = dict(width_ratios=[2, 1.2, 1, 1], height_ratios=[1, 1])
    fig, axd = plt.subplot_mosaic([['up_left', 'up_right1', 'up_right2', 'up_right3'],
                                   ['do_left', 'do_right1', 'do_right2', 'do_right3']],
                                  gridspec_kw=gs_kw, figsize=(10, 4),
                                  layout="constrained")
    axd['up_left'].imshow(edges_all, cmap='gray'), \
        axd['up_left'].set_title('Sobel Edge Detection - Gray'), \
        axd['up_left'].set(xticks=[], yticks=[])
    axd['do_left'].imshow(cv2.bitwise_not(blackAndWhiteImage), cmap='gray'), \
        axd['do_left'].set_title('Sobel Edge Detection - B&W'), \
        axd['do_left'].set(xticks=[], yticks=[])
    axd['up_right1'].imshow(edges_ver, cmap='gray'), \
        axd['up_right1'].set_title('Sobel Edge Vertical'), \
        axd['up_right1'].set(xticks=[], yticks=[])
    axd['up_right2'].imshow(edges_vr, cmap='gray'), \
        axd['up_right2'].set_title('Vertical from Right'), \
        axd['up_right2'].set(xticks=[], yticks=[])
    axd['up_right3'].imshow(edges_vl, cmap='gray'), \
        axd['up_right3'].set_title('Vertical from Left'), \
        axd['up_right3'].set(xticks=[], yticks=[])
    axd['do_right1'].imshow(edges_hor, cmap='gray'), \
        axd['do_right1'].set_title('Sobel Edge Horizontal'), \
        axd['do_right1'].set(xticks=[], yticks=[])
    axd['do_right2'].imshow(edges_hu, cmap='gray'), \
        axd['do_right2'].set_title('Horizontal from Up'), \
        axd['do_right2'].set(xticks=[], yticks=[])
    axd['do_right3'].imshow(edges_hd, cmap='gray'), \
        axd['do_right3'].set_title('Horizontal from Down'), \
        axd['do_right3'].set(xticks=[], yticks=[])

    plt.show()

    return 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test_environment()
    # basics_of_image_processing()
    # from_color_to_black_and_white()
    filters()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
